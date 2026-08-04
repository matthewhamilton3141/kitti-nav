#!/usr/bin/env python3
"""Compare every policy on synthetic scenes and on real recorded KITTI geometry.

    python3 scripts/eval_policies.py --episodes 200

Reports the full arc the way `gsplat-rt`'s nav flagship did: heuristic -> heuristic+shield ->
learned -> learned+shield -> learned *trained through* the shield. Success rate and collision
count are the two numbers that matter, and they trade off against each other, so both are
always shown together.

The KITTI column is the transfer test: every policy is trained only on synthetic obstacle
fields, then evaluated on occupancy grids built from real Velodyne scans.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.nav_env import (                     # noqa: E402
    DriveNavConfig,
    DriveNavEnv,
    KittiScenes,
    SyntheticScenes,
    evaluate,
    gap_following_policy,
)


def load_ppo(path: Path):
    """Wrap a saved stable-baselines3 policy as a plain `obs -> action` callable."""
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def scene_sources(args):
    """Scene sets to score every policy against, in increasing order of realism.

    The two KITTI columns are the same recorded geometry seen two ways: one frozen scan, and
    a window of scans fused through the drive's poses. The fused map is *harder* rather than
    better-behaved — it contains obstacles a single scan is simply blind to — so showing both
    is the honest way to report what connecting odometry to the map costs the planner.
    """
    sources = {"synthetic": lambda: SyntheticScenes()}
    if not args.no_kitti:
        from kitti_nav.kitti import KittiDrive
        from kitti_nav.mapping import MapConfig

        drive = KittiDrive(args.date, args.drive)
        w = args.fused_window
        sources["KITTI (single scan)"] = lambda: KittiScenes(drive=drive)
        if w > 1:
            sources[f"KITTI (fused, {w} scans)"] = lambda: KittiScenes(
                drive=drive, map_config=MapConfig(window=w))
            # Free-space carving, two ways: with unknown cells assumed free (isolates what
            # carving's obstacle removal does to the planner) and with unknown cells blocking
            # (adds the honest reading that unobserved space is not certified drivable). The
            # gap between the two is the price of that honesty.
            if args.carve:
                sources[f"KITTI (carved, {w} scans)"] = lambda: KittiScenes(
                    drive=drive, map_config=MapConfig(window=w, carve=True))
                sources[f"KITTI (carved+unknown, {w} scans)"] = lambda: KittiScenes(
                    drive=drive, map_config=MapConfig(window=w, carve=True),
                    unknown_blocks=True)
    return sources


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--no-kitti", action="store_true", help="synthetic scenes only")
    p.add_argument("--fused-window", type=int, default=5,
                   help="scans fused for the accumulated-map column; 1 disables it")
    p.add_argument("--carve", action="store_true",
                   help="add free-space-carved KITTI columns (unknown free, then blocking)")
    args = p.parse_args()

    base = DriveNavConfig()
    shielded = replace(base, use_shield=True)

    # (label, config used at evaluation time, policy factory)
    rows: list[tuple[str, DriveNavConfig, object]] = [
        ("gap-following", base, lambda: (lambda o: gap_following_policy(o, base))),
        ("gap-following + shield", shielded,
         lambda: (lambda o: gap_following_policy(o, shielded))),
    ]
    raw, shield_trained = args.models / "ppo_raw", args.models / "ppo_shielded"
    if raw.with_suffix(".zip").exists():
        rows.append(("PPO (raw)", base, lambda: load_ppo(raw)))
        rows.append(("PPO (raw) + shield at eval", shielded, lambda: load_ppo(raw)))
    if shield_trained.with_suffix(".zip").exists():
        rows.append(("PPO trained through shield", shielded, lambda: load_ppo(shield_trained)))

    for scene_name, make_scenes in scene_sources(args).items():
        print(f"\n=== {scene_name} scenes, {args.episodes} episodes ===")
        print(f"{'policy':<30} {'success':>8} {'collisions':>11} {'reward':>8} {'steps':>7}")
        print("-" * 68)
        for label, cfg, make_policy in rows:
            env = DriveNavEnv(make_scenes(), cfg)
            stats = evaluate(env, make_policy(), n_episodes=args.episodes)
            print(f"{label:<30} {stats['success_rate']:>7.0%} {stats['collisions']:>11} "
                  f"{stats['mean_reward']:>8.1f} {stats['mean_steps_when_reached']:>7.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
