#!/usr/bin/env python3
"""Does training on real KITTI geometry beat synthetic transfer — on held-out road?

    python3 scripts/eval_kitti_trained.py --episodes 200 \
        --model synthetic=models/ppo_shielded.zip --model kitti=models/ppo_kitti_shielded.zip

    # cross-drive: score 0009-trained policies on a whole different drive they never saw
    python3 scripts/eval_kitti_trained.py --drive 0093 --all-frames --episodes 200 \
        --model synthetic=models/ppo_shielded.zip --model kitti-0009=models/ppo_kitti_shielded.zip

Every published policy here was trained on synthetic obstacle fields and *transferred* to KITTI.
The obvious untried lever is to train on the drive's own recorded occupancy instead. This scores
each policy on the **held-out** half of a contiguous frame split (`kitti_frame_split`) — the
stretch of the drive no policy trained on — so the comparison is genuine generalisation, not
memorisation. With `--all-frames` on a *different* drive it becomes a cross-drive test: nothing
was trained on any of that drive, so the whole of it is fair to score on. Each model is run three
ways, the same arc as `eval_policies.py`:

  * **raw** — the policy alone;
  * **+ shield at eval** — the braking shield bolted on only at evaluation;
  * (the policy trained through the shield is scored as its own "raw", since the shield is then
    already part of what it learned).

Success and collisions are shown together. The shield column should stay at **0 collisions**
regardless of what the policy learned — that is the whole point of a runtime certificate — while
the interesting question is whether KITTI-trained *success* beats synthetic-trained success on
road neither has seen.
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
    evaluate,
    gap_following_policy,
    kitti_frame_split,
)


def load_ppo(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", type=float, default=0.3,
                   help="must match the split the policies were trained with")
    p.add_argument("--all-frames", action="store_true",
                   help="score on the whole drive, not a held-out split — the honest choice for "
                        "a cross-drive test, where no policy was trained on any of this drive")
    p.add_argument("--fused-window", type=int, default=5)
    p.add_argument("--model", action="append", default=[], metavar="NAME=PATH",
                   help="a named policy to score (repeatable), e.g. synthetic=models/ppo.zip")
    args = p.parse_args()

    from kitti_nav.kitti import KittiDrive
    from kitti_nav.mapping import MapConfig

    drive = KittiDrive(args.date, args.drive)
    mapdesc = f"fused {args.fused_window} scans" if args.fused_window > 1 else "single scan"
    if args.all_frames:
        eval_frames = np.arange(drive.n_velodyne)
        print(f"drive {args.drive}: scoring on ALL {drive.n_velodyne} frames "
              f"(cross-drive — nothing trained here), {mapdesc}, {args.episodes} episodes\n")
    else:
        train, eval_frames = kitti_frame_split(drive.n_velodyne, holdout=args.holdout)
        print(f"drive {args.drive}: {drive.n_velodyne} frames -> "
              f"{len(train)} train / {len(eval_frames)} held-out (last {args.holdout:.0%})\n"
              f"scoring on held-out, {mapdesc}, {args.episodes} episodes\n")

    # window <= 1 means the single-scan map (no fusion). This matters cross-drive: fusing a fast
    # drive's wide-baseline window can paint phantom geometry into the ego's own spawn region
    # (a ground-removal-under-multiple-viewpoints artifact), which single-scan avoids.
    map_config = MapConfig(window=args.fused_window) if args.fused_window > 1 else None
    scenes = KittiScenes(drive=drive, frames=eval_frames, map_config=map_config,
                         grid_cache_size=len(eval_frames) + 1)
    base = DriveNavConfig()
    shielded = replace(base, use_shield=True)

    def score(policy, cfg) -> str:
        s = evaluate(DriveNavEnv(scenes, cfg), policy, n_episodes=args.episodes, seed=args.seed)
        return f"{s['success_rate']:>5.0%} / {s['collisions']:>3} coll"

    print(f"{'policy':<28}{'raw':>18}{'+ shield at eval':>20}")
    print(f"{'gap-following (baseline)':<28}"
          f"{score(lambda o: gap_following_policy(o, base), base):>18}"
          f"{score(lambda o: gap_following_policy(o, shielded), shielded):>20}")

    for spec in args.model:
        name, _, path = spec.partition("=")
        policy = load_ppo(Path(path))
        print(f"{name:<28}{score(policy, base):>18}{score(policy, shielded):>20}")

    print("\nThe shield column stays at 0 collisions whatever the policy learned. The comparison "
          "that matters is\nwhether KITTI-trained success beats synthetic transfer on road "
          "neither policy was trained on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
