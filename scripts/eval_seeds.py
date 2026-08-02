#!/usr/bin/env python3
"""Multi-seed comparison: does training *through* the shield actually help?

    python3 scripts/eval_seeds.py --seeds 5 --episodes 200

Why this exists. `eval_policies.py` evaluates one trained policy over many episodes, which
measures **evaluation variance** — how much the score moves with the luck of the scenes. It
says nothing about **training variance**: how much the score moves with the luck of the
training run (weight initialisation, exploration sampling). A single training run per
configuration cannot distinguish a real effect from one lucky initialisation, and deep RL is
notorious for exactly this failure (Henderson et al., *Deep Reinforcement Learning That
Matters*, 2018, found seed-to-seed spread routinely exceeding algorithm-to-algorithm spread).

The claim under test is about a *method*, so the unit of replication has to be the training
run, not the episode. This script trains nothing — it evaluates N independently seeded
policies per configuration and compares the distributions across seeds.

Every policy is scored on the **same** evaluation scenes (episode seeds are fixed, not
derived from the training seed), so scene difficulty is held constant and cannot contribute
to a difference between configurations.
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
)


def load_ppo(path: Path):
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def welch(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch's t-test plus the 95% CI half-width of the difference in means.

    Welch rather than Student because the two configurations have no reason to share a
    variance — a shielded policy's scores may well be more tightly clustered.
    """
    from scipy import stats

    t, p = stats.ttest_ind(a, b, equal_var=False)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                    + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    return float(t), float(p), float(stats.t.ppf(0.975, df) * se)


def run(scene_name: str, make_scenes, cfg: DriveNavConfig, seeds: int,
        episodes: int, models: Path) -> None:
    groups = {
        "PPO (raw) + shield at eval": "ppo_raw_s",
        "PPO trained through shield": "ppo_shielded_s",
    }

    print(f"\n=== {scene_name}: {seeds} training seeds x {episodes} episodes ===")
    scores: dict[str, np.ndarray] = {}

    for label, prefix in groups.items():
        per_seed, collisions = [], []
        for s in range(seeds):
            path = models / f"{prefix}{s}"
            if not path.with_suffix(".zip").exists():
                print(f"  missing {path}.zip — run scripts/train_ppo.py --seed {s}")
                return
            stats_ = evaluate(DriveNavEnv(make_scenes(), cfg), load_ppo(path),
                              n_episodes=episodes, seed=0)
            per_seed.append(stats_["success_rate"])
            collisions.append(stats_["collisions"])
        scores[label] = np.array(per_seed)
        pretty = "  ".join(f"{v:.1%}" for v in per_seed)
        print(f"{label:<28} {pretty}   mean {np.mean(per_seed):.1%} "
              f"+/- {np.std(per_seed, ddof=1):.1%} sd   collisions {sum(collisions)}")

    a, b = scores["PPO trained through shield"], scores["PPO (raw) + shield at eval"]
    t, p, ci = welch(a, b)
    diff = a.mean() - b.mean()
    print(f"\n  difference (in-loop - at-eval): {diff:+.1%}  95% CI [{diff - ci:+.1%}, "
          f"{diff + ci:+.1%}]   Welch t={t:.2f}, p={p:.3f}")
    print(f"  -> {'SIGNIFICANT at 0.05' if p < 0.05 else 'not significant at 0.05'}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--models", type=Path, default=Path("models"))
    p.add_argument("--no-kitti", action="store_true")
    args = p.parse_args()

    cfg = replace(DriveNavConfig(), use_shield=True)
    run("synthetic scenes", lambda: SyntheticScenes(), cfg,
        args.seeds, args.episodes, args.models)

    if not args.no_kitti:
        from kitti_nav.kitti import KittiDrive

        drive = KittiDrive("2011_09_26", "0009")
        run("KITTI (real) scenes", lambda: KittiScenes(drive=drive), cfg,
            args.seeds, args.episodes, args.models)
    return 0


if __name__ == "__main__":
    sys.exit(main())
