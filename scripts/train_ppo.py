#!/usr/bin/env python3
"""Train a PPO driving policy, optionally *through* the braking safety shield.

    python3 scripts/train_ppo.py --steps 400000 --out models/ppo_raw
    python3 scripts/train_ppo.py --steps 400000 --shield --out models/ppo_shielded

Training through the shield (rather than only wrapping it at evaluation time) was the
capstone result in `gsplat-rt`: the shielded-in-the-loop policy dominated the raw one on
every axis, because it learns against the dynamics it will actually be deployed with instead
of being surprised by a filter at test time.

MLP-PPO on a handful of observations trains faster on CPU than on a GPU here — the batches
are far too small to amortise kernel launches — so the device is pinned to CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.nav_env import DriveNavConfig, SyntheticScenes   # noqa: E402


def make_env(shield: bool, seed: int):
    from stable_baselines3.common.monitor import Monitor

    from kitti_nav.nav_gym import NavGymEnv

    def _init():
        cfg = DriveNavConfig(use_shield=shield)
        env = Monitor(NavGymEnv(SyntheticScenes(), cfg))
        env.reset(seed=seed)
        return env
    return _init


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", type=int, default=400_000)
    p.add_argument("--shield", action="store_true",
                   help="filter every action through the braking shield during training")
    p.add_argument("--out", type=Path, default=Path("models/ppo"))
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([make_env(args.shield, args.seed + i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                n_steps=512, batch_size=1024, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.005, learning_rate=3e-4)

    print(f"training {'WITH' if args.shield else 'WITHOUT'} the shield "
          f"for {args.steps:,} steps on {args.n_envs} envs")
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, progress_bar=False)
    elapsed = time.perf_counter() - t0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"trained in {elapsed / 60:.1f} min -> {args.out}.zip")
    return 0


if __name__ == "__main__":
    sys.exit(main())
