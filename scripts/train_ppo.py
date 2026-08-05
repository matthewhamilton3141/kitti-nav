#!/usr/bin/env python3
"""Train a PPO driving policy, optionally *through* the braking safety shield.

    python3 scripts/train_ppo.py --steps 400000 --out models/ppo_raw
    python3 scripts/train_ppo.py --steps 400000 --shield --out models/ppo_shielded

    # train on real recorded geometry instead of synthetic obstacle fields
    python3 scripts/train_ppo.py --scenes kitti-fused --shield --out models/ppo_kitti_shielded

Training through the shield (rather than only wrapping it at evaluation time) was the
capstone result in `gsplat-rt`: the shielded-in-the-loop policy dominated the raw one on
every axis, because it learns against the dynamics it will actually be deployed with instead
of being surprised by a filter at test time.

**Scene source.** The default `synthetic` trains on random obstacle fields — every published
policy here did — so the KITTI columns are a pure transfer test. `kitti` / `kitti-fused` instead
train on the drive's own recorded occupancy (single scan / a fused window), restricted to the
**train half** of a contiguous frame split (`kitti_frame_split`); the held-out half is what
`eval_kitti_trained.py` scores on, so training never sees the road it is graded on.

MLP-PPO on a handful of observations trains faster on CPU than on a GPU here — the batches
are far too small to amortise kernel launches — so the device is pinned to CPU.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.nav_env import (                          # noqa: E402
    DriveNavConfig,
    KittiScenes,
    SyntheticScenes,
    kitti_frame_split,
)


def make_scenes(args):
    """Build the training scene source (synthetic, or a drive's train-split occupancy)."""
    if args.scenes == "synthetic":
        return SyntheticScenes()

    from kitti_nav.kitti import KittiDrive
    from kitti_nav.mapping import MapConfig

    drive = KittiDrive(args.date, args.drive)
    train, _ = kitti_frame_split(drive.n_velodyne, holdout=args.holdout,
                                 stride=args.train_stride)
    map_config = (MapConfig(window=args.fused_window)
                  if args.scenes == "kitti-fused" and args.fused_window > 1 else None)
    # A generous grid cache so the strided train pool stays resident (fusing + the distance
    # transform per frame is what would otherwise dominate a KITTI training run).
    return KittiScenes(drive=drive, frames=train, map_config=map_config,
                       grid_cache_size=max(len(train) + 1, 64))


def make_env(args, seed: int):
    from stable_baselines3.common.monitor import Monitor

    from kitti_nav.nav_gym import NavGymEnv

    def _init():
        cfg = DriveNavConfig(use_shield=args.shield)
        env = Monitor(NavGymEnv(make_scenes(args), cfg))
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
    p.add_argument("--scenes", choices=("synthetic", "kitti", "kitti-fused"),
                   default="synthetic", help="training geometry (default: synthetic fields)")
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--fused-window", type=int, default=5,
                   help="scans fused per map for --scenes kitti-fused")
    p.add_argument("--holdout", type=float, default=0.3,
                   help="fraction of the drive held out for evaluation (trained on the rest)")
    p.add_argument("--train-stride", type=int, default=2,
                   help="subsample the KITTI train frames by this stride (cache-friendly)")
    args = p.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv

    venv = SubprocVecEnv([make_env(args, args.seed + i) for i in range(args.n_envs)])
    model = PPO("MlpPolicy", venv, verbose=1, seed=args.seed, device="cpu",
                n_steps=512, batch_size=1024, gae_lambda=0.95, gamma=0.99,
                ent_coef=0.005, learning_rate=3e-4)

    print(f"training {'WITH' if args.shield else 'WITHOUT'} the shield on {args.scenes} "
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
