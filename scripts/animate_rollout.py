#!/usr/bin/env python3
"""Animate one shielded-policy episode driving across a real KITTI scan.

Unlike render_bev.py --rollout (which draws every episode's path as a static swept
bundle), this plays a *single* episode as a movie: the vehicle footprint, rotated to
its heading, drives frame-by-frame across the frozen lidar occupancy toward its goal.
The footprint is coloured by speed and flashes when the braking shield intervenes.

    python3 scripts/animate_rollout.py --frame 60 --model models/ppo_kitti_shielded \
        --out docs/drive.mp4

The world is one recorded scan (the car moves; the scene is static). Use --search to
scan seeds for an episode that reaches the goal.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.kitti import KittiDrive  # noqa: E402
from kitti_nav.nav_env import DriveNavConfig, DriveNavEnv, KittiScenes  # noqa: E402


def run_episode(env: DriveNavEnv, model, cfg: DriveNavConfig, seed: int):
    """Roll one deterministic episode, returning per-step telemetry."""
    obs, _ = env.reset(seed)
    xs, ys, yaws, vs, shielded = [env.state.x], [env.state.y], [env.state.yaw], [env.state.v], [False]
    info = {"reached": False, "collided": False, "distance": np.inf}
    for _ in range(cfg.max_steps):
        obs, _, terminated, truncated, info = env.step(model.predict(obs, deterministic=True)[0])
        xs.append(env.state.x)
        ys.append(env.state.y)
        yaws.append(env.state.yaw)
        vs.append(env.state.v)
        shielded.append(bool(info["shield_intervened"]))
        if terminated or truncated:
            break
    return (np.array(xs), np.array(ys), np.array(yaws), np.array(vs),
            np.array(shielded), info)


def footprint_polygon(x: float, y: float, yaw: float, v) -> np.ndarray:
    """Vehicle footprint corners in lidar frame (x forward, y left), rotated to yaw."""
    # body frame: reference point is the rear axle; box spans -rear_overhang..length-rear_overhang
    fx0, fx1 = -v.rear_overhang, v.length - v.rear_overhang
    fy0, fy1 = -v.width / 2, v.width / 2
    corners = np.array([[fx0, fy0], [fx1, fy0], [fx1, fy1], [fx0, fy1]])
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    world = corners @ rot.T + np.array([x, y])
    # plot axes are (y-left horizontal, x-forward vertical), so return (col=y, row=x)
    return np.column_stack([world[:, 1], world[:, 0]])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--frame", type=int, default=60)
    p.add_argument("--model", type=Path, default=Path("models/ppo_kitti_shielded"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--search", type=int, default=0,
                   help="scan this many seeds for an episode that reaches the goal")
    p.add_argument("--out", type=Path, default=Path("docs/drive.mp4"))
    p.add_argument("--fps", type=int, default=12)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    from matplotlib.patches import Polygon
    from stable_baselines3 import PPO

    cfg = DriveNavConfig(use_shield=True)
    drive = KittiDrive(args.date, args.drive)
    scenes = KittiScenes(drive=drive, frames=np.array([args.frame]))
    env = DriveNavEnv(scenes, cfg)
    model = PPO.load(str(args.model), device="cpu")

    seed = args.seed
    xs, ys, yaws, vs, shielded, info = run_episode(env, model, cfg, seed)
    for extra in range(args.search):
        if info["reached"]:
            break
        seed = args.seed + 1 + extra
        xs, ys, yaws, vs, shielded, info = run_episode(env, model, cfg, seed)
    print(f"seed {seed}: {len(xs)} steps, reached={info['reached']}, "
          f"collided={info['collided']}, shield fired {int(shielded.sum())}x")

    bev = scenes.bev
    occ = env.scenes.sample(np.random.default_rng(0)).grid.occupancy
    goal = env.scene.goal

    fig, ax = plt.subplots(figsize=(8, 10))
    masked = np.ma.masked_where(occ == 0, np.ones(occ.shape))
    ax.imshow(masked, origin="lower", extent=[bev.y_min, bev.y_max, bev.x_min, bev.x_max],
              cmap="autumn_r", vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax.invert_xaxis()   # +y (left) on the visual left; matches the true-y car & trail
    ax.scatter([goal[1]], [goal[0]], marker="*", s=260, c="k", zorder=6)
    trail, = ax.plot([], [], "-", color="tab:green", lw=1.4, alpha=0.7, zorder=4)
    car = Polygon(footprint_polygon(xs[0], ys[0], yaws[0], cfg.vehicle),
                  closed=True, zorder=5)
    ax.add_patch(car)
    hud = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                  fontsize=11, family="monospace",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_xlabel("y, left (m)")
    ax.set_ylabel("x, forward (m)")
    ax.set_title(f"Shielded PPO driving real KITTI geometry (frame {args.frame})")
    ax.grid(alpha=0.15)

    vmax = cfg.vehicle.max_speed

    def update(k: int):
        car.set_xy(footprint_polygon(xs[k], ys[k], yaws[k], cfg.vehicle))
        # green = fast, red = slow/stopped; orange edge when the shield is braking
        frac = np.clip(vs[k] / vmax, 0, 1)
        car.set_facecolor((1 - frac, 0.35 + 0.45 * frac, 0.2))
        car.set_edgecolor("orange" if shielded[k] else "black")
        car.set_linewidth(2.5 if shielded[k] else 1.2)
        trail.set_data(ys[:k + 1], xs[:k + 1])
        tag = "  SHIELD BRAKING" if shielded[k] else ""
        hud.set_text(f"t = {k * cfg.vehicle.dt:4.1f} s\n"
                     f"v = {vs[k]:4.1f} m/s{tag}")
        return car, trail, hud

    anim = FuncAnimation(fig, update, frames=len(xs), interval=1000 / args.fps, blit=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gif":
        anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    else:
        anim.save(str(args.out), writer=FFMpegWriter(fps=args.fps, bitrate=2400))
    print(f"wrote {args.out}  ({len(xs)} frames @ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
