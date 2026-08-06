#!/usr/bin/env python3
"""Replay a real recorded KITTI drive as a movie: camera main view + BEV mini-map.

The main panel is the left colour camera streaming through the recorded street, so the
*world* actually moves. The mini-map on the right is the lidar BEV the planner sees —
occupancy over a distance field, the ego footprint fixed at the sensor origin, and the
braking envelope at the speed the human actually drove. The mini-map title / HUD reports
the shield-permitted speed each frame and flags the frames where the shield would slow
the driver.

    python3 scripts/animate_drive.py --drive 0009 --start 40 --stop 200 --out docs/drive_scene.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.bev import BEVConfig, BEVGrid  # noqa: E402
from kitti_nav.kitti import KittiDrive  # noqa: E402
from kitti_nav.vehicle import (  # noqa: E402
    VehicleConfig,
    max_safe_speed,
    stopping_distance,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--start", type=int, default=40)
    p.add_argument("--stop", type=int, default=200)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--out", type=Path, default=Path("docs/drive_scene.mp4"))
    p.add_argument("--fps", type=int, default=10)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.patches import Rectangle

    drive = KittiDrive(args.date, args.drive)
    vcfg = VehicleConfig()
    bcfg = BEVConfig()
    n = drive.n_velodyne
    stop = min(args.stop, n)
    frames = list(range(args.start, stop, args.stride))
    # imshow draws column 0 at extent[0]. The rasteriser puts world y_min in column 0, so
    # extent[0] must be y_min for data-x to equal true world y (else the BEV is L/R mirrored).
    # We then invert the x-axis below so +y (left) still shows on the visual left of the panel.
    extent = [bcfg.y_min, bcfg.y_max, bcfg.x_min, bcfg.x_max]

    # A fixed ego pose (sensor origin); only its driven speed changes frame to frame.
    base_state = drive.vehicle_state_in_lidar()

    def frame_data(i: int):
        grid = BEVGrid.from_scan(drive.velodyne(i), bcfg)
        state = drive.vehicle_state_in_lidar(speed=float(drive.speeds[i]))
        permitted = max_safe_speed(grid, vcfg, state=state)
        dist = np.where(np.isinf(grid.distance_field), np.nan, grid.distance_field)
        occ = np.ma.masked_where(grid.occupancy == 0, grid.occupancy)
        return grid, state, permitted, dist, occ

    grid, state, permitted, dist, occ = frame_data(frames[0])

    fig, (ax_img, ax_bev) = plt.subplots(
        1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.4, 1]})

    im = ax_img.imshow(drive.rgb_left(frames[0]), aspect="auto")
    ax_img.axis("off")
    cam_title = ax_img.set_title(f"{args.date} drive {args.drive} — frame {frames[0]}",
                                 fontsize=13)

    dist_im = ax_bev.imshow(dist, origin="lower", extent=extent, cmap="Blues",
                            vmin=0, vmax=15, aspect="equal")
    occ_im = ax_bev.imshow(occ, origin="lower", extent=extent, cmap="autumn_r",
                           vmin=0, vmax=1, aspect="equal", interpolation="nearest")
    ax_bev.add_patch(Rectangle((base_state.y - vcfg.width / 2,
                                base_state.x - vcfg.rear_overhang),
                               vcfg.width, vcfg.length, fill=False, ec="lime", lw=2,
                               zorder=5))
    d_stop = stopping_distance(state.v, vcfg)
    (envelope,) = ax_bev.plot(
        [base_state.y, base_state.y],
        [base_state.x + vcfg.front_overhang, base_state.x + vcfg.front_overhang + d_stop],
        "-", lw=3, alpha=0.85, zorder=6)
    hud = ax_bev.text(0.03, 0.97, "", transform=ax_bev.transAxes, va="top", ha="left",
                      fontsize=11, family="monospace",
                      bbox=dict(boxstyle="round", fc="white", alpha=0.85))
    ax_bev.set_xlabel("y, left (m)")
    ax_bev.set_ylabel("x, forward (m)")
    ax_bev.set_title("lidar BEV — the planner's mini-map", fontsize=13)
    ax_bev.grid(alpha=0.15)
    ax_bev.invert_xaxis()   # +y (left) on the visual left, matching the camera's left
    fig.tight_layout()

    def draw(k: int):
        i = frames[k]
        grid, state, permitted, dist, occ = frame_data(i)
        im.set_data(drive.rgb_left(i))
        cam_title.set_text(f"{args.date} drive {args.drive} — frame {i}")
        dist_im.set_data(dist)
        occ_im.set_data(occ)
        d_stop = stopping_distance(state.v, vcfg)
        envelope.set_data([base_state.y, base_state.y],
                          [base_state.x + vcfg.front_overhang,
                           base_state.x + vcfg.front_overhang + d_stop])
        binding = permitted < state.v - 1e-3
        envelope.set_color("red" if binding else "lime")
        tag = "  SHIELD SLOWS" if binding else ""
        hud.set_text(f"driven    {state.v:4.1f} m/s\n"
                     f"permitted {permitted:4.1f} m/s{tag}")
        return im, dist_im, occ_im, envelope, cam_title, hud

    anim = FuncAnimation(fig, draw, frames=len(frames),
                         interval=1000 / args.fps, blit=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.suffix == ".gif":
        anim.save(str(args.out), writer=PillowWriter(fps=args.fps))
    else:
        anim.save(str(args.out), writer=FFMpegWriter(fps=args.fps, bitrate=3200))
    print(f"wrote {args.out}  ({len(frames)} frames [{frames[0]}..{frames[-1]}] "
          f"@ {args.fps} fps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
