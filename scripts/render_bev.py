#!/usr/bin/env python3
"""Render a KITTI frame as camera view + lidar BEV occupancy with the shield's verdict.

    python3 scripts/render_bev.py --frame 0 --out docs/bev.png
    python3 scripts/render_bev.py --speed-profile docs/speed_profile.png

The BEV panel shows what the planner actually sees: occupied cells from the lidar, the
vehicle footprint as the shield models it (the disc cover, not the rectangle), and the
distance-to-obstacle field the braking check queries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.bev import BEVConfig, BEVGrid            # noqa: E402
from kitti_nav.kitti import KittiDrive                  # noqa: E402
from kitti_nav.vehicle import (                         # noqa: E402
    VehicleConfig,
    footprint_discs,
    max_safe_speed,
    stopping_distance,
)


def render_frame(drive: KittiDrive, i: int, out: Path, vcfg: VehicleConfig) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Rectangle

    cfg = BEVConfig()
    grid = BEVGrid.from_scan(drive.velodyne(i), cfg)
    state = drive.vehicle_state_in_lidar(speed=float(drive.speeds[i]))
    permitted = max_safe_speed(grid, vcfg, state=state)

    fig, (ax_img, ax_bev) = plt.subplots(2, 1, figsize=(12, 11),
                                         gridspec_kw={"height_ratios": [1, 2.2]})

    ax_img.imshow(drive.rgb_left(i))
    ax_img.set_title(f"frame {i} — left colour camera")
    ax_img.axis("off")

    # Distance field as background, occupancy on top. Forward (+x) is up, left (+y) is left,
    # so the plot reads like a map from the driver's seat.
    dist = np.where(np.isinf(grid.distance_field), np.nan, grid.distance_field)
    extent = [cfg.y_max, cfg.y_min, cfg.x_min, cfg.x_max]      # y reversed: +y is left
    ax_bev.imshow(dist, origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=15,
                  aspect="equal")
    occ = np.ma.masked_where(grid.occupancy == 0, grid.occupancy)
    ax_bev.imshow(occ, origin="lower", extent=extent, cmap="autumn_r", vmin=0, vmax=1,
                  aspect="equal", interpolation="nearest")

    centres, radius = footprint_discs(state, vcfg)
    for cx, cy in centres:
        ax_bev.add_patch(Circle((cy, cx), radius, fill=False, ec="k",
                                lw=0.8, ls=":", alpha=0.7))
    ax_bev.add_patch(Rectangle((state.y - vcfg.width / 2, state.x - vcfg.rear_overhang),
                               vcfg.width, vcfg.length, fill=False, ec="lime", lw=2))

    # How far the car would travel before stopping, at the speed actually driven.
    d_stop = stopping_distance(state.v, vcfg)
    ax_bev.plot([state.y, state.y], [state.x + vcfg.front_overhang,
                                     state.x + vcfg.front_overhang + d_stop],
                "g-", lw=2.5, alpha=0.8,
                label=f"braking envelope at {state.v:.1f} m/s ({d_stop:.0f} m)")

    ax_bev.set_xlabel("y, left (m)")
    ax_bev.set_ylabel("x, forward (m)")
    ax_bev.set_title(f"lidar BEV occupancy — driven {state.v:.1f} m/s, "
                     f"shield permits {permitted:.1f} m/s")
    ax_bev.legend(loc="upper right")
    ax_bev.grid(alpha=0.15)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  (driven {state.v:.1f} m/s, permitted {permitted:.1f} m/s)")


def render_speed_profile(drive: KittiDrive, out: Path, vcfg: VehicleConfig) -> None:
    """Shield-permitted speed vs the speed a human actually drove, over the whole drive."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = drive.n_velodyne
    state = drive.vehicle_state_in_lidar()
    permitted = np.array([max_safe_speed(BEVGrid.from_scan(drive.velodyne(i)), vcfg,
                                         state=state) for i in range(n)])
    driven = drive.speeds[:n]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(permitted, color="tab:blue", lw=1.2, label="shield-permitted speed")
    ax.plot(driven, color="k", lw=1.6, label="speed actually driven (OXTS)")
    binding = permitted < driven
    ax.fill_between(np.arange(n), 0, 40, where=binding, color="tab:red", alpha=0.25,
                    label=f"shield would slow the driver ({binding.sum()} frames)")
    ax.set_xlabel("frame")
    ax.set_ylabel("speed (m/s)")
    ax.set_ylim(0, 40)
    ax.set_title("Braking-shield speed limit vs a real human driver")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"wrote {out}  ({binding.sum()}/{n} frames where the shield binds, "
          f"median permitted {np.median(permitted):.1f} m/s)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("docs/bev.png"))
    p.add_argument("--speed-profile", type=Path, default=None,
                   help="also write the whole-drive permitted-vs-driven speed plot here")
    p.add_argument("--max-speed", type=float, default=35.0,
                   help="raise the vehicle speed cap so geometry, not the cap, is what binds")
    args = p.parse_args()

    drive = KittiDrive(args.date, args.drive)
    vcfg = VehicleConfig(max_speed=args.max_speed)
    render_frame(drive, args.frame, args.out, vcfg)
    if args.speed_profile:
        render_speed_profile(drive, args.speed_profile, vcfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
