#!/usr/bin/env python3
"""What does reasoning about obstacle motion change? Static vs dynamic permitted speed on KITTI.

    python3 scripts/eval_dynamic_shield.py
    python3 scripts/eval_dynamic_shield.py --forward-only   # only movers ahead of the ego

The static shield certifies a braking stop against occupancy frozen where it is *now*. The
dynamic shield (dynamics.py) advances each moving obstacle along its velocity through the
braking rollout, so it brakes for where a crossing car is *going* — and, symmetrically, permits
more speed behind a car that is *leaving*. This measures that gap directly on drive 0009's
labelled movers, using the tracklet velocity (the clean feed; label-free velocity is too
false-positive-heavy to drive the shield — see `eval_dynamics.py`).

Per moving actor and frame it is visible through, with the ego at its true pose in the lidar
frame:

  * **v_static** — `max_safe_speed` against the frame's occupancy, the actor frozen in it.
  * **v_dynamic** — `max_safe_speed_dynamic` against the same occupancy with the actor lifted
    out and re-inserted as a constant-velocity box.

A negative gap (dynamic slower) is the shield catching a car entering the path; a positive gap
is it clearing a car that has left. Both are the static shield being wrong about a moving world.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.bev import BEVConfig, BEVGrid, occupancy_from_scan, rasterize_box
from kitti_nav.dynamics import MovingObstacle, max_safe_speed_dynamic
from kitti_nav.kitti import KittiDrive
from kitti_nav.mapping import (
    KITTI_EGO_BOX,
    drop_ego_returns,
    relative_lidar_transform,
    transform_points,
)
from kitti_nav.vehicle import VehicleConfig, max_safe_speed

DT = 0.1


def _actor_velocity(drive: KittiDrive, t, f: int) -> np.ndarray | None:
    """Tracklet `t`'s velocity `(vx, vy)` in frame `f`'s velo frame, or None if unavailable."""
    if t.index_of(f) is None or t.index_of(f - 1) is None:
        return None
    Tcv = drive.T_cam2_velo
    cf = t.box_at(f)[:2]
    kprev = t.index_of(f - 1)
    prev3 = np.array([[t.tx[kprev], t.ty[kprev], t.tz[kprev]]])
    T = relative_lidar_transform(drive.gt_poses[f], drive.gt_poses[f - 1], Tcv)
    cprev = transform_points(prev3, T)[0, :2]
    return (cf - cprev) / DT


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--forward-only", action="store_true",
                   help="only score movers ahead of the ego (x > rear axle)")
    p.add_argument("--eps", type=float, default=0.3, help="m/s gap that counts as a difference")
    args = p.parse_args()

    drive = KittiDrive(args.date, args.drive)
    cfg = BEVConfig()
    vcfg = VehicleConfig()
    ego = drive.vehicle_state_in_lidar()
    movers = drive.moving_tracklets(2.0)

    gaps, n_slower, n_faster, worst_brake = [], 0, 0, 0.0
    for t in movers:
        for f in range(t.first_frame + 1, min(t.last_frame + 1, drive.n_velodyne)):
            box = t.box_at(f)
            if box is None:
                continue
            if not (cfg.x_min < box[0] < cfg.x_max and cfg.y_min < box[1] < cfg.y_max):
                continue
            if args.forward_only and box[0] < ego.x:
                continue
            vel = _actor_velocity(drive, t, f)
            if vel is None:
                continue

            occ = occupancy_from_scan(drop_ego_returns(drive.velodyne(f), KITTI_EGO_BOX), cfg)
            grid_with = BEVGrid(occ, cfg)
            v_static = max_safe_speed(grid_with, vcfg, state=ego)

            occ_without = (occ.astype(bool) & ~rasterize_box(box, cfg)).astype(np.uint8)
            grid_without = BEVGrid(occ_without, cfg)
            v_dyn = max_safe_speed_dynamic(grid_without, [MovingObstacle(box, vel)], vcfg, ego)

            gap = v_dyn - v_static
            gaps.append(gap)
            if gap < -args.eps:
                n_slower += 1
                worst_brake = min(worst_brake, gap)
            elif gap > args.eps:
                n_faster += 1

    if not gaps:
        print("no scorable mover frames")
        return 1
    gaps = np.array(gaps)
    n = len(gaps)
    print(f"drive {args.drive}: {len(movers)} moving actors, {n} mover-frames scored"
          f"{' (forward only)' if args.forward_only else ''}\n")
    print(f"dynamic vs static permitted speed:")
    print(f"  more conservative (dynamic slower): {n_slower}/{n} = {n_slower / n:.0%}"
          f"   worst {worst_brake:+.1f} m/s")
    print(f"  more permissive  (dynamic faster): {n_faster}/{n} = {n_faster / n:.0%}"
          f"   best {gaps.max():+.1f} m/s")
    print(f"  unchanged (|gap| <= {args.eps} m/s):  {n - n_slower - n_faster}/{n}")
    print(f"  mean gap {gaps.mean():+.2f} m/s")
    print("\nA negative gap is the dynamic shield braking for a car entering the ego's path; a\n"
          "positive gap is it clearing a car that has left. The static shield sees neither.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
