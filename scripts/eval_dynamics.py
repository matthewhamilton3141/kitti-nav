#!/usr/bin/env python3
"""How good is label-free BEV velocity estimation? Scored against KITTI tracklets.

    python3 scripts/eval_dynamics.py                    # detection rate + velocity error
    python3 scripts/eval_dynamics.py --min-speed 1.5

The dynamic shield (next) needs each obstacle's velocity, and the deployable way to get it is
from occupancy alone — `dynamics.estimate_obstacle_velocities`. This checks whether that
actually recovers the motion, using the tracklet labels as ground truth (they are *not* used
by the estimator). Both scans of a consecutive pair are fused into the later frame through
ground-truth poses, so ego motion is removed and a static wall reads as stationary; whatever
velocity remains is the object's own.

Reported per moving actor present in a consecutive pair and inside the grid:

  * **detected** — did a label-free moving blob land on it (centroid within a gate)?
  * **speed / heading error** — of the matched estimate against the tracklet's own displacement.

Plus the false-positive count: estimated movers that match no labelled moving object (usually
static geometry smeared by registration noise past the `min_speed` floor).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.bev import BEVConfig, occupancy_from_scan
from kitti_nav.dynamics import estimate_obstacle_velocities
from kitti_nav.kitti import KittiDrive
from kitti_nav.mapping import (
    KITTI_EGO_BOX,
    drop_ego_returns,
    relative_lidar_transform,
    transform_points,
)

DT = 0.1                     # KITTI Velodyne spins at ~10 Hz


def _occ_window(drive: KittiDrive, f: int, cfg: BEVConfig, window: int):
    """Occupancy of scans `f-window+1 .. f`, all rasterised in frame `f`'s velodyne frame."""
    Tcv = drive.T_cam2_velo
    grids = []
    for k in range(f - window + 1, f + 1):
        pts = drop_ego_returns(drive.velodyne(k), KITTI_EGO_BOX)
        if k != f:
            pts = transform_points(pts, relative_lidar_transform(
                drive.gt_poses[f], drive.gt_poses[k], Tcv))
        grids.append(occupancy_from_scan(pts, cfg))
    return grids


def _actor_gt(drive: KittiDrive, t, f: int, cfg: BEVConfig):
    """Ground-truth `(centre_xy, velocity_xy)` of tracklet `t` at frame `f`, in frame-`f` velo."""
    if t.index_of(f) is None or t.index_of(f - 1) is None:
        return None
    Tcv = drive.T_cam2_velo
    cf = t.box_at(f)[:2]
    prev3 = np.array([[t.tx[t.index_of(f - 1)], t.ty[t.index_of(f - 1)],
                       t.tz[t.index_of(f - 1)]]])
    T = relative_lidar_transform(drive.gt_poses[f], drive.gt_poses[f - 1], Tcv)
    cprev = transform_points(prev3, T)[0, :2]
    if not (cfg.x_min < cf[0] < cfg.x_max and cfg.y_min < cf[1] < cfg.y_max):
        return None
    return cf, (cf - cprev) / DT


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--min-speed", type=float, default=1.0, help="m/s gate for a moving blob")
    p.add_argument("--gate", type=float, default=3.0, help="match radius (m) estimate<->actor")
    p.add_argument("--window", type=int, default=4, help="frames tracked for coherence")
    p.add_argument("--coherence", type=float, default=0.7, help="min net/path displacement ratio")
    p.add_argument("--gt-min-speed", type=float, default=1.0,
                   help="only score actors whose true speed exceeds this")
    args = p.parse_args()

    drive = KittiDrive(args.date, args.drive)
    cfg = BEVConfig()
    movers = drive.moving_tracklets(2.0)
    frames = sorted({f for t in movers for f in range(t.first_frame + 1, t.last_frame + 1)})

    n_gt = n_det = 0
    speed_err, head_err, gt_speeds = [], [], []
    n_est = n_fp = 0
    for f in frames:
        if f >= drive.n_velodyne or f - args.window + 1 < 0:
            continue
        window = _occ_window(drive, f, cfg, args.window)
        est = estimate_obstacle_velocities(window, cfg, DT, min_speed=args.min_speed,
                                           coherence=args.coherence)
        n_est += len(est)

        actors = []
        for t in movers:
            g = _actor_gt(drive, t, f, cfg)
            if g is not None and np.hypot(*g[1]) >= args.gt_min_speed:
                actors.append(g)

        matched_est = set()
        for centre, vel in actors:
            n_gt += 1
            gt_speeds.append(float(np.hypot(*vel)))
            if not est:
                continue
            d = [float(np.linalg.norm(o.centre - centre)) for o in est]
            j = int(np.argmin(d))
            if d[j] <= args.gate:
                n_det += 1
                matched_est.add(j)
                o = est[j]
                speed_err.append(abs(o.speed - np.hypot(*vel)))
                head_err.append(np.degrees(abs(np.arctan2(*o.velocity[::-1])
                                               - np.arctan2(*vel[::-1]))))
        n_fp += len(est) - len(matched_est)

    if n_gt == 0:
        print("no scorable actor pairs found")
        return 1
    print(f"drive {args.drive}: {len(movers)} moving actors, {len(frames)} frames scored\n")
    print(f"ground-truth actor-frames (speed > {args.gt_min_speed:g} m/s): {n_gt}")
    print(f"  true speed:     mean {np.mean(gt_speeds):.1f}  max {np.max(gt_speeds):.1f} m/s")
    print(f"detection rate:   {n_det}/{n_gt} = {n_det / n_gt:.0%}")
    if speed_err:
        head = np.array(head_err)
        head = np.minimum(head, 360 - head)                  # wrap to [0, 180]
        print(f"  speed error:    mean {np.mean(speed_err):.2f}  median "
              f"{np.median(speed_err):.2f} m/s")
        print(f"  heading error:  median {np.median(head):.0f} deg")
    print(f"estimated movers: {n_est} total, {n_fp} matching no labelled mover "
          f"({n_fp / max(n_est, 1):.0%} false-positive)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
