#!/usr/bin/env python3
"""Run stereo visual odometry over a KITTI drive and score it against OXTS ground truth.

    python3 scripts/eval_odometry.py                      # full default drive
    python3 scripts/eval_odometry.py --frames 150         # quick pass
    python3 scripts/eval_odometry.py --plot docs/traj.png

Reports ATE and final drift as a percentage of distance travelled. No Sim(3) alignment is
applied — stereo VO is metric, so scale error is real error and is not fitted away.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.kitti import KittiDrive                       # noqa: E402
from kitti_nav.odometry import (                             # noqa: E402
    OdometryConfig,
    ORBFrontend,
    StereoOdometry,
    evaluate_trajectory,
)
from kitti_nav.stereo import StereoDepth, StereoDepthConfig  # noqa: E402


def plot_trajectory(est: np.ndarray, gt: np.ndarray, out: Path) -> None:
    """Top-down (BEV) overlay of estimated vs ground-truth path.

    KITTI camera axes are x-right, y-down, z-forward, so the ground plane is (x, z) — not
    (x, y), which would plot the vertical axis and look like a flat line.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(gt[:, 0], gt[:, 2], "k-", lw=2, label="ground truth (OXTS)")
    ax.plot(est[:, 0], est[:, 2], "r--", lw=2, label="stereo VO (ORB + PnP)")
    ax.scatter([0], [0], c="g", s=80, marker="o", zorder=5, label="start")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z, forward (m)")
    ax.set_title("Stereo visual odometry vs ground truth")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--frames", type=int, default=None, help="limit frame count")
    p.add_argument("--features", type=int, default=2000, help="ORB features per frame")
    p.add_argument("--ratio", type=float, default=0.8, help="Lowe ratio-test threshold")
    p.add_argument("--max-depth", type=float, default=40.0,
                   help="reject stereo depth beyond this (m)")
    p.add_argument("--plot", type=Path, default=None, help="write a trajectory PNG here")
    args = p.parse_args()

    drive = KittiDrive(args.date, args.drive, n_frames=args.frames)
    K = drive.intrinsics
    print(f"drive {args.date}/{args.drive}: {len(drive)} frames, "
          f"fx={K.fx:.1f} baseline={drive.baseline:.4f} m")
    print(f"ground-truth path {drive.path_length:.1f} m, "
          f"speed up to {drive.speeds.max():.2f} m/s")

    stereo = StereoDepth(K.fx, drive.baseline, StereoDepthConfig(max_depth=args.max_depth))
    odo = StereoOdometry(K, cfg=OdometryConfig(),
                         frontend=ORBFrontend(n_features=args.features, ratio=args.ratio))

    t0 = time.perf_counter()
    matches, inliers = [], []
    for i, left, right in drive.frames():
        res = odo.track(left, stereo.depth(left, right))
        if i:
            matches.append(res.n_matches)
            inliers.append(res.n_inliers)
        if i % 100 == 0:
            print(f"  frame {i:4d}/{len(drive)}  matches={res.n_matches:4d} "
                  f"inliers={res.n_inliers:4d}")
    elapsed = time.perf_counter() - t0

    est = odo.positions
    gt = drive.gt_poses[:, :3, 3]
    err = evaluate_trajectory(est, gt)

    print()
    print(f"{err}")
    print(f"tracked {len(drive)} frames in {elapsed:.1f}s "
          f"({len(drive) / elapsed:.1f} fps, CPU)")
    print(f"mean matches {np.mean(matches):.0f}, mean PnP inliers {np.mean(inliers):.0f}, "
          f"fallback steps {odo.n_fallbacks}")

    if args.plot:
        plot_trajectory(est, gt, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
