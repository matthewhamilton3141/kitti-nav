#!/usr/bin/env python3
"""Does fusing lidar scans with *estimated* poses help the planner, or hurt it?

    python3 scripts/eval_mapping.py                      # the headline comparison
    python3 scripts/eval_mapping.py --sweep 1 3 5 10 20  # coverage-vs-drift tradeoff
    python3 scripts/eval_mapping.py --plot docs/mapping.png

Until now the planner drove on a single frozen lidar scan, and `odometry.py`'s trajectory
went unused. `mapping.py` joins them. That join is not obviously a win: an accumulated map
sees more, but it inherits the odometry's drift, and a map built from wrong poses can be
worse than one built from no poses at all.

Three maps are built at each sampled frame and compared:

  * **single** — one scan, the previous behaviour, the thing to beat.
  * **gt** — accumulated using OXTS ground-truth poses. The ceiling: what fusion is worth
    with a perfect trajectory. Not achievable by a real system.
  * **vo** — accumulated using this repo's stereo VO. The honest number.

The comparison that matters is `vo` against `gt`, and it is deliberately **not** reported as
one similarity score. Drift corrupts a map in two directions that are not equally bad:

  * **phantom** cells (occupied in `vo`, free in `gt`) — obstacles invented from
    mis-registration. They cost speed. They cannot cause a crash.
  * **missed** cells (free in `vo`, occupied in `gt`) — real geometry lost. **These can.**

The same asymmetry decides the safety metric. `max_safe_speed` is evaluated on every map,
and the number to watch is not the average difference but how often the VO map permits a
*higher* speed than the ground-truth map does — because that is the map claiming clearance
the world does not have. An aggregate IoU would let exactly that failure hide inside a pile
of harmless phantom cells.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.bev import BEVConfig, BEVGrid, occupancy_from_scan   # noqa: E402
from kitti_nav.kitti import KittiDrive                              # noqa: E402
from kitti_nav.mapping import (                                     # noqa: E402
    MapConfig,
    drop_ego_returns,
    fuse_scans,
    occupancy_agreement,
    window_indices,
)
from kitti_nav.odometry import StereoOdometry, evaluate_trajectory  # noqa: E402
from kitti_nav.stereo import StereoDepth, StereoDepthConfig         # noqa: E402
from kitti_nav.vehicle import VehicleConfig, max_safe_speed         # noqa: E402

CACHE = Path(__file__).resolve().parents[1] / "data" / "cache"


def vo_poses(drive: KittiDrive, date: str, drv: str, refresh: bool = False) -> np.ndarray:
    """Stereo-VO camera-to-world poses for the whole drive, cached to disk.

    Cached because the window sweep needs the same trajectory a dozen times and re-running
    ORB + SGBM each pass would dominate the runtime. The cache lives under `data/`, which is
    gitignored — it is derived, and reproducible in one pass.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"vo_poses_{date}_{drv}_{len(drive)}.npz"
    if path.exists() and not refresh:
        return np.load(path)["poses"]

    K = drive.intrinsics
    stereo = StereoDepth(K.fx, drive.baseline, StereoDepthConfig())
    odo = StereoOdometry(K)

    t0 = time.perf_counter()
    for i, left, right in drive.frames():
        odo.track(left, stereo.depth(left, right))
        if i % 100 == 0:
            print(f"  VO frame {i:4d}/{len(drive)}")
    poses = np.stack(odo.trajectory)

    err = evaluate_trajectory(odo.positions, drive.gt_poses[:, :3, 3])
    print(f"  VO done in {time.perf_counter() - t0:.1f}s — {err}")
    np.savez_compressed(path, poses=poses)
    return poses


def relative_pose_error(est: np.ndarray, gt: np.ndarray, gap: int) -> float:
    """RMS translation error (m) of the pose change across `gap` frames.

    The metric that governs accumulation, and *not* the one the README headlines. Global ATE
    measures how far the trajectory has wandered since the start; a fused map never composes
    poses across more than its own window, so what corrupts it is how wrong the odometry is
    over `window * stride` frames — a few tenths of a second. The two can differ by orders
    of magnitude, which is the point.
    """
    if gap <= 0:
        return 0.0
    errs = []
    for i in range(len(est) - gap):
        d_est = np.linalg.inv(est[i]) @ est[i + gap]
        d_gt = np.linalg.inv(gt[i]) @ gt[i + gap]
        errs.append(np.linalg.norm(d_est[:3, 3] - d_gt[:3, 3]))
    return float(np.sqrt(np.mean(np.square(errs)))) if errs else 0.0


def audit_ego(drive: KittiDrive, bcfg: BEVConfig, every: int = 7) -> None:
    """Re-derive the ego self-return box from the data, and price the filter on one scan.

    The box in `mapping.KITTI_EGO_BOX` is a measured constant, so it needs a way to be
    re-measured. Self-returns are found by their defining property: they recur at a *fixed
    position in the sensor frame* in **every** scan, because the bodywork does not move
    relative to the sensor.

    Two constraints keep this honest, and both are needed. Presence is required in every
    scan rather than most, since rigid structure is not intermittent. And the search is
    bounded by what the vehicle could physically be — a 1.82 m-wide car with mirrors cannot
    reach |y| > 2 m. Roadside structure *also* persists in sensor-frame coordinates while
    the car holds its lane: without the lateral bound this drive reports a solid band at
    y in [-2.5, -2.1] spanning every x from -4 to +4 m, present in 100% of scans, which is a
    kerb line at constant offset and obviously not part of the car.
    """
    from kitti_nav.mapping import KITTI_EGO_BOX

    frames = list(range(0, drive.n_velodyne, every))
    pts = []
    for f in frames:
        p = drive.velodyne(f)
        # Above the road, and inside a box the car itself could plausibly occupy.
        pts.append(p[(np.abs(p[:, 0]) < 4.0) & (np.abs(p[:, 1]) < 2.0)
                     & (p[:, 2] > -1.35)][:, :3])

    cells = np.round(np.concatenate(pts)[:, :2] / 0.1).astype(int)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    fixed = uniq[counts >= len(frames)] * 0.1

    print(f"ego self-return audit over {len(frames)} scans")
    if len(fixed):
        print(f"  returns fixed in the sensor frame: {len(fixed)} cells, "
              f"x [{fixed[:, 0].min():+.2f}, {fixed[:, 0].max():+.2f}]  "
              f"y [{fixed[:, 1].min():+.2f}, {fixed[:, 1].max():+.2f}]")
    print(f"  KITTI_EGO_BOX = {KITTI_EGO_BOX}")

    f0 = drive.velodyne(frames[len(frames) // 2])
    raw = occupancy_from_scan(f0, bcfg).sum()
    filt = occupancy_from_scan(drop_ego_returns(f0, KITTI_EGO_BOX), bcfg).sum()
    print(f"  cost on a single scan: {raw} -> {filt} occupied cells "
          f"({len(f0) - len(drop_ego_returns(f0, KITTI_EGO_BOX))} returns dropped)")


def scan_loader(drive: KittiDrive, maxsize: int):
    """Velodyne reader with a bounded LRU cache.

    Worth the wrapper because the sweep asks for the same scans once per window size, and a
    scan is ~2 MB on disk. The cache is bounded rather than a plain dict: holding the whole
    drive would be ~900 MB, and the access pattern below never needs more than the largest
    window at a time anyway.
    """
    from functools import lru_cache

    @lru_cache(maxsize=maxsize)
    def load(i: int) -> np.ndarray:
        return drive.velodyne(i)

    return load


def evaluate(drive: KittiDrive, gt: np.ndarray, vo: np.ndarray, mcfgs: list[MapConfig],
             bcfg: BEVConfig, vcfg: VehicleConfig, frames: np.ndarray,
             verbose: bool = True) -> list[dict]:
    """Compare the three maps over `frames`, for every window config; one pass over the data.

    Frames are the **outer** loop and window sizes the inner one, so every scan a frame
    needs is read once and reused across all the windows that overlap it. The reverse
    nesting reloads the whole drive per window size.
    """
    rows: dict[int, list] = {i: [] for i in range(len(mcfgs))}
    load = scan_loader(drive, maxsize=max(m.window * m.stride for m in mcfgs) + 4)
    state = drive.vehicle_state_in_lidar()

    for ref in frames:
        ref = int(ref)
        # The baseline gets the same ego self-filtering as the fused maps, so the comparison
        # isolates accumulation rather than crediting it with the self-filter. On a single
        # scan the filter is a no-op anyway — it drops ~41 returns that the per-cell ground
        # estimate was already treating as free, changing zero occupied cells. `--audit-ego`
        # prints the delta.
        single = BEVGrid(occupancy_from_scan(
            drop_ego_returns(load(ref), mcfgs[0].ego_box), bcfg), bcfg)
        v_single = max_safe_speed(single, vcfg, state=state)

        for j, mcfg in enumerate(mcfgs):
            idx = window_indices(ref, mcfg, drive.n_velodyne)
            scans = [load(i) for i in idx]
            grids = {}
            for name, poses in (("gt", gt), ("vo", vo)):
                pts = fuse_scans(scans, [poses[i] for i in idx], drive.T_cam2_velo)
                grids[name] = BEVGrid(occupancy_from_scan(pts, bcfg), bcfg)

            agree_vo = occupancy_agreement(grids["vo"].occupancy, grids["gt"].occupancy)
            agree_single = occupancy_agreement(single.occupancy, grids["gt"].occupancy)

            rows[j].append({
                "frame": ref,
                "n_scans": len(idx),
                "occ_single": int(single.occupancy.sum()),
                "occ_gt": int(grids["gt"].occupancy.sum()),
                "occ_vo": int(grids["vo"].occupancy.sum()),
                "iou": agree_vo["iou"],
                "phantom_rate": agree_vo["phantom_rate"],
                "missed_rate": agree_vo["missed_rate"],
                "missed_rate_single": agree_single["missed_rate"],
                "v_single": v_single,
                "v_gt": max_safe_speed(grids["gt"], vcfg, state=state),
                "v_vo": max_safe_speed(grids["vo"], vcfg, state=state),
            })
            if verbose:
                r = rows[j][-1]
                print(f"  frame {r['frame']:4d}  cells {r['occ_single']:5d}->"
                      f"{r['occ_vo']:5d}  IoU {r['iou']:.3f}  "
                      f"missed {r['missed_rate']:6.2%}  permitted "
                      f"{r['v_single']:5.2f}/{r['v_gt']:5.2f}/{r['v_vo']:5.2f} m/s")

    return [aggregate(rows[j], m,
                      relative_pose_error(vo, gt, m.window * m.stride))
            for j, m in enumerate(mcfgs)]


def aggregate(rows: list[dict], mcfg: MapConfig, rpe: float) -> dict:
    """Reduce per-frame rows to the numbers worth reporting."""
    col = lambda k: np.array([r[k] for r in rows], float)      # noqa: E731

    v_gt, v_vo, v_single = col("v_gt"), col("v_vo"), col("v_single")
    # The dangerous direction: the VO map permitting more speed than the true map allows,
    # i.e. claiming clearance the world does not have. Tolerance is the bisection's own
    # resolution (0.05 m/s), so numerical noise is not counted as an unsafe reading.
    optimistic = v_vo > v_gt + 0.05

    return {
        "window": mcfg.window,
        "stride": mcfg.stride,
        "n_frames": len(rows),
        "rpe_over_window": rpe,
        "occ_single": float(np.mean(col("occ_single"))),
        "occ_gt": float(np.mean(col("occ_gt"))),
        "occ_vo": float(np.mean(col("occ_vo"))),
        "coverage_gain": float(np.mean(col("occ_vo")) / np.mean(col("occ_single"))),
        "iou": float(np.mean(col("iou"))),
        "phantom_rate": float(np.mean(col("phantom_rate"))),
        "missed_rate": float(np.mean(col("missed_rate"))),
        "missed_rate_single": float(np.mean(col("missed_rate_single"))),
        "v_single": float(np.mean(v_single)),
        "v_gt": float(np.mean(v_gt)),
        "v_vo": float(np.mean(v_vo)),
        "v_vo_minus_gt": float(np.mean(v_vo - v_gt)),
        "n_optimistic": int(optimistic.sum()),
        "max_optimistic": float(np.max(v_vo - v_gt)) if len(rows) else 0.0,
        "rows": rows,
    }


def report(res: dict) -> None:
    print()
    print(f"=== window {res['window']} (stride {res['stride']}), "
          f"{res['n_frames']} frames ===")
    print(f"  relative pose error over the window : {res['rpe_over_window']*100:.1f} cm")
    print(f"  occupied cells  single {res['occ_single']:7.0f}   "
          f"gt {res['occ_gt']:7.0f}   vo {res['occ_vo']:7.0f}"
          f"   ({res['coverage_gain']:.2f}x more mapped)")
    print(f"  vo vs gt map    IoU {res['iou']:.3f}   "
          f"phantom {res['phantom_rate']:.2%}   missed {res['missed_rate']:.2%}")
    print(f"  (for scale, the single-scan map misses "
          f"{res['missed_rate_single']:.2%} of the same cells)")
    print(f"  shield-permitted speed  single {res['v_single']:.2f}   "
          f"gt {res['v_gt']:.2f}   vo {res['v_vo']:.2f} m/s")
    print(f"  UNSAFE direction: vo permitted more than gt on "
          f"{res['n_optimistic']}/{res['n_frames']} frames "
          f"(worst {res['max_optimistic']:+.2f} m/s)")


def plot_sweep(results: list[dict], out: Path) -> None:
    """Coverage, map fidelity, and the safety gap, all against window size."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w = [r["window"] for r in results]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    axes[0].plot(w, [r["occ_gt"] for r in results], "o-", label="ground-truth poses")
    axes[0].plot(w, [r["occ_vo"] for r in results], "s-", label="stereo VO poses")
    axes[0].axhline(results[0]["occ_single"], color="k", ls="--", label="single scan")
    axes[0].set_ylabel("occupied cells")
    axes[0].set_title("Fusion maps several times more of the scene")

    axes[1].plot(w, [100 * r["phantom_rate"] for r in results], "o-",
                 color="tab:orange", label="phantom (costs speed)")
    axes[1].plot(w, [100 * r["missed_rate"] for r in results], "s-",
                 color="tab:red", label="missed (dangerous)")
    axes[1].set_ylabel("% of true occupied cells")
    axes[1].set_title("VO drift corrupts the map, asymmetrically")

    axes[2].plot(w, [r["v_gt"] for r in results], "o-", label="ground-truth poses")
    axes[2].plot(w, [r["v_vo"] for r in results], "s-", label="stereo VO poses")
    axes[2].axhline(results[0]["v_single"], color="k", ls="--", label="single scan")
    axes[2].set_ylabel("shield-permitted speed (m/s)")
    axes[2].set_title("Better informed, so slower\n(the single scan was optimistic)")

    # The safety-critical panel: how often the VO map claims clearance the ground-truth map
    # says is not there. Everything else can be traded off; this one cannot.
    frac = [100 * r["n_optimistic"] / r["n_frames"] for r in results]
    ax3b = axes[3].twinx()
    axes[3].bar(range(len(w)), frac, color="tab:red", alpha=0.55,
                label="frames where VO permits more than truth")
    ax3b.plot(range(len(w)), [r["max_optimistic"] for r in results], "k^-",
              label="worst single excursion")
    axes[3].set_xticks(range(len(w)))
    axes[3].set_xticklabels(w)
    axes[3].set_ylabel("% of frames optimistic")
    ax3b.set_ylabel("worst excursion (m/s)")
    axes[3].set_title("The unsafe direction")
    axes[3].set_xlabel("scans fused")
    h1, l1 = axes[3].get_legend_handles_labels()
    h2, l2 = ax3b.get_legend_handles_labels()
    axes[3].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")

    for ax in axes[:3]:
        ax.set_xlabel("scans fused")
        ax.set_xscale("log")
        ax.set_xticks(w)
        ax.set_xticklabels(w)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[3].grid(alpha=0.3, axis="y")

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--window", type=int, default=5, help="scans fused per map")
    p.add_argument("--stride", type=int, default=1, help="keep every Nth scan")
    p.add_argument("--sweep", type=int, nargs="+", default=None,
                   help="window sizes to sweep instead of a single run")
    p.add_argument("--every", type=int, default=10, help="evaluate every Nth frame")
    p.add_argument("--max-speed", type=float, default=35.0,
                   help="raise the speed cap so geometry, not the cap, is what binds")
    p.add_argument("--plot", type=Path, default=None, help="write the sweep figure here")
    p.add_argument("--refresh-vo", action="store_true", help="ignore the cached VO poses")
    p.add_argument("--audit-ego", action="store_true",
                   help="re-derive the ego self-return box from the data and exit")
    args = p.parse_args()

    drive = KittiDrive(args.date, args.drive)
    bcfg, vcfg = BEVConfig(), VehicleConfig(max_speed=args.max_speed)
    print(f"drive {args.date}/{args.drive}: {len(drive)} frames, "
          f"{drive.n_velodyne} scans, {drive.path_length:.1f} m")

    if args.audit_ego:
        audit_ego(drive, bcfg)
        return 0

    gt = drive.gt_poses
    vo = vo_poses(drive, args.date, args.drive, args.refresh_vo)

    windows = args.sweep or [args.window]
    # Skip the warm-up frames: a window ending before frame `max(window)` is truncated, and
    # averaging a thin map in with the full ones would understate accumulation.
    start = max(windows) * args.stride
    frames = np.arange(start, drive.n_velodyne, args.every)
    print(f"evaluating {len(frames)} frames from {start} "
          f"(ATE over the whole drive: "
          f"{evaluate_trajectory(vo[:, :3, 3], gt[:, :3, 3]).final_drift:.2f} m)")

    mcfgs = [MapConfig(window=w, stride=args.stride) for w in windows]
    results = evaluate(drive, gt, vo, mcfgs, bcfg, vcfg, frames,
                       verbose=len(windows) == 1)
    for res in results:
        report(res)

    if args.plot and len(results) > 1:
        plot_sweep(results, args.plot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
