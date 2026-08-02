"""Fuse a stream of lidar scans into one accumulated BEV map, using estimated poses.

This is the join between the repo's two halves. `odometry.py` estimates where the car is;
`bev.py` turns *one* scan into the grid the planner drives on. Until this module existed
they never spoke: every planning decision was made against a single frozen scan, which
throws away everything the sensor saw a tenth of a second ago and leaves the map full of
holes the lidar simply could not see from one vantage point.

**What accumulation buys.** A single HDL-64E scan is sparse at range (rings spread apart,
so a wall at 30 m lands in a handful of cells) and shadowed (nothing behind a parked car is
mapped). Driving past the same geometry from a slightly different position fills both in.

**What it costs, and why that is the interesting part.** Accumulation is only as good as
the poses. An accumulated map inherits the odometry's drift, and drift does not merely blur
the map — it corrupts it in two opposite and asymmetric ways:

  * **Phantom obstacles.** A mis-registered road surface is a road surface at two slightly
    different heights in the same cell, and `bev.py`'s `height_diff` ground removal reads a
    vertical spread as an obstacle. Drift therefore manufactures obstacles out of open road,
    and the shield brakes for them. Expensive, but *safe*.
  * **Smeared free space.** The same mis-registration drags real obstacle returns across
    cells. That widens obstacles (safe) but can also displace them off the cell the planner
    queries (**unsafe** — the map claims clearance that is not there).

Only the second kind can hurt someone, so the evaluation in `scripts/eval_mapping.py`
measures them separately rather than reporting one aggregate map-similarity score. A single
IoU number would let a genuinely dangerous error hide behind a pile of harmless ones.

**Frames.** Scans live in the Velodyne frame (+x forward, +y left, +z up); poses from
`odometry.py` and `kitti.py` are 4x4 **camera-to-world** SE(3) in the *camera* frame
(+x right, +y down, +z forward). Fusion therefore has to hop frames twice, and
`relative_lidar_transform` is the only place that happens — see its docstring for the
composition. Everything this module returns is back in the Velodyne frame of the reference
frame, so `BEVGrid`, the vehicle model, and the shield all consume it unchanged.

**Point-level, not grid-level, fusion.** Scans are transformed as point clouds and
rasterised once, rather than rasterised individually and OR-ed together. Two reasons: the
ground estimate gets *better* with more returns per cell (the whole weakness of
`height_diff` is cells with no ground return of their own), and OR-ing binary grids would
bake each scan's ground-removal mistakes in permanently, with no evidence left to revise
them. The trade is the phantom-obstacle sensitivity described above — deliberate, because
that sensitivity is the measurement this module exists to make.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np

from .bev import BEVConfig, BEVGrid
from .kitti import invert_se3


# Returns inside this box (lidar frame, metres, `(x_min, x_max, y_min, y_max)`) are the
# recording car itself and are discarded before fusion. See `drop_ego_returns` for why this
# is mandatory once scans are accumulated, and why the box is *measured* rather than derived
# from `VehicleConfig`'s body rectangle.
#
# Measured on drive 0009 by pooling near-field returns above the road across the drive and
# keeping those that recur at a fixed position in the sensor frame: a hood/roof-edge return
# at (2.51, 0.46, -0.91) present in every scan, a right-side cluster spanning
# x [-1.0, 1.6], y [-1.20, -0.95], and a left-side cluster at x [-1.0, -0.8], y [1.01, 1.19],
# all at z about -0.5 to -0.95 (roof-rail height). `test_ego_box_covers_the_self_returns`
# re-derives the extent from the data and fails if it drifts outside this box.
KITTI_EGO_BOX = (-2.0, 3.1, -1.6, 1.5)


def drop_ego_returns(points: np.ndarray,
                     box: Optional[Tuple[float, float, float, float]] = KITTI_EGO_BOX
                     ) -> np.ndarray:
    """Remove returns that hit the recording vehicle itself, in the sensor's own frame.

    Self-filtering is standard in any real lidar stack, and this repo got away without it
    for exactly as long as it looked at one scan at a time. The reason is worth stating,
    because the bug it caused was invisible until fusion existed:

    A roof-mounted Velodyne sees its own car — the hood, the roof rails, the mirrors. In a
    **single** scan those cells contain *only* the bodywork, so `bev.py`'s per-cell ground
    estimate takes the bodywork itself as the ground, measures no height above it, and calls
    the cell free. The map is right by accident. **Accumulate** two scans and an earlier
    viewpoint supplies the road surface at that same ground position — seen from behind,
    before the car arrived — so the cell now holds road at -1.73 m and bodywork at -0.92 m.
    That 0.81 m spread is read as an obstacle, and the car maps itself as a wall.

    Measured on drive 0009, this put phantom obstacles inside the vehicle's own footprint on
    ~80% of frames and drove the shield-permitted speed from 25.1 m/s to 2.2 m/s. It is not
    a small effect, and it is not drift — it happens with perfect ground-truth poses.

    The box is deliberately **not** derived from `VehicleConfig`'s body rectangle. The body
    rectangle is positioned at the rear axle, which sits 0.32 m off the lidar's centreline,
    whereas the self-returns are symmetric about the sensor; and they reach ~1.5 m laterally,
    past the 0.91 m half-width, because roof rails and mirrors are not part of the body
    rectangle. Deriving the filter from the body would leave the right-hand rail unfiltered.

    Filtering must happen in the **sensor's own frame**, before any pose transform — the
    bodywork is fixed relative to the sensor, not to the world. `None` disables it.
    """
    p = np.asarray(points, float)
    if box is None or p.size == 0:
        return p
    x_min, x_max, y_min, y_max = box
    inside = ((p[:, 0] >= x_min) & (p[:, 0] <= x_max)
              & (p[:, 1] >= y_min) & (p[:, 1] <= y_max))
    return p[~inside]


def relative_lidar_transform(pose_ref: np.ndarray, pose_i: np.ndarray,
                             T_cam_velo: np.ndarray) -> np.ndarray:
    """4x4 transform taking points from frame `i`'s Velodyne frame into frame `ref`'s.

    `pose_ref` and `pose_i` are camera-to-world SE(3); `T_cam_velo` is the fixed extrinsic
    mapping Velodyne coordinates into camera coordinates (`KittiDrive.T_cam2_velo`).

    Read the composition right to left — it is four hops, each undoing or applying one
    known relationship::

        T = T_velo_cam @ inv(pose_ref) @ pose_i @ T_cam_velo
            ^^^^^^^^^^   ^^^^^^^^^^^^^   ^^^^^^   ^^^^^^^^^^
            back to      world into      cam_i    lidar_i into
            lidar_ref    cam_ref         into     cam_i
                                         world

    The two extrinsic hops are what make this easy to get wrong: the poses are *camera*
    poses, so applying them directly to lidar points would rotate the cloud by the ~90
    degrees between the camera and Velodyne axis conventions and offset it by the ~0.3 m
    between the two sensors. Both errors are large enough to wreck a 0.2 m grid.

    Sanity check worth keeping in mind: `pose_ref is pose_i` gives
    `T_velo_cam @ T_cam_velo` = identity, regardless of what the extrinsic is.
    """
    T_velo_cam = invert_se3(np.asarray(T_cam_velo, float))
    return T_velo_cam @ invert_se3(np.asarray(pose_ref, float)) \
        @ np.asarray(pose_i, float) @ np.asarray(T_cam_velo, float)


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to `(N, 3)` or `(N, 4+)` points.

    Columns past the third (reflectance, and anything else the sensor appends) are dropped
    rather than transformed — they are not spatial and rotating them would be nonsense.
    Returns float32: a lidar point is metre-scale and the grid quantises at 0.2 m, so
    float64 buys nothing here and doubles the memory a long accumulation window holds.
    """
    p = np.asarray(points, float)
    if p.size == 0:
        return np.zeros((0, 3), np.float32)
    M = np.asarray(T, float)

    # The errstate is a workaround for the platform, not for this maths. NumPy 2.x on Apple
    # Accelerate raises spurious divide/overflow/invalid warnings from inside `matmul` on
    # large arrays — its SIMD kernels trip the FP status flags on the padding lanes of the
    # final vector. A bare `points @ M` warns identically, and the result here is verified
    # finite and exact to float32 (`test_transform_points_*`). Rigid transforms of finite
    # metre-scale points cannot genuinely overflow, so nothing real is being hidden.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (p[:, :3] @ M[:3, :3].T + M[:3, 3]).astype(np.float32)


@dataclass(frozen=True)
class MapConfig:
    """How much history to fuse, and how aggressively to thin it."""

    # Scans fused per map, counting the current one. 1 reproduces the single-scan behaviour,
    # which is what makes "does accumulation help?" a fair comparison — and is the control
    # the evaluation leans on. Agreement there is near-exact rather than exact: fusion routes
    # points through a float64 matmul and casts to float32, so a return within a rounding
    # error of a cell boundary can land either side of it (~2 cells in 2600, IoU 0.997).
    window: int = 5

    # Keep every `stride`-th scan. At 10 Hz and city speeds consecutive scans overlap almost
    # completely, so a stride spreads the same number of scans over a longer baseline — more
    # new viewpoint per scan retained, but more accumulated drift between the ends.
    stride: int = 1

    # Drop points further than this (m) from their own sensor origin before storing. Far
    # returns are the least accurate part of a lidar scan and the most likely to fall
    # outside the grid anyway; pruning at insert keeps the window's memory bounded.
    # None keeps everything.
    max_range: Optional[float] = 70.0

    # Returns inside this lidar-frame box are the recording car itself. Filtering them is
    # not optional for a fused map — see `drop_ego_returns`. `None` disables it.
    ego_box: Optional[Tuple[float, float, float, float]] = KITTI_EGO_BOX

    def __post_init__(self) -> None:
        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")


class ScanAccumulator:
    """Sliding window of recent scans, fused on demand into the current Velodyne frame.

    Stateful and streaming by design — `add` once per frame as the drive replays, then ask
    for `grid()` whenever the planner needs a map. Points are stored in their **own** sensor
    frame alongside the pose they were taken at, and transformed only at fusion time. That
    ordering matters: re-transforming on demand means a later pose correction (a loop
    closure, a smoothed trajectory) would change the map, whereas transforming eagerly into
    a fixed world frame would bake the drift in at insert time and make it unrecoverable.

    Usage::

        acc = ScanAccumulator(drive.T_cam2_velo, MapConfig(window=5))
        for i in range(n):
            acc.add(drive.velodyne(i), poses[i])
            grid = acc.grid()          # fused, in frame i's Velodyne frame
    """

    def __init__(self, T_cam_velo: np.ndarray, cfg: Optional[MapConfig] = None,
                 bev_cfg: Optional[BEVConfig] = None):
        self.cfg = cfg or MapConfig()
        self.bev_cfg = bev_cfg or BEVConfig()
        self.T_cam_velo = np.asarray(T_cam_velo, float)
        if self.T_cam_velo.shape != (4, 4):
            raise ValueError(f"T_cam_velo must be 4x4, got {self.T_cam_velo.shape}")

        # Bounded by window so old scans fall out on their own; entries are (points, pose).
        self._scans: Deque[Tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.cfg.window)
        self._n_added = 0

    def __len__(self) -> int:
        """Scans currently held — less than `window` while the window is still filling."""
        return len(self._scans)

    @property
    def n_added(self) -> int:
        """Scans offered so far, including ones `stride` skipped."""
        return self._n_added

    @property
    def latest_pose(self) -> np.ndarray:
        """Pose of the most recent stored scan — the default fusion reference."""
        if not self._scans:
            raise RuntimeError("accumulator is empty: add() at least one scan first")
        return self._scans[-1][1]

    def add(self, points: np.ndarray, pose: np.ndarray) -> bool:
        """Offer a scan with its camera-to-world pose. Returns whether it was stored.

        Scans skipped by `stride` still advance the counter, so striding thins the window
        without changing which frame is "current" for the caller.
        """
        self._n_added += 1
        if (self._n_added - 1) % self.cfg.stride:
            return False

        # Self-filter first, in the sensor's own frame — the only frame the bodywork is
        # fixed in. Everything after this point may be transformed freely.
        p = drop_ego_returns(points, self.cfg.ego_box)[:, :3].astype(np.float32)
        if self.cfg.max_range is not None and len(p):
            p = p[np.linalg.norm(p, axis=1) <= self.cfg.max_range]

        pose = np.asarray(pose, float)
        if pose.shape != (4, 4):
            raise ValueError(f"pose must be 4x4, got {pose.shape}")
        self._scans.append((p, pose.copy()))
        return True

    def fused_points(self, pose_ref: Optional[np.ndarray] = None) -> np.ndarray:
        """Every stored scan, transformed into `pose_ref`'s Velodyne frame, as `(N, 3)`.

        Defaults to the newest pose, which is what a planner running live wants: a map
        centred on where the car is *now*, in the frame the vehicle model already uses.
        """
        if not self._scans:
            return np.zeros((0, 3), np.float32)
        ref = self.latest_pose if pose_ref is None else np.asarray(pose_ref, float)

        out = []
        for pts, pose in self._scans:
            if np.array_equal(pose, ref):
                out.append(pts)                  # identity hop; skip the matrix multiply
            else:
                out.append(transform_points(
                    pts, relative_lidar_transform(ref, pose, self.T_cam_velo)))
        return np.concatenate(out, axis=0)

    def grid(self, pose_ref: Optional[np.ndarray] = None,
             outside_is_free: bool = True) -> BEVGrid:
        """Fused occupancy + distance field, ready for the shield. See `fused_points`."""
        return BEVGrid.from_scan(self.fused_points(pose_ref), self.bev_cfg,
                                 outside_is_free=outside_is_free)


def fuse_scans(scans: Sequence[np.ndarray], poses: Sequence[np.ndarray],
               T_cam_velo: np.ndarray, ref: int = -1,
               ego_box: Optional[Tuple[float, float, float, float]] = KITTI_EGO_BOX
               ) -> np.ndarray:
    """One-shot fusion of exactly these scans into `scans[ref]`'s Velodyne frame, `(N, 3)`.

    The batch counterpart to `ScanAccumulator`, for evaluation code that already has the
    drive in hand and wants a map at an arbitrary frame without replaying the stream.
    Deliberately applies no window or stride of its own — pick the frames with
    `window_indices` and pass them in, so there is exactly one place that decides which
    scans a map is allowed to contain.
    """
    if len(scans) != len(poses):
        raise ValueError(f"{len(scans)} scans but {len(poses)} poses")
    if not scans:
        return np.zeros((0, 3), np.float32)

    pose_ref = np.asarray(poses[ref], float)
    out = [transform_points(drop_ego_returns(pts, ego_box),
                            relative_lidar_transform(pose_ref, pose, T_cam_velo))
           for pts, pose in zip(scans, poses)]
    return np.concatenate(out, axis=0)


def window_indices(ref: int, cfg: MapConfig, n_available: int) -> list[int]:
    """Frame indices a window ending at `ref` would fuse, oldest first.

    Looks **backwards only**. A live planner cannot fuse scans it has not received yet, and
    an evaluation that quietly centred the window on `ref` would be reporting a map no
    real-time system could build.
    """
    idx = [i for i in range(ref, -1, -cfg.stride)][:cfg.window]
    return [i for i in reversed(idx) if 0 <= i < n_available]


def occupancy_agreement(estimated: np.ndarray, reference: np.ndarray) -> dict:
    """Compare two occupancy grids cell-by-cell, split by which errors are dangerous.

    `estimated` is the map actually built (from odometry poses); `reference` is the best
    available map of the same scene (from ground-truth poses). Returns IoU plus the two
    error classes broken out, because they are not interchangeable:

      * ``phantom`` — occupied in `estimated`, free in `reference`. Obstacles invented out
        of drift. Costs speed and comfort; cannot cause a collision.
      * ``missed`` — free in `estimated`, occupied in `reference`. Real geometry the map
        lost. **This is the one that can hurt**, and it is the number to watch even when
        IoU looks healthy.

    Rates are expressed against the reference's occupied count, so they read as "x% of the
    true obstacle cells were lost" rather than as a fraction of a grid that is ~98% empty
    and would make every error look negligible.
    """
    est = np.asarray(estimated, bool)
    ref = np.asarray(reference, bool)
    if est.shape != ref.shape:
        raise ValueError(f"grid shapes differ: {est.shape} vs {ref.shape}")

    inter = int(np.count_nonzero(est & ref))
    union = int(np.count_nonzero(est | ref))
    phantom = int(np.count_nonzero(est & ~ref))
    missed = int(np.count_nonzero(~est & ref))
    n_ref = int(np.count_nonzero(ref))

    return {
        "iou": float(inter / union) if union else 1.0,
        "phantom_cells": phantom,
        "missed_cells": missed,
        "n_estimated": int(np.count_nonzero(est)),
        "n_reference": n_ref,
        "phantom_rate": float(phantom / n_ref) if n_ref else 0.0,
        "missed_rate": float(missed / n_ref) if n_ref else 0.0,
    }


def stream_maps(scans: Iterable[np.ndarray], poses: Sequence[np.ndarray],
                T_cam_velo: np.ndarray, cfg: Optional[MapConfig] = None,
                bev_cfg: Optional[BEVConfig] = None):
    """Yield `(index, BEVGrid)` per frame, exactly as a live planner would see them.

    Streaming rather than batch: the window fills as the drive progresses, so the first few
    maps are genuinely thinner than the later ones — the same warm-up a real system has.
    """
    acc = ScanAccumulator(T_cam_velo, cfg, bev_cfg)
    for i, pts in enumerate(scans):
        acc.add(pts, poses[i])
        yield i, acc.grid()
