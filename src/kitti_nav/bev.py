"""Velodyne lidar -> top-down BEV occupancy grid, and a grid-native clearance field.

This is the representation the planner actually consumes — the deliberate choice this repo
is built around. Production AV stacks plan on occupancy/BEV, not on photorealistic
reconstructions; see the README.

**Frame.** Everything here stays in the KITTI *Velodyne* frame: **+x forward, +y left,
+z up**, origin at the lidar. That is already the standard robotics convention, and it is
exactly the frame `vehicle.py`'s bicycle model works in — so the planner consumes this grid
with no axis juggling at all. (The *camera* frame is the awkward one: +x right, +y down,
+z forward. Converting lidar into camera coordinates and then back out to a ground plane
would be two conversions to accomplish nothing.)

**Ground removal.** The single most important step. A raw scan is dominated by road surface
returns, and a grid built without removing them is uniformly occupied and useless. Two
strategies are implemented, and the default was chosen by measuring, not by assumption:

  * ``"plane"`` — keep points inside a height band above one global ground height. Simple
    and fast, but it assumes the road is a single flat plane at a known height.
  * ``"height_diff"`` (**default**) — estimate the ground *locally*, as the lowest return in
    a small neighbourhood of each cell, and mark the cell occupied when its highest return
    rises far enough above that floor. Flat road barely rises above its own floor; a car,
    pole, or wall does. The estimate is taken over a neighbourhood rather than the single
    cell because many cells hold no ground return at all — lidar rings spread apart with
    range, and nothing beneath an overhanging canopy reaches the road — and such a cell
    would otherwise treat the structure above it as its own ground.

The global-plane assumption turned out to be shaky on this data. KITTI documents the lidar
at 1.73 m above the ground, but the near-field returns on drive 0009 spread over ~0.4 m in
``z`` and are visibly bimodal — sensor pitch plus real road slope — so no single constant is
right everywhere in the scan. ``"height_diff"`` sidesteps the problem: because each cell is
compared against its own floor, a sloping or pitched road costs nothing, and there is no
ground height to tune. Overhead structure is excluded *before* the spread is measured, so a
tree canopy over open road does not turn that cell into a 4 m-tall phantom obstacle.

**Occupancy convention:** 1 = occupied, 0 = free. Cells no lidar ray reached are *not*
distinguished from free space here — a known simplification, discussed in `README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

# KITTI's Velodyne HDL-64E is mounted 1.73 m above the ground (per the sensor-setup diagram
# in Geiger et al.), so road returns land near z = -1.73 in the lidar frame.
KITTI_LIDAR_HEIGHT = 1.73


@dataclass(frozen=True)
class BEVConfig:
    """Grid extent, resolution, and the ground/height band used to filter the scan."""

    # Grid extent in metres, in the lidar frame. Forward-biased: the planner needs to see
    # far enough ahead to cover the braking envelope (at 15 m/s that is ~25 m).
    x_min: float = -10.0     # behind the car
    x_max: float = 50.0      # ahead
    y_min: float = -20.0     # right
    y_max: float = 20.0      # left
    resolution: float = 0.2  # metres per cell

    # Ground-removal strategy: "height_diff" (per-cell, default) or "plane" (global).
    ground_mode: str = "height_diff"

    # Height band above the ground that counts as an obstacle. `min_height` applies only to
    # "plane" mode; `max_height` applies to both (it is the drive-under clearance).
    min_height: float = 0.25   # below this is road surface / kerb noise
    max_height: float = 2.5    # above this the car passes underneath (canopy, signs)

    # "height_diff": a cell is occupied when its highest and lowest returns differ by at
    # least this much. 0.3 m is comfortably above road roughness and lidar noise (~0.02 m
    # here) while still catching a kerb-height obstacle.
    height_diff_threshold: float = 0.3

    # "height_diff": side length, in cells, of the neighbourhood a cell borrows its ground
    # estimate from. 5 cells = 1 m at the default resolution — wide enough to cover the gap
    # between lidar rings at range, narrow enough not to flatten a real kerb.
    ground_window: int = 5

    # "plane": ground height in the lidar frame. None estimates it per scan.
    ground_z: float | None = -KITTI_LIDAR_HEIGHT

    @property
    def shape(self) -> tuple[int, int]:
        """Grid shape as (rows along +x forward, cols along +y left)."""
        return (int(round((self.x_max - self.x_min) / self.resolution)),
                int(round((self.y_max - self.y_min) / self.resolution)))


def estimate_ground_z(points: np.ndarray, near_radius: float = 20.0,
                      percentile: float = 5.0) -> float:
    """Estimate the ground plane height (m, lidar frame) from a scan.

    Takes a low percentile of `z` over near-field returns. The percentile (rather than the
    minimum) rejects the handful of spurious low returns every lidar produces, and the
    near-field restriction keeps a downhill road far ahead from dragging the estimate down.
    Flat-world assumption — fine for KITTI's city drives, wrong on a steep hill.
    """
    pts = np.asarray(points, float)
    near = pts[np.hypot(pts[:, 0], pts[:, 1]) < near_radius]
    if len(near) < 100:
        near = pts
    return float(np.percentile(near[:, 2], percentile))


def _in_bounds_cells(pts: np.ndarray, cfg: BEVConfig) -> tuple[np.ndarray, np.ndarray]:
    """Points inside the grid extent, plus their flat cell indices.

    The bounds check must happen *before* the cast to integers: negative coordinates floor
    toward zero and would silently wrap around to the far edge of the grid.
    """
    inside = ((pts[:, 0] >= cfg.x_min) & (pts[:, 0] < cfg.x_max)
              & (pts[:, 1] >= cfg.y_min) & (pts[:, 1] < cfg.y_max))
    sel = pts[inside]
    _, cols = cfg.shape
    r = ((sel[:, 0] - cfg.x_min) / cfg.resolution).astype(np.int64)
    c = ((sel[:, 1] - cfg.y_min) / cfg.resolution).astype(np.int64)
    return sel, r * cols + c


def occupancy_from_scan(points: np.ndarray, cfg: BEVConfig | None = None) -> np.ndarray:
    """Rasterise a Velodyne scan into a `(rows, cols)` uint8 occupancy grid.

    `points` is `(N, 3)` or `(N, 4)` (x, y, z[, reflectance]) in the lidar frame.
    """
    cfg = cfg or BEVConfig()
    pts = np.asarray(points, float)[:, :3]
    rows, cols = cfg.shape
    if len(pts) == 0:
        return np.zeros((rows, cols), np.uint8)

    sel, idx = _in_bounds_cells(pts, cfg)
    if len(sel) == 0:
        return np.zeros((rows, cols), np.uint8)

    if cfg.ground_mode == "plane":
        ground = cfg.ground_z if cfg.ground_z is not None else estimate_ground_z(pts)
        height = sel[:, 2] - ground
        keep = (height >= cfg.min_height) & (height <= cfg.max_height)
        flat = np.zeros(rows * cols, np.uint8)
        flat[idx[keep]] = 1
        return flat.reshape(rows, cols)

    if cfg.ground_mode != "height_diff":
        raise ValueError(f"unknown ground_mode {cfg.ground_mode!r}")

    import cv2

    z = sel[:, 2]
    n_cells = rows * cols
    NO_GROUND = 1e6                     # finite sentinel; cv2 min-filtering dislikes inf

    # Pass 1: each cell's lowest return.
    cell_min = np.full(n_cells, NO_GROUND, np.float32)
    np.minimum.at(cell_min, idx, z.astype(np.float32))

    # Widen that to the lowest return in a small neighbourhood. A cell often has no ground
    # return of its own — lidar rings spread apart with range, and nothing under an
    # overhanging canopy reaches the road — and such a cell would otherwise take its own
    # floor as "ground" and report the structure above it as an obstacle. Eroding with a
    # square kernel is exactly a local-minimum filter.
    k = max(int(cfg.ground_window), 1)
    ground = cv2.erode(cell_min.reshape(rows, cols), np.ones((k, k), np.uint8)).ravel()

    # Drop overhead structure *before* measuring spread, or a canopy over open road would
    # register as a several-metre obstacle in an otherwise drivable cell.
    keep = (z - ground[idx]) <= cfg.max_height

    # Pass 2: the highest remaining return per cell. Spread = local obstacle height.
    cell_max = np.full(n_cells, -np.inf)
    np.maximum.at(cell_max, idx[keep], z[keep])

    spread = cell_max - ground
    occupied = np.isfinite(spread) & (ground < NO_GROUND) & \
        (spread >= cfg.height_diff_threshold)
    return occupied.reshape(rows, cols).astype(np.uint8)


class BEVGrid:
    """An occupancy grid plus the Euclidean distance field the safety shield queries.

    Implements the `ObstacleField` contract in `vehicle.py` (`distance_to_obstacles`), so the
    braking shield can run directly against real lidar with no conversion into circles.
    Going grid-native rather than fitting circles to occupied cells is the point: circles
    would discard exactly the arbitrary obstacle shape that occupancy represents well.
    """

    def __init__(self, occupancy: np.ndarray, cfg: BEVConfig | None = None,
                 outside_is_free: bool = True, unknown: np.ndarray | None = None):
        self.cfg = cfg or BEVConfig()
        self.occupancy = np.asarray(occupancy, np.uint8)
        self.outside_is_free = outside_is_free
        if self.occupancy.shape != self.cfg.shape:
            raise ValueError(f"grid {self.occupancy.shape} != config shape {self.cfg.shape}")

        # Optional third occupancy class: cells no lidar ray ever passed through, as opposed
        # to cells observed and found clear. `None` (the default) is the binary map every
        # existing consumer sees — occupied vs "everything else is free". Free-space carving
        # (`mapping.fuse_map` with carving on) fills this in; it is carried here for measurement and
        # rendering only. The shield and the RL rays still read `occupancy` alone this pass,
        # so an un-consumed `unknown` mask cannot change any existing number — folding it into
        # `distance_field`/`ray_distances` is a deliberate later step.
        self.unknown = None if unknown is None else np.asarray(unknown, bool)
        if self.unknown is not None and self.unknown.shape != self.cfg.shape:
            raise ValueError(f"unknown {self.unknown.shape} != config shape {self.cfg.shape}")

    @classmethod
    def from_scan(cls, points: np.ndarray, cfg: BEVConfig | None = None,
                  outside_is_free: bool = True) -> "BEVGrid":
        cfg = cfg or BEVConfig()
        return cls(occupancy_from_scan(points, cfg), cfg, outside_is_free)

    # -- geometry ------------------------------------------------------------------------

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """World `(n, 2)` metres -> integer `(n, 2)` `(row, col)`. May fall outside the grid."""
        p = np.asarray(xy, float).reshape(-1, 2)
        return np.stack([(p[:, 0] - self.cfg.x_min) / self.cfg.resolution,
                         (p[:, 1] - self.cfg.y_min) / self.cfg.resolution],
                        axis=1).astype(np.int32)

    def cell_to_world(self, rc: np.ndarray) -> np.ndarray:
        """Integer `(n, 2)` `(row, col)` -> world `(n, 2)` metres at the **cell centre**."""
        c = np.asarray(rc, float).reshape(-1, 2)
        return np.stack([self.cfg.x_min + (c[:, 0] + 0.5) * self.cfg.resolution,
                         self.cfg.y_min + (c[:, 1] + 0.5) * self.cfg.resolution], axis=1)

    @property
    def occupied_fraction(self) -> float:
        return float(self.occupancy.mean())

    # -- distance field ------------------------------------------------------------------

    @cached_property
    def distance_field(self) -> np.ndarray:
        """Metres from each cell to the nearest occupied cell, as a float32 grid.

        Computed once with OpenCV's exact Euclidean distance transform and cached, because
        the shield queries it thousands of times per decision (candidate actions x braking
        horizon) and recomputing would dominate the runtime.

        A half-cell-diagonal is subtracted so the result is a **conservative** lower bound:
        the transform measures to the nearest occupied cell's *centre*, while the obstacle
        may extend to that cell's corner. Under-reporting clearance can only make the shield
        brake earlier than strictly necessary, never later.
        """
        import cv2

        if not self.occupancy.any():
            return np.full(self.occupancy.shape, np.inf, np.float32)

        # distanceTransform measures to the nearest ZERO pixel, so free cells must be nonzero.
        free = (1 - self.occupancy).astype(np.uint8) * 255
        dist = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        dist *= self.cfg.resolution
        half_diag = self.cfg.resolution * np.sqrt(2.0) / 2.0
        return np.maximum(dist - half_diag, 0.0).astype(np.float32)

    def distance_to_obstacles(self, points: np.ndarray) -> np.ndarray:
        """Distance (m) from each query point to the nearest obstacle — the shield's hook.

        Points outside the grid return `inf` when `outside_is_free` (the default: the grid
        is assumed to cover the planning horizon), else `-inf` to forbid leaving it.
        """
        p = np.asarray(points, float).reshape(-1, 2)
        rows, cols = self.cfg.shape
        rc = self.world_to_cell(p)
        inside = ((rc[:, 0] >= 0) & (rc[:, 0] < rows)
                  & (rc[:, 1] >= 0) & (rc[:, 1] < cols))

        out = np.full(len(p), np.inf if self.outside_is_free else -np.inf, float)
        if inside.any():
            sel = rc[inside]
            out[inside] = self.distance_field[sel[:, 0], sel[:, 1]]
        return out

    def ray_distances(self, origin: np.ndarray, yaw: float, angles: np.ndarray,
                      max_range: float = 30.0,
                      outside_blocks: bool = True) -> np.ndarray:
        """Cast a fan of rays from `origin` and return the free distance (m) along each.

        The observation a learned policy consumes. Ray marching, not analytic intersection,
        because the obstacles are a raster: stepping at half a cell guarantees no occupied
        cell is skipped (Nyquist on the grid), and the whole fan is vectorised into one
        array lookup rather than a Python loop per ray.

        `angles` are relative to `yaw`. Rays leaving the grid stop there when
        `outside_blocks`, which treats the edge of what the sensor mapped as a wall — the
        honest reading, since unmapped space is unknown rather than known-clear.
        """
        rows, cols = self.cfg.shape
        step = self.cfg.resolution * 0.5
        t = np.arange(0.0, max_range + step, step)                    # (S,)
        dirs = yaw + np.asarray(angles, float).reshape(-1)            # (R,)

        px = origin[0] + np.cos(dirs)[:, None] * t[None, :]           # (R, S)
        py = origin[1] + np.sin(dirs)[:, None] * t[None, :]

        r = ((px - self.cfg.x_min) / self.cfg.resolution).astype(np.int32)
        c = ((py - self.cfg.y_min) / self.cfg.resolution).astype(np.int32)
        inside = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)

        blocked = np.zeros(px.shape, bool)
        blocked[inside] = self.occupancy[r[inside], c[inside]] > 0
        if outside_blocks:
            blocked |= ~inside

        # First blocked sample per ray; rays that never hit return the full range.
        any_hit = blocked.any(axis=1)
        first = np.where(any_hit, blocked.argmax(axis=1), len(t) - 1)
        return np.minimum(t[first], max_range)

    def covers_stopping_distance(self, distance: float) -> bool:
        """Does the grid extend far enough ahead to certify a stop of `distance` metres?

        `outside_is_free=True` assumes unseen space is drivable, which is only defensible
        while the braking envelope stays inside the grid. Callers should assert this.
        """
        return self.cfg.x_max >= distance
