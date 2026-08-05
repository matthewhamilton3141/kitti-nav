"""Label-free estimation of moving obstacles from BEV occupancy — the feed a dynamic shield needs.

The static shield treats every occupied cell as a frozen wall. A real one has to know which
cells are *moving* and how fast, so it can reason about where an obstacle will be during the
braking maneuver rather than where it is now. This module recovers that from occupancy alone —
no object labels — so it works on any drive; the KITTI tracklets (`kitti.py`) are kept for
*validating* it (`scripts/eval_dynamics.py`), not for running it.

The method is deliberately simple and auditable: given a short window of occupancy grids **all in
a common frame** (the caller compensates ego motion by fusing each scan through the poses), it
finds connected components in each, traces every current component backward through the window by
nearest-centroid matching, and fits a constant velocity to that little trajectory.

The reason it needs a *window*, not just two frames, is the finding that motivated it: a single
frame-pair cannot tell a parked car from a moving one. Both are object-sized blobs, and a parked
car's centroid still jitters between scans as the sweep lands on different points — enough to read
as several m/s. Against KITTI labels the pair method was ~40% detection at **98% false positives**
(drive 0009 has 89 parked cars). The discriminator is **temporal coherence**: real motion is
directionally consistent, so its trajectory's net displacement is close to its path length, while
jitter random-walks and its path dwarfs its net. Requiring `net / path >= coherence` collapses the
false movers while keeping the real ones.

Honest limits that remain: it is blind to rotation and reports the constant velocity the shield's
reachable set consumes, not a full track; and it still leans on registration being good within the
short window (true for GT poses and for VO's small within-window drift). The residual
false-positive rate is reported by `scripts/eval_dynamics.py`, and it costs the shield speed, not
safety — a spurious slow mover is braked for like any obstacle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bev import BEVConfig


@dataclass(frozen=True)
class MovingObstacle:
    """A moving occupancy blob: an oriented footprint and a constant velocity, in world metres.

    `box` is `(cx, cy, yaw, l, w)` — the same tuple `bev.rasterize_box` consumes — so a
    predicted future footprint (`box` translated by `velocity * t`) drops straight back into the
    grid the shield reads. Coordinates are in the frame the input occupancy was given in.
    """

    box: np.ndarray                 # (cx, cy, yaw, l, w)
    velocity: np.ndarray            # (vx, vy) m/s
    n_cells: int

    @property
    def centre(self) -> np.ndarray:
        return self.box[:2]

    @property
    def speed(self) -> float:
        return float(np.hypot(*self.velocity))

    def box_at(self, t: float) -> np.ndarray:
        """The footprint predicted `t` seconds ahead at constant velocity."""
        b = self.box.copy()
        b[:2] = b[:2] + self.velocity * t
        return b


@dataclass(frozen=True)
class _Blob:
    centre: np.ndarray              # centroid (world xy)
    box: np.ndarray                 # (cx, cy, yaw, l, w) min-area footprint
    n_cells: int


def _components(occ: np.ndarray, cfg: BEVConfig, min_cells: int) -> list[_Blob]:
    """Connected occupancy components as world-frame blobs, dropping specks below `min_cells`."""
    import cv2

    n, labels = cv2.connectedComponents(occ.astype(np.uint8), connectivity=8)
    blobs: list[_Blob] = []
    for lab in range(1, n):
        rows, cols = np.where(labels == lab)
        if len(rows) < min_cells:
            continue
        # Cell (row, col) centres -> world (x forward from rows, y left from cols).
        xs = cfg.x_min + (rows + 0.5) * cfg.resolution
        ys = cfg.y_min + (cols + 0.5) * cfg.resolution
        centre = np.array([xs.mean(), ys.mean()])
        pts = np.stack([ys, xs], axis=1).astype(np.float32)      # cv2 wants (x=y_world, y=x_world)
        (bx, by), (bw, bh), ang = cv2.minAreaRect(pts)
        # cv2's rect is in (y_world, x_world); map back and fold the longer side into `l`.
        yaw = np.deg2rad(ang)
        l, w = (bh, bw) if bh >= bw else (bw, bh)
        if bh < bw:
            yaw += np.pi / 2
        box = np.array([by, bx, yaw, max(l, cfg.resolution), max(w, cfg.resolution)])
        blobs.append(_Blob(centre, box, len(rows)))
    return blobs


def estimate_obstacle_velocities(occ_window: list[np.ndarray], cfg: BEVConfig, dt: float, *,
                                 min_speed: float = 1.0, gate: float = 2.0, min_cells: int = 6,
                                 max_extent: float = 14.0,
                                 coherence: float = 0.7) -> list[MovingObstacle]:
    """Moving obstacles from a window of common-frame occupancy grids, oldest first.

    Each blob in the newest grid is traced backward through the window by nearest-centroid
    matching (step gate `gate` metres, ~ the most a vehicle moves per frame), and a constant
    velocity is fit to its centroid trajectory. Four filters keep it off static clutter:

      * `min_cells` drops the specks the ray raster leaves;
      * `max_extent` (m) drops components longer than any vehicle — a building or kerb run is
        one huge component whose centroid wanders meaninglessly;
      * `min_speed` drops everything below a speed floor;
      * `coherence` = required `net displacement / path length`, the temporal-consistency test
        that separates a real mover (goes straight, ratio near 1) from centroid jitter on a
        parked car (random-walks, path >> net). This is what the window is *for*.

    Needs at least two grids; the trace requires a match in every window frame, so a blob that
    appears mid-window (a newly-revealed object) is not yet reported. Returns constant-velocity
    `MovingObstacle`s in the window's common frame.
    """
    if len(occ_window) < 2:
        return []
    frames = [[c for c in _components(o, cfg, min_cells) if c.box[3] <= max_extent]
              for o in occ_window]
    if any(len(f) == 0 for f in frames[:-1]):
        return []

    span = (len(occ_window) - 1) * dt
    out: list[MovingObstacle] = []
    for c in frames[-1]:
        traj = [c.centre]
        cur = c.centre
        for earlier in reversed(frames[:-1]):
            centres = np.stack([b.centre for b in earlier])
            d = np.linalg.norm(centres - cur, axis=1)
            j = int(np.argmin(d))
            if d[j] > gate:
                break                                            # lost the trace this far back
            cur = earlier[j].centre
            traj.append(cur)
        if len(traj) < len(occ_window):
            continue
        traj = np.asarray(traj[::-1])                            # oldest -> newest
        net = float(np.linalg.norm(traj[-1] - traj[0]))
        path = float(np.linalg.norm(np.diff(traj, axis=0), axis=1).sum())
        if net / span < min_speed or path <= 0 or net / path < coherence:
            continue
        out.append(MovingObstacle(c.box, (traj[-1] - traj[0]) / span, c.n_cells))
    return out
