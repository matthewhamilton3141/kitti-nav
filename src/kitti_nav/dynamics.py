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
from .vehicle import (
    ShieldResult,
    VehicleConfig,
    VehicleState,
    evasive_steer_candidates,
    footprint_discs,
    step_state,
)


@dataclass(frozen=True)
class MovingObstacle:
    """A moving occupancy blob: an oriented footprint and a constant velocity, in world metres.

    `box` is `(cx, cy, yaw, l, w)` — the same tuple `bev.rasterize_box` consumes — so a
    predicted future footprint (`box` translated by `velocity * t`) drops straight back into the
    grid the shield reads. Coordinates are in the frame the input occupancy was given in.
    """

    box: np.ndarray                 # (cx, cy, yaw, l, w)
    velocity: np.ndarray            # (vx, vy) m/s
    n_cells: int = 0                # occupancy support, when it came from the estimator

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


# --- the dynamic shield: braking against where obstacles *will be* --------------------------

def point_box_distance(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Distance (m) from each `(N, 2)` point to an oriented box `(cx, cy, yaw, l, w)`, 0 inside.

    Rotates the points into the box's own frame and takes the axis-aligned point-to-rectangle
    distance — the exact clearance the shield needs against a moving footprint, without
    rasterising it into a grid every timestep.
    """
    p = np.asarray(points, float).reshape(-1, 2)
    cx, cy, yaw, l, w = (float(v) for v in np.asarray(box, float).reshape(5))
    d = p - np.array([cx, cy])
    c, s = np.cos(yaw), np.sin(yaw)
    xr = c * d[:, 0] + s * d[:, 1]                        # world -> box frame (rotate by -yaw)
    yr = -s * d[:, 0] + c * d[:, 1]
    dx = np.maximum(np.abs(xr) - l / 2.0, 0.0)
    dy = np.maximum(np.abs(yr) - w / 2.0, 0.0)
    return np.hypot(dx, dy)


class BoxField:
    """An `ObstacleField` of oriented boxes — lets the static shield run against boxes directly.

    The `distance_to_obstacles` seam `vehicle.py` already speaks, backed by boxes instead of a
    grid. Used to put a moving obstacle's *current* footprint in front of the ordinary shield,
    which is the control condition the dynamic shield is measured against.
    """

    def __init__(self, boxes):
        self.boxes = [np.asarray(b, float).reshape(5) for b in boxes]

    def distance_to_obstacles(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float).reshape(-1, 2)
        if not self.boxes:
            return np.full(len(p), np.inf)
        return np.min([point_box_distance(p, b) for b in self.boxes], axis=0)


def moving_clearance(state: VehicleState, static_field, movers: list[MovingObstacle],
                     cfg: VehicleConfig, t: float) -> float:
    """Signed clearance (m) of the footprint against static geometry plus movers at time `t`.

    The moving obstacles are advanced to `box_at(t)` — where constant velocity predicts them —
    so a rollout that threads increasing `t` checks the vehicle against where each obstacle
    *will be*, not where it is now.
    """
    centres, radius = footprint_discs(state, cfg)
    d = (static_field.distance_to_obstacles(centres) if static_field is not None
         else np.full(len(centres), np.inf))
    for m in movers:
        d = np.minimum(d, point_box_distance(centres, m.box_at(t)))
    return float(np.min(d) - radius)


def can_stop_safely_dynamic(state: VehicleState, static_field, movers: list[MovingObstacle],
                            cfg: VehicleConfig, t0: float = 0.0) -> bool:
    """Full-braking rollout that also advances the obstacles — the dynamic inductive invariant.

    Identical to `vehicle.can_stop_safely` but the obstacles move with the clock: at braking
    step `k` the vehicle is checked against the obstacles at `t0 + k*dt`. Holding steer stays
    the conservative reading. Because both vehicle and obstacles are rolled forward under the
    *predicted* constant velocity, certifying a state means the car can brake clear of the
    obstacles' predicted paths — sound to the extent the prediction holds, which is the honest
    scope of any behaviour-predicting planner (cf. RSS's bounded-behaviour assumption).
    """
    s = state
    t = t0
    for _ in range(cfg.max_brake_steps):
        if moving_clearance(s, static_field, movers, cfg, t) < cfg.safety_margin:
            return False
        if s.v <= 1e-9:
            return True
        s = step_state(s, -cfg.max_decel, s.steer, cfg)
        t += cfg.dt
    return False


def dynamic_safety_shield(accel_cmd: float, steer_cmd: float, state: VehicleState,
                          static_field, movers: list[MovingObstacle],
                          cfg: VehicleConfig) -> ShieldResult:
    """Braking-aware shield that reasons about a moving obstacle's predicted path.

    The exact structure of `vehicle.safety_shield` — search accel from commanded down to full
    braking, over the commanded steer then the held steer, admit the first level that is clear
    now *and* leaves a certifiable dynamic stop — but every clearance is time-indexed: the
    successor state sits one step ahead, so obstacles are advanced by one `dt` for the
    immediate check and by the rollout clock thereafter.

    Where the static shield sees only where an obstacle *is*, this brakes for where it is
    *going*, which is the difference between stopping short of a car crossing ahead and driving
    into where it will be. With `cfg.n_evasive_steers > 0` it also **swerves** for a predicted
    path it cannot brake clear of — the moving-world case that matters most, since a car cutting
    in is exactly what braking alone cannot always escape. Falls back to held-steer maximum
    braking with `ics=True` only when neither slowing nor steering can certify a clear stop.
    """
    hi = float(np.clip(accel_cmd, -cfg.max_decel, cfg.max_accel))
    accels = np.linspace(hi, -cfg.max_decel, max(int(cfg.n_accel_candidates), 2))

    def admissible(steer: float, a: float) -> bool:
        nxt = step_state(state, float(a), float(steer), cfg)
        return (moving_clearance(nxt, static_field, movers, cfg, cfg.dt) >= cfg.safety_margin
                and can_stop_safely_dynamic(nxt, static_field, movers, cfg, t0=cfg.dt))

    steer_options = [steer_cmd]
    if not np.isclose(steer_cmd, state.steer):
        steer_options.append(state.steer)

    for steer in steer_options:
        for a in accels:
            if admissible(steer, a):
                intervened = not (np.isclose(a, hi) and np.isclose(steer, steer_cmd))
                return ShieldResult(accel=float(a), steer=float(steer),
                                    intervened=bool(intervened), ics=False)

    # Neither commanded nor held steer leaves a certifiable stop against the predicted paths:
    # swerve before giving up. Sound for the same reason as the static shield's evasive pass —
    # every candidate carries its own dynamic braking certificate.
    for steer in evasive_steer_candidates(steer_cmd, state.steer, cfg):
        for a in accels:
            if admissible(steer, a):
                return ShieldResult(accel=float(a), steer=float(steer),
                                    intervened=True, ics=False)

    return ShieldResult(accel=-cfg.max_decel, steer=state.steer, intervened=True, ics=True)


def max_safe_speed_dynamic(static_field, movers: list[MovingObstacle], cfg: VehicleConfig,
                           state: VehicleState, tol: float = 0.05) -> float:
    """Highest speed from which a dynamic stop is still certifiable at `state`'s pose.

    The moving counterpart of `vehicle.max_safe_speed`: bisected because `can_stop_safely_dynamic`
    has no closed form, and valid because the predicate is monotone in speed — a faster car
    travels further into the obstacles' predicted paths. Directly comparable to the static
    permitted speed, so the gap is exactly what reasoning about motion costs (or saves).
    """
    def ok(v: float) -> bool:
        return can_stop_safely_dynamic(
            VehicleState(state.x, state.y, state.yaw, v, state.steer), static_field, movers, cfg)

    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, cfg.max_speed
    if ok(hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo
