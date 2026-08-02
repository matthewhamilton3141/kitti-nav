"""Kinematic bicycle (Ackermann) vehicle model + a braking-aware hard safety shield.

Pure NumPy, no GPU / torch / gym — runs and is fully unit-tested on a laptop, the same
discipline that made the diff-drive nav core in this author's `gsplat-rt` repo portable
across physics backends.

Provenance: the *design* of the shield — a runtime filter that wraps any policy, sharing
one `step_state` with the simulator so lookahead and integration can't disagree — comes
from `gsplat-rt`'s `src/isaac/nav_sim.py` (`predict_pose` / `clearance_at` /
`safety_shield`). The kinematics and the safety argument are re-derived here, because a
car is not a differential-drive robot in two ways that matter:

  1. **No rotation in place.** The diff-drive shield's fallback was "forbid forward motion
     and let it spin, which cannot collide." A bicycle model's yaw rate is `v/L·tan δ`, so
     at zero speed it cannot turn either. Stopping is still the safe fallback, but it is no
     longer a *free* one — you have to already be able to stop.
  2. **One-step lookahead is unsound at speed.** At 15 m/s with 4.5 m/s² braking the car
     needs ~25 m to stop, but one 0.1 s step advances it only 1.5 m. A filter that checks
     only the next pose will happily drive into a wall 10 m away, finding every individual
     step "clear" until none of them are. So the shield here asks a strictly stronger
     question: *after taking this action, does a full-braking trajectory from the resulting
     state stop without hitting anything?* An action is admitted only if the answer is yes,
     which makes the invariant inductive — every admitted state retains a safe stop.

That is the inevitable-collision-state / reachability argument (Fraichard & Asama 2004;
the reasoning behind RSS, Shalev-Shwartz et al. 2017), implemented from the concept — see
`ATTRIBUTION.md`.

Steering needs care, and an early version of this file got it wrong: it modulated only
throttle and let steering pass through untouched. That breaks the induction, because the
braking trajectory was certified holding the *previous* wheel angle — command a different
angle on the next step and full braking can curve into an obstacle the certificate never
covered. A randomised rollout test caught it. The shield therefore falls back to *holding
the last certified steer*, which the induction hypothesis guarantees is stoppable.

Known limitation, stated honestly: the shield chooses between the commanded steer and the
held steer; it never searches for an *evasive* one. It will brake for an obstacle it might
have swerved around. That is a harder search and is not claimed here.

Conventions: SI throughout (m, s, rad). The pose `(x, y)` is the **rear-axle centre**, yaw
is 0 along +x and increases counter-clockwise, and positive steer is a left turn.
Obstacles are `(N, 3)` circles `(cx, cy, r)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Union

import numpy as np


class ObstacleField(Protocol):
    """Anything that can answer "how far is the nearest obstacle from these points?".

    The seam that lets the shield run unchanged against either hand-made circular obstacles
    or a real lidar BEV occupancy grid (`bev.BEVGrid`). The shield only ever needs distances,
    never the obstacle representation itself — so no conversion between the two is required,
    and in particular occupancy never has to be approximated by circles.
    """

    def distance_to_obstacles(self, points: np.ndarray) -> np.ndarray:
        """Distance (m) from each `(n, 2)` query point to the nearest obstacle surface."""
        ...


class CircleField:
    """Obstacle field backed by an `(N, 3)` array of circles `(cx, cy, r)`."""

    def __init__(self, circles: Optional[np.ndarray]):
        arr = np.zeros((0, 3), float) if circles is None else \
            np.asarray(circles, float).reshape(-1, 3)
        self.circles = arr

    def distance_to_obstacles(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, float).reshape(-1, 2)
        if self.circles.size == 0:
            return np.full(len(p), np.inf)
        d = np.linalg.norm(p[:, None, :] - self.circles[None, :, :2], axis=2)
        return np.min(d - self.circles[None, :, 2], axis=1)


Obstacles = Union[np.ndarray, ObstacleField, None]


def as_field(obstacles: Obstacles) -> ObstacleField:
    """Coerce circles-or-field into a field. Pass-through if it already is one."""
    if hasattr(obstacles, "distance_to_obstacles"):
        return obstacles                                  # already an ObstacleField
    return CircleField(obstacles)


@dataclass(frozen=True)
class VehicleConfig:
    """Vehicle geometry, actuation limits, and shield parameters.

    Defaults describe the KITTI recording platform (a VW Passat B6 station wagon), so the
    model is dimensionally consistent with the drives we replay.
    """

    # --- geometry (m) ---
    wheelbase: float = 2.71        # front-to-rear axle distance
    length: float = 4.77           # bumper to bumper
    width: float = 1.82
    rear_overhang: float = 0.97    # rear axle back to the rear bumper
    # The body rectangle is approximated by this many covering discs. The cover is always
    # conservative, but its lateral excess over the true half-width shrinks as discs are
    # added: 3 discs inflate the car by 0.30 m per side, 5 by 0.12 m, 7 by 0.06 m. At 3 the
    # inflation was large enough to report contact with real KITTI roadside geometry the car
    # actually cleared, so 5 is the default; cost is linear in this number.
    n_footprint_discs: int = 5

    # --- integration ---
    dt: float = 0.1

    # --- actuation limits ---
    max_speed: float = 15.0        # m/s (~54 km/h, KITTI city/residential range)
    min_speed: float = 0.0         # no reverse by default
    max_accel: float = 2.0         # m/s^2
    max_decel: float = 4.5         # m/s^2, firm-but-not-emergency braking
    max_steer: float = 0.52        # rad (~30 deg) road-wheel angle
    max_steer_rate: float = 0.6    # rad/s — the steering rack cannot snap instantly

    # --- shield ---
    safety_margin: float = 0.30    # clearance (m) the shield keeps free
    n_accel_candidates: int = 11   # throttle/brake levels searched, commanded -> full brake
    max_brake_steps: int = 200     # hard cap on the braking rollout horizon

    @property
    def front_overhang(self) -> float:
        """Rear axle forward to the front bumper."""
        return self.length - self.rear_overhang


@dataclass(frozen=True)
class VehicleState:
    """Planar vehicle state: rear-axle pose, forward speed, and current road-wheel angle.

    Steering is *state*, not a direct input, because the rack is rate-limited — the shield
    has to reason about the angle the car will actually have, not the one commanded.
    """

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    steer: float = 0.0

    @property
    def xy(self) -> np.ndarray:
        return np.array([self.x, self.y], float)


def _wrap(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def step_state(state: VehicleState, accel: float, steer_cmd: float,
               cfg: VehicleConfig) -> VehicleState:
    """Advance one control step under the kinematic bicycle model.

    Shared verbatim by the simulator and by the shield's lookahead, so a state the shield
    certified as safe is the state the sim actually reaches — a filter that predicts
    differently than the world integrates is unsound.

    Speed and yaw are integrated at the midpoint of the step (rather than with the values
    at its start), which keeps curved motion accurate at the fairly coarse `dt = 0.1 s`
    used for control.
    """
    # Steering rack: slew toward the command, bounded by rate and by the mechanical stop.
    max_delta = cfg.max_steer_rate * cfg.dt
    target = float(np.clip(steer_cmd, -cfg.max_steer, cfg.max_steer))
    steer = float(np.clip(target, state.steer - max_delta, state.steer + max_delta))

    accel = float(np.clip(accel, -cfg.max_decel, cfg.max_accel))
    v_new = float(np.clip(state.v + accel * cfg.dt, cfg.min_speed, cfg.max_speed))
    v_mid = 0.5 * (state.v + v_new)

    # Bicycle yaw rate: v / L * tan(delta). At v = 0 this is 0 — no rotation in place.
    yaw_rate = v_mid / cfg.wheelbase * np.tan(steer)
    yaw_mid = state.yaw + 0.5 * yaw_rate * cfg.dt

    return VehicleState(
        x=state.x + v_mid * cfg.dt * float(np.cos(yaw_mid)),
        y=state.y + v_mid * cfg.dt * float(np.sin(yaw_mid)),
        yaw=_wrap(state.yaw + yaw_rate * cfg.dt),
        v=v_new,
        steer=steer,
    )


def turning_radius(steer: float, cfg: VehicleConfig) -> float:
    """Signed path radius (m) of the rear axle at road-wheel angle `steer`; inf when straight."""
    t = np.tan(steer)
    return float("inf") if abs(t) < 1e-12 else cfg.wheelbase / float(t)


def stopping_distance(v: float, cfg: VehicleConfig) -> float:
    """Distance (m) to come to rest from `v` under `max_decel` — the v²/2a the shield exists for."""
    return float(max(v, 0.0) ** 2 / (2.0 * cfg.max_decel))


def footprint_discs(state: VehicleState, cfg: VehicleConfig) -> tuple[np.ndarray, float]:
    """Cover the vehicle rectangle with equal discs; returns (centres `(n, 2)`, radius).

    Exact rectangle-vs-circle tests are cheap individually but the shield runs thousands of
    them per decision (candidate actions x braking horizon x obstacles), so we use the
    standard planning-stack trick: split the body into `n` equal segments and bound each by
    its circumscribed disc. Radius `sqrt((L/2n)^2 + (W/2)^2)` guarantees full coverage, so
    the approximation is **conservative** — it can refuse a tight gap the car would squeeze
    through, but it can never miss a collision. For safety code, erring outward is correct.
    """
    n = max(int(cfg.n_footprint_discs), 1)
    seg = cfg.length / n
    radius = float(np.hypot(seg / 2.0, cfg.width / 2.0))

    # Segment centres along the body axis, measured forward from the rear axle.
    offsets = -cfg.rear_overhang + seg * (np.arange(n) + 0.5)
    c, s = np.cos(state.yaw), np.sin(state.yaw)
    centres = np.stack([state.x + offsets * c, state.y + offsets * s], axis=1)
    return centres, radius


def clearance(state: VehicleState, obstacles: Obstacles, cfg: VehicleConfig) -> float:
    """Signed distance (m) from the vehicle footprint to the nearest obstacle.

    Positive is free space, 0 is touching, negative is overlap. Pure geometry with no env
    state, so the simulator and the shield share one definition of "how close are we."
    Accepts either an `(N, 3)` circle array or any `ObstacleField` (e.g. a lidar BEV grid);
    returns `inf` when there are no obstacles.
    """
    field = as_field(obstacles)
    centres, radius = footprint_discs(state, cfg)
    return float(np.min(field.distance_to_obstacles(centres)) - radius)


def can_stop_safely(state: VehicleState, obstacles: Obstacles, cfg: VehicleConfig) -> bool:
    """Does a full-braking rollout from `state`, holding the current steer, stay clear?

    This is the inductive invariant the shield maintains. Holding steer (rather than
    assuming the wheel straightens) is the conservative reading: it certifies a stop the
    car can execute with no further steering input.
    """
    field = as_field(obstacles)              # coerce once; this loop runs thousands of times
    s = state
    for _ in range(cfg.max_brake_steps):
        if clearance(s, field, cfg) < cfg.safety_margin:
            return False
        if s.v <= 1e-9:
            return True                      # at rest and clear: the state is safe forever
        s = step_state(s, -cfg.max_decel, s.steer, cfg)
    return False                             # never stopped within the horizon: don't certify


@dataclass(frozen=True)
class ShieldResult:
    """What the shield decided, and why — the `intervened` / `ics` flags are the eval stats."""

    accel: float
    steer: float
    intervened: bool      # the shield altered the commanded acceleration
    ics: bool             # inevitable-collision state: even full braking was not certifiable

    @property
    def action(self) -> np.ndarray:
        return np.array([self.accel, self.steer], np.float32)


def safety_shield(accel_cmd: float, steer_cmd: float, state: VehicleState,
                  obstacles: Obstacles, cfg: VehicleConfig) -> ShieldResult:
    """Hard braking-aware safety filter over a commanded `(accel, steer)`.

    Searches acceleration from the commanded value down to full braking and returns the
    first (i.e. least restrictive) level whose resulting state is both clear *and* still
    able to brake to a stop safely. Because admission requires the successor to retain a
    safe stop, the invariant is inductive: from a certified state the shield can always
    certify at least full braking, so it never has to admit a collision.

    That induction only closes if the fallback uses the wheel angle the certificate was
    issued under. So the search runs over two steering options: the commanded angle first,
    and — if nothing is admissible there — the vehicle's *current* angle, under which the
    previous step's certificate guarantees a safe stop exists. Passing the commanded steer
    through unconditionally is unsound, and was a real bug here (see the module docstring).

    A runtime layer over *any* policy — learned or hand-written — needing no retraining.
    If even held-steer maximum braking fails to certify, the state is already an
    inevitable-collision state (only reachable by starting from an uncertified state, or
    from an obstacle appearing inside the stopping envelope); the shield then commands
    maximum braking as the best available action and flags `ics=True` rather than
    pretending the situation is safe.
    """
    field = as_field(obstacles)              # coerce once, then reuse across every candidate
    hi = float(np.clip(accel_cmd, -cfg.max_decel, cfg.max_accel))
    lo = -cfg.max_decel
    accels = np.linspace(hi, lo, max(int(cfg.n_accel_candidates), 2))

    # Commanded steer first; held steer is the certified fallback. Skip the duplicate pass
    # when the policy already asked for the angle we are holding.
    steer_options = [steer_cmd]
    if not np.isclose(steer_cmd, state.steer):
        steer_options.append(state.steer)

    for steer in steer_options:
        for a in accels:
            nxt = step_state(state, float(a), float(steer), cfg)
            if clearance(nxt, field, cfg) >= cfg.safety_margin and \
                    can_stop_safely(nxt, field, cfg):
                intervened = not (np.isclose(a, hi) and np.isclose(steer, steer_cmd))
                return ShieldResult(accel=float(a), steer=float(steer),
                                    intervened=bool(intervened), ics=False)

    return ShieldResult(accel=lo, steer=state.steer, intervened=True, ics=True)


def max_safe_speed(obstacles: Obstacles, cfg: VehicleConfig,
                   state: Optional[VehicleState] = None, tol: float = 0.05) -> float:
    """Highest speed from which the shield could still certify a stop at `state`'s pose.

    Bisected rather than solved, because `can_stop_safely` integrates a braking rollout
    against arbitrary obstacle geometry and has no closed form. Bisection is valid because
    the predicate is **monotone in speed**: a faster car travels strictly further along the
    same braking path, so if it cannot stop at `v` it cannot stop at anything above `v`.

    This is the shield's opinion expressed as a speed limit, which makes it directly
    comparable to what a human driver actually did on the same road.
    """
    field = as_field(obstacles)
    base = state or VehicleState()

    def ok(v: float) -> bool:
        return can_stop_safely(VehicleState(base.x, base.y, base.yaw, v, base.steer),
                               field, cfg)

    if not ok(0.0):
        return 0.0                       # already too close to stop clear even at rest
    lo, hi = 0.0, cfg.max_speed
    if ok(hi):
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ok(mid) else (lo, mid)
    return lo


def shielded_rollout(policy, state: VehicleState, obstacles: Obstacles,
                     cfg: VehicleConfig, n_steps: int,
                     shield: bool = True) -> tuple[list[VehicleState], dict]:
    """Roll `policy` forward for `n_steps`, optionally through the shield; returns states + stats.

    `policy(state) -> (accel, steer)`. The unshielded path is not dead weight — it is the
    control condition that shows the shield is doing something, and the harness the
    braking-vs-one-step comparison in the tests runs through.
    """
    field = as_field(obstacles)
    states = [state]
    n_intervened = n_ics = 0
    collided = False

    for _ in range(n_steps):
        accel, steer = policy(state)
        if shield:
            res = safety_shield(float(accel), float(steer), state, field, cfg)
            accel, steer = res.accel, res.steer
            n_intervened += int(res.intervened)
            n_ics += int(res.ics)
        state = step_state(state, float(accel), float(steer), cfg)
        states.append(state)
        if clearance(state, field, cfg) < 0.0:
            collided = True
            break

    return states, {
        "collided": collided,
        "n_interventions": n_intervened,
        "n_ics": n_ics,
        "final_speed": state.v,
        "distance_travelled": float(np.linalg.norm(states[-1].xy - states[0].xy)),
    }
