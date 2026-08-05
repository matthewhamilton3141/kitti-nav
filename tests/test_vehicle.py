"""Tests for the kinematic bicycle model and the braking-aware safety shield.

The load-bearing test is `test_one_step_shield_crashes_where_braking_shield_stops`: it is
the evidence that porting the diff-drive one-step filter unchanged would have been unsafe.
"""

import numpy as np
import pytest

from kitti_nav.vehicle import (
    CircleField,
    ShieldResult,
    VehicleConfig,
    VehicleState,
    can_stop_safely,
    clearance,
    evasive_steer_candidates,
    footprint_discs,
    max_safe_speed,
    safety_shield,
    shielded_rollout,
    step_state,
    stopping_distance,
    turning_radius,
)


@pytest.fixture
def cfg():
    return VehicleConfig()


def wall(x: float, cfg: VehicleConfig, half_width: float = 10.0, r: float = 1.0):
    """A line of overlapping circles at `x`, spanning `+/- half_width` in y — a barrier."""
    ys = np.arange(-half_width, half_width + r, r)
    return np.stack([np.full_like(ys, x), ys, np.full_like(ys, r)], axis=1)


# --- kinematics -------------------------------------------------------------------------

def test_at_rest_with_no_throttle_nothing_moves(cfg):
    s = VehicleState(x=1.0, y=2.0, yaw=0.5, v=0.0, steer=0.4)
    nxt = step_state(s, 0.0, 0.4, cfg)
    assert nxt.x == pytest.approx(1.0)
    assert nxt.y == pytest.approx(2.0)
    # A bicycle model cannot rotate in place: yaw rate is v/L*tan(delta), and v is 0.
    assert nxt.yaw == pytest.approx(0.5)


def test_straight_driving_advances_along_heading(cfg):
    s = VehicleState(v=10.0)
    nxt = step_state(s, 0.0, 0.0, cfg)
    assert nxt.x == pytest.approx(10.0 * cfg.dt)
    assert nxt.y == pytest.approx(0.0)
    assert nxt.yaw == pytest.approx(0.0)


def test_acceleration_integrates_at_the_midpoint(cfg):
    s = VehicleState(v=0.0)
    nxt = step_state(s, cfg.max_accel, 0.0, cfg)
    assert nxt.v == pytest.approx(cfg.max_accel * cfg.dt)
    # Midpoint speed over the step is half the final speed, not the full value.
    assert nxt.x == pytest.approx(0.5 * nxt.v * cfg.dt)


def test_positive_steer_turns_left(cfg):
    s = VehicleState(v=5.0, steer=0.2)
    assert step_state(s, 0.0, 0.2, cfg).yaw > 0.0
    s_right = VehicleState(v=5.0, steer=-0.2)
    assert step_state(s_right, 0.0, -0.2, cfg).yaw < 0.0


def test_turning_radius_matches_bicycle_geometry(cfg):
    """Path radius traced over a step should equal the analytic L/tan(delta)."""
    steer, v = 0.3, 5.0
    s = VehicleState(v=v, steer=steer)
    nxt = step_state(s, 0.0, steer, cfg)
    arc = float(np.linalg.norm(nxt.xy - s.xy))
    measured = arc / abs(nxt.yaw - s.yaw)
    assert measured == pytest.approx(abs(turning_radius(steer, cfg)), rel=1e-3)


def test_steering_is_straight_when_wheels_are_centred(cfg):
    assert np.isinf(turning_radius(0.0, cfg))


def test_steering_rack_is_rate_limited(cfg):
    """A full-lock command cannot be met in one step — the rack has a finite slew rate."""
    s = VehicleState(v=5.0, steer=0.0)
    nxt = step_state(s, 0.0, cfg.max_steer, cfg)
    assert nxt.steer == pytest.approx(cfg.max_steer_rate * cfg.dt)
    assert nxt.steer < cfg.max_steer


def test_speed_and_steer_respect_limits(cfg):
    fast = step_state(VehicleState(v=cfg.max_speed), cfg.max_accel, 0.0, cfg)
    assert fast.v == pytest.approx(cfg.max_speed)
    # Braking harder than the remaining speed clips at rest rather than reversing.
    assert step_state(VehicleState(v=0.2), -cfg.max_decel, 0.0, cfg).v \
        == pytest.approx(cfg.min_speed)

    # Commanding past the mechanical stop saturates there, given enough steps to slew.
    s = VehicleState(v=1.0)
    for _ in range(50):
        s = step_state(s, 0.0, 10.0, cfg)
    assert s.steer == pytest.approx(cfg.max_steer)


def test_stopping_distance_is_v_squared_over_2a(cfg):
    assert stopping_distance(15.0, cfg) == pytest.approx(15.0 ** 2 / (2 * cfg.max_decel))
    assert stopping_distance(0.0, cfg) == 0.0


def test_braking_rollout_travels_the_predicted_stopping_distance(cfg):
    """The closed-form v^2/2a must agree with what the integrator actually does."""
    s = VehicleState(v=12.0)
    start = s.xy
    while s.v > 1e-9:
        s = step_state(s, -cfg.max_decel, 0.0, cfg)
    travelled = float(np.linalg.norm(s.xy - start))
    assert travelled == pytest.approx(stopping_distance(12.0, cfg), rel=0.02)


# --- footprint & clearance --------------------------------------------------------------

def test_footprint_discs_cover_every_corner_of_the_vehicle_rectangle(cfg):
    """The disc cover must be conservative: no corner may stick out of the union."""
    s = VehicleState(x=3.0, y=-2.0, yaw=0.7)
    centres, radius = footprint_discs(s, cfg)
    c, sn = np.cos(s.yaw), np.sin(s.yaw)
    for along in (-cfg.rear_overhang, cfg.front_overhang):
        for lateral in (-cfg.width / 2, cfg.width / 2):
            corner = np.array([s.x + along * c - lateral * sn,
                               s.y + along * sn + lateral * c])
            assert np.min(np.linalg.norm(centres - corner, axis=1)) <= radius + 1e-9


def test_clearance_is_infinite_with_no_obstacles(cfg):
    assert np.isinf(clearance(VehicleState(), np.zeros((0, 3)), cfg))


def test_clearance_is_signed_and_shrinks_as_the_obstacle_nears(cfg):
    far = clearance(VehicleState(), np.array([[50.0, 0.0, 1.0]]), cfg)
    near = clearance(VehicleState(), np.array([[20.0, 0.0, 1.0]]), cfg)
    assert far > near > 0.0
    # An obstacle sitting on the rear axle overlaps the body: clearance goes negative.
    assert clearance(VehicleState(), np.array([[0.0, 0.0, 1.0]]), cfg) < 0.0


# --- the shield -------------------------------------------------------------------------

def test_shield_passes_through_on_an_empty_road(cfg):
    res = safety_shield(cfg.max_accel, 0.1, VehicleState(v=5.0), np.zeros((0, 3)), cfg)
    assert res.accel == pytest.approx(cfg.max_accel)
    assert res.steer == pytest.approx(0.1)
    assert not res.intervened and not res.ics


def test_shield_brakes_when_a_wall_enters_the_stopping_envelope(cfg):
    """Approaching a barrier at exactly the stopping distance, full throttle is refused."""
    v = 12.0
    barrier_x = cfg.front_overhang + stopping_distance(v, cfg) + 1.0
    res = safety_shield(cfg.max_accel, 0.0, VehicleState(v=v), wall(barrier_x, cfg), cfg)
    assert res.intervened
    assert res.accel < cfg.max_accel


def test_shield_flags_an_inevitable_collision_state(cfg):
    """Too fast, too close: no action is certifiable, so brake hard and say so."""
    res = safety_shield(cfg.max_accel, 0.0, VehicleState(v=cfg.max_speed),
                        wall(cfg.front_overhang + 2.0, cfg), cfg)
    assert res.ics and res.intervened
    assert res.accel == pytest.approx(-cfg.max_decel)


def test_can_stop_safely_agrees_with_the_stopping_distance(cfg):
    v = 10.0
    d = stopping_distance(v, cfg)
    obstacles = wall(cfg.front_overhang + d + 2.0, cfg)
    assert can_stop_safely(VehicleState(v=v), obstacles, cfg)
    assert not can_stop_safely(VehicleState(v=v), wall(cfg.front_overhang + d - 2.0, cfg), cfg)


# --- evasive steering: swerving out of an otherwise-inevitable collision -----------------

# A lone obstacle 16 m ahead while doing 13 m/s: stopping distance is ~18.8 m, so braking
# straight cannot stop in time (an ICS), but the lane beside it is open — a swerve clears it.
_EVASIVE_STATE = VehicleState(x=0.0, y=0.0, yaw=0.0, v=13.0, steer=0.0)
_EVASIVE_OBSTACLE = CircleField(np.array([[16.0, 0.0, 1.2]]))


def test_evasive_steering_is_off_by_default(cfg):
    """The default shield is its two-option self: it brakes and declares the ICS, never swerves."""
    assert cfg.n_evasive_steers == 0
    assert evasive_steer_candidates(0.0, 0.0, cfg).size == 0
    res = safety_shield(cfg.max_accel, 0.0, _EVASIVE_STATE, _EVASIVE_OBSTACLE, cfg)
    assert res.ics and res.steer == pytest.approx(0.0)


def test_shield_swerves_around_an_obstacle_it_cannot_brake_for():
    """The headline: with evasive steering on, an ICS becomes a certified swerve, not a crash."""
    cfg = VehicleConfig(n_evasive_steers=15)
    res = safety_shield(cfg.max_accel, 0.0, _EVASIVE_STATE, _EVASIVE_OBSTACLE, cfg)
    assert not res.ics, "a swerve should have been certifiable"
    assert res.intervened and abs(res.steer) > 0.0, "the shield should have steered away"
    # The admitted action is genuinely safe: its successor still carries a braking certificate.
    nxt = step_state(_EVASIVE_STATE, res.accel, res.steer, cfg)
    assert can_stop_safely(nxt, _EVASIVE_OBSTACLE, cfg)


def test_evasive_steering_avoids_a_collision_the_braking_shield_drives_into():
    """End to end: the same policy (floor it, straight) crashes without evasion, clears with it."""
    def collided(n_evasive):
        cfg = VehicleConfig(n_evasive_steers=n_evasive)
        s = _EVASIVE_STATE
        for _ in range(120):
            res = safety_shield(cfg.max_accel, 0.0, s, _EVASIVE_OBSTACLE, cfg)
            s = step_state(s, res.accel, res.steer, cfg)
            if clearance(s, _EVASIVE_OBSTACLE, cfg) < 0.0:
                return True
            if s.x > 30.0:
                return False
        return False

    assert collided(0), "the braking-only shield should drive into the obstacle it cannot stop for"
    assert not collided(15), "the evasive shield should swerve clear"


def test_evasive_steer_candidates_are_ordered_and_exclude_the_tried_angles():
    cfg = VehicleConfig(n_evasive_steers=9)
    cand = evasive_steer_candidates(steer_cmd=0.0, state_steer=0.0, cfg=cfg)
    assert np.all(np.abs(cand) <= cfg.max_steer + 1e-9)
    # Ordered by proximity to the commanded angle (smallest swerve first).
    assert np.all(np.diff(np.abs(cand - 0.0)) >= -1e-9)
    # The two angles the main search already tried are dropped.
    tried = evasive_steer_candidates(steer_cmd=0.13, state_steer=-0.13, cfg=cfg)
    assert not np.any(np.isclose(tried, 0.13)) and not np.any(np.isclose(tried, -0.13))


# --- why the diff-drive shield could not simply be ported --------------------------------

def _one_step_shield(accel_cmd, steer_cmd, state, obstacles, cfg) -> ShieldResult:
    """The naive port: `gsplat-rt`'s one-step-lookahead filter, unchanged for a car.

    Checks only that the *next* pose is clear — sound for a diff-drive robot that can stop
    dead within a step, unsound for a vehicle whose stopping distance spans many steps.
    Present only to be measured against; it is not exported from the package.
    """
    hi = float(np.clip(accel_cmd, -cfg.max_decel, cfg.max_accel))
    for a in np.linspace(hi, -cfg.max_decel, cfg.n_accel_candidates):
        if clearance(step_state(state, float(a), steer_cmd, cfg),
                     obstacles, cfg) >= cfg.safety_margin:
            return ShieldResult(float(a), steer_cmd, not np.isclose(a, hi), False)
    return ShieldResult(-cfg.max_decel, steer_cmd, True, True)


def test_one_step_shield_crashes_where_braking_shield_stops(cfg):
    """The load-bearing result: at speed, one-step lookahead is not a safety guarantee.

    Same scene, same throttle-happy policy. The one-step filter sees a clear next pose all
    the way in and only reacts within ~1.5 m of the barrier — far inside the ~25 m it needs
    to stop. The braking-aware shield refuses throttle early enough to come to rest.
    """
    obstacles = wall(60.0, cfg)
    start = VehicleState(v=10.0)
    flat_out = lambda s: (cfg.max_accel, 0.0)   # noqa: E731 — deliberately reckless policy

    # Control: unshielded, the policy drives straight into the barrier.
    _, raw = shielded_rollout(flat_out, start, obstacles, cfg, n_steps=200, shield=False)
    assert raw["collided"]

    # Naive port: still collides, despite "having a safety filter".
    def one_step_policy(s):
        res = _one_step_shield(cfg.max_accel, 0.0, s, obstacles, cfg)
        return res.accel, res.steer

    _, naive = shielded_rollout(one_step_policy, start, obstacles, cfg,
                                n_steps=200, shield=False)
    assert naive["collided"], "one-step lookahead unexpectedly survived; scene too easy"

    # Braking-aware shield: no collision, and the car actually comes to a stop.
    _, shielded = shielded_rollout(flat_out, start, obstacles, cfg, n_steps=200, shield=True)
    assert not shielded["collided"]
    assert shielded["final_speed"] == pytest.approx(0.0, abs=1e-6)
    assert shielded["n_interventions"] > 0


def test_shield_falls_back_to_the_held_steer_rather_than_obeying_a_fatal_one(cfg):
    """Regression: unconditional steer pass-through broke the shield's induction.

    The car is certified stoppable going straight, with an obstacle off to the left. A
    policy that suddenly demands left lock would curve the braking trajectory into it — no
    acceleration can save that steer, so the shield must reject the angle and hold its own.
    """
    obstacles = np.array([[14.0, 6.0, 3.0]])
    state = VehicleState(v=10.0, steer=0.0)
    assert can_stop_safely(state, obstacles, cfg)

    res = safety_shield(0.0, cfg.max_steer, state, obstacles, cfg)
    assert res.intervened
    assert res.steer == pytest.approx(state.steer), "shield obeyed an uncertifiable steer"
    assert not res.ics
    assert can_stop_safely(step_state(state, res.accel, res.steer, cfg), obstacles, cfg)


def test_max_safe_speed_is_unbounded_on_an_open_road(cfg):
    assert max_safe_speed(np.zeros((0, 3)), cfg) == pytest.approx(cfg.max_speed)


def test_max_safe_speed_shrinks_as_an_obstacle_closes_in(cfg):
    far = max_safe_speed(wall(60.0, cfg), cfg)
    near = max_safe_speed(wall(25.0, cfg), cfg)
    closer = max_safe_speed(wall(12.0, cfg), cfg)
    assert far > near > closer >= 0.0


def test_max_safe_speed_agrees_with_the_stopping_distance_formula(cfg):
    """The bisection must land where v^2/2a says it should, given the footprint and margin.

    The braking room is measured from the *disc cover's* forward reach, not the front
    bumper: circumscribed discs also bulge past the ends of the rectangle (~0.55 m here), so
    the shield conservatively treats the car as slightly longer than it is. The barrier is
    placed close enough that geometry, not `max_speed`, is what binds.
    """
    barrier = 25.0
    centres, radius = footprint_discs(VehicleState(), cfg)
    reach = float(centres[:, 0].max() + radius)
    room = barrier - 1.0 - reach - cfg.safety_margin        # wall circles have radius 1
    expected = np.sqrt(2 * cfg.max_decel * room)
    assert expected < cfg.max_speed, "test scene must be geometry-limited, not speed-capped"
    assert max_safe_speed(wall(barrier, cfg), cfg) == pytest.approx(expected, rel=0.05)


def test_max_safe_speed_is_zero_when_already_touching(cfg):
    assert max_safe_speed(wall(cfg.front_overhang + 0.5, cfg), cfg) == 0.0


def test_speeds_below_the_max_safe_speed_are_certifiable(cfg):
    """The bisection's contract: it really is the boundary of `can_stop_safely`."""
    obstacles = wall(30.0, cfg)
    v = max_safe_speed(obstacles, cfg)
    assert can_stop_safely(VehicleState(v=v * 0.95), obstacles, cfg)
    assert not can_stop_safely(VehicleState(v=v + 1.0), obstacles, cfg)


def test_more_footprint_discs_approximate_the_body_more_tightly(cfg):
    """Adding discs must shrink the conservative excess, never flip to optimistic."""
    from dataclasses import replace

    excess = []
    for n in (3, 5, 9):
        _, radius = footprint_discs(VehicleState(), replace(cfg, n_footprint_discs=n))
        excess.append(radius - cfg.width / 2)
    assert excess[0] > excess[1] > excess[2] > 0.0


def test_shielded_rollouts_never_collide_across_random_scenes(cfg):
    """Fuzz: any policy, any obstacle field — from a certified start, the shield holds.

    Starts are rejection-sampled to be certifiably safe, since the shield's guarantee is
    inductive and cannot rescue a state that was already an inevitable collision.
    """
    rng = np.random.default_rng(0)
    tested = 0

    for _ in range(60):
        n = int(rng.integers(1, 6))
        obstacles = np.stack([rng.uniform(10.0, 70.0, n),
                              rng.uniform(-8.0, 8.0, n),
                              rng.uniform(0.5, 2.5, n)], axis=1)
        start = VehicleState(v=float(rng.uniform(0.0, cfg.max_speed)),
                             steer=float(rng.uniform(-cfg.max_steer, cfg.max_steer)))
        if not can_stop_safely(start, obstacles, cfg):
            continue
        tested += 1

        def erratic(s, rng=rng):
            return (float(rng.uniform(-cfg.max_decel, cfg.max_accel)),
                    float(rng.uniform(-cfg.max_steer, cfg.max_steer)))

        _, stats = shielded_rollout(erratic, start, obstacles, cfg, n_steps=120, shield=True)
        assert not stats["collided"]

    assert tested >= 20, f"too few certifiable starts sampled ({tested}) to be meaningful"
