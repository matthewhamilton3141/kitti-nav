"""Label-free obstacle-velocity estimation — pinned on synthetic occupancy where truth is exact.

The estimator's *accuracy on real data* is a separate, measured story (~35% detection at a high
false-positive rate; `scripts/eval_dynamics.py` and the notes in `dynamics.py`). These tests pin
the mechanics instead: a clean translating box is caught with the right velocity, and each filter
(static, jitter, over-size) rejects what it should — so a regression in the logic is caught even
though real urban clutter defeats the method.
"""

import numpy as np
import pytest

from kitti_nav.bev import BEVConfig, rasterize_box
from kitti_nav.dynamics import (
    BoxField,
    MovingObstacle,
    can_stop_safely_dynamic,
    dynamic_safety_shield,
    estimate_obstacle_velocities,
    point_box_distance,
)
from kitti_nav.vehicle import (
    VehicleConfig,
    VehicleState,
    footprint_discs,
    safety_shield,
    step_state,
)

CFG = BEVConfig()
DT = 0.1


def _window(boxes):
    """Occupancy grids, one per (cx, cy) centre, for a 4 x 2 m box."""
    return [rasterize_box([cx, cy, 0.0, 4.0, 2.0], CFG) for cx, cy in boxes]


def test_a_translating_box_is_detected_with_its_velocity():
    # Box moving +x at 6 m/s: 0.6 m per 0.1 s frame, over four frames.
    window = _window([(10.0, 0.0), (10.6, 0.0), (11.2, 0.0), (11.8, 0.0)])
    movers = estimate_obstacle_velocities(window, CFG, DT, min_speed=1.0)
    assert len(movers) == 1
    assert movers[0].speed == pytest.approx(6.0, abs=0.5)
    assert abs(movers[0].velocity[0] - 6.0) < 0.3 and abs(movers[0].velocity[1]) < 0.3


def test_a_static_box_is_not_a_mover():
    window = _window([(10.0, 0.0)] * 4)
    assert estimate_obstacle_velocities(window, CFG, DT, min_speed=1.0) == []


def test_jitter_fails_the_coherence_test():
    # Centroid wanders back and forth: large path, ~zero net displacement.
    window = _window([(10.0, 0.0), (10.5, 0.0), (10.0, 0.0), (10.5, 0.0)])
    assert estimate_obstacle_velocities(window, CFG, DT, min_speed=1.0, coherence=0.7) == []


def test_an_oversize_component_is_ignored():
    # A 30 m wall translating: real displacement, but too big to be a vehicle.
    window = [rasterize_box([12.0 + 0.6 * k, 0.0, 0.0, 30.0, 2.0], CFG) for k in range(4)]
    assert estimate_obstacle_velocities(window, CFG, DT, min_speed=1.0, max_extent=14.0) == []


def test_below_the_speed_floor_is_dropped():
    # 0.05 m per frame = 0.5 m/s, under a 1 m/s floor.
    window = _window([(10.0, 0.0), (10.05, 0.0), (10.1, 0.0), (10.15, 0.0)])
    assert estimate_obstacle_velocities(window, CFG, DT, min_speed=1.0) == []


def test_a_single_frame_yields_nothing():
    assert estimate_obstacle_velocities(_window([(10.0, 0.0)]), CFG, DT) == []


def test_box_at_predicts_the_future_footprint():
    window = _window([(10.0, 0.0), (10.6, 0.0), (11.2, 0.0), (11.8, 0.0)])
    m = estimate_obstacle_velocities(window, CFG, DT, min_speed=1.0)[0]
    future = m.box_at(1.0)                                    # one second ahead at ~6 m/s
    assert abs(future[0] - (m.box[0] + 6.0)) < 0.4
    assert future[3] == m.box[3] and future[4] == m.box[4]   # size unchanged


# --- the dynamic shield: braking for where an obstacle is going --------------------------------

def test_point_box_distance_is_zero_inside_and_grows_outside():
    box = np.array([0.0, 0.0, 0.0, 4.0, 2.0])                 # 4 x 2 at the origin
    assert point_box_distance([[0.0, 0.0]], box)[0] == 0.0    # centre is inside
    assert point_box_distance([[2.0, 1.0]], box)[0] == 0.0    # a corner is on the edge
    assert point_box_distance([[5.0, 0.0]], box)[0] == pytest.approx(3.0)   # 5 - half-length
    assert point_box_distance([[0.0, 4.0]], box)[0] == pytest.approx(3.0)   # 4 - half-width
    # Rotated 90 deg: the long side now runs along y.
    turned = np.array([0.0, 0.0, np.pi / 2, 4.0, 2.0])
    assert point_box_distance([[0.0, 5.0]], turned)[0] == pytest.approx(3.0)


def test_box_field_is_the_min_over_boxes():
    field = BoxField([[10.0, 0.0, 0.0, 4.0, 2.0], [0.0, 10.0, 0.0, 4.0, 2.0]])
    d = field.distance_to_obstacles(np.array([[0.0, 0.0]]))
    assert d[0] == pytest.approx(8.0)                         # nearer box edge at x=8
    assert np.isinf(BoxField([]).distance_to_obstacles(np.zeros((1, 2)))[0])


def test_dynamic_shield_reduces_to_the_static_one_without_movers():
    """With no movers, a time-invariant field must give exactly the ordinary shield's decision."""
    vcfg = VehicleConfig()
    field = BoxField([[14.0, 0.0, 0.0, 4.0, 2.0]])
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=10.0)
    stat = safety_shield(vcfg.max_accel, 0.0, state, field, vcfg)
    dyn = dynamic_safety_shield(vcfg.max_accel, 0.0, state, field, [], vcfg)
    assert dyn.accel == pytest.approx(stat.accel) and dyn.steer == pytest.approx(stat.steer)


def _run_crossing(dynamic: bool, vcfg: VehicleConfig):
    """Drive straight at full throttle while a car crosses the path; return (collided, states).

    The obstacle starts off to the right and moves +y across the lane, timed to reach it as the
    vehicle arrives. The static shield sees only its current (off-path) position; the dynamic
    shield is handed its velocity.
    """
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=9.0)
    obox = np.array([20.0, -9.0, np.pi / 2, 4.0, 2.0])
    vel = np.array([0.0, 4.0])
    collided = False
    for _ in range(150):
        if dynamic:
            r = dynamic_safety_shield(vcfg.max_accel, 0.0, state, None,
                                      [MovingObstacle(obox.copy(), vel)], vcfg)
        else:
            r = safety_shield(vcfg.max_accel, 0.0, state, BoxField([obox.copy()]), vcfg)
        state = step_state(state, r.accel, r.steer, vcfg)
        obox[1] += vel[1] * vcfg.dt
        centres, radius = footprint_discs(state, vcfg)
        if point_box_distance(centres, obox).min() - radius < 0.0:
            collided = True
            break
        if state.x > 32.0:
            break
    return collided, state


def test_static_shield_hits_a_crossing_car_that_the_dynamic_shield_avoids():
    """The headline: predicting the obstacle's path is the difference between a crash and a stop.

    The static shield brakes only once the car is already in the lane — too late — and drives
    into it. The dynamic shield brakes for where the car *will be* and stays clear. This is the
    moving-obstacle analogue of the braking-vs-one-step demonstration in the shield tests.
    """
    vcfg = VehicleConfig()
    static_collided, _ = _run_crossing(dynamic=False, vcfg=vcfg)
    dynamic_collided, _ = _run_crossing(dynamic=True, vcfg=vcfg)
    assert static_collided, "the static shield should not have anticipated the crossing car"
    assert not dynamic_collided, "the dynamic shield should have braked for the predicted path"


def test_can_stop_safely_dynamic_refuses_a_stop_that_an_obstacle_crosses_into():
    """A stop that is clear against the frozen obstacle is refused once the obstacle is moving."""
    vcfg = VehicleConfig()
    state = VehicleState(x=0.0, y=0.0, yaw=0.0, v=10.0)
    # A car clear of the braking path when frozen, but sweeping straight into it when moving.
    box = np.array([14.0, -5.0, np.pi / 2, 4.0, 2.0])
    frozen = [MovingObstacle(box, np.array([0.0, 0.0]))]
    crossing = [MovingObstacle(box, np.array([0.0, 3.5]))]
    assert can_stop_safely_dynamic(state, None, frozen, vcfg)
    assert not can_stop_safely_dynamic(state, None, crossing, vcfg)
