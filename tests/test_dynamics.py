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
from kitti_nav.dynamics import estimate_obstacle_velocities

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
