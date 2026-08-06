"""Pure-side tests for the CARLA bridge: the handedness/actuator seams, no `carla` needed.

The bridge's only subtle logic is converting between CARLA's left-handed frame (+y right,
+steer right) and kitti-nav's right-handed one (+y left, +steer left). A silent sign error
here is the exact class of bug that mirrored the BEV renders, so it is worth pinning.
"""
import numpy as np
import pytest

from kitti_nav.bev import BEVConfig
from kitti_nav.carla_bridge import (
    ForwardGoalPlanner,
    ShieldController,
    carla_lidar_to_velodyne,
    shield_to_carla_control,
)
from kitti_nav.vehicle import ShieldResult, VehicleConfig


def test_lidar_y_is_flipped_left_handed_to_right_handed():
    # A CARLA point to the vehicle's right (+y in Unreal) must land to the left (+y) nowhere:
    # it becomes -y in the velodyne frame. x, z, intensity are untouched.
    raw = np.array([1.0, 2.0, 3.0, 0.5], np.float32)   # x, y_right, z, intensity
    out = carla_lidar_to_velodyne(raw)
    assert out.shape == (1, 4)
    np.testing.assert_allclose(out[0], [1.0, -2.0, 3.0, 0.5])


def test_lidar_conversion_is_an_involution():
    raw = np.random.default_rng(0).normal(size=(50, 4)).astype(np.float32)
    twice = carla_lidar_to_velodyne(carla_lidar_to_velodyne(raw))
    np.testing.assert_allclose(twice, raw.reshape(-1, 4), atol=0)


def test_positive_accel_is_throttle_negative_is_brake():
    cfg = VehicleConfig()
    thr, brk, _ = shield_to_carla_control(ShieldResult(cfg.max_accel, 0.0, False, False), cfg)
    assert thr == pytest.approx(1.0) and brk == 0.0
    thr, brk, _ = shield_to_carla_control(ShieldResult(-cfg.max_decel, 0.0, True, False), cfg)
    assert thr == 0.0 and brk == pytest.approx(1.0)


def test_left_steer_maps_to_negative_carla_steer():
    # kitti-nav positive steer = left; CARLA positive steer = right, so the sign flips.
    cfg = VehicleConfig()
    _, _, steer = shield_to_carla_control(ShieldResult(0.0, cfg.max_steer, False, False), cfg)
    assert steer == pytest.approx(-1.0)
    _, _, steer = shield_to_carla_control(ShieldResult(0.0, -cfg.max_steer, False, False), cfg)
    assert steer == pytest.approx(1.0)


def test_control_outputs_stay_in_carla_range():
    cfg = VehicleConfig()
    rng = np.random.default_rng(1)
    for _ in range(200):
        res = ShieldResult(float(rng.uniform(-10, 10)), float(rng.uniform(-2, 2)), False, False)
        thr, brk, steer = shield_to_carla_control(res, cfg)
        assert 0.0 <= thr <= 1.0 and 0.0 <= brk <= 1.0 and -1.0 <= steer <= 1.0


def test_controller_brakes_for_a_wall_dead_ahead():
    # An obstacle wall straight ahead: the shield must not let the forward planner accelerate
    # into it — the closed-loop analogue of the offline "0 collisions" guarantee.
    cfg = VehicleConfig()
    ctrl = ShieldController(planner=ForwardGoalPlanner(cfg, target_speed=8.0),
                            vcfg=cfg, bcfg=BEVConfig())
    ctrl.reset()
    # Dense points ~6 m ahead spanning the lane, elevated so ground removal keeps them.
    xs = np.full(400, 6.0)
    ys = np.linspace(-2.0, 2.0, 400)
    wall = np.stack([xs, ys, np.full(400, 0.4), np.ones(400)], axis=1).astype(np.float32)
    # Stack a low ground return under each so height_diff flags the wall as occupied.
    ground = wall.copy(); ground[:, 2] = -1.7
    scan = np.concatenate([wall, ground], axis=0)
    throttle, brake, _, result = ctrl.step(scan, speed=8.0)
    assert brake > 0.0 and throttle == 0.0
    assert result.intervened


def test_controller_lets_a_clear_road_accelerate():
    cfg = VehicleConfig()
    ctrl = ShieldController(planner=ForwardGoalPlanner(cfg, target_speed=8.0),
                            vcfg=cfg, bcfg=BEVConfig())
    ctrl.reset()
    empty = np.zeros((0, 4), np.float32)
    throttle, brake, _, result = ctrl.step(empty, speed=2.0)
    assert throttle > 0.0 and brake == 0.0
    assert not result.ics
