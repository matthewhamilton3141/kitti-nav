"""Tests for stereo depth and the PnP visual odometry.

The odometry tests inject a synthetic front-end with *known* correspondences, which
separates two things that are easy to conflate: whether the pose geometry
(back-projection -> PnP -> composition, and the camera-to-world convention) is correct, and
whether ORB happens to find good features on a given image. Only the first is a bug in this
code; the second is a property of the data. Image-driven behaviour is covered by the
dataset-gated tests in `test_kitti_integration.py`.
"""

import numpy as np
import pytest

from kitti_nav.kitti import Intrinsics, invert_se3
from kitti_nav.odometry import (
    OdometryConfig,
    StereoOdometry,
    evaluate_trajectory,
)
from kitti_nav.stereo import StereoDepth, StereoDepthConfig, depth_from_disparity

K = Intrinsics(fx=721.5, fy=721.5, cx=609.6, cy=172.9)
IMG_H, IMG_W = 375, 1242


# --- stereo depth -------------------------------------------------------------------------

def test_depth_from_disparity_is_the_textbook_relation():
    fx, b = 721.5, 0.53
    disp = np.array([[721.5 * 0.53, 10.0]])          # first gives exactly 1.0 m
    depth = depth_from_disparity(disp, fx, b)
    assert depth[0, 0] == pytest.approx(1.0)
    assert depth[0, 1] == pytest.approx(fx * b / 10.0)


def test_non_positive_disparity_is_marked_invalid_not_infinite():
    depth = depth_from_disparity(np.array([[0.0, -3.0, 20.0]]), 721.5, 0.53)
    assert depth[0, 0] == 0.0 and depth[0, 1] == 0.0
    assert depth[0, 2] > 0.0
    assert np.all(np.isfinite(depth))                 # never divide by zero into an inf


def test_disparity_and_depth_are_inversely_related():
    d = depth_from_disparity(np.array([[5.0, 50.0]]), 721.5, 0.53)
    assert d[0, 0] > d[0, 1], "smaller disparity must mean farther away"


def test_stereo_depth_recovers_a_known_shift_on_a_synthetic_pair():
    """Shift a random-texture image by a known disparity; SGBM should recover that depth.

    Random texture is the easiest possible case for a block matcher — this checks the
    plumbing (scaling by 1/16, the fx*b/d conversion, the validity sentinel), not SGBM's
    performance on hard scenes.
    """
    rng = np.random.default_rng(0)
    shift = 32
    right = rng.integers(0, 255, size=(200, 400), dtype=np.uint8)
    left = np.roll(right, shift, axis=1)              # left sees content shifted right

    fx, baseline = 721.5, 0.53
    depth = StereoDepth(fx, baseline, StereoDepthConfig(max_depth=1000.0)).depth(left, right)

    interior = depth[50:150, 150:350]                 # avoid borders, where SGBM has no support
    valid = interior[interior > 0]
    assert valid.size > 0.5 * interior.size, "matcher failed on trivially matchable texture"
    assert np.median(valid) == pytest.approx(fx * baseline / shift, rel=0.05)


def test_depth_outside_the_trusted_range_is_rejected():
    cfg = StereoDepthConfig(min_depth=2.0, max_depth=50.0)
    s = StereoDepth(721.5, 0.53, cfg)
    # Bypass the matcher: check the range gate on a hand-made disparity map.
    disp = np.array([[721.5 * 0.53 / 1.0, 721.5 * 0.53 / 10.0, 721.5 * 0.53 / 100.0]],
                    dtype=np.float32)
    depth = np.zeros_like(disp)
    depth[disp > 0] = s.fx * s.baseline / disp[disp > 0]
    depth[(depth < cfg.min_depth) | (depth > cfg.max_depth)] = 0.0
    assert depth[0, 0] == 0.0 and depth[0, 2] == 0.0  # too near / too far
    assert depth[0, 1] == pytest.approx(10.0)


# --- synthetic-correspondence odometry ----------------------------------------------------

def project(points_cam: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Pinhole projection of `(N, 3)` camera-frame points to `(N, 2)` pixels."""
    z = points_cam[:, 2]
    return np.stack([K.fx * points_cam[:, 0] / z + K.cx,
                     K.fy * points_cam[:, 1] / z + K.cy], axis=1).astype(np.float32)


def transform(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ T[:3, :3].T + T[:3, 3]


def depth_map_from(points_cam: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    """A depth image carrying each point's depth at its own pixel; 0 (invalid) elsewhere."""
    depth = np.zeros((IMG_H, IMG_W), np.float32)
    rows = np.clip(np.rint(pixels[:, 1]).astype(int), 0, IMG_H - 1)
    cols = np.clip(np.rint(pixels[:, 0]).astype(int), 0, IMG_W - 1)
    depth[rows, cols] = points_cam[:, 2]
    return depth


class ScriptedFrontend:
    """Returns pre-computed keypoints per call, with the identity as the match set."""

    def __init__(self, per_frame_xy):
        self._per_frame = list(per_frame_xy)
        self._i = 0

    def detect(self, gray):
        xy = self._per_frame[self._i]
        self._i += 1
        return xy, np.arange(len(xy))          # "descriptors" are just indices

    def match(self, desc0, desc1):
        n = min(len(desc0), len(desc1))
        return np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.int32)


def make_scene(T_rel: np.ndarray, n: int = 300, seed: int = 0):
    """A point cloud plus its projections in two views related by `T_rel` (cam0 -> cam1)."""
    rng = np.random.default_rng(seed)
    pts0 = np.stack([rng.uniform(-12.0, 12.0, n),
                     rng.uniform(-3.0, 3.0, n),
                     rng.uniform(6.0, 45.0, n)], axis=1)
    pts1 = transform(T_rel, pts0)
    keep = pts1[:, 2] > 1.0                     # must stay in front of the second camera
    pts0, pts1 = pts0[keep], pts1[keep]
    return pts0, project(pts0, K), project(pts1, K)


def run_two_frames(T_rel: np.ndarray):
    pts0, px0, px1 = make_scene(T_rel)
    odo = StereoOdometry(K, frontend=ScriptedFrontend([px0, px1]))
    blank = np.zeros((IMG_H, IMG_W), np.uint8)
    odo.track(blank, depth_map_from(pts0, px0))
    return odo, odo.track(blank, np.zeros((IMG_H, IMG_W), np.float32))


def test_first_frame_seeds_the_trajectory_at_identity():
    odo = StereoOdometry(K, frontend=ScriptedFrontend([np.zeros((0, 2), np.float32)]))
    res = odo.track(np.zeros((IMG_H, IMG_W), np.uint8), np.zeros((IMG_H, IMG_W), np.float32))
    assert np.allclose(res.pose, np.eye(4))
    assert res.ok and len(odo.trajectory) == 1


def test_pure_forward_motion_is_recovered_with_the_right_sign():
    """A camera advancing 1 m must end up at +1 m along its own initial +z (forward)."""
    T_rel = np.eye(4)
    T_rel[2, 3] = -1.0                          # points fall 1 m closer in the new camera
    odo, res = run_two_frames(T_rel)
    assert res.ok and res.n_inliers >= 8
    np.testing.assert_allclose(res.pose[:3, 3], [0.0, 0.0, 1.0], atol=1e-3)


def test_sideways_motion_is_recovered():
    T_rel = np.eye(4)
    T_rel[0, 3] = -0.5                          # camera moves +0.5 m along its own +x (right)
    _, res = run_two_frames(T_rel)
    np.testing.assert_allclose(res.pose[:3, 3], [0.5, 0.0, 0.0], atol=1e-3)


def test_yaw_rotation_is_recovered():
    """A car turning is mostly yaw about the camera's +y (down) axis."""
    angle = np.deg2rad(5.0)
    c, s = np.cos(angle), np.sin(angle)
    T_rel = np.eye(4)
    T_rel[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    _, res = run_two_frames(T_rel)
    np.testing.assert_allclose(res.pose[:3, :3], invert_se3(T_rel)[:3, :3], atol=1e-3)


def test_composition_over_many_steps_matches_the_analytic_pose():
    """Repeated identical motion must compose, not accumulate a convention error.

    A sign or inverse mistake in `P @ inv(T_rel)` can look fine for one step and diverge
    over many, so this walks 10 steps of combined translation + yaw.
    """
    angle = np.deg2rad(2.0)
    c, s = np.cos(angle), np.sin(angle)
    T_rel = np.eye(4)
    T_rel[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    T_rel[:3, 3] = [0.0, 0.0, -1.2]

    n_steps = 10
    scenes = [make_scene(T_rel, seed=i) for i in range(n_steps)]
    odo = StereoOdometry(K, frontend=ScriptedFrontend(
        [scenes[0][1]] + [sc[2] for sc in scenes]))

    blank = np.zeros((IMG_H, IMG_W), np.uint8)
    odo.track(blank, depth_map_from(scenes[0][0], scenes[0][1]))
    for i in range(n_steps):
        pts0, px0, _ = scenes[i]
        odo.track(blank, depth_map_from(pts0, px0))

    expected = np.linalg.matrix_power(invert_se3(T_rel), n_steps)
    np.testing.assert_allclose(odo.trajectory[-1], expected, atol=1e-2)


def test_too_few_matches_falls_back_without_stalling():
    """Degenerate frames must coast on the last motion, and be reported as not-ok."""
    T_rel = np.eye(4)
    T_rel[2, 3] = -1.0
    pts0, px0, px1 = make_scene(T_rel)
    empty = np.zeros((0, 2), np.float32)

    odo = StereoOdometry(K, frontend=ScriptedFrontend([px0, px1, empty]))
    blank = np.zeros((IMG_H, IMG_W), np.uint8)
    odo.track(blank, depth_map_from(pts0, px0))
    odo.track(blank, depth_map_from(pts0, px0))
    before = odo.trajectory[-1][:3, 3].copy()

    res = odo.track(blank, np.zeros((IMG_H, IMG_W), np.float32))
    assert not res.ok and odo.n_fallbacks == 1
    # Constant-velocity: it kept moving forward by roughly the previous step.
    assert odo.trajectory[-1][2, 3] > before[2]


def test_implausibly_fast_steps_are_rejected_as_blunders():
    """A PnP solution implying 400 m/s is a bad solve, not a motion — gate it out."""
    T_rel = np.eye(4)
    T_rel[2, 3] = -40.0                         # 40 m in one 0.1 s frame = 400 m/s
    cfg = OdometryConfig(max_speed=40.0, dt=0.1)
    pts0, px0, px1 = make_scene(T_rel)

    odo = StereoOdometry(K, cfg=cfg, frontend=ScriptedFrontend([px0, px1]))
    blank = np.zeros((IMG_H, IMG_W), np.uint8)
    odo.track(blank, depth_map_from(pts0, px0))
    res = odo.track(blank, np.zeros((IMG_H, IMG_W), np.float32))
    assert not res.ok, "blunder gate let through a 400 m/s step"


# --- trajectory evaluation ----------------------------------------------------------------

def test_identical_trajectories_have_zero_error():
    gt = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
    err = evaluate_trajectory(gt.copy(), gt)
    assert err.ate_rmse == pytest.approx(0.0)
    assert err.drift_percent == pytest.approx(0.0)
    assert err.path_length == pytest.approx(9.0)


def test_error_metrics_match_a_hand_computed_case():
    gt = np.stack([np.arange(11.0), np.zeros(11), np.zeros(11)], axis=1)
    est = gt.copy()
    est[-1, 1] = 2.0                            # 2 m lateral error at the final pose only
    err = evaluate_trajectory(est, gt)
    assert err.final_drift == pytest.approx(2.0)
    assert err.ate_max == pytest.approx(2.0)
    assert err.path_length == pytest.approx(10.0)
    assert err.drift_percent == pytest.approx(20.0)
    assert err.ate_rmse == pytest.approx(np.sqrt(4.0 / 11))


def test_evaluation_does_not_secretly_rescale_the_estimate():
    """Guard the deliberate absence of Sim(3) alignment: stereo VO is metric, so a
    trajectory that is uniformly 10% short must be reported as wrong, not aligned away."""
    gt = np.stack([np.arange(11.0), np.zeros(11), np.zeros(11)], axis=1)
    err = evaluate_trajectory(gt * 0.9, gt)
    assert err.final_drift == pytest.approx(1.0)
    assert err.ate_rmse > 0.0
