"""Integration tests against a real KITTI drive — skipped cleanly when the data is absent.

The unit tests elsewhere prove the maths in isolation with synthetic inputs. These prove
the parts that only real data can: that the calibration is read correctly, that the OXTS ->
camera ground-truth conversion is right, and that ORB + SGBM actually find enough structure
in road imagery for PnP to work. Thresholds are deliberately loose — they are regression
guards against something breaking, not performance claims.
"""

import numpy as np
import pytest

from kitti_nav.kitti import DEFAULT_DATA_DIR, KittiDrive
from kitti_nav.odometry import StereoOdometry, evaluate_trajectory
from kitti_nav.stereo import StereoDepth

pytestmark = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "2011_09_26" / "2011_09_26_drive_0009_sync").exists(),
    reason="KITTI drive not downloaded; run scripts/fetch_kitti.py",
)


@pytest.fixture(scope="module")
def drive():
    return KittiDrive("2011_09_26", "0009", n_frames=40)


# --- calibration ---------------------------------------------------------------------------

def test_intrinsics_match_the_known_kitti_calibration(drive):
    K = drive.intrinsics
    assert K.fx == pytest.approx(721.5, abs=1.0)
    assert K.fx == pytest.approx(K.fy, abs=1e-6)      # rectified pixels are square
    assert 0 < K.cx < 1242 and 0 < K.cy < 375         # principal point inside the image


def test_stereo_baseline_is_the_documented_half_metre(drive):
    """KITTI's colour pair sits ~0.54 m apart; a wrong baseline scales all depth linearly."""
    assert drive.baseline == pytest.approx(0.54, abs=0.02)


# --- ground truth --------------------------------------------------------------------------

def test_ground_truth_starts_at_the_origin(drive):
    # atol is 1e-7, not 0: the first pose is `inv(P0) @ P0`, so it is identity only up to
    # float64 round-off in the matrix product (observed ~3e-9), never exactly.
    np.testing.assert_allclose(drive.gt_poses[0], np.eye(4), atol=1e-7)


def test_ground_truth_poses_are_valid_rigid_transforms(drive):
    for T in drive.gt_poses[::10]:
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(T[3], [0, 0, 0, 1], atol=1e-12)


def test_ground_truth_path_length_agrees_with_integrated_oxts_speed(drive):
    """Cross-check the pose conversion against an independent OXTS channel.

    Path length comes from differencing the converted camera poses; the speed integral comes
    straight from the `vf` packets. If `T_cam2_imu` were composed wrongly the two would
    disagree, so this catches the single most likely error in `gt_poses`.
    """
    integrated = float(np.sum(drive.speeds[:-1] * 0.1))    # KITTI raw is 10 Hz
    assert drive.path_length == pytest.approx(integrated, rel=0.05)


def test_the_car_moves_mostly_forward_not_vertically(drive):
    """Sanity on the axis convention: KITTI camera +z is forward, +y is down."""
    disp = drive.gt_poses[-1, :3, 3] - drive.gt_poses[0, :3, 3]
    assert abs(disp[2]) > 5.0 * abs(disp[1]), "vertical motion dominates; axes look wrong"


# --- imagery and stereo --------------------------------------------------------------------

def test_stereo_pairs_load_as_matching_grayscale_images(drive):
    left, right = drive.gray_pair(0)
    assert left.shape == right.shape == (375, 1242)
    assert left.dtype == np.uint8
    assert not np.array_equal(left, right), "left and right frames are identical"


def test_stereo_depth_on_a_real_frame_is_dense_and_plausible(drive):
    left, right = drive.gray_pair(0)
    depth = StereoDepth(drive.intrinsics.fx, drive.baseline).depth(left, right)

    valid = depth[depth > 0]
    assert valid.size > 0.2 * depth.size, "stereo produced almost no valid depth"
    # Road-scene depths should sit in the tens of metres, not centimetres or kilometres.
    assert 3.0 < np.median(valid) < 60.0


def test_velodyne_scan_loads_with_reflectance(drive):
    scan = drive.velodyne(0)
    assert scan.ndim == 2 and scan.shape[1] == 4
    assert scan.shape[0] > 10_000
    assert np.all((scan[:, 3] >= 0.0) & (scan[:, 3] <= 1.0))   # reflectance is normalised


# --- end-to-end odometry -------------------------------------------------------------------

def test_odometry_tracks_a_real_sequence_without_falling_back(drive):
    """The whole chain on real imagery: enough ORB matches, valid depth, converging PnP."""
    stereo = StereoDepth(drive.intrinsics.fx, drive.baseline)
    odo = StereoOdometry(drive.intrinsics)

    for _, left, right in drive.frames():
        res = odo.track(left, stereo.depth(left, right))

    assert odo.n_fallbacks == 0, "PnP failed on real road imagery"
    assert res.n_inliers > 50

    err = evaluate_trajectory(odo.positions, drive.gt_poses[:, :3, 3])
    assert err.drift_percent < 10.0, f"drift regressed badly: {err}"
    assert err.path_length > 20.0        # the clip really does contain motion
