"""Integration tests against a real KITTI drive — skipped cleanly when the data is absent.

The unit tests elsewhere prove the maths in isolation with synthetic inputs. These prove
the parts that only real data can: that the calibration is read correctly, that the OXTS ->
camera ground-truth conversion is right, and that ORB + SGBM actually find enough structure
in road imagery for PnP to work. Thresholds are deliberately loose — they are regression
guards against something breaking, not performance claims.
"""

import numpy as np
import pytest

from kitti_nav.bev import BEVConfig, BEVGrid
from kitti_nav.kitti import DEFAULT_DATA_DIR, KittiDrive
from kitti_nav.mapping import MapConfig
from kitti_nav.nav_env import KittiScenes
from kitti_nav.odometry import StereoOdometry, evaluate_trajectory
from kitti_nav.stereo import StereoDepth
from kitti_nav.vehicle import (
    VehicleConfig,
    VehicleState,
    clearance,
    max_safe_speed,
    safety_shield,
    stopping_distance,
)

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


# --- BEV occupancy from real lidar ----------------------------------------------------------

def test_real_scans_produce_a_sparse_but_non_empty_grid(drive):
    """Occupancy on a city street should be a few percent — neither empty nor a solid block.

    An all-zero grid means ground removal ate the obstacles; a mostly-full grid means the
    road leaked in. Both have happened during development, hence the two-sided bound.
    """
    cfg = BEVConfig()
    for i in (0, 10, 20):
        grid = BEVGrid.from_scan(drive.velodyne(i), cfg)
        assert 0.002 < grid.occupied_fraction < 0.20, \
            f"frame {i}: {grid.occupied_fraction:.3%} occupied"


def test_per_cell_ground_beats_a_fixed_plane_on_real_road(drive):
    """The measured justification for the default `ground_mode`.

    On real scans the fixed-plane band marks substantially more of the grid occupied, and
    the extra cells are road surface drifting out of the band with slope and sensor pitch —
    phantom obstacles that would make the shield brake for open road.
    """
    scan = drive.velodyne(0)
    per_cell = BEVGrid.from_scan(scan, BEVConfig(ground_mode="height_diff"))
    plane = BEVGrid.from_scan(scan, BEVConfig(ground_mode="plane"))
    assert per_cell.occupied_fraction < plane.occupied_fraction

    # Straight ahead down the lane, the plane mode reports markedly less clearance.
    ahead = np.array([[30.0, 0.0]])
    assert (per_cell.distance_to_obstacles(ahead)[0]
            > plane.distance_to_obstacles(ahead)[0])


def test_the_lane_ahead_is_not_reported_as_blocked(drive):
    """Sanity that the car's own lane is drivable — the scan was recorded while driving it."""
    grid = BEVGrid.from_scan(drive.velodyne(0))
    ahead = np.stack([np.arange(3.0, 25.0, 1.0), np.zeros(22)], axis=1)
    assert np.median(grid.distance_to_obstacles(ahead)) > 1.0


def test_lidar_frame_count_can_be_short_of_the_image_count(drive):
    """Drive 0009 ships 447 images and OXTS packets but only 443 Velodyne scans.

    Real datasets have holes. Anything iterating lidar must bound on `n_velodyne`, and
    overrunning it should say so rather than raising from inside pykitti's file list.
    """
    full = KittiDrive("2011_09_26", "0009")
    assert full.n_velodyne == 443 and len(full) == 447
    with pytest.raises(IndexError, match="out of range"):
        full.velodyne(full.n_velodyne)


def test_the_rear_axle_is_offset_from_the_lidar_origin(drive):
    """The Velodyne is roof-mounted ahead of the rear axle; conflating them misplaces the car.

    Treating the lidar origin as the vehicle reference point pushed a 4.77 m car 0.8 m too
    far forward, which corrupted every clearance query against real scans.
    """
    offset = drive.rear_axle_in_lidar
    assert offset.shape == (2,)
    assert offset[0] == pytest.approx(-0.81, abs=0.05)   # axle sits behind the lidar
    assert np.linalg.norm(offset) > 0.5

    state = drive.vehicle_state_in_lidar(speed=5.0)
    assert (state.x, state.y) == pytest.approx(tuple(offset))
    assert state.v == 5.0


def test_correct_axle_placement_reports_more_clearance_than_the_lidar_origin(drive):
    """Regression on the fix: the mistake made the car appear ~0.8 m deeper into the scene."""
    grid = BEVGrid.from_scan(drive.velodyne(0))
    vcfg = VehicleConfig()
    at_origin = clearance(VehicleState(), grid, vcfg)
    at_axle = clearance(drive.vehicle_state_in_lidar(), grid, vcfg)
    assert at_axle > at_origin


def test_shield_permits_more_speed_than_the_driver_used_on_open_road(drive):
    """The shield expressed as a speed limit, compared against a real human driver.

    On the open lane at the start of this drive the constraint should not bind — a shield
    that already overrules the recorded driver here would be uselessly timid.
    """
    vcfg = VehicleConfig(max_speed=35.0)
    state = drive.vehicle_state_in_lidar()
    for i in (0, 5, 10):
        permitted = max_safe_speed(BEVGrid.from_scan(drive.velodyne(i)), vcfg, state=state)
        assert permitted > drive.speeds[i], f"frame {i}: shield would slow an open-road drive"


def test_shield_runs_on_real_lidar_within_its_grid(drive):
    """End to end on real data: grid extent covers the braking envelope and the shield runs."""
    vcfg, grid = VehicleConfig(), BEVGrid.from_scan(drive.velodyne(0))
    assert grid.covers_stopping_distance(stopping_distance(vcfg.max_speed, vcfg))

    res = safety_shield(vcfg.max_accel, 0.0, VehicleState(v=float(drive.speeds[0])),
                        grid, vcfg)
    assert np.isfinite(res.accel) and not res.ics
    assert -vcfg.max_decel <= res.accel <= vcfg.max_accel


def test_kitti_scene_carves_a_tri_state_map(drive):
    """A carved KittiScenes builds an occupied/free/unknown grid, not a binary one."""
    scenes = KittiScenes(drive=drive, map_config=MapConfig(window=5, carve=True))
    grid = scenes._build_grid(20)
    assert grid.unknown is not None and grid.unknown.any()      # holes are marked, not free
    assert grid.occupancy.any()                                 # real geometry survives


def test_unknown_blocks_is_more_conservative_but_not_degenerate(drive):
    """Honest unknown never permits *more* speed, and the near-field exemption keeps it usable.

    Treating unobserved space as blocked can only lower the shield's permitted speed. The
    near-field free assumption (`carve_near_field`) is what stops it collapsing to 0 on a roof
    lidar's ground blind spot — so the honest reading is slower than assuming unknown is free,
    but the car can still move.
    """
    mcfg = MapConfig(window=5, carve=True)
    free = KittiScenes(drive=drive, map_config=mcfg, unknown_blocks=False)._build_grid(20)
    blocking = KittiScenes(drive=drive, map_config=mcfg, unknown_blocks=True)._build_grid(20)

    vcfg = VehicleConfig(max_speed=21.0)
    state = drive.vehicle_state_in_lidar()
    v_free = max_safe_speed(free, vcfg, state=state)
    v_block = max_safe_speed(blocking, vcfg, state=state)
    assert 0.0 < v_block <= v_free + 0.05, f"honest={v_block:.2f} free={v_free:.2f}"


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


# --- the planner driving an accumulated map --------------------------------------------------

def test_kitti_scenes_defaults_to_a_single_scan(drive):
    """The default must stay the frozen-scan behaviour, or every published policy number
    silently changes meaning."""
    from kitti_nav.mapping import drop_ego_returns
    from kitti_nav.nav_env import KittiScenes

    scenes = KittiScenes(drive=drive, frames=np.array([20]))
    grid = scenes.sample(np.random.default_rng(0)).grid
    expected = BEVGrid.from_scan(drop_ego_returns(drive.velodyne(20)))
    assert np.array_equal(grid.occupancy, expected.occupancy)


def test_kitti_scenes_can_serve_an_accumulated_map(drive):
    """Opting in to fusion gives the planner a materially denser scene of the same street."""
    from kitti_nav.mapping import MapConfig
    from kitti_nav.nav_env import KittiScenes

    single = KittiScenes(drive=drive, frames=np.array([20]))
    fused = KittiScenes(drive=drive, frames=np.array([20]),
                        map_config=MapConfig(window=5))

    n_single = single.sample(np.random.default_rng(0)).grid.occupancy.sum()
    n_fused = fused.sample(np.random.default_rng(0)).grid.occupancy.sum()
    assert n_fused > 1.4 * n_single, f"fusion added little: {n_single} -> {n_fused}"


def test_the_shield_still_admits_no_collision_on_a_fused_map(drive):
    """The guarantee is geometry-agnostic, and a harder map is exactly where that matters.

    The fused map holds obstacles a single scan cannot see, so policies meet genuinely
    unfamiliar geometry here. The shield is not a policy and does not care: it re-derives a
    braking certificate from whatever occupancy it is handed.
    """
    from kitti_nav.mapping import MapConfig
    from kitti_nav.nav_env import (
        DriveNavConfig,
        DriveNavEnv,
        KittiScenes,
        evaluate,
        gap_following_policy,
    )

    cfg = DriveNavConfig(use_shield=True)
    env = DriveNavEnv(KittiScenes(drive=drive, map_config=MapConfig(window=5)), cfg)
    stats = evaluate(env, lambda o: gap_following_policy(o, cfg), n_episodes=25)

    assert stats["collisions"] == 0
    assert stats["success_rate"] > 0.2, "the fused scenes should still be solvable"


# --- object tracklets (labels ship separately from the drive) ------------------------------

_HAS_TRACKLETS = (DEFAULT_DATA_DIR / "2011_09_26" / "2011_09_26_drive_0009_sync"
                  / "tracklet_labels.xml").exists()
needs_tracklets = pytest.mark.skipif(
    not _HAS_TRACKLETS, reason="tracklets absent; run fetch_kitti.py --tracklets")


@needs_tracklets
def test_tracklets_load_in_the_velodyne_frame(drive):
    """0009 ships 98 labelled objects; a present box sits inside the BEV grid, in velo coords."""
    ts = drive.tracklets
    assert len(ts) == 98
    assert {t.object_type for t in ts} >= {"Car", "Pedestrian"}

    bcfg = BEVConfig()
    seen = 0
    for i in range(30):
        for t, box in drive.tracklet_boxes(i):
            assert t.index_of(i) is not None                  # only present objects returned
            cx, cy = box[0], box[1]
            if bcfg.x_min < cx < bcfg.x_max and bcfg.y_min < cy < bcfg.y_max:
                seen += 1
    assert seen > 0, "no labelled object fell inside the grid in the first 30 frames"


@needs_tracklets
def test_moving_actors_are_distinguished_from_parked_ones():
    """The world-displacement test: a full 0009 has a dozen genuine movers among 98 objects."""
    full = KittiDrive("2011_09_26", "0009")               # full pose stream, not the 40-frame fixture
    movers = full.moving_tracklets(min_disp=2.0)
    assert 8 <= len(movers) <= 20                          # measured 12; loose regression guard
    assert len(movers) < len(full.tracklets)              # most objects are parked
    # A mover's world track really translates; the classifier agrees with its own threshold.
    t = max(movers, key=lambda t: len(t.tx))
    track = full.tracklet_world_track(t)
    assert np.linalg.norm(track[-1] - track[0]) > 2.0
