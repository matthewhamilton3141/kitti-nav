"""Tests for multi-scan fusion — the join between odometry and the BEV map.

Two groups, testing two different things.

The first is the transform algebra. It is easy to write pose composition that looks right
and is silently wrong by a fixed extrinsic, and the symptom on real data (a map that is
merely a bit blurry) looks exactly like ordinary drift. So these tests pin the composition
against a construction where the answer is known analytically, and check it is *invariant*
to the camera/lidar extrinsic — a version that dropped or mis-ordered `T_cam_velo` passes
neither.

The second is the behaviour the module claims: fusion fills in what one scan could not see,
and pose error corrupts the result in two asymmetric ways. Those tests use synthetic scenes
where ground truth is exact, so the numbers mean something; the real-data version of the
same question lives in `scripts/eval_mapping.py`.

Everything here is pure NumPy — no KITTI download, no torch.
"""

import numpy as np
import pytest

from kitti_nav.bev import BEVConfig, BEVGrid, occupancy_from_scan
from kitti_nav.kitti import DEFAULT_DATA_DIR, KittiDrive, invert_se3
from kitti_nav.mapping import (
    KITTI_EGO_BOX,
    MapConfig,
    ScanAccumulator,
    drop_ego_returns,
    fuse_scans,
    occupancy_agreement,
    relative_lidar_transform,
    stream_maps,
    transform_points,
    window_indices,
)
from kitti_nav.vehicle import VehicleConfig, VehicleState, clearance, max_safe_speed

GROUND = -1.73

# A KITTI-shaped lidar->camera extrinsic: the Velodyne frame (+x forward, +y left, +z up)
# rotated into the camera frame (+x right, +y down, +z forward), plus a small lever arm.
# Any bug that ignores this rotation is off by 90 degrees and cannot hide.
T_CAM_VELO = np.array([[0.0, -1.0, 0.0, 0.06],
                       [0.0, 0.0, -1.0, -0.08],
                       [1.0, 0.0, 0.0, -0.27],
                       [0.0, 0.0, 0.0, 1.0]])


def se3(x=0.0, y=0.0, z=0.0, yaw=0.0):
    """Rigid transform with a yaw about +z — a car's motion in the Velodyne/world frame."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0, x],
                     [s, c, 0.0, y],
                     [0.0, 0.0, 1.0, z],
                     [0.0, 0.0, 0.0, 1.0]])


def cam_pose(T_velo_world, T_cam_velo=T_CAM_VELO):
    """Convert a sensor-rig pose into the camera-to-world pose the module consumes.

    The rig pose is the intuitive thing to write a test scene in ("the car is 5 m further
    along"); the API takes camera poses, because that is what `odometry.py` produces.
    """
    return T_velo_world @ invert_se3(T_cam_velo)


def observe(world_points, T_velo_world):
    """What a lidar at rig pose `T_velo_world` measures of a static world, in its own frame."""
    return transform_points(world_points, invert_se3(T_velo_world))


def road(spacing=0.25, x=(-10, 40), y=(-12, 12), z=GROUND, seed=0, noise=0.02):
    """Flat road surface in world coordinates — returns that must never become obstacles."""
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(*x, spacing), np.arange(*y, spacing), indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    return np.stack([gx, gy, np.full(gx.size, z) + rng.normal(0, noise, gx.size)], axis=1)


def wall(x, y0, y1, n=400, height=2.0):
    """A vertical wall segment at `x`, spanning `[y0, y1]`, standing on the road."""
    ys = np.linspace(y0, y1, n)
    zs = np.linspace(GROUND + 0.3, GROUND + height, 12)
    gy, gz = np.meshgrid(ys, zs, indexing="ij")
    return np.stack([np.full(gy.size, float(x)), gy.ravel(), gz.ravel()], axis=1)


# --- transform algebra ---------------------------------------------------------------------

def test_identical_poses_give_the_identity_whatever_the_extrinsic():
    """The composition must cancel `T_cam_velo` exactly, or every map is offset by it."""
    P = cam_pose(se3(x=12.0, yaw=0.4))
    T = relative_lidar_transform(P, P, T_CAM_VELO)
    assert np.allclose(T, np.eye(4), atol=1e-12)


def test_a_static_point_lands_in_the_same_place_from_every_viewpoint():
    """The defining property of fusion: geometry is fixed, only the sensor moves."""
    target = np.array([[20.0, 3.0, 0.0]])
    rig_ref = se3(x=4.0, yaw=0.15)
    truth = observe(target, rig_ref)                     # where the ref frame sees it

    for rig in (se3(), se3(x=9.0, y=-1.0, yaw=-0.3), se3(x=-6.0, yaw=1.1)):
        seen = observe(target, rig)                      # a different viewpoint's raw scan
        moved = transform_points(seen, relative_lidar_transform(
            cam_pose(rig_ref), cam_pose(rig), T_CAM_VELO))
        assert np.allclose(moved, truth, atol=1e-4)


def test_fusion_is_invariant_to_the_camera_lidar_extrinsic():
    """Same rig motion, different extrinsic, identical fused cloud.

    The extrinsic describes where the sensors sit relative to each other, not where the
    world is. If it leaks into the output, the composition is unbalanced somewhere.
    """
    world = np.concatenate([road(spacing=1.0), wall(25.0, -4.0, 4.0)])
    rigs = [se3(x=d, yaw=0.05 * i) for i, d in enumerate([0.0, 2.0, 4.0])]
    other = se3(y=0.5, z=0.2, yaw=0.9) @ T_CAM_VELO      # a differently-mounted rig

    fused = []
    for extr in (T_CAM_VELO, other):
        scans = [observe(world, r) for r in rigs]
        poses = [cam_pose(r, extr) for r in rigs]
        fused.append(fuse_scans(scans, poses, extr, ref=-1))

    assert np.allclose(fused[0], fused[1], atol=1e-3)


def test_transform_points_ignores_the_reflectance_column():
    """Velodyne scans are `(N, 4)`; rotating the intensity channel would be nonsense."""
    pts = np.array([[1.0, 2.0, 3.0, 0.7]])
    out = transform_points(pts, se3(x=1.0))
    assert out.shape == (1, 3)
    assert np.allclose(out, [[2.0, 2.0, 3.0]])


def test_transform_points_handles_an_empty_scan():
    assert transform_points(np.zeros((0, 4)), np.eye(4)).shape == (0, 3)


# --- window selection ----------------------------------------------------------------------

def test_window_looks_backwards_only():
    """A live planner cannot fuse scans it has not received yet."""
    idx = window_indices(50, MapConfig(window=4), n_available=200)
    assert idx == [47, 48, 49, 50]
    assert max(idx) == 50


def test_window_stride_spreads_the_same_scan_count_over_a_longer_baseline():
    assert window_indices(50, MapConfig(window=4, stride=3), 200) == [41, 44, 47, 50]


def test_window_truncates_at_the_start_of_the_drive():
    """Early frames have less history; the map is thinner rather than the code failing."""
    assert window_indices(2, MapConfig(window=5), 200) == [0, 1, 2]


def test_window_of_one_is_exactly_the_single_scan_case():
    assert window_indices(7, MapConfig(window=1), 200) == [7]


@pytest.mark.parametrize("bad", [dict(window=0), dict(stride=0)])
def test_map_config_rejects_degenerate_windows(bad):
    with pytest.raises(ValueError):
        MapConfig(**bad)


# --- accumulator behaviour -----------------------------------------------------------------

def test_accumulator_drops_scans_beyond_the_window():
    acc = ScanAccumulator(T_CAM_VELO, MapConfig(window=3))
    for i in range(10):
        acc.add(np.zeros((5, 3)), cam_pose(se3(x=float(i))))
    assert len(acc) == 3
    assert acc.n_added == 10


def test_accumulator_stride_skips_scans_but_still_counts_them():
    acc = ScanAccumulator(T_CAM_VELO, MapConfig(window=4, stride=2))
    stored = [acc.add(np.zeros((5, 3)), cam_pose(se3(x=float(i)))) for i in range(6)]
    assert stored == [True, False, True, False, True, False]
    assert len(acc) == 3 and acc.n_added == 6


def test_accumulator_fuses_into_the_latest_frame_by_default():
    """The map a live planner wants is centred where the car is *now*."""
    target = np.array([[30.0, 0.0, 0.0]])
    acc = ScanAccumulator(T_CAM_VELO, MapConfig(window=2))
    for d in (0.0, 5.0):
        acc.add(observe(target, se3(x=d)), cam_pose(se3(x=d)))

    fused = acc.fused_points()
    assert np.allclose(fused[:, 0], 25.0, atol=1e-3)     # 30 m away, after driving 5 m


def test_empty_accumulator_yields_no_points_and_refuses_a_reference_pose():
    acc = ScanAccumulator(T_CAM_VELO)
    assert acc.fused_points().shape == (0, 3)
    with pytest.raises(RuntimeError):
        _ = acc.latest_pose


def test_accumulator_rejects_a_malformed_pose():
    acc = ScanAccumulator(T_CAM_VELO)
    with pytest.raises(ValueError):
        acc.add(np.zeros((5, 3)), np.eye(3))


def test_max_range_prunes_far_returns_at_insert():
    pts = np.array([[10.0, 0.0, 0.0], [90.0, 0.0, 0.0]])
    acc = ScanAccumulator(T_CAM_VELO, MapConfig(window=1, max_range=70.0))
    acc.add(pts, cam_pose(se3()))
    assert len(acc.fused_points()) == 1


# --- what fusion actually buys -------------------------------------------------------------

def test_fusing_sparse_scans_maps_more_of_the_wall_than_any_one_scan():
    """Models lidar ring sparsity: each scan samples the same wall incompletely.

    Sparsity is applied per *column* rather than per point, because that is how the sensor
    misses things — a vertical fan either hits a patch of wall and returns its whole height
    or sweeps past it entirely. Dropping individual points instead would thin each column
    until its height spread vanished, and would be testing ground removal, not fusion.
    """
    rng = np.random.default_rng(0)
    ys = np.linspace(-6.0, 6.0, 60)                      # one column per 0.2 m cell
    cfg = BEVConfig()

    scans, poses = [], []
    for i in range(6):
        rig = se3(x=1.0 * i)
        seen = ys[rng.random(len(ys)) < 0.25]            # a quarter of the wall per scan
        cols = np.concatenate([wall(25.0, y, y, n=1) for y in seen])
        scans.append(observe(cols, rig))
        poses.append(cam_pose(rig))

    single = occupancy_from_scan(scans[-1], cfg).sum()
    fused = occupancy_from_scan(fuse_scans(scans, poses, T_CAM_VELO), cfg).sum()
    assert fused > single * 1.5, f"fusion added little: {single} -> {fused} cells"


def test_fusion_fills_a_shadow_the_current_scan_cannot_see():
    """The headline claim: geometry hidden behind an obstacle *now* was visible earlier.

    Occlusion is modelled analytically — a point is hidden when the straight line from the
    sensor to it passes through the blocker's span — which is enough to create a real
    shadow without simulating ray casting.
    """
    blocker_x, span = 12.0, (-1.0, 3.0)
    hidden = wall(20.0, 0.0, 2.0, n=200)                 # sits inside the shadow at the end

    def visible(pts, sensor_xy):
        """Points not cut off by the blocker, from a sensor at `sensor_xy`."""
        sx, sy = sensor_xy
        t = (blocker_x - sx) / np.where(pts[:, 0] == sx, 1e-9, pts[:, 0] - sx)
        crossing = sy + t * (pts[:, 1] - sy)
        blocked = (t > 0) & (t < 1) & (crossing >= span[0]) & (crossing <= span[1])
        return pts[~blocked]

    cfg = BEVConfig()
    scans, poses = [], []
    for x, y in [(0.0, -6.0), (1.0, -4.0), (2.0, -2.0), (3.0, 0.0)]:
        rig = se3(x=x, y=y)
        scans.append(observe(visible(hidden, (x, y)), rig))
        poses.append(cam_pose(rig))

    # By the last viewpoint the wall is behind the blocker and mostly unseen.
    current = occupancy_from_scan(scans[-1], cfg).sum()
    fused = occupancy_from_scan(fuse_scans(scans, poses, T_CAM_VELO), cfg).sum()
    assert current < fused, f"shadow was not filled: {current} vs {fused} cells"
    assert fused >= 8, "the hidden wall should be substantially recovered"


def test_a_fused_map_still_drives_the_shield_with_no_conversion():
    """The whole point of staying in the Velodyne frame: the seam holds end to end."""
    world = np.concatenate([road(), wall(18.0, -6.0, 6.0)])
    acc = ScanAccumulator(T_CAM_VELO, MapConfig(window=3))
    for d in (0.0, 0.6, 1.2):
        acc.add(observe(world, se3(x=d)), cam_pose(se3(x=d)))

    grid = acc.grid()
    vcfg = VehicleConfig(max_speed=25.0)
    state = VehicleState()
    assert grid.occupancy.sum() > 0
    assert clearance(state, grid, vcfg) > 0              # not already in contact
    # The wall sits ~16.8 m ahead of the front bumper; at 4.5 m/s^2 that caps the shield
    # near sqrt(2*a*d) ~ 12 m/s, so it must permit something, but far from the 25 m/s cap.
    permitted = max_safe_speed(grid, vcfg, state=state)
    assert 3.0 < permitted < 20.0, permitted


def test_stream_maps_warms_up_like_a_live_system():
    world = np.concatenate([road(spacing=0.5), wall(20.0, -4.0, 4.0)])
    rigs = [se3(x=0.5 * i) for i in range(5)]
    scans = [observe(world, r) for r in rigs]
    poses = [cam_pose(r) for r in rigs]

    out = list(stream_maps(scans, poses, T_CAM_VELO, MapConfig(window=3)))
    assert [i for i, _ in out] == [0, 1, 2, 3, 4]
    assert all(isinstance(g, BEVGrid) for _, g in out)


# --- pose error, and which kind of error it makes -------------------------------------------

def test_drift_free_fusion_of_flat_road_stays_empty():
    """Control for the next test: with exact poses, accumulation invents nothing."""
    rigs = [se3(x=0.5 * i) for i in range(6)]
    scans = [observe(road(), r) for r in rigs]
    poses = [cam_pose(r) for r in rigs]
    assert occupancy_from_scan(fuse_scans(scans, poses, T_CAM_VELO)).sum() == 0


def test_vertical_pose_error_manufactures_phantom_obstacles_from_open_road():
    """The failure mode the module docstring warns about, made to happen on purpose.

    Mis-registering the road vertically puts one surface at two heights in the same cell,
    and `height_diff` ground removal reads that spread as an obstacle. This is why the
    evaluation cannot just report map similarity: drift's *first* effect is inventing
    obstacles on open road.
    """
    rigs = [se3(x=0.5 * i) for i in range(6)]
    scans = [observe(road(), r) for r in rigs]

    # Same scans, but the poses believe the car rose 8 cm per frame.
    drifted = [cam_pose(se3(x=0.5 * i, z=0.08 * i)) for i in range(6)]
    phantom = occupancy_from_scan(fuse_scans(scans, drifted, T_CAM_VELO)).sum()
    assert phantom > 100, f"expected drift to invent obstacles, got {phantom} cells"


def test_occupancy_agreement_separates_phantom_from_missed():
    """Missed cells are the dangerous class; an aggregate score would hide them."""
    ref = np.zeros((10, 10), np.uint8)
    ref[2:5, 2:5] = 1                                    # 9 truly occupied cells
    est = np.zeros((10, 10), np.uint8)
    est[2:4, 2:5] = 1                                    # lost one row of 3
    est[8, 8] = 1                                        # invented one

    m = occupancy_agreement(est, ref)
    assert m["missed_cells"] == 3 and m["phantom_cells"] == 1
    assert m["n_reference"] == 9 and m["n_estimated"] == 7
    assert m["missed_rate"] == pytest.approx(3 / 9)
    assert m["iou"] == pytest.approx(6 / 10)


def test_occupancy_agreement_is_perfect_on_identical_grids():
    g = np.zeros((6, 6), np.uint8)
    g[1:3, 1:3] = 1
    m = occupancy_agreement(g, g)
    assert m["iou"] == 1.0 and m["missed_cells"] == 0 and m["phantom_cells"] == 0


def test_occupancy_agreement_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        occupancy_agreement(np.zeros((4, 4)), np.zeros((5, 5)))


def test_fuse_scans_rejects_mismatched_scan_and_pose_counts():
    with pytest.raises(ValueError):
        fuse_scans([np.zeros((3, 3))], [np.eye(4), np.eye(4)], T_CAM_VELO)


# --- ego self-filtering, against the real sensor ---------------------------------------------
#
# These need KITTI: the whole point is that the recording car's own bodywork is a property of
# the physical rig, not something synthetic data can stand in for. They skip cleanly without it.

kitti = pytest.mark.skipif(
    not (DEFAULT_DATA_DIR / "2011_09_26" / "2011_09_26_drive_0009_sync").exists(),
    reason="KITTI drive not downloaded; run scripts/fetch_kitti.py",
)


@pytest.fixture(scope="module")
def drive():
    return KittiDrive("2011_09_26", "0009", n_frames=60)


@kitti
def test_ego_box_covers_the_self_returns():
    """Re-derive the self-return extent from the data and check the constant still bounds it.

    `KITTI_EGO_BOX` is a measured number, so it needs a test that measures rather than one
    that restates it. Self-returns are identified as cells holding a return in *every* scan,
    searched only within what the vehicle could physically occupy — roadside structure also
    persists in sensor-frame coordinates while the car holds its lane, so without the
    lateral bound this finds the kerb instead of the car.

    Uses the **whole** drive rather than the shared 60-frame fixture, and deliberately so: a
    short window is not long enough for roadside structure to wash out. Over 12 scans this
    still reports a stray cell at the search boundary; by ~45 it has converged to
    x [-1.20, 2.50], y [-1.50, 1.40] and stays there.
    """
    full = KittiDrive("2011_09_26", "0009")
    frames = list(range(0, full.n_velodyne, 10))
    near = [full.velodyne(f)[:, :3] for f in frames]
    near = [p[(np.abs(p[:, 0]) < 4.0) & (np.abs(p[:, 1]) < 2.0) & (p[:, 2] > -1.35)]
            for p in near]

    cells = np.round(np.concatenate(near)[:, :2] / 0.1).astype(int)
    uniq, counts = np.unique(cells, axis=0, return_counts=True)
    fixed = uniq[counts >= len(frames)] * 0.1
    assert len(fixed) > 0, "no rigidly-attached returns found at all — check the sensor frame"

    x_min, x_max, y_min, y_max = KITTI_EGO_BOX
    assert fixed[:, 0].min() >= x_min and fixed[:, 0].max() <= x_max
    assert fixed[:, 1].min() >= y_min and fixed[:, 1].max() <= y_max


@kitti
def test_ego_filter_barely_touches_a_single_scan(drive):
    """The filter is nearly free on the un-fused path, so it can be on by default.

    Single-scan occupancy is unchanged: those cells contained *only* bodywork, so the
    per-cell ground estimate was already treating them as free. This is exactly why the bug
    stayed hidden until scans were accumulated.
    """
    scan = drive.velodyne(30)
    filtered = drop_ego_returns(scan)
    assert 0 < len(scan) - len(filtered) < 200          # a few dozen returns, not a swathe
    assert (occupancy_from_scan(scan).sum()
            == occupancy_from_scan(filtered).sum())


@kitti
def test_fusion_does_not_map_the_car_on_top_of_itself(drive):
    """Regression for the bug this module was built to find.

    Without ego self-filtering, fusing scans places the recording car's own roof rails and
    hood into cells that an earlier viewpoint supplied road for, reads the 0.8 m difference
    as an obstacle, and parks a phantom wall inside the vehicle footprint. Measured on this
    drive it drove the shield-permitted speed from 25.1 m/s to 2.2 m/s.
    """
    ref = 40
    idx = window_indices(ref, MapConfig(window=5), drive.n_velodyne)
    scans = [drive.velodyne(i) for i in idx]
    poses = [drive.gt_poses[i] for i in idx]
    state = drive.vehicle_state_in_lidar()
    vcfg = VehicleConfig()

    unfiltered = BEVGrid(occupancy_from_scan(
        fuse_scans(scans, poses, drive.T_cam2_velo, ego_box=None)))
    filtered = BEVGrid(occupancy_from_scan(
        fuse_scans(scans, poses, drive.T_cam2_velo)))

    assert clearance(state, unfiltered, vcfg) < 0, \
        "expected the unfiltered fusion to bury the car in its own returns"
    assert clearance(state, filtered, vcfg) > vcfg.safety_margin, \
        "self-filtering must leave the vehicle footprint clear"


@kitti
def test_fusion_with_ground_truth_poses_maps_more_than_one_scan(drive):
    """The payoff, on real data: accumulation is a large coverage gain, not a rounding error."""
    ref = 40
    idx = window_indices(ref, MapConfig(window=5), drive.n_velodyne)
    fused = occupancy_from_scan(fuse_scans(
        [drive.velodyne(i) for i in idx],
        [drive.gt_poses[i] for i in idx], drive.T_cam2_velo)).sum()
    single = occupancy_from_scan(drop_ego_returns(drive.velodyne(ref))).sum()
    assert fused > 1.5 * single, f"only {fused / single:.2f}x coverage from 5 scans"


@kitti
def test_a_window_of_one_reproduces_the_single_scan_map(drive):
    """The control that makes the whole comparison trustworthy: with one scan the pose cannot
    matter, so a difference here would be a bug in the transform, not a real effect.

    Agreement is near-exact rather than exact, and the residual is understood: `fuse_scans`
    routes points through a float64 matmul and casts the result to float32, so a return
    sitting within a rounding error of a cell boundary can land on either side of it. That
    moves a couple of cells out of ~2600 and nothing else — the same 0.997 IoU the
    window-1 row of `scripts/eval_mapping.py` reports.
    """
    for poses in (drive.gt_poses, drive.gt_poses[::-1]):
        fused = occupancy_from_scan(fuse_scans(
            [drive.velodyne(20)], [poses[20]], drive.T_cam2_velo))
        single = occupancy_from_scan(drop_ego_returns(drive.velodyne(20)))
        assert occupancy_agreement(fused, single)["iou"] > 0.99
