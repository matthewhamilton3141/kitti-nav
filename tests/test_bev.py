"""Tests for the lidar BEV occupancy grid and its distance field.

The final group is the one that matters architecturally: the braking shield running
directly against a grid, with no conversion to circles, proving the `ObstacleField` seam
actually holds.
"""

import numpy as np
import pytest

from kitti_nav.bev import (
    BEVConfig,
    BEVGrid,
    estimate_ground_z,
    occupancy_from_scan,
)
from kitti_nav.vehicle import (
    VehicleConfig,
    VehicleState,
    can_stop_safely,
    clearance,
    safety_shield,
    shielded_rollout,
    stopping_distance,
)

GROUND = -1.73


@pytest.fixture
def cfg():
    return BEVConfig()


def ground_points(spacing=0.25, seed=0):
    """A flat road surface: the returns that must NOT become obstacles.

    A dense regular lattice with noise, rather than a sparse random scatter, because real
    near-field lidar road returns *are* dense — and the per-cell ground estimate's whole
    job is to sit on them. Testing it against an unrealistically sparse road would measure
    the wrong thing.
    """
    rng = np.random.default_rng(seed)
    xs = np.arange(-10, 50, spacing)
    ys = np.arange(-20, 20, spacing)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    return np.stack([gx, gy, np.full(gx.size, GROUND) + rng.normal(0, 0.02, gx.size)],
                    axis=1)


def pillar(x, y, n=200, z0=0.3, z1=2.0):
    """A vertical obstacle at (x, y), spanning a height range above the ground."""
    return np.stack([np.full(n, float(x)), np.full(n, float(y)),
                     GROUND + np.linspace(z0, z1, n)], axis=1)


# --- grid construction ---------------------------------------------------------------------

def test_config_shape_follows_extent_and_resolution():
    c = BEVConfig(x_min=0, x_max=10, y_min=-5, y_max=5, resolution=0.5)
    assert c.shape == (20, 20)


def test_road_surface_alone_produces_an_empty_grid(cfg):
    """Ground removal is the whole ballgame: without it every cell is occupied."""
    grid = occupancy_from_scan(ground_points(), cfg)
    assert grid.sum() == 0, "road returns leaked through as obstacles"


def test_a_pillar_becomes_occupied_cells_at_the_right_place(cfg):
    scan = np.vstack([ground_points(), pillar(20.0, 4.0)])
    grid = occupancy_from_scan(scan, cfg)
    assert grid.sum() > 0

    rows, cols = np.nonzero(grid)
    x = cfg.x_min + rows * cfg.resolution
    y = cfg.y_min + cols * cfg.resolution
    assert np.allclose(x, 20.0, atol=cfg.resolution)
    assert np.allclose(y, 4.0, atol=cfg.resolution)


def test_overhead_structures_are_not_obstacles(cfg):
    """A bridge or canopy 4 m up is something the car drives under, not into."""
    canopy = pillar(20.0, 0.0, z0=4.0, z1=5.0)
    assert occupancy_from_scan(np.vstack([ground_points(), canopy]), cfg).sum() == 0


def test_low_kerb_noise_below_the_band_is_ignored(cfg):
    kerb = pillar(20.0, 0.0, z0=0.02, z1=0.10)      # below min_height = 0.25
    assert occupancy_from_scan(np.vstack([ground_points(), kerb]), cfg).sum() == 0


def test_points_outside_the_extent_do_not_wrap_around(cfg):
    """Negative coordinates cast to int floor toward zero and would wrap to the far edge."""
    far_behind = pillar(-500.0, 0.0)
    far_left = pillar(20.0, 500.0)
    grid = occupancy_from_scan(np.vstack([ground_points(), far_behind, far_left]), cfg)
    assert grid.sum() == 0, "out-of-range points wrapped into the grid"


def test_empty_scan_yields_an_empty_grid(cfg):
    assert occupancy_from_scan(np.zeros((0, 3)), cfg).sum() == 0


def test_reflectance_column_is_tolerated(cfg):
    """Velodyne scans arrive as (N, 4); the fourth column must not be read as a coordinate."""
    scan = np.vstack([ground_points(), pillar(20.0, 0.0)])
    with_reflectance = np.hstack([scan, np.random.default_rng(0).uniform(0, 1, (len(scan), 1))])
    np.testing.assert_array_equal(occupancy_from_scan(scan, cfg),
                                  occupancy_from_scan(with_reflectance, cfg))


def test_ground_height_is_estimated_when_not_supplied():
    """`ground_z=None` only means anything in "plane" mode — the default ignores it."""
    scan = np.vstack([ground_points(), pillar(20.0, 0.0)])
    assert estimate_ground_z(scan) == pytest.approx(GROUND, abs=0.1)

    auto = BEVConfig(ground_mode="plane", ground_z=None)
    grid = occupancy_from_scan(scan, auto)
    assert grid.sum() > 0                            # pillar still found
    assert grid.sum() < 100                          # but the road did not leak in


def test_unknown_ground_mode_is_rejected():
    with pytest.raises(ValueError, match="ground_mode"):
        occupancy_from_scan(ground_points(), BEVConfig(ground_mode="magic"))


def test_a_canopy_with_no_ground_return_beneath_it_is_still_drivable():
    """The reason ground is estimated over a *neighbourhood* and not per single cell.

    Lidar rings spread apart with range and nothing under an overhang reaches the road, so
    cells routinely hold no ground return of their own. Such a cell must borrow its floor
    from its neighbours; taking its own lowest return would make every overhang an obstacle.
    """
    cfg = BEVConfig()
    road = ground_points()
    hole = ~((np.abs(road[:, 0] - 20.0) < 0.3) & (np.abs(road[:, 1]) < 0.3))
    road = road[hole]                                # no road returns under the canopy

    canopy = pillar(20.0, 0.0, z0=4.0, z1=5.0)
    assert occupancy_from_scan(np.vstack([road, canopy]), cfg).sum() == 0


def test_an_overhang_wider_than_the_ground_window_is_flagged_conservatively():
    """A stated limitation, pinned so it stays a known trade-off rather than a surprise.

    If no ground return exists anywhere within `ground_window` of a cell, there is no floor
    to borrow and the structure above becomes its own ground — so a *wide* overhang (a
    bridge deck, a tunnel mouth) reads as an obstacle. That errs toward braking for
    something drivable, which is the safe direction to be wrong, but it is an error.
    Widening the window trades this against flattening real kerbs.
    """
    cfg = BEVConfig()
    road = ground_points()
    wide = ~((np.abs(road[:, 0] - 20.0) < 2.0) & (np.abs(road[:, 1]) < 2.0))
    canopy = np.vstack([pillar(x, y, z0=4.0, z1=5.0)
                        for x in (19.0, 20.0, 21.0) for y in (-1.0, 0.0, 1.0)])
    assert occupancy_from_scan(np.vstack([road[wide], canopy]), cfg).sum() > 0


def test_height_diff_tolerates_a_sloping_road_where_a_fixed_plane_does_not():
    """Measured on real data: a global plane invents obstacles once the road leaves it.

    A road that climbs 1.5 m across the grid is entirely drivable, but leaves any fixed
    height band and reads as a wall of obstacles. The per-cell estimate follows it.
    (Downhill fails the opposite way — the road drops *below* the band and becomes
    invisible, which is the more dangerous direction of the same bug.)
    """
    road = ground_points()
    road[:, 2] += 0.025 * (road[:, 0] - road[:, 0].min())     # gentle uphill

    sloped = occupancy_from_scan(road, BEVConfig(ground_mode="height_diff"))
    flat_assumption = occupancy_from_scan(road, BEVConfig(ground_mode="plane"))

    assert sloped.sum() == 0, "per-cell ground failed on a sloping road"
    assert flat_assumption.sum() > 0, "expected the fixed plane to mis-fire here"


# --- coordinates ---------------------------------------------------------------------------

def test_world_and_cell_coordinates_round_trip(cfg):
    grid = BEVGrid(np.zeros(cfg.shape, np.uint8), cfg)
    pts = np.array([[0.0, 0.0], [20.0, -5.0], [-9.0, 19.0]])
    back = grid.cell_to_world(grid.world_to_cell(pts))
    assert np.all(np.abs(back - pts) <= cfg.resolution)   # exact to within one cell


def test_forward_is_rows_and_left_is_columns(cfg):
    """Pin the axis convention: +x forward -> row, +y left -> col."""
    grid = BEVGrid(np.zeros(cfg.shape, np.uint8), cfg)
    origin = grid.world_to_cell(np.array([[0.0, 0.0]]))[0]
    ahead = grid.world_to_cell(np.array([[10.0, 0.0]]))[0]
    left = grid.world_to_cell(np.array([[0.0, 10.0]]))[0]
    assert ahead[0] > origin[0] and ahead[1] == origin[1]
    assert left[1] > origin[1] and left[0] == origin[0]


# --- distance field ------------------------------------------------------------------------

def test_distance_field_is_infinite_when_nothing_is_occupied(cfg):
    grid = BEVGrid(np.zeros(cfg.shape, np.uint8), cfg)
    assert np.all(np.isinf(grid.distance_field))
    assert np.all(np.isinf(grid.distance_to_obstacles(np.array([[5.0, 0.0]]))))


def test_distance_field_grows_with_range_from_an_obstacle(cfg):
    grid = BEVGrid.from_scan(np.vstack([ground_points(), pillar(20.0, 0.0)]), cfg)
    d = grid.distance_to_obstacles(np.array([[19.0, 0.0], [15.0, 0.0], [5.0, 0.0]]))
    assert d[0] < d[1] < d[2]


def test_distance_field_is_a_conservative_underestimate(cfg):
    """It may under-report clearance (braking early), never over-report (braking late)."""
    grid = BEVGrid.from_scan(np.vstack([ground_points(), pillar(20.0, 0.0)]), cfg)
    query = np.array([[10.0, 0.0], [18.0, 3.0], [0.0, 0.0]])
    true_d = np.linalg.norm(query - np.array([20.0, 0.0]), axis=1)
    reported = grid.distance_to_obstacles(query)
    assert np.all(reported <= true_d + 1e-6), "distance field over-reported clearance"
    assert np.all(reported > true_d - 2 * cfg.resolution), "estimate is uselessly loose"


def test_points_outside_the_grid_follow_the_outside_policy(cfg):
    scan = np.vstack([ground_points(), pillar(20.0, 0.0)])
    far = np.array([[500.0, 0.0]])
    assert np.isinf(BEVGrid.from_scan(scan, cfg).distance_to_obstacles(far)[0])
    strict = BEVGrid.from_scan(scan, cfg, outside_is_free=False)
    assert strict.distance_to_obstacles(far)[0] == -np.inf


# --- the third occupancy class: unknown -----------------------------------------------------

def test_unknown_mask_is_inert_by_default(cfg):
    """Carrying an `unknown` mask changes nothing until `unknown_blocks` is asked for.

    This is the reproducibility guarantee: a carved map can hand its mask to every consumer and
    no existing number moves unless the honest reading is explicitly turned on.
    """
    occ = np.zeros(cfg.shape, np.uint8)
    occ[100, 100] = 1
    unknown = np.zeros(cfg.shape, bool)
    unknown[:20, :] = True                          # a big unobserved band
    plain = BEVGrid(occ, cfg)
    carried = BEVGrid(occ, cfg, unknown=unknown)    # mask present, unknown_blocks off
    assert np.array_equal(plain.blocking, carried.blocking)
    assert np.allclose(plain.distance_field, carried.distance_field)


def test_unknown_blocks_makes_unknown_cells_obstacles(cfg):
    """With the flag on, an unknown cell is as impassable as an occupied one."""
    occ = np.zeros(cfg.shape, np.uint8)
    unknown = np.zeros(cfg.shape, bool)
    r, c = cfg.shape[0] // 2, cfg.shape[1] // 2
    unknown[r, c] = True

    free = BEVGrid(occ, cfg, unknown=unknown, unknown_blocks=False)
    blocking = BEVGrid(occ, cfg, unknown=unknown, unknown_blocks=True)
    assert not free.blocking.any()                              # unknown ignored
    assert blocking.blocking[r, c] and blocking.blocking.sum() == 1
    xy = blocking.cell_to_world(np.array([[r, c]]))
    assert np.isinf(free.distance_to_obstacles(xy)[0])          # no obstacle to free
    assert blocking.distance_to_obstacles(xy)[0] == 0.0         # standing on the obstacle


def test_unknown_blocks_stops_a_ray(cfg):
    """A forward ray halts at an unknown cell the same way it halts at an occupied one."""
    occ = np.zeros(cfg.shape, np.uint8)
    unknown = np.zeros(cfg.shape, bool)
    wall_r = int((15.0 - cfg.x_min) / cfg.resolution)
    unknown[wall_r, :] = True                                   # unknown wall across +x at 15 m

    origin, angles = np.array([0.0, 0.0]), np.zeros(1)
    free = BEVGrid(occ, cfg, unknown=unknown, unknown_blocks=False)
    blocking = BEVGrid(occ, cfg, unknown=unknown, unknown_blocks=True)
    assert free.ray_distances(origin, 0.0, angles, max_range=30.0)[0] == 30.0
    assert 14.0 < blocking.ray_distances(origin, 0.0, angles, max_range=30.0)[0] < 16.0


# --- confident-clear distance: the geometry the speed governor caps against -----------------

def test_cautious_blocking_ignores_the_unknown_blocks_flag(cfg):
    """`cautious_blocking` is occupied|unknown regardless of how the shield treats unknown."""
    occ = np.zeros(cfg.shape, np.uint8)
    occ[100, 100] = 1
    unknown = np.zeros(cfg.shape, bool)
    unknown[50, 50] = True

    for flag in (False, True):
        grid = BEVGrid(occ, cfg, unknown=unknown, unknown_blocks=flag)
        assert grid.cautious_blocking[100, 100] and grid.cautious_blocking[50, 50]
        assert grid.cautious_blocking.sum() == 2

    # With no mask it collapses to occupancy, so an un-carved map is unaffected.
    assert np.array_equal(BEVGrid(occ, cfg).cautious_blocking, occ.astype(bool))


def test_confident_clear_distance_stops_at_unknown_not_just_occupied(cfg):
    """The frontier distance halts at an unknown wall even when the shield lets rays through.

    This is the whole point of measuring against `cautious_blocking` rather than `blocking`:
    the governor must see the edge of confidently-free space regardless of `unknown_blocks`.
    """
    occ = np.zeros(cfg.shape, np.uint8)
    unknown = np.zeros(cfg.shape, bool)
    wall_r = int((15.0 - cfg.x_min) / cfg.resolution)
    unknown[wall_r, :] = True                                   # unknown wall at +x = 15 m

    grid = BEVGrid(occ, cfg, unknown=unknown, unknown_blocks=False)
    # A collision ray sees nothing (unknown_blocks off); the frontier ray still stops at 15 m.
    assert grid.ray_distances(np.zeros(2), 0.0, np.zeros(1), max_range=30.0)[0] == 30.0
    d = grid.confident_clear_distance(np.zeros(2), 0.0, np.zeros(1), max_range=30.0)[0]
    assert 14.0 < d < 16.0


def test_confident_clear_distance_takes_the_cone_minimum(cfg):
    """A fan returns the nearest frontier across the cone, so the cap is the conservative one."""
    occ = np.zeros(cfg.shape, np.uint8)
    unknown = np.zeros(cfg.shape, bool)
    # An unknown block off to the +y side, closer than anything straight ahead.
    r0, r1 = int((7.0 - cfg.x_min) / cfg.resolution), int((9.0 - cfg.x_min) / cfg.resolution)
    c0, c1 = int((2.0 - cfg.y_min) / cfg.resolution), int((4.0 - cfg.y_min) / cfg.resolution)
    unknown[r0:r1, c0:c1] = True

    grid = BEVGrid(occ, cfg, unknown=unknown)
    wide = grid.confident_clear_distance(np.zeros(2), 0.0, np.linspace(-0.6, 0.6, 9),
                                         max_range=30.0)
    ahead = grid.confident_clear_distance(np.zeros(2), 0.0, np.zeros(1), max_range=30.0)[0]
    assert ahead == 30.0                          # a straight ray misses the off-axis block
    assert float(wide.min()) < 12.0               # a wide cone reaches it and reports it


def test_grid_extent_is_checked_against_the_braking_envelope(cfg):
    grid = BEVGrid(np.zeros(cfg.shape, np.uint8), cfg)
    assert grid.covers_stopping_distance(25.0)
    assert not grid.covers_stopping_distance(80.0)


def test_mismatched_grid_and_config_are_rejected(cfg):
    with pytest.raises(ValueError):
        BEVGrid(np.zeros((5, 5), np.uint8), cfg)


# --- the shield, running natively on a grid -------------------------------------------------

def test_shield_runs_directly_against_a_lidar_grid(cfg):
    """The `ObstacleField` seam: no conversion of occupancy into circles anywhere."""
    vcfg = VehicleConfig()
    open_road = BEVGrid.from_scan(ground_points(), cfg)
    free = safety_shield(vcfg.max_accel, 0.0, VehicleState(v=8.0), open_road, vcfg)
    assert not free.intervened, "shield braked on an empty grid"

    # Wall placement is not arbitrary. At 12 m/s one step of throttle covers 1.2 m and
    # leaves 16.5 m of braking, with the front bumper 3.8 m ahead of the rear axle — so a
    # wall must sit inside ~21.8 m before full throttle is genuinely unsafe. Putting it
    # further out and asserting intervention would be testing timidity, not safety.
    wall = np.vstack([pillar(20.0, y) for y in np.arange(-6.0, 6.0, 0.4)])
    grid = BEVGrid.from_scan(np.vstack([ground_points(), wall]), cfg)
    blocked = safety_shield(vcfg.max_accel, 0.0, VehicleState(v=12.0), grid, vcfg)
    assert blocked.intervened, "shield ignored a lidar wall inside its stopping distance"

    # ...and it stays permissive when the same wall is comfortably beyond that envelope.
    far = np.vstack([pillar(45.0, y) for y in np.arange(-6.0, 6.0, 0.4)])
    far_grid = BEVGrid.from_scan(np.vstack([ground_points(), far]), cfg)
    assert not safety_shield(vcfg.max_accel, 0.0, VehicleState(v=12.0),
                             far_grid, vcfg).intervened


def test_vehicle_stops_short_of_a_lidar_wall(cfg):
    """End to end: reckless policy + real grid geometry -> stops clear, every time."""
    vcfg = VehicleConfig()
    wall = np.vstack([pillar(35.0, y) for y in np.arange(-8.0, 8.0, 0.4)])
    grid = BEVGrid.from_scan(np.vstack([ground_points(), wall]), cfg)

    start = VehicleState(v=10.0)
    assert grid.covers_stopping_distance(stopping_distance(vcfg.max_speed, vcfg))
    assert can_stop_safely(start, grid, vcfg)

    states, stats = shielded_rollout(lambda s: (vcfg.max_accel, 0.0), start, grid,
                                     vcfg, n_steps=200, shield=True)
    assert not stats["collided"]
    assert stats["final_speed"] == pytest.approx(0.0, abs=1e-6)
    assert stats["n_interventions"] > 0
    # It stopped short of the wall, and did not simply refuse to move.
    assert 5.0 < states[-1].x < 35.0
    assert clearance(states[-1], grid, vcfg) >= 0.0


def test_unshielded_control_run_hits_the_same_wall(cfg):
    """Control condition — without the shield this scene really is a collision."""
    vcfg = VehicleConfig()
    wall = np.vstack([pillar(35.0, y) for y in np.arange(-8.0, 8.0, 0.4)])
    grid = BEVGrid.from_scan(np.vstack([ground_points(), wall]), cfg)
    _, stats = shielded_rollout(lambda s: (vcfg.max_accel, 0.0), VehicleState(v=10.0),
                                grid, vcfg, n_steps=200, shield=False)
    assert stats["collided"]
