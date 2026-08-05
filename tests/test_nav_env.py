"""Tests for the driving navigation environment and its scene sources.

The headline test is `test_shield_eliminates_collisions_from_certifiable_starts`: the shield
guarantee, end to end, through the full env loop rather than in isolation.
"""

import numpy as np
import pytest
from dataclasses import replace

from kitti_nav.bev import BEVConfig, BEVGrid
from kitti_nav.dynamics import MovingObstacle
from kitti_nav.nav_env import (
    DriveNavConfig,
    DriveNavEnv,
    DynamicNavConfig,
    DynamicNavEnv,
    DynamicScene,
    Scene,
    SyntheticScenes,
    cautious_speed_cap,
    certifiable_start,
    evaluate,
    gap_following_policy,
    kitti_frame_split,
    rasterize_circles,
    rollout,
)
from kitti_nav.vehicle import (
    VehicleConfig,
    VehicleState,
    can_stop_safely,
    clearance,
    stopping_distance,
)


@pytest.fixture
def cfg():
    return DriveNavConfig()


class FixedScene:
    """A scene source that always returns the same world — makes episodes deterministic."""

    def __init__(self, scene: Scene):
        self.scene = scene

    def sample(self, rng):
        return self.scene


def empty_scene(start_speed=5.0, goal=(30.0, 0.0), circles=None):
    bev = BEVConfig()
    occ = rasterize_circles(np.array(circles if circles is not None else [],
                                     float).reshape(-1, 3), bev)
    return Scene(BEVGrid(occ, bev),
                 VehicleState(x=0.0, y=0.0, yaw=0.0, v=start_speed),
                 np.array(goal, float))


# --- rasterisation and rays -----------------------------------------------------------------

def test_rasterize_circles_paints_the_right_cells():
    bev = BEVConfig()
    grid = rasterize_circles(np.array([[10.0, 2.0, 1.0]]), bev)
    rows, cols = np.nonzero(grid)
    x = bev.x_min + rows * bev.resolution
    y = bev.y_min + cols * bev.resolution
    assert grid.sum() > 0
    assert np.all(np.hypot(x - 10.0, y - 2.0) <= 1.0 + 2 * bev.resolution)


def test_rays_reach_full_range_on_an_empty_scene(cfg):
    grid = empty_scene().grid
    rays = grid.ray_distances(np.array([0.0, 0.0]), 0.0, np.array([0.0]), max_range=25.0)
    assert rays[0] == pytest.approx(25.0, abs=0.2)


def test_rays_stop_at_an_obstacle(cfg):
    grid = empty_scene(circles=[[12.0, 0.0, 1.5]]).grid
    ahead = grid.ray_distances(np.array([0.0, 0.0]), 0.0, np.array([0.0]), max_range=30.0)
    assert ahead[0] == pytest.approx(10.5, abs=0.5)      # obstacle surface at 12 - 1.5


def test_rays_are_blocked_by_the_grid_edge_when_asked(cfg):
    grid = empty_scene().grid
    behind = grid.ray_distances(np.array([0.0, 0.0]), 0.0, np.array([np.pi]), max_range=30.0)
    # The grid only extends 10 m behind the origin, so a rearward ray stops there.
    assert behind[0] == pytest.approx(10.0, abs=0.3)


def test_rays_follow_the_heading(cfg):
    """A ray fan is expressed relative to yaw, so turning must move what it sees."""
    grid = empty_scene(circles=[[0.0, 12.0, 1.5]]).grid       # obstacle to the LEFT
    straight = grid.ray_distances(np.array([0.0, 0.0]), 0.0, np.array([0.0]), 30.0)[0]
    turned = grid.ray_distances(np.array([0.0, 0.0]), np.pi / 2, np.array([0.0]), 30.0)[0]
    assert turned < straight


# --- env mechanics ---------------------------------------------------------------------------

def test_observation_has_the_declared_shape(cfg):
    env = DriveNavEnv(FixedScene(empty_scene()), cfg)
    obs, info = env.reset(0)
    assert obs.shape == (cfg.obs_dim,) == (5 + cfg.n_rays,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert info["reached"] is False


def test_goal_is_reported_in_the_ego_frame(cfg):
    """Turning the car must change the goal bearing even though the goal never moves."""
    scene = empty_scene(goal=(30.0, 0.0))
    env = DriveNavEnv(FixedScene(scene), cfg)
    obs, _ = env.reset(0)
    assert obs[0] > 0 and obs[1] == pytest.approx(0.0, abs=1e-6)   # dead ahead

    turned = Scene(scene.grid, replace(scene.start, yaw=np.pi / 2), scene.goal)
    obs2, _ = DriveNavEnv(FixedScene(turned), cfg).reset(0)
    assert obs2[1] < 0        # goal is now to the right of the heading


def test_action_denormalisation_uses_asymmetric_braking(cfg):
    """Cars brake harder than they accelerate; a symmetric scale would waste the brakes."""
    env = DriveNavEnv(FixedScene(empty_scene()), cfg)
    assert env.denormalize_action(np.array([1.0, 0.0]))[0] == cfg.vehicle.max_accel
    assert env.denormalize_action(np.array([-1.0, 0.0]))[0] == -cfg.vehicle.max_decel
    assert env.denormalize_action(np.array([0.0, 1.0]))[1] == cfg.vehicle.max_steer


def test_driving_straight_at_a_goal_ahead_succeeds(cfg):
    """Sanity that the env is solvable: full throttle across an empty scene reaches it."""
    env = DriveNavEnv(FixedScene(empty_scene(goal=(30.0, 0.0))), cfg)
    result = rollout(env, lambda obs: np.array([1.0, 0.0]))
    assert result["reached"] and not result["collided"]


def test_driving_into_an_obstacle_is_a_collision(cfg):
    env = DriveNavEnv(FixedScene(empty_scene(circles=[[15.0, 0.0, 3.0]])), cfg)
    result = rollout(env, lambda obs: np.array([1.0, 0.0]))
    assert result["collided"] and not result["reached"]


def test_reward_is_positive_for_progress_and_negative_for_a_collision(cfg):
    env = DriveNavEnv(FixedScene(empty_scene(goal=(40.0, 0.0))), cfg)
    env.reset(0)
    _, reward, _, _, _ = env.step(np.array([1.0, 0.0]))
    assert reward > 0                                     # closed distance

    crash = DriveNavEnv(FixedScene(empty_scene(circles=[[15.0, 0.0, 3.0]])), cfg)
    total = rollout(crash, lambda obs: np.array([1.0, 0.0]))["reward"]
    assert total < 0


def test_episodes_truncate_at_the_step_limit(cfg):
    """Sitting still must end the episode by truncation, not run forever."""
    short = replace(cfg, max_steps=20)
    env = DriveNavEnv(FixedScene(empty_scene(start_speed=0.0, goal=(40.0, 0.0))), short)
    env.reset(0)
    for _ in range(short.max_steps):
        _, _, terminated, truncated, info = env.step(np.array([-1.0, 0.0]))
    assert truncated and not terminated
    assert info["step"] == short.max_steps


def test_reset_is_reproducible_for_a_seed(cfg):
    a = DriveNavEnv(SyntheticScenes(), cfg).reset(7)[0]
    b = DriveNavEnv(SyntheticScenes(), cfg).reset(7)[0]
    np.testing.assert_array_equal(a, b)


# --- scene generation -------------------------------------------------------------------------

def test_synthetic_starts_are_always_certifiable():
    """Every episode must begin somewhere the shield can still guarantee a stop.

    Unclamped, 36% of starts were already inevitable-collision states, which surfaced as
    shield-on collisions and is easy to misread as the shield failing.
    """
    scenes, vehicle = SyntheticScenes(), VehicleConfig()
    rng = np.random.default_rng(0)
    for _ in range(40):
        scene = scenes.sample(rng)
        assert can_stop_safely(scene.start, scene.grid, vehicle)


def test_certifiable_start_only_ever_lowers_the_speed():
    bev = BEVConfig()
    grid = BEVGrid(rasterize_circles(np.array([[12.0, 0.0, 2.0]]), bev), bev)
    fast = VehicleState(v=14.0)
    clamped = certifiable_start(fast, grid, VehicleConfig())
    assert clamped.v < fast.v
    assert (clamped.x, clamped.y, clamped.yaw) == (fast.x, fast.y, fast.yaw)


def test_certifiable_start_leaves_a_safe_speed_alone():
    bev = BEVConfig()
    slow = VehicleState(v=2.0)
    unchanged = certifiable_start(slow, BEVGrid(np.zeros(bev.shape, np.uint8), bev),
                                  VehicleConfig())
    assert unchanged.v == pytest.approx(2.0)


def test_synthetic_scenes_keep_the_goal_reachable():
    scenes = SyntheticScenes()
    rng = np.random.default_rng(1)
    for _ in range(20):
        scene = scenes.sample(rng)
        assert scene.grid.distance_to_obstacles(scene.goal[None, :])[0] > 0.0


def test_kitti_frame_split_is_contiguous_disjoint_and_holds_out_the_tail():
    """Train/test split must not leak: held-out frames are the drive's last stretch, unseen."""
    train, test = kitti_frame_split(100, holdout=0.3, stride=1)
    assert set(train).isdisjoint(set(test)), "a frame trained on must not be scored on"
    assert test.min() > train.max(), "held-out frames should be a later stretch, not interleaved"
    assert list(test) == list(range(70, 100)), "the last 30% is held out"
    assert list(train) == list(range(70))


def test_kitti_frame_split_stride_thins_only_the_train_pool():
    train, test = kitti_frame_split(100, holdout=0.3, stride=3)
    assert list(train) == list(range(0, 70, 3)), "stride subsamples train"
    assert list(test) == list(range(70, 100)), "test stays dense"


# --- the shield in the loop ---------------------------------------------------------------------

def test_shield_eliminates_collisions_from_certifiable_starts(cfg):
    """The headline guarantee, end to end through the env: zero collisions, no exceptions.

    Deliberately run against the *weak* gap-following baseline, which crashes constantly on
    its own — the point is that the shield's guarantee does not depend on the policy being
    any good.
    """
    shielded = replace(cfg, use_shield=True)
    env = DriveNavEnv(SyntheticScenes(), shielded)
    stats = evaluate(env, lambda o: gap_following_policy(o, shielded), n_episodes=60)
    assert stats["collisions"] == 0, "shield admitted a collision"


def test_unshielded_control_run_does_collide(cfg):
    """Control condition: without the shield these same scenes really are dangerous."""
    env = DriveNavEnv(SyntheticScenes(), cfg)
    stats = evaluate(env, lambda o: gap_following_policy(o, cfg), n_episodes=60)
    assert stats["collisions"] > 0


def test_shield_trades_success_for_safety(cfg):
    """An honest characterisation, not a win: the shield stops the car more often.

    It converts crashes into safe stops, so collisions vanish but some episodes end without
    reaching the goal. Recovering that success while keeping zero collisions is exactly the
    job of the learned policy.
    """
    shielded = replace(cfg, use_shield=True)
    with_shield = evaluate(DriveNavEnv(SyntheticScenes(), shielded),
                           lambda o: gap_following_policy(o, shielded), n_episodes=60)
    assert with_shield["collisions"] == 0
    assert with_shield["success_rate"] < 0.9      # room left for a learned policy


def test_random_actions_never_collide_behind_the_shield(cfg):
    """The strongest form of the claim: even a policy with no idea what it is doing."""
    shielded = replace(cfg, use_shield=True)
    env = DriveNavEnv(SyntheticScenes(), shielded)
    rng = np.random.default_rng(0)
    for i in range(15):
        result = rollout(env, lambda obs: rng.uniform(-1, 1, 2), seed=i)
        assert not result["collided"]


# --- the cautious-unknown speed governor -------------------------------------------------------

def _unknown_forward_of(x: float, bev: BEVConfig) -> np.ndarray:
    """An unknown half-plane: every cell at or beyond `x` metres ahead is unobserved."""
    unknown = np.zeros(bev.shape, bool)
    unknown[int((x - bev.x_min) / bev.resolution):, :] = True
    return unknown


def test_speed_cap_keeps_the_bumper_short_of_an_unknown_frontier():
    """The governor's guarantee: the car never enters unmapped space faster than it could halt.

    Driven at full throttle with the collision shield off (so the cap is the *only* thing
    limiting speed), the front bumper's braking-stop point must never cross the unknown
    frontier at x = 15 m — through the discretisation, not merely in the continuous limit (see
    the one-step reservation in `cautious_speed_cap`). The car then stalls short of it rather
    than driving on blind.
    """
    bev, vcfg = BEVConfig(), VehicleConfig()
    grid = BEVGrid(np.zeros(bev.shape, np.uint8), bev,
                   unknown=_unknown_forward_of(15.0, bev), unknown_speed_cap=True)
    env = DriveNavEnv(FixedScene(Scene(grid, VehicleState(v=8.0), np.array([30.0, 0.0]))),
                      DriveNavConfig())
    env.reset(0)
    max_x = 0.0
    for _ in range(150):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0]))
        s = info["state"]
        assert s.x + vcfg.front_overhang + stopping_distance(s.v, vcfg) <= 15.0
        max_x = max(max_x, s.x)
        if term or trunc:
            break
    # Rear axle halts ~ front_overhang + margin (≈ 4.1 m) short of the frontier at x = 15 m.
    assert 9.0 < max_x < 12.0


def test_speed_cap_is_inert_without_an_unknown_mask():
    """Reproducibility: the flag is a no-op off a carved map, so no existing number moves."""
    bev = BEVConfig()

    def run(grid):
        env = DriveNavEnv(
            FixedScene(Scene(grid, VehicleState(v=5.0), np.array([30.0, 0.0]))),
            DriveNavConfig())
        env.reset(0)
        xs = []
        for _ in range(40):
            _, _, term, trunc, info = env.step(np.array([1.0, 0.0]))
            xs.append(info["state"].x)
            if term or trunc:
                break
        return xs

    capped = run(BEVGrid(np.zeros(bev.shape, np.uint8), bev, unknown_speed_cap=True))
    plain = run(BEVGrid(np.zeros(bev.shape, np.uint8), bev))
    assert capped == plain


def test_speed_cap_leaves_the_shield_sound():
    """The governor only ever lowers acceleration, so it cannot break the braking certificate.

    Full throttle straight at a real occupied wall, shield on and the cap on as well: a slower
    car on the same braking path still stops clear, so the collision count stays zero.
    """
    bev, vcfg = BEVConfig(), VehicleConfig()
    occ = rasterize_circles(np.array([[18.0, 0.0, 3.0]]), bev)   # a genuine obstacle
    grid = BEVGrid(occ, bev, unknown=_unknown_forward_of(30.0, bev), unknown_speed_cap=True)
    shielded = replace(DriveNavConfig(), use_shield=True)
    start = certifiable_start(VehicleState(v=8.0), grid, vcfg)
    env = DriveNavEnv(FixedScene(Scene(grid, start, np.array([40.0, 0.0]))), shielded)
    env.reset(0)
    for _ in range(80):
        _, _, term, trunc, info = env.step(np.array([1.0, 0.0]))
        assert not info["collided"], "the cap must not undo the shield"
        if term or trunc:
            break


def test_cap_does_not_turn_unknown_into_collision_geometry():
    """Why the cap is drivable where hard `unknown_blocks` is not.

    Hard-blocking makes an unknown cell an obstacle, so unknown hugging a tight corridor puts
    the car's own footprint in collision and walls it in. The cap leaves collision a function
    of occupancy alone — the car may sit in and cross unknown space — which is the whole point.
    """
    bev, vcfg = BEVConfig(), VehicleConfig()
    occ = np.zeros(bev.shape, np.uint8)
    unknown = np.ones(bev.shape, bool)
    c_lo = int((-1.0 - bev.y_min) / bev.resolution)
    c_hi = int((1.0 - bev.y_min) / bev.resolution)
    unknown[:, c_lo:c_hi] = False                       # a free corridor tighter than the car
    state = VehicleState(x=5.0, y=0.0, yaw=0.0, v=3.0)

    blocking = BEVGrid(occ, bev, unknown=unknown, unknown_blocks=True)
    capped = BEVGrid(occ, bev, unknown=unknown, unknown_speed_cap=True)
    assert clearance(state, blocking, vcfg) < 0.0       # footprint overlaps the unknown walls
    assert np.isinf(clearance(state, capped, vcfg))     # occupancy is empty: no collision


def test_cautious_speed_cap_is_max_speed_where_the_way_ahead_is_confidently_clear():
    """A carved map whose forward corridor is all observed-free imposes no cap at all."""
    bev, vcfg = BEVConfig(), VehicleConfig()
    unknown = np.zeros(bev.shape, bool)                 # nothing unobserved ahead
    grid = BEVGrid(np.zeros(bev.shape, np.uint8), bev, unknown=unknown, unknown_speed_cap=True)
    assert cautious_speed_cap(VehicleState(v=5.0), grid, vcfg) == vcfg.max_speed


# --- closed-loop dynamic environment -----------------------------------------------------------

def crossing_scene(mover_vy=4.0, ego_speed=9.0, background=None):
    """An empty (or given) background with one car sweeping +y across the lane 20 m ahead.

    Timed like the `test_dynamics` crossing test: the car starts off to the right and crosses
    into the ego's straight-ahead path just as the ego arrives, so a shield that sees only its
    current (off-path) position brakes too late.
    """
    bev = BEVConfig()
    occ = np.zeros(bev.shape, np.uint8) if background is None else background
    mover = MovingObstacle(np.array([20.0, -9.0, np.pi / 2, 4.0, 2.0]),
                           np.array([0.0, mover_vy]))
    start = VehicleState(x=0.0, y=0.0, yaw=0.0, v=ego_speed)
    return DynamicScene(BEVGrid(occ, bev), [mover], start, np.array([32.0, 0.0]))


def _drive_straight(env, throttle=1.0):
    """Full-throttle straight run; returns the final info dict."""
    env.reset(0)
    info = {}
    for _ in range(env.cfg.max_steps):
        _, _, terminated, truncated, info = env.step(np.array([throttle, 0.0]))
        if terminated or truncated:
            break
    return info


def test_static_shield_drives_into_a_crossing_mover_the_dynamic_shield_stops_for():
    """The closed-loop headline: the moving-world analogue of the "0 collisions" result.

    Same scene, same policy (drive straight), differing only in which shield filters the
    action. The static shield sees the crossing car only once it is already in the lane —
    too late — and drives into it; the dynamic shield brakes for where the car *will be* and
    lets it pass. This is `test_dynamics`'s open-loop crossing demonstration closed into a
    full episode with the environment stepping the obstacle.
    """
    src = FixedScene(crossing_scene())
    static = _drive_straight(DynamicNavEnv(src, DynamicNavConfig(use_shield=True)))
    dynamic = _drive_straight(
        DynamicNavEnv(src, DynamicNavConfig(use_shield=True, dynamic_shield=True)))
    assert static["hit_mover"], "the static shield should have driven into the crossing car"
    assert not dynamic["collided"], "the dynamic shield should have braked for it"
    assert dynamic["reached"], "and then reached the goal once the car had passed"


def test_a_dynamic_shield_collision_is_always_an_inevitable_collision_state():
    """Soundness in the moving world: the dynamic shield never *drives* into a mover.

    A car sweeping straight into the ego from close range is unavoidable — but the shield must
    brake and flag it, not silently admit it. Any collision it does suffer is a mover reaching
    a stopping envelope it could not escape, so `ics` is set: the analogue of the static
    shield's guarantee that its only collisions are inevitable-collision starts.
    """
    bev = BEVConfig()
    mover = MovingObstacle(np.array([8.0, -3.0, np.pi / 2, 4.0, 2.0]), np.array([0.0, 6.0]))
    scene = DynamicScene(BEVGrid(np.zeros(bev.shape, np.uint8), bev), [mover],
                         VehicleState(x=0.0, y=0.0, yaw=0.0, v=10.0), np.array([32.0, 0.0]))
    env = DynamicNavEnv(FixedScene(scene), DynamicNavConfig(use_shield=True, dynamic_shield=True))
    info = _drive_straight(env)
    if info["collided"]:
        assert env.last_shield.ics, "a dynamic-shield collision must be an ICS, never a silent hit"


def test_dynamic_env_advances_the_mover_each_step():
    """Mechanics: the obstacle really moves, so the grid the policy reads changes with time."""
    env = DynamicNavEnv(FixedScene(crossing_scene()),
                        DynamicNavConfig(use_shield=False))
    env.reset(0)
    before = int(env.grid.occupancy.sum())
    cells_before = env.grid.occupancy.copy()
    for _ in range(5):
        env.step(np.array([0.0, 0.0]))
    # The footprint is still on the grid (same cell count, box just translated) but in new cells.
    assert int(env.grid.occupancy.sum()) == pytest.approx(before, abs=6)
    assert not np.array_equal(env.grid.occupancy, cells_before), "the mover did not move"


def test_dynamic_env_with_no_movers_matches_a_static_drive():
    """With an empty mover list the dynamic env is just a static drive across `static_grid`."""
    bev = BEVConfig()
    scene = DynamicScene(BEVGrid(np.zeros(bev.shape, np.uint8), bev), [],
                         VehicleState(x=0.0, y=0.0, yaw=0.0, v=5.0), np.array([30.0, 0.0]))
    env = DynamicNavEnv(FixedScene(scene), DynamicNavConfig(use_shield=False))
    info = _drive_straight(env)
    assert info["reached"] and not info["collided"]
    assert np.array_equal(env.grid.occupancy, scene.static_grid.occupancy)
    assert info["hit_mover"] is False


# --- Gymnasium wrapper -------------------------------------------------------------------------

def test_gym_wrapper_matches_the_core_env_contract(cfg):
    """The wrapper must be a view, not a reimplementation — spaces derived from the core."""
    gym = pytest.importorskip("gymnasium")
    from kitti_nav.nav_gym import NavGymEnv

    env = NavGymEnv(SyntheticScenes(), cfg)
    assert env.observation_space.shape == (cfg.obs_dim,)
    assert env.action_space.shape == (2,)

    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert "state" not in info, "unslimmed info would drag a whole BEVGrid through vec envs"
    assert gym.Env in type(env).__mro__
