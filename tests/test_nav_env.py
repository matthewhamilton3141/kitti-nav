"""Tests for the driving navigation environment and its scene sources.

The headline test is `test_shield_eliminates_collisions_from_certifiable_starts`: the shield
guarantee, end to end, through the full env loop rather than in isolation.
"""

import numpy as np
import pytest
from dataclasses import replace

from kitti_nav.bev import BEVConfig, BEVGrid
from kitti_nav.nav_env import (
    DriveNavConfig,
    DriveNavEnv,
    Scene,
    SyntheticScenes,
    certifiable_start,
    evaluate,
    gap_following_policy,
    rasterize_circles,
    rollout,
)
from kitti_nav.vehicle import VehicleConfig, VehicleState, can_stop_safely


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
