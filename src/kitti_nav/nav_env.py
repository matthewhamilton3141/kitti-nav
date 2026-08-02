"""Driving navigation environment: a bicycle vehicle crossing a BEV occupancy scene.

Pure NumPy and duck-typed to the Gymnasium API (`reset(seed) -> (obs, info)`,
`step(action) -> (obs, reward, terminated, truncated, info)`), so it runs and is fully
unit-tested with no RL dependency at all; `nav_gym.py` adds the thin Gymnasium wrapper that
stable-baselines3 needs. That split is inherited from `gsplat-rt`, where keeping the env
core dependency-free is what let the same environment run on a laptop and a GPU box.

**Scenes are pluggable, and that is the point.** `SyntheticScenes` generates random obstacle
fields for training; `KittiScenes` serves *real occupancy grids built from recorded Velodyne
scans*. Training on synthetic scenes and evaluating on real ones is the transfer test — the
driving analogue of the kinematic-to-PyBullet transfer that de-risked `gsplat-rt`'s policy.

**Observation** is a forward ray fan plus the goal in the ego frame, speed, and steering
angle. Rays rather than an occupancy crop because `gsplat-rt` measured the two and found the
crop marginal at this obstacle density — the cheaper observation was not worse.

**The shield is optional and orthogonal.** With `use_shield`, every action is filtered before
it reaches the dynamics, so a policy can be *trained through* the shield. That was the
capstone result there: the shielded-in-the-loop policy dominated the raw one on every axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

import numpy as np

from .bev import BEVConfig, BEVGrid
from .vehicle import (
    ShieldResult,
    VehicleConfig,
    VehicleState,
    clearance,
    max_safe_speed,
    safety_shield,
    step_state,
)


def certifiable_start(state: VehicleState, grid: BEVGrid,
                      vehicle: VehicleConfig) -> VehicleState:
    """Clamp a start speed to one the car could actually stop from in this scene.

    Without this, episodes routinely begin in an inevitable-collision state — spawned fast
    with an obstacle already inside the braking envelope — and *no* shield can rescue those,
    because the guarantee is inductive from a safe state. Measured on the synthetic scenes,
    36% of unclamped starts were already doomed, which showed up as shield-on collisions and
    would have been easy to misread as the shield failing.

    A real car is never teleported to a speed it cannot stop from, so clamping is the
    physically honest fix rather than a convenience.
    """
    ceiling = max_safe_speed(grid, vehicle, state=state)
    return VehicleState(state.x, state.y, state.yaw, min(state.v, ceiling), state.steer)


@dataclass(frozen=True)
class Scene:
    """One episode's world: static occupancy, where the car starts, where it must reach."""

    grid: BEVGrid
    start: VehicleState
    goal: np.ndarray          # (2,) metres, in the grid frame


class SceneSource(Protocol):
    """Supplies a fresh `Scene` per episode."""

    def sample(self, rng: np.random.Generator) -> Scene: ...


@dataclass
class DriveNavConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    # --- observation ---
    n_rays: int = 24
    ray_fov: float = np.pi          # forward half-plane, centred on the heading
    ray_range: float = 30.0

    # --- episode ---
    goal_radius: float = 4.0        # reaching within this counts as success
    max_steps: int = 150

    # --- reward ---
    progress_weight: float = 1.0    # per metre of progress toward the goal
    step_penalty: float = 0.02      # mild time pressure, so dawdling is not free
    collision_penalty: float = 20.0
    goal_bonus: float = 20.0

    # --- safety ---
    use_shield: bool = False        # filter every action through the braking shield

    @property
    def obs_dim(self) -> int:
        return 5 + self.n_rays

    @property
    def ray_angles(self) -> np.ndarray:
        if self.n_rays == 1:
            return np.zeros(1)
        return np.linspace(-self.ray_fov / 2, self.ray_fov / 2, self.n_rays)


class DriveNavEnv:
    """A bicycle vehicle driving to a goal across a static BEV occupancy scene."""

    def __init__(self, scenes: SceneSource, cfg: Optional[DriveNavConfig] = None):
        self.cfg = cfg or DriveNavConfig()
        self.scenes = scenes
        self.obs_dim = self.cfg.obs_dim
        self.act_dim = 2

        # Actions are normalised to [-1, 1] on both channels, which is what policy networks
        # expect; the env owns the mapping to physical accel (m/s^2) and steer (rad).
        self.action_low = -np.ones(2, np.float32)
        self.action_high = np.ones(2, np.float32)

        self._rng = np.random.default_rng()
        self._scene: Optional[Scene] = None
        self._state = VehicleState()
        self._prev_dist = 0.0
        self._step = 0
        self.last_shield: Optional[ShieldResult] = None

    # -- helpers -------------------------------------------------------------------------

    @property
    def state(self) -> VehicleState:
        return self._state

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise RuntimeError("call reset() before using the environment")
        return self._scene

    def denormalize_action(self, action: np.ndarray) -> tuple[float, float]:
        """Map a policy's `[-1, 1]^2` output to (accel m/s^2, steer rad).

        Acceleration is asymmetric — cars brake harder than they accelerate — so the
        negative half of the channel maps to `max_decel` and the positive half to
        `max_accel`, rather than a single symmetric scale that would under-use the brakes.
        """
        v = self.cfg.vehicle
        a = float(np.clip(action[0], -1.0, 1.0))
        accel = a * (v.max_accel if a >= 0 else v.max_decel)
        steer = float(np.clip(action[1], -1.0, 1.0)) * v.max_steer
        return accel, steer

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self._state.xy - self.scene.goal))

    def _observation(self) -> np.ndarray:
        cfg, s = self.cfg, self._state
        rays = self.scene.grid.ray_distances(s.xy, s.yaw, cfg.ray_angles, cfg.ray_range)

        # Goal in the ego frame: what the policy can act on without knowing world coords.
        d = self.scene.goal - s.xy
        c, sn = np.cos(-s.yaw), np.sin(-s.yaw)
        fwd, left = c * d[0] - sn * d[1], sn * d[0] + c * d[1]
        dist = float(np.linalg.norm(d))

        return np.concatenate([
            [fwd / cfg.ray_range, left / cfg.ray_range, dist / cfg.ray_range,
             s.v / cfg.vehicle.max_speed, s.steer / cfg.vehicle.max_steer],
            rays / cfg.ray_range,
        ]).astype(np.float32)

    def _info(self, collided: bool, reached: bool) -> dict:
        return {
            "state": self._state,
            "distance": self._distance_to_goal(),
            "collided": collided,
            "reached": reached,
            "step": self._step,
            "clearance": clearance(self._state, self.scene.grid, self.cfg.vehicle),
            "shield_intervened": bool(self.last_shield.intervened) if self.last_shield
            else False,
        }

    # -- Gymnasium-style API -------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._scene = self.scenes.sample(self._rng)
        self._state = self._scene.start
        self._prev_dist = self._distance_to_goal()
        self._step = 0
        self.last_shield = None
        return self._observation(), self._info(collided=False, reached=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg
        accel, steer = self.denormalize_action(np.asarray(action, float).reshape(-1))

        if cfg.use_shield:
            self.last_shield = safety_shield(accel, steer, self._state,
                                             self.scene.grid, cfg.vehicle)
            accel, steer = self.last_shield.accel, self.last_shield.steer

        self._state = step_state(self._state, accel, steer, cfg.vehicle)
        self._step += 1

        dist = self._distance_to_goal()
        collided = clearance(self._state, self.scene.grid, cfg.vehicle) < 0.0
        reached = dist <= cfg.goal_radius

        reward = cfg.progress_weight * (self._prev_dist - dist) - cfg.step_penalty
        if collided:
            reward -= cfg.collision_penalty
        if reached:
            reward += cfg.goal_bonus
        self._prev_dist = dist

        terminated = bool(collided or reached)
        truncated = bool(self._step >= cfg.max_steps and not terminated)
        return (self._observation(), float(reward), terminated, truncated,
                self._info(collided, reached))


# --- scene sources --------------------------------------------------------------------------

def rasterize_circles(circles: np.ndarray, cfg: BEVConfig) -> np.ndarray:
    """Paint circular obstacles into an occupancy grid, for synthetic scenes."""
    rows, cols = cfg.shape
    grid = np.zeros((rows, cols), np.uint8)
    ri = np.arange(rows) * cfg.resolution + cfg.x_min + cfg.resolution / 2
    ci = np.arange(cols) * cfg.resolution + cfg.y_min + cfg.resolution / 2
    gx, gy = np.meshgrid(ri, ci, indexing="ij")
    for cx, cy, r in np.asarray(circles, float).reshape(-1, 3):
        grid |= ((gx - cx) ** 2 + (gy - cy) ** 2 <= r * r).astype(np.uint8)
    return grid


@dataclass
class SyntheticScenes:
    """Random circular-obstacle fields between the car and a goal straight ahead.

    Obstacles are rejection-sampled to leave the start and goal clear, so every episode is
    solvable; the corridor between them is not guaranteed clear, which is the point.
    """

    bev: BEVConfig = field(default_factory=BEVConfig)
    n_obstacles: tuple[int, int] = (4, 12)
    radius: tuple[float, float] = (0.6, 2.5)
    goal_distance: tuple[float, float] = (25.0, 40.0)
    goal_lateral: float = 6.0
    start_speed: tuple[float, float] = (3.0, 10.0)
    keep_clear: float = 6.0          # radius kept free around start and goal
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    def sample(self, rng: np.random.Generator) -> Scene:
        start = VehicleState(x=0.0, y=0.0, yaw=0.0,
                             v=float(rng.uniform(*self.start_speed)), steer=0.0)
        goal = np.array([float(rng.uniform(*self.goal_distance)),
                         float(rng.uniform(-self.goal_lateral, self.goal_lateral))])

        n = int(rng.integers(*self.n_obstacles))
        placed: list[tuple[float, float, float]] = []
        for _ in range(n * 40):
            if len(placed) >= n:
                break
            r = float(rng.uniform(*self.radius))
            cx = float(rng.uniform(4.0, goal[0] + 5.0))
            cy = float(rng.uniform(self.bev.y_min / 2, self.bev.y_max / 2))
            if np.hypot(cx - start.x, cy - start.y) < self.keep_clear + r:
                continue
            if np.hypot(cx - goal[0], cy - goal[1]) < self.keep_clear + r:
                continue
            placed.append((cx, cy, r))

        occ = rasterize_circles(np.array(placed, float).reshape(-1, 3), self.bev)
        grid = BEVGrid(occ, self.bev)
        return Scene(grid, certifiable_start(start, grid, self.vehicle), goal)


@dataclass
class KittiScenes:
    """Real recorded street geometry: BEV grids built from a drive's Velodyne scans.

    The scan is frozen as a static obstacle field and the car is asked to cross it. The
    geometry is genuinely real — parked cars, kerbs, buildings — which is what makes this a
    transfer test rather than more of the same synthetic distribution.
    """

    drive: object                              # KittiDrive; untyped to avoid a hard import
    bev: BEVConfig = field(default_factory=BEVConfig)
    goal_distance: tuple[float, float] = (20.0, 35.0)
    goal_lateral: float = 3.0
    start_speed: tuple[float, float] = (3.0, 10.0)
    frames: Optional[np.ndarray] = None        # restrict to these frames (e.g. a held-out split)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    def sample(self, rng: np.random.Generator) -> Scene:
        pool = self.frames if self.frames is not None else np.arange(self.drive.n_velodyne)
        i = int(rng.choice(pool))
        grid = BEVGrid.from_scan(self.drive.velodyne(i), self.bev)

        start = self.drive.vehicle_state_in_lidar(speed=float(rng.uniform(*self.start_speed)))
        goal = np.array([start.x + float(rng.uniform(*self.goal_distance)),
                         start.y + float(rng.uniform(-self.goal_lateral, self.goal_lateral))])
        return Scene(grid, certifiable_start(start, grid, self.vehicle), goal)


# --- baseline policy ------------------------------------------------------------------------

def gap_following_policy(obs: np.ndarray, cfg: Optional[DriveNavConfig] = None) -> np.ndarray:
    """Observation-only baseline: steer into the freest ray pointing roughly goalward.

    A follow-the-gap / vector-field-histogram controller, the driving counterpart of
    `gsplat-rt`'s `avoidance_action`. It exists to prove the environment is solvable without
    learning and to give the learned policy something to beat. Uses no privileged state.
    """
    cfg = cfg or DriveNavConfig()
    n = cfg.n_rays
    goal_fwd, goal_left = float(obs[0]), float(obs[1])
    speed = float(obs[3]) * cfg.vehicle.max_speed
    rays = np.asarray(obs[5:5 + n], float) * cfg.ray_range

    angles = cfg.ray_angles
    goal_bearing = float(np.arctan2(goal_left, goal_fwd))
    target = float(np.clip(goal_bearing, angles[0], angles[-1]))

    # Score each ray by how open it is, penalised by how far it points from the goal.
    room = np.clip(rays / cfg.ray_range, 0.0, 1.0)
    penalty = np.abs(angles - target) / (cfg.ray_fov + 1e-9)
    steer = float(angles[int(np.argmax(room - 1.5 * penalty))])

    # Is the straight-ahead corridor clear? A ray at angle t and distance d sits d*sin(t) to
    # the side, so it only threatens the car's path if that offset is within its half-width.
    half_width = cfg.vehicle.width / 2 + 0.3
    ahead = np.cos(angles) > 0
    in_path = ahead & (rays * np.abs(np.sin(angles)) < half_width)
    corridor = float((rays * np.cos(angles))[in_path].min()) if in_path.any() else cfg.ray_range

    # Target the speed we could still stop from within the visible corridor.
    room_to_stop = max(corridor - cfg.vehicle.front_overhang - cfg.vehicle.safety_margin, 0.0)
    v_target = min(np.sqrt(2 * cfg.vehicle.max_decel * room_to_stop), cfg.vehicle.max_speed)
    v_target *= max(np.cos(steer), 0.2)          # slow down for hard turns

    accel = np.clip((v_target - speed) / (cfg.vehicle.max_accel * cfg.vehicle.dt), -1.0, 1.0)
    return np.array([accel, steer / cfg.vehicle.max_steer], np.float32)


def rollout(env: DriveNavEnv, policy: Callable[[np.ndarray], np.ndarray],
            seed: Optional[int] = None) -> dict:
    """Run one episode and return its outcome; the unit of every evaluation here."""
    obs, _ = env.reset(seed)
    total = 0.0
    for _ in range(env.cfg.max_steps):
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        if terminated or truncated:
            break
    return {
        "reward": total,
        "reached": bool(info["reached"]),
        "collided": bool(info["collided"]),
        "steps": int(info["step"]),
        "distance": float(info["distance"]),
    }


def evaluate(env: DriveNavEnv, policy: Callable[[np.ndarray], np.ndarray],
             n_episodes: int = 100, seed: int = 0) -> dict:
    """Aggregate rollouts into the success / collision / steps summary used throughout."""
    results = [rollout(env, policy, seed=seed + i) for i in range(n_episodes)]
    reached = [r for r in results if r["reached"]]
    return {
        "episodes": n_episodes,
        "success_rate": len(reached) / n_episodes,
        "collisions": sum(r["collided"] for r in results),
        "mean_reward": float(np.mean([r["reward"] for r in results])),
        "mean_steps_when_reached": float(np.mean([r["steps"] for r in reached]))
        if reached else float("nan"),
    }
