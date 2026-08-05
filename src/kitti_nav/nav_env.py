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

from .bev import BEVConfig, BEVGrid, occupancy_from_scan, rasterize_box
from .dynamics import MovingObstacle, dynamic_safety_shield, point_box_distance
from .mapping import (
    KITTI_EGO_BOX,
    MapConfig,
    drop_ego_returns,
    fuse_map,
    fuse_scans,
    window_indices,
)
from .vehicle import (
    ShieldResult,
    VehicleConfig,
    VehicleState,
    clearance,
    footprint_discs,
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


def cautious_speed_cap(state: VehicleState, grid: BEVGrid, vehicle: VehicleConfig,
                       half_cone: float = 0.15, n_rays: int = 3) -> float:
    """Speed (m/s) the car may run given how far confidently-free space extends ahead.

    The drivable reading of an accumulated map's unknown holes. Hard `unknown_blocks` walls
    the car out of every unobserved cell and is unnavigable (a single drive is >50% unknown);
    this instead lets the car *traverse* unknown space but never enter it faster than it could
    brake to rest at its threshold — `v_cap = sqrt(2 * max_decel * d)`, the inverse of
    `stopping_distance`. `d` is the distance ahead to the nearest occupied-or-unknown cell,
    measured over a narrow forward cone (the braking path is roughly straight; a wide cone
    would see the unknown *beside* the corridor and cap to a standstill), shrunk by the front
    overhang and safety margin so the bumper — not the rear axle — is what stops short.

    This is a governor layered on top of the collision shield, never a replacement: it only
    ever lowers the commanded acceleration, so a state the shield certified stays certified (a
    slower car on the same braking path still stops clear). Returns `max_speed` when the map
    carries no `unknown` mask, i.e. it is a no-op off a carved map.

    One control step of travel (`v * dt`) is reserved in the effective distance: the cap is
    read at the step's start but the car moves a full step before it can re-brake, so without
    the reservation it would lag the shrinking frontier by a step and nose past it. With it,
    the bumper stays short of the frontier through the discretisation, not merely in the limit.
    """
    if grid.unknown is None:
        return vehicle.max_speed
    angles = np.linspace(-half_cone, half_cone, n_rays) if n_rays > 1 else np.zeros(1)
    d = float(grid.confident_clear_distance(state.xy, state.yaw, angles).min())
    reach = max(state.v, 0.0) * vehicle.dt
    d_eff = max(0.0, d - vehicle.front_overhang - vehicle.safety_margin - reach)
    return float(min(np.sqrt(2.0 * vehicle.max_decel * d_eff), vehicle.max_speed))


@dataclass(frozen=True)
class Scene:
    """One episode's world: static occupancy, where the car starts, where it must reach."""

    grid: BEVGrid
    start: VehicleState
    goal: np.ndarray          # (2,) metres, in the grid frame


@dataclass(frozen=True)
class DynamicScene:
    """A moving-world episode: static background + constant-velocity movers + start + goal.

    `static_grid` is the frozen geometry with the movers **lifted out** — every obstacle that
    moves is represented instead by a `dynamics.MovingObstacle` in `movers`, so the env can put
    each one where it is *this* step. Everything is in one common frame (the velodyne frame for
    a KITTI scene), and the movers travel at constant velocity for the whole episode — which is
    exactly the model the dynamic shield reasons against, so a scene here is honest to what the
    shield claims rather than to a replayed recording.
    """

    static_grid: BEVGrid
    movers: list              # list[dynamics.MovingObstacle], velo/world frame
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

    @property
    def grid(self) -> BEVGrid:
        """The occupancy the observation and collision check read this step.

        For a static scene it is just `scene.grid`; `DynamicNavEnv` overrides it with a grid
        rebuilt each step from the movers' current positions, so the shared observation and
        info code sees the right obstacles without knowing the world is moving.
        """
        return self.scene.grid

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
        rays = self.grid.ray_distances(s.xy, s.yaw, cfg.ray_angles, cfg.ray_range)

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
            "clearance": clearance(self._state, self.grid, self.cfg.vehicle),
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

        # Cautious-unknown governor (see `cautious_speed_cap`): on a carved map that requests
        # it, cap speed by the frontier distance. It only ever *lowers* accel, so it cannot
        # undo the shield's braking certificate — a slower car on the same path still stops.
        if getattr(self.scene.grid, "unknown_speed_cap", False):
            v_cap = cautious_speed_cap(self._state, self.scene.grid, cfg.vehicle)
            accel = min(accel, (v_cap - self._state.v) / cfg.vehicle.dt)

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


@dataclass
class DynamicNavConfig(DriveNavConfig):
    """`DriveNavConfig` plus the choice of which shield reasons about the moving obstacles."""

    # With `use_shield`, pick the dynamic shield (brakes for where a mover is *going*) over the
    # ordinary static one (which sees only where it is now). This flag is the whole experiment:
    # the same policy, scenes, and observations, differing only in which shield filters actions.
    dynamic_shield: bool = False


class DynamicNavEnv(DriveNavEnv):
    """A bicycle vehicle crossing a scene whose obstacles move at constant velocity.

    The closed-loop counterpart to `DriveNavEnv`, and the moving-world analogue of the static
    shield's "0 collisions" result. Each `step` advances every `MovingObstacle` to `box_at(t)`
    and rebuilds the obstacle field, so the collision check, the observation rays, and the
    shield all see the movers where they are this step. With `cfg.dynamic_shield` the actions
    are filtered by `dynamics.dynamic_safety_shield`; otherwise by the ordinary `safety_shield`
    against the movers frozen at their current position — the control condition that shows what
    treating a moving world as static costs.

    The collision check and both shields see one obstacle set — static grid ∪ current mover
    boxes — so the comparison is clean: only the shield's *model of the future* differs.
    """

    def __init__(self, scenes: SceneSource, cfg: Optional[DynamicNavConfig] = None):
        super().__init__(scenes, cfg or DynamicNavConfig())
        self._sim_grid: Optional[BEVGrid] = None

    @property
    def grid(self) -> BEVGrid:
        return self._sim_grid

    def _movers_at(self, t: float) -> list:
        """Each scene mover as a `MovingObstacle` whose footprint sits at time `t`."""
        return [MovingObstacle(m.box_at(t), m.velocity) for m in self.scene.movers]

    def _combined_grid(self, t: float) -> BEVGrid:
        """Static background ∪ every mover's footprint at time `t`, as one occupancy grid.

        This is what the collision check, the rays, and the *static* shield read; the dynamic
        shield instead takes the static grid plus the moving obstacles directly, so it can
        advance them through its braking rollout.
        """
        static: BEVGrid = self.scene.static_grid
        occ = static.occupancy.astype(bool).copy()
        for m in self.scene.movers:
            occ |= rasterize_box(m.box_at(t), static.cfg)
        return BEVGrid(occ.astype(np.uint8), static.cfg)

    def _hit_mover(self, t: float) -> bool:
        """Is the footprint overlapping a mover (not just static geometry) at time `t`?

        Lets the eval attribute a collision to the moving car specifically, which is the
        precise claim: the dynamic shield removes collisions *with crossing movers* that the
        static shield admits.
        """
        movers = self.scene.movers
        if not movers:
            return False
        centres, radius = footprint_discs(self._state, self.cfg.vehicle)
        d = min(point_box_distance(centres, m.box_at(t)).min() for m in movers)
        return bool(d - radius < 0.0)

    def reset(self, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._scene = self.scenes.sample(self._rng)
        self._state = self._scene.start
        self._prev_dist = self._distance_to_goal()
        self._step = 0
        self.last_shield = None
        self._sim_grid = self._combined_grid(0.0)
        return self._observation(), self._info(collided=False, reached=False)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        cfg = self.cfg
        vcfg = cfg.vehicle
        accel, steer = self.denormalize_action(np.asarray(action, float).reshape(-1))

        # The movers are frozen at the step's start; `_sim_grid` already holds them there.
        t_now = self._step * vcfg.dt
        if cfg.use_shield:
            if cfg.dynamic_shield:
                self.last_shield = dynamic_safety_shield(
                    accel, steer, self._state, self.scene.static_grid,
                    self._movers_at(t_now), vcfg)
            else:
                self.last_shield = safety_shield(accel, steer, self._state,
                                                 self._sim_grid, vcfg)
            accel, steer = self.last_shield.accel, self.last_shield.steer

        self._state = step_state(self._state, accel, steer, vcfg)
        self._step += 1

        # Advance the world one step and re-read it: the successor state is checked against the
        # movers where they now are, matching the shield's one-step-ahead certificate.
        t_next = self._step * vcfg.dt
        self._sim_grid = self._combined_grid(t_next)

        dist = self._distance_to_goal()
        collided = clearance(self._state, self._sim_grid, vcfg) < 0.0
        reached = dist <= cfg.goal_radius

        reward = cfg.progress_weight * (self._prev_dist - dist) - cfg.step_penalty
        if collided:
            reward -= cfg.collision_penalty
        if reached:
            reward += cfg.goal_bonus
        self._prev_dist = dist

        terminated = bool(collided or reached)
        truncated = bool(self._step >= cfg.max_steps and not terminated)
        info = self._info(collided, reached)
        info["hit_mover"] = collided and self._hit_mover(t_next)
        return self._observation(), float(reward), terminated, truncated, info


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


def kitti_frame_split(n_frames: int, holdout: float = 0.3,
                      stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic **contiguous** train/test split of a drive's frames.

    Contiguous rather than interleaved so the held-out frames are a genuinely different stretch
    of the drive: consecutive frames of a moving vehicle are near-duplicate views of the same
    street, and an interleaved split would leak them across the boundary and flatter the
    transfer. Training on the first `1 - holdout` and testing on the last `holdout` therefore
    measures generalisation to road the policy never saw. `stride` subsamples the *train* pool
    only (so per-frame fused grids stay cache-resident during training); the test set is dense.
    Both `train` and `test` are computed the same way here so a trainer and an evaluator agree
    on the split without passing frame arrays around.
    """
    frames = np.arange(int(n_frames))
    cut = int(round(int(n_frames) * (1.0 - holdout)))
    return frames[:cut][::max(int(stride), 1)], frames[cut:]


@dataclass
class KittiScenes:
    """Real recorded street geometry: BEV grids built from a drive's Velodyne scans.

    The geometry is frozen as a static obstacle field and the car is asked to cross it. It is
    genuinely real — parked cars, kerbs, buildings — which is what makes this a transfer test
    rather than more of the same synthetic distribution.

    **Single scan or accumulated map.** With `map_config` set, the scene is built by fusing a
    window of scans through estimated poses (`mapping.py`) instead of freezing one. This is
    not a free upgrade and should not be treated as one: the fused map contains obstacles a
    single scan could not see — `scripts/eval_mapping.py` measures the single-scan map as
    missing **44%** of the occupied cells a five-scan map holds — so scenes get genuinely
    harder, and policy success rates move. It defaults to `None` (single scan) so the
    existing results stay reproducible, and the comparison is the point.

    `poses` defaults to the drive's OXTS ground truth. Passing this repo's stereo-VO
    trajectory instead is the more deployable choice and measures the same: at a five-scan
    window the two agree to 0.01 m/s of shield-permitted speed, because fusion is governed by
    the *relative* pose error over the window (24 cm) rather than by global drift (12 m).
    """

    drive: object                              # KittiDrive; untyped to avoid a hard import
    bev: BEVConfig = field(default_factory=BEVConfig)
    goal_distance: tuple[float, float] = (20.0, 35.0)
    goal_lateral: float = 3.0
    start_speed: tuple[float, float] = (3.0, 10.0)
    frames: Optional[np.ndarray] = None        # restrict to these frames (e.g. a held-out split)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    # None = one frozen scan (the original behaviour). Set it to accumulate a window; set
    # `MapConfig(carve=True)` to also free-space-carve it into an occupied/free/unknown map.
    map_config: Optional[MapConfig] = None
    poses: Optional[np.ndarray] = None         # camera-to-world; defaults to OXTS ground truth

    # Only meaningful on a carved map: treat never-observed cells as obstacles for the shield
    # and the policy's rays, rather than assuming them free. The honest reading of an
    # accumulated map's interior holes; off keeps the old `outside_is_free` behaviour.
    unknown_blocks: bool = False

    # The drivable alternative to `unknown_blocks` (and independent of it): let the car
    # traverse unknown cells but have the env govern its speed by the frontier distance, so it
    # never enters unmapped space faster than it could brake out of. See `cautious_speed_cap`.
    unknown_speed_cap: bool = False

    # Grids are cached by frame, which matters more than it looks: every episode resamples a
    # frame, and building a grid means loading scans *and* computing the distance transform
    # the shield queries. Without this a training run rebuilds the same few grids thousands
    # of times. Bounded so a long `frames` pool cannot exhaust memory.
    grid_cache_size: int = 64

    def __post_init__(self) -> None:
        from functools import lru_cache

        self._grid = lru_cache(maxsize=self.grid_cache_size)(self._build_grid)

    def _build_grid(self, i: int) -> BEVGrid:
        """The obstacle field at frame `i`, fused or single-scan."""
        if self.map_config is None:
            # Self-filtering even here: it costs zero occupancy cells on a single scan (the
            # bodywork was already its own ground) and keeps both paths preprocessed alike.
            return BEVGrid(occupancy_from_scan(
                drop_ego_returns(self.drive.velodyne(i)), self.bev), self.bev)

        poses = self.drive.gt_poses if self.poses is None else self.poses
        idx = window_indices(i, self.map_config, self.drive.n_velodyne)
        scans = [self.drive.velodyne(j) for j in idx]
        window_poses = [poses[j] for j in idx]

        if self.map_config.carve:
            fm = fuse_map(scans, window_poses, self.drive.T_cam2_velo, ref=-1,
                          cfg=self.map_config, bev_cfg=self.bev)
            return fm.to_bev_grid(unknown_blocks=self.unknown_blocks,
                                  unknown_speed_cap=self.unknown_speed_cap)

        pts = fuse_scans(scans, window_poses, self.drive.T_cam2_velo)
        return BEVGrid(occupancy_from_scan(pts, self.bev), self.bev)

    def sample(self, rng: np.random.Generator) -> Scene:
        pool = self.frames if self.frames is not None else np.arange(self.drive.n_velodyne)
        i = int(rng.choice(pool))
        grid = self._grid(i)

        start = self.drive.vehicle_state_in_lidar(speed=float(rng.uniform(*self.start_speed)))
        goal = np.array([start.x + float(rng.uniform(*self.goal_distance)),
                         start.y + float(rng.uniform(-self.goal_lateral, self.goal_lateral))])
        return Scene(grid, certifiable_start(start, grid, self.vehicle), goal)


@dataclass
class KittiDynamicScenes:
    """Closed-loop crossing encounters mined from a drive's labelled moving objects.

    The dynamic counterpart to `KittiScenes`, and the source the static-vs-dynamic-shield table
    is measured on. It scans the drive's genuinely-moving tracklets (`moving_tracklets`) for
    frames where an actor sits **ahead of the ego** and is **crossing into** the forward
    corridor at a real lateral speed — precisely the 8% of mover-frames where treating the world
    as frozen is a safety error (`scripts/eval_dynamic_shield.py`). Each such `(frame, tracklet)`
    becomes a `DynamicScene`: the frame's occupancy with that actor lifted out as the static
    background, the actor re-inserted as a constant-velocity `MovingObstacle` seeded from its
    tracklet motion (`KittiDrive.actor_velocity`), the ego at its true lidar-frame pose driving
    +x, and a goal ahead. Tracklet poses are native to the velo frame, so boxes drop in with no
    transform.

    Velocity comes from the **tracklet**, not label-free estimation: label-free BEV velocity is
    ~95% false-positive on 0009's parked cars (`dynamics` docstring), which would have the shield
    braking for phantoms. The perception gap is real and measured; the *feed* here is the clean
    one, so the scene tests the shield rather than the estimator.
    """

    drive: object                              # KittiDrive; untyped to avoid a hard import
    bev: BEVConfig = field(default_factory=BEVConfig)
    goal_distance: tuple[float, float] = (20.0, 35.0)
    goal_lateral: float = 3.0
    start_speed: tuple[float, float] = (3.0, 10.0)
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)

    min_disp: float = 2.0            # world displacement (m) for a tracklet to count as moving
    min_lateral_speed: float = 1.0   # crossing speed (m/s) for a frame to count as an encounter
    ahead_margin: float = 5.0        # a mover must be at least this far ahead of the ego
    edge_margin: float = 5.0         # ...and this far inside the grid's forward/side edges

    def __post_init__(self) -> None:
        self._ego = self.drive.vehicle_state_in_lidar()
        self._encounters = self._find_encounters()
        if not self._encounters:
            raise ValueError(
                "no crossing encounters found on this drive — needs labelled moving actors "
                "crossing ahead of the ego (fetch tracklets, or relax min_lateral_speed).")

    def _find_encounters(self) -> list[tuple]:
        """`(frame, box, velocity)` for every mover-frame crossing into the forward corridor."""
        ego, cfg = self._ego, self.bev
        out: list[tuple] = []
        for t in self.drive.moving_tracklets(self.min_disp):
            last = min(t.last_frame + 1, self.drive.n_velodyne)
            for f in range(t.first_frame + 1, last):
                box = t.box_at(f)
                if box is None:
                    continue
                cx, cy = float(box[0]), float(box[1])
                if not (ego.x + self.ahead_margin < cx < cfg.x_max - self.edge_margin):
                    continue
                if not (cfg.y_min + self.edge_margin < cy < cfg.y_max - self.edge_margin):
                    continue
                vel = self.drive.actor_velocity(t, f)
                if vel is None or abs(vel[1]) < self.min_lateral_speed:
                    continue
                # Crossing *toward* the ego's line, not away from it: vy points at ego.y.
                if np.sign(vel[1]) != np.sign(ego.y - cy):
                    continue
                out.append((f, box.copy(), np.asarray(vel, float)))
        return out

    def sample(self, rng: np.random.Generator) -> DynamicScene:
        f, box, vel = self._encounters[int(rng.integers(len(self._encounters)))]

        occ = occupancy_from_scan(
            drop_ego_returns(self.drive.velodyne(f), KITTI_EGO_BOX), self.bev).astype(bool)
        actor = rasterize_box(box, self.bev)
        static_grid = BEVGrid((occ & ~actor).astype(np.uint8), self.bev)
        mover = MovingObstacle(box.copy(), vel)

        ego = self.drive.vehicle_state_in_lidar(speed=float(rng.uniform(*self.start_speed)))
        goal = np.array([ego.x + float(rng.uniform(*self.goal_distance)),
                         ego.y + float(rng.uniform(-self.goal_lateral, self.goal_lateral))])

        # Certify the start against the world at t=0 (background + mover in place), so no episode
        # begins already unable to stop — the moving-world version of `certifiable_start`.
        grid0 = BEVGrid((occ | actor).astype(np.uint8), self.bev)
        start = certifiable_start(ego, grid0, self.vehicle)
        return DynamicScene(static_grid, [mover], start, goal)


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
