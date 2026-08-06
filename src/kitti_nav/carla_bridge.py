"""Run the kitti-nav BEV perception + braking safety shield as a CARLA ego controller.

This is the **client** side of a CARLA closed loop. CARLA's simulator + NuRec neural
rendering run on a GPU box (Linux, NVIDIA RTX); this module connects to that server over
CARLA's Python API, and every tick it:

    CARLA lidar  ->  velodyne-frame points  ->  BEVGrid occupancy  ->  safety_shield  ->  CARLA control

so the exact `bev` + `vehicle` code that produced the offline KITTI numbers now drives a car
in closed loop, where the ego's own steering changes what the sensors see next — the thing a
frozen log can never give you. The heavy simulator does not run here; only this client does,
which is why the pure pieces below (`carla_lidar_to_velodyne`, `shield_to_carla_control`,
`ShieldController`) import and unit-test with no `carla` package and no GPU.

Coordinate conventions — the part to get right, and where the easy bugs live
----------------------------------------------------------------------------
* **kitti-nav / Velodyne** is right-handed: +x forward, **+y left**, +z up; yaw 0 along +x
  increasing counter-clockwise; positive steer turns **left**. The BEV grid and the shield
  live entirely in this frame, ego-centric (the sensor is the origin).
* **CARLA / Unreal** is left-handed: +x forward, **+y right**, +z up; positive steer turns
  **right**. So converting CARLA -> kitti-nav means negating `y` (points) and negating the
  steer command. `carla_lidar_to_velodyne` and `shield_to_carla_control` are the two seams
  that do exactly that, and nothing else in the loop touches handedness.

The learned PPO policy is intentionally *not* wired in yet: the shield is a runtime layer over
*any* base command, so the scaffold ships with a trivial `ForwardGoalPlanner` (drive toward a
point, let the shield handle safety). Swap in the sb3 policy later behind the `BasePlanner`
protocol — the shield and the CARLA plumbing are unchanged by that choice.

    python3 -m kitti_nav.carla_bridge --host <box-ip> --port 2000 --target-speed 8
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from .bev import BEVConfig, BEVGrid
from .vehicle import ShieldResult, VehicleConfig, VehicleState, safety_shield

# ---------------------------------------------------------------------------------------
# Pure conversions (no `carla`, no GPU — unit-tested in tests/test_carla_bridge.py)
# ---------------------------------------------------------------------------------------


def carla_lidar_to_velodyne(raw_points: np.ndarray) -> np.ndarray:
    """CARLA lidar buffer -> `(N, 4)` velodyne-frame points `(x_forward, y_left, z_up, i)`.

    CARLA delivers `float32` `(x, y, z, intensity)` in the sensor's Unreal (left-handed, +y
    **right**) frame. `BEVGrid.from_scan` expects the Velodyne convention (+y **left**), so the
    single required change is to negate `y`; `x`, `z`, and intensity pass through untouched.
    """
    pts = np.asarray(raw_points, np.float32).reshape(-1, 4)
    out = pts.copy()
    out[:, 1] = -out[:, 1]      # +y right (CARLA) -> +y left (velodyne)
    return out


def shield_to_carla_control(result: ShieldResult,
                            cfg: VehicleConfig) -> tuple[float, float, float]:
    """`ShieldResult` (accel m/s^2, steer rad, +left) -> CARLA `(throttle, brake, steer)`.

    CARLA control is normalised: throttle/brake in `[0, 1]`, steer in `[-1, 1]` with **+1 =
    full right**. We map acceleration onto throttle *or* brake (never both) by the vehicle's
    own `max_accel` / `max_decel`, and flip the steer sign because our positive steer is a left
    turn. This is a deliberately simple actuator model — CARLA's true throttle->accel curve is
    non-linear and vehicle-specific; a per-vehicle calibration or a small PID is the honest
    upgrade, and the place to add it is right here, not in the shield.
    """
    if result.accel >= 0.0:
        throttle = float(min(result.accel / cfg.max_accel, 1.0))
        brake = 0.0
    else:
        throttle = 0.0
        brake = float(min(-result.accel / cfg.max_decel, 1.0))
    steer = float(np.clip(-result.steer / cfg.max_steer, -1.0, 1.0))   # +left -> -right
    return throttle, brake, steer


# ---------------------------------------------------------------------------------------
# Base planner: the command the shield filters. Trivial by design; swap the PPO policy in here.
# ---------------------------------------------------------------------------------------


class BasePlanner(Protocol):
    """Produces an *unshielded* `(accel_cmd, steer_cmd)` from the current state and BEV.

    The seam for the actual driver. A learned sb3 PPO policy, a gap-follower, or the trivial
    planner below all satisfy this; the shield wraps whatever it returns without retraining.
    """

    def __call__(self, state: VehicleState, grid: BEVGrid) -> tuple[float, float]:
        ...


@dataclass
class ForwardGoalPlanner:
    """Placeholder driver: accelerate toward `target_speed`, steer toward a point ahead.

    Enough to exercise the closed loop and let the shield demonstrably brake for obstacles.
    `goal_xy` is in the ego frame (metres, +x forward / +y left); the default drives straight.
    Pure-pursuit-lite: a proportional heading command, clamped to the mechanical steer limit.
    Replace this with the trained policy once the loop is up.
    """

    cfg: VehicleConfig
    target_speed: float = 8.0
    goal_xy: tuple[float, float] = (30.0, 0.0)
    heading_gain: float = 1.0

    def __call__(self, state: VehicleState, grid: BEVGrid) -> tuple[float, float]:
        gx, gy = self.goal_xy
        # Bearing to the goal in the ego frame; ego heading is 0 along +x by construction.
        bearing = float(np.arctan2(gy, gx))
        steer_cmd = float(np.clip(self.heading_gain * bearing,
                                  -self.cfg.max_steer, self.cfg.max_steer))
        # Proportional approach to target speed, clamped to the actuation envelope.
        accel_cmd = float(np.clip((self.target_speed - state.v) / self.cfg.dt,
                                  -self.cfg.max_decel, self.cfg.max_accel))
        return accel_cmd, steer_cmd


# ---------------------------------------------------------------------------------------
# ShieldController: BEV -> shield -> CARLA control. Pure (no carla); the loop's brain.
# ---------------------------------------------------------------------------------------


@dataclass
class ShieldController:
    """Turns a lidar scan + measured speed into a shielded CARLA control command.

    Holds the one piece of state the shield needs across ticks: the vehicle's current
    road-wheel angle. The rack is rate-limited, so the shield reasons about the angle the car
    will *actually* have, not the last one commanded — we track it as the shield's own accepted
    steer, which is what `step_state` would realise. `rear_axle_x` places the rear axle in the
    lidar/BEV frame: the lidar sits forward of the rear axle by that many metres, so the axle is
    at `x = -rear_axle_x` (KITTI's Velodyne is ~1.9 m forward; a rear-axle mount is 0.0).
    """

    planner: BasePlanner
    vcfg: VehicleConfig
    bcfg: BEVConfig
    rear_axle_x: float = 0.0
    _steer: float = 0.0        # current road-wheel angle (rad), carried between ticks

    def reset(self) -> None:
        self._steer = 0.0

    def step(self, velodyne_points: np.ndarray, speed: float
             ) -> tuple[float, float, float, ShieldResult]:
        """One control tick. Returns `(throttle, brake, steer_norm, ShieldResult)`."""
        grid = BEVGrid.from_scan(velodyne_points, self.bcfg)
        state = VehicleState(x=-self.rear_axle_x, y=0.0, yaw=0.0,
                             v=float(speed), steer=self._steer)
        accel_cmd, steer_cmd = self.planner(state, grid)
        result = safety_shield(accel_cmd, steer_cmd, state, grid, self.vcfg)
        self._steer = result.steer      # what the car will hold going into the next tick
        throttle, brake, steer_norm = shield_to_carla_control(result, self.vcfg)
        return throttle, brake, steer_norm, result


# ---------------------------------------------------------------------------------------
# CARLA glue (requires the `carla` package + a running server; runs on the GPU box).
# ---------------------------------------------------------------------------------------


class CarlaBridge:
    """Spawns an ego + roof lidar in a running CARLA world and drives it with a ShieldController.

    Runs the world in **synchronous** mode at the vehicle's control `dt` so perception, the
    shield, and the physics step advance in lockstep — the same determinism the offline env
    relies on. Everything CARLA-specific is confined to this class; import it only where the
    `carla` package is installed (the box), never as a hard dependency of the shield.
    """

    def __init__(self, controller: ShieldController, host: str = "127.0.0.1",
                 port: int = 2000, lidar_z: float = 1.9, timeout: float = 20.0) -> None:
        import carla   # noqa: F401  — deferred: only the box has it, and only for the loop

        self.carla = carla
        self.controller = controller
        self.lidar_z = lidar_z
        self.client = carla.Client(host, port)
        self.client.set_timeout(timeout)
        self.world = self.client.get_world()
        self.ego = None
        self.lidar = None
        self._latest_scan: Optional[np.ndarray] = None

    def _setup_sync(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.controller.vcfg.dt
        self.world.apply_settings(settings)

    def spawn(self) -> None:
        """Spawn the ego vehicle and a roof-mounted lidar streaming into `_latest_scan`."""
        carla = self.carla
        bp = self.world.get_blueprint_library()
        ego_bp = bp.filter("vehicle.*")[0]
        spawn = self.world.get_map().get_spawn_points()[0]
        self.ego = self.world.spawn_actor(ego_bp, spawn)

        lidar_bp = bp.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("range", "50")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "600000")
        lidar_bp.set_attribute("rotation_frequency", str(1.0 / self.controller.vcfg.dt))
        mount = carla.Transform(carla.Location(x=0.0, z=self.lidar_z))
        self.lidar = self.world.spawn_actor(lidar_bp, mount, attach_to=self.ego)
        self.lidar.listen(self._on_scan)

    def _on_scan(self, data) -> None:
        raw = np.frombuffer(data.raw_data, dtype=np.float32)
        self._latest_scan = carla_lidar_to_velodyne(raw)

    def _speed(self) -> float:
        v = self.ego.get_velocity()
        return float(np.hypot(np.hypot(v.x, v.y), v.z))

    def run(self, max_ticks: int = 600) -> None:
        """Closed loop: tick the world, read lidar, shield, apply control. Ctrl-C to stop."""
        self._setup_sync()
        self.spawn()
        self.controller.reset()
        try:
            for _ in range(max_ticks):
                self.world.tick()
                if self._latest_scan is None:
                    continue
                throttle, brake, steer, result = self.controller.step(
                    self._latest_scan, self._speed())
                self.ego.apply_control(self.carla.VehicleControl(
                    throttle=throttle, brake=brake, steer=steer))
                if result.ics:
                    print("shield: inevitable-collision state — commanding full brake")
        finally:
            self.close()

    def close(self) -> None:
        for actor in (self.lidar, self.ego):
            if actor is not None:
                actor.destroy()
        settings = self.world.get_settings()
        settings.synchronous_mode = False
        self.world.apply_settings(settings)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="CARLA server IP (the GPU box)")
    p.add_argument("--port", type=int, default=2000)
    p.add_argument("--target-speed", type=float, default=8.0, help="m/s the planner seeks")
    p.add_argument("--evasive", type=int, default=0,
                   help="n_evasive_steers: >0 lets the shield swerve, not just brake")
    p.add_argument("--rear-axle-x", type=float, default=0.0,
                   help="metres the lidar is mounted forward of the rear axle")
    p.add_argument("--ticks", type=int, default=600)
    args = p.parse_args()

    import dataclasses
    vcfg = dataclasses.replace(VehicleConfig(), n_evasive_steers=args.evasive)
    controller = ShieldController(
        planner=ForwardGoalPlanner(vcfg, target_speed=args.target_speed),
        vcfg=vcfg, bcfg=BEVConfig(), rear_axle_x=args.rear_axle_x)
    CarlaBridge(controller, host=args.host, port=args.port).run(max_ticks=args.ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
