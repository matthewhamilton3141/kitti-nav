#!/usr/bin/env python3
"""Closed-loop dynamic traffic: the static shield vs the dynamic shield on moving cars.

    python3 scripts/eval_dynamic_policies.py
    python3 scripts/eval_dynamic_policies.py --episodes 200 --ppo models/ppo_shielded.zip

The moving-world analogue of `eval_policies.py`'s "0 collisions" table. `KittiDynamicScenes`
mines drive 0009's labelled movers for frames where a car crosses **into the ego's forward
corridor**, lifts that actor out of the frozen occupancy, and re-inserts it as a
constant-velocity `MovingObstacle`. A policy then drives to a goal ahead while the car crosses,
under three conditions:

  * **unshielded** — the raw policy;
  * **static shield** — `vehicle.safety_shield` against the movers frozen where they are now;
  * **dynamic shield** — `dynamics.dynamic_safety_shield`, which brakes for where a mover is
    *going*.

The headline is not just the collision count but *how* a collision happens. A mover-collision is
split by the ego's speed at impact:

  * **drove in** (ego still moving): the shield steered the car into the crossing mover — the
    failure the dynamic shield exists to remove;
  * **run into** (ego stopped): the car had already braked to a halt clear of the mover's
    predicted path, and the mover drove into the stationary ego — which *no* braking shield can
    prevent, and which the dynamic shield correctly flags as an inevitable-collision state.

Expected: the static shield drives into crossing cars at speed; the dynamic shield cuts that to
~0, its residual collisions being movers striking an already-stopped ego. Velocity is fed from
the tracklet, not label-free estimation (~95% false-positive on parked cars — see `dynamics`).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kitti_nav.nav_env import (                     # noqa: E402
    DynamicNavConfig,
    DynamicNavEnv,
    KittiDynamicScenes,
    gap_following_policy,
)

STOPPED = 1.5   # m/s below which a mover-collision counts as "the mover ran into a stopped ego"


def load_ppo(path: Path):
    """Wrap a saved stable-baselines3 policy as a plain `obs -> action` callable."""
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")

    def policy(obs: np.ndarray) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=True)
        return action
    return policy


def evaluate_dynamic(scenes, policy, *, use_shield: bool, dynamic: bool,
                     n_episodes: int, seed: int = 0, n_evasive: int = 0) -> dict:
    """Roll `policy` over the dynamic scenes, tallying how each mover-collision happened."""
    from kitti_nav.vehicle import VehicleConfig

    cfg = DynamicNavConfig(use_shield=use_shield, dynamic_shield=dynamic,
                           vehicle=VehicleConfig(n_evasive_steers=n_evasive))
    env = DynamicNavEnv(scenes, cfg)

    reached = collided = mover_hits = drove_in = run_into = ics_hits = 0
    for i in range(n_episodes):
        obs, info = env.reset(seed + i)
        for _ in range(cfg.max_steps):
            obs, _, terminated, truncated, info = env.step(policy(obs))
            if terminated or truncated:
                break
        reached += bool(info["reached"])
        collided += bool(info["collided"])
        if info.get("hit_mover"):
            mover_hits += 1
            if env.state.v > STOPPED:
                drove_in += 1
            else:
                run_into += 1
            if env.last_shield is not None and env.last_shield.ics:
                ics_hits += 1
    return {
        "episodes": n_episodes,
        "reached": reached,
        "collisions": collided,
        "mover_hits": mover_hits,
        "drove_in": drove_in,
        "run_into": run_into,
        "ics_hits": ics_hits,
    }


def _row(name: str, r: dict) -> str:
    n = r["episodes"]
    return (f"  {name:<22} success {r['reached']:>3}/{n}   collisions {r['collisions']:>3}   "
            f"mover-hits {r['mover_hits']:>3}  (drove in {r['drove_in']:>3}, "
            f"run into {r['run_into']:>3}, ics-flagged {r['ics_hits']:>3})")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date", default="2011_09_26")
    p.add_argument("--drive", default="0009")
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ppo", type=Path, default=None,
                   help="also score a saved PPO policy (else gap-following only)")
    p.add_argument("--min-lateral-speed", type=float, default=1.0,
                   help="crossing speed (m/s) for a mover-frame to count as an encounter")
    p.add_argument("--evasive", type=int, default=0, metavar="N",
                   help="also add a dynamic-shield row that swerves (N steer candidates) when "
                        "braking cannot certify a stop; 0 disables it")
    args = p.parse_args()

    from kitti_nav.kitti import KittiDrive

    drive = KittiDrive(args.date, args.drive)
    scenes = KittiDynamicScenes(drive, min_lateral_speed=args.min_lateral_speed)
    n_enc = len(scenes._encounters)
    print(f"drive {args.drive}: {n_enc} crossing encounters "
          f"({len(scenes.drive.moving_tracklets(scenes.min_disp))} moving actors)\n")

    gap_cfg = DynamicNavConfig()
    policies = [("gap-following", lambda o: gap_following_policy(o, gap_cfg))]
    if args.ppo is not None:
        policies.append((f"PPO ({args.ppo.name})", load_ppo(args.ppo)))

    # (label, use_shield, dynamic, n_evasive); the evasive row is opt-in via --evasive.
    conditions = [("unshielded", False, False, 0),
                  ("static shield", True, False, 0),
                  ("dynamic shield", True, True, 0)]
    if args.evasive > 0:
        conditions.append(("dynamic + evasive", True, True, args.evasive))

    for name, pol in policies:
        print(f"{name}:")
        for label, use_shield, dyn, ev in conditions:
            r = evaluate_dynamic(scenes, pol, use_shield=use_shield, dynamic=dyn,
                                 n_episodes=args.episodes, seed=args.seed, n_evasive=ev)
            print(_row(label, r))
        print()

    print("'drove in' = the shield steered the ego into the crossing mover (the failure the\n"
          "dynamic shield removes); 'run into' = the ego had stopped clear and the mover drove\n"
          "into it (unavoidable by braking, and correctly ics-flagged). The dynamic shield's\n"
          "guarantee is sound to the extent the constant-velocity prediction holds. With\n"
          "--evasive the shield may swerve out of an ICS rather than brake into it; on real\n"
          "cluttered traffic the room to do so is usually absent, so the gain is small.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
