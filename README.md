# kitti-nav

Autonomous-driving navigation on real recorded drives: replay a KITTI sequence, build the
occupancy/BEV representation a real AV stack actually plans on, and drive a vehicle through
it behind a **hard safety shield** that is provably unable to admit a collision it could
have braked out of.

Runs entirely on a laptop — pure NumPy core, no GPU, no simulator install.

> **Why occupancy and not splats?** Production AV stacks make live driving decisions on
> lidar/occupancy/BEV grids, not photorealistic reconstructions. Gaussian splatting's real
> role in AV work is *offline* — turning recorded drives into replayable digital twins for
> closed-loop testing. This repo keeps the live path on occupancy from the start.

## Status

| Piece | State |
| --- | --- |
| Kinematic bicycle (Ackermann) vehicle model | done, 20 tests |
| Braking-aware safety shield | done, tested — incl. a fuzz test that found a real soundness bug |
| KITTI raw loading (stereo + lidar + OXTS ground truth) | verified on drive `2011_09_26_0009` |
| Stereo/ORB visual odometry | next |
| Lidar → BEV occupancy grid | next |
| Learned planner behind the shield | after the above |

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/fetch_kitti.py          # ~1.7 GB, drive 0009; --list for other options
python3 -m pytest tests/ -q
```

The dataset lands in `data/kitti_raw/` and is gitignored. Drive `2011_09_26_0009` is 447
frames covering **332.3 m** at up to **11.4 m/s**, with a 0.533 m stereo baseline and
~122k Velodyne points per frame.

## The safety shield

`src/kitti_nav/vehicle.py` wraps *any* controller — hand-written or learned, no retraining
— and filters its commands. The design descends from the diff-drive shield in this author's
[`gsplat-rt`](../gsplat-rt) repo, but the safety argument had to be rebuilt, because a car
breaks both of the assumptions that made the original sound:

**1. A car cannot rotate in place.** The diff-drive shield's escape hatch was "forbid
forward motion and let it spin, which cannot collide." A bicycle model's yaw rate is
`v/L·tan δ` — at zero speed it cannot turn either. Stopping is still safe, but no longer free.

**2. One-step lookahead is not a safety guarantee at speed.** At 12 m/s with 4.5 m/s²
braking the vehicle needs ~16 m to stop, while one 0.1 s control step advances it 1.2 m. A
filter that checks only the next pose finds every individual step clear right up until none
of them are. `test_one_step_shield_crashes_where_braking_shield_stops` demonstrates exactly
this: the naive port, faithfully reimplemented, drives into a wall.

So the shield asks a strictly stronger question — *after this action, does a full-braking
rollout from the resulting state still stop clear?* Admission requires the successor to
retain a safe stop, which makes the invariant inductive: from a certified state, full
braking is always still certifiable, so the shield never has to admit a collision. This is
the inevitable-collision-state / reachability argument (Fraichard & Asama 2004; the
reasoning behind RSS), implemented from the concept.

### A bug worth keeping in the record

The first version modulated throttle only and passed steering straight through. That is
unsound, and the randomised rollout test caught it: the braking certificate is issued under
the *current* wheel angle, so if the policy commands a different angle on the next step,
full braking can curve into an obstacle the certificate never covered. The shield now falls
back to holding the last certified steer, which induction guarantees is stoppable. The
failing test is kept as `test_shield_falls_back_to_the_held_steer_rather_than_obeying_a_fatal_one`.

Known limitation, stated plainly: the shield chooses between the commanded steer and the
held steer — it never searches for an *evasive* one, so it will brake for obstacles it might
have swerved around.

## Layout

```
src/kitti_nav/vehicle.py   bicycle model, footprint geometry, braking shield
scripts/fetch_kitti.py     dataset download (data is never committed)
tests/                     pure-NumPy, no GPU, no dataset needed
```

## Attribution

Built on KITTI (Geiger et al., **CC BY-NC-SA 3.0 — non-commercial**) and pykitti (MIT).
Full credits, licenses, and citations in [`ATTRIBUTION.md`](ATTRIBUTION.md).
