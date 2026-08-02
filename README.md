# kitti-nav

Autonomous-driving navigation on real recorded drives: replay a KITTI sequence, build the
occupancy/BEV representation a real AV stack plans on, and drive a vehicle through it behind
a **hard safety shield** that provably cannot admit a collision it could have braked out of.

Runs entirely on a laptop — pure NumPy + OpenCV, no GPU, no simulator install.

> **Why occupancy and not Gaussian splats?** Production AV stacks make live driving decisions
> on lidar/occupancy/BEV grids, not photorealistic reconstructions. Splatting's real role in
> AV work is *offline* — turning recorded drives into replayable digital twins for
> closed-loop testing. This repo keeps the live path on occupancy from the start.

## Status

| Piece | State |
| --- | --- |
| Kinematic bicycle (Ackermann) vehicle model | done |
| Braking-aware safety shield | done — a fuzz test found a real soundness bug in it |
| KITTI raw loading (stereo + lidar + OXTS ground truth) | done |
| Stereo visual odometry (ORB + PnP) | done — **3.55% drift over 332.8 m at 26 fps CPU** |
| Lidar → BEV occupancy + distance field | done — **1.8% occupancy, 1.9 ms/frame** |
| Shield running natively on real lidar | done — **6 ms/frame end to end** |
| Learned planner behind the shield | next |

**83 tests pass.** Dataset-backed tests skip cleanly when KITTI isn't downloaded.

## Quickstart

```bash
pip install -r requirements.txt
python3 scripts/fetch_kitti.py               # ~1.7 GB, drive 0009; --list for other options
python3 -m pytest tests/ -q

python3 scripts/eval_odometry.py --plot docs/trajectory.png
python3 scripts/render_bev.py --frame 294 --speed-profile docs/speed_profile.png
```

Data lands in gitignored `data/kitti_raw/`. Drive `2011_09_26_0009` is 447 frames covering
**332.8 m** at up to **11.4 m/s**, with a 0.533 m stereo baseline and ~122k Velodyne points
per frame. (It ships only **443** lidar scans for those 447 images — real datasets have
holes, so anything iterating lidar bounds on `n_velodyne`.)

---

## What this takes from gsplat-rt, and what it doesn't

This repo is a deliberate branch off [**gsplat-rt**](https://github.com/matthewhamilton3141/gsplat-rt),
a real-time TensorRT Gaussian-splat SLAM project by the same author. That project ended with
two finished arcs: a VGGT-style reconstruction model taken to TensorRT (1.187× whole-model
speedup from two compounding levers), and a navigation-RL flagship whose capstone result was
that **training a policy through a hard safety shield dominated the unshielded policy on
every axis** (100% success / 0 collisions / fewer steps, vs 98% / 4 / more).

That shield result is the seed of this repo. The interesting question was whether the idea
survives contact with *driving* — a different vehicle, different speeds, different sensors.
Mostly it did, but almost nothing transferred unchanged.

### Reused as-is (the design, not the code)

- **The shield concept**: a runtime filter wrapping *any* controller, learned or hand-written,
  requiring no retraining. `predict_pose` / `clearance_at` / `safety_shield` from
  `src/isaac/nav_sim.py` set the shape of `vehicle.py`.
- **One integrator, shared by the simulator and the shield's lookahead.** A filter that
  predicts differently than the world integrates is unsound, so `step_state` is used by both.
- **Pure-NumPy, laptop-testable cores.** gsplat-rt's discipline of keeping the durable logic
  free of GPU/torch/gym dependencies is what makes this repo runnable with no setup.
- **The pluggable front-end seam** in the odometry. gsplat-rt used exactly this to swap ORB
  for a TensorRT SuperPoint+LightGlue front-end without touching any geometry.
- **Measure, don't assume; correct claims downward.** Every number in this README was
  produced by a script in `scripts/`.

### Rebuilt, because a car breaks the assumptions

**The safety shield could not be ported.** Two independent reasons:

1. **A car cannot rotate in place.** The diff-drive shield's escape hatch was "forbid forward
   motion and let it spin — spinning can't collide." But a bicycle model's yaw rate is
   `v/L·tan δ`, which is *also* zero at `v = 0`. Stopping is still safe; it is no longer free.
2. **One-step lookahead is not a safety guarantee at speed.** At 12 m/s braking at 4.5 m/s²,
   stopping takes ~16 m while one 0.1 s control step covers 1.2 m. A next-pose check finds
   every step clear right up until none are.

`test_one_step_shield_crashes_where_braking_shield_stops` reimplements the naive port
faithfully and **drives it into a wall**, in the same scene where the rebuilt shield stops
clean. The failure is demonstrated, not asserted.

**Stereo deleted an entire subsystem.** gsplat-rt estimated depth with a monocular network,
whose output is only defined up to scale — it needed a whole `monocular_scale.py` stage, and
residual scale drift dominated its error. Here `depth = fx·b/disparity` is metric by
construction. Nothing to estimate, nothing to drift. Consequently the evaluation applies **no
Sim(3) alignment**: fitting a scale factor is correct for monocular VO and would be
self-flattery here.

**Circular obstacles became an occupancy grid.** gsplat-rt's world was circles on a plane.
Real lidar is not, so the shield now talks to an `ObstacleField` interface, and `BEVGrid`
implements it directly. Occupancy is never approximated by circles — that would discard
exactly the arbitrary shape occupancy represents well.

### Deliberately left behind

Gaussian splats as a live representation (see the note at the top), the whole monocular
scale-recovery stack, the TensorRT/CUDA layer (this is CPU-only by design), and the
diff-drive kinematics.

---

## The safety shield

`src/kitti_nav/vehicle.py` wraps any controller and filters its commands. It asks a strictly
stronger question than a next-pose check:

> After taking this action, does a **full-braking rollout** from the resulting state still
> stop clear?

An action is admitted only if the answer is yes. That makes the invariant **inductive**: from
a certified state, full braking is always still certifiable, so the shield never has to admit
a collision. This is the inevitable-collision-state / reachability argument (Fraichard &
Asama 2004; the reasoning behind RSS), implemented from the concept.

### A bug worth keeping in the record

The first version modulated throttle and passed steering straight through. That is unsound,
and a randomised rollout test caught it: the braking certificate is issued under the
*current* wheel angle, so if the policy commands a different angle next step, full braking
can curve into an obstacle the certificate never covered. The shield now falls back to
holding the last certified steer, which induction guarantees is stoppable. Kept as
`test_shield_falls_back_to_the_held_steer_rather_than_obeying_a_fatal_one`.

I would not have found this by hand — every scenario worth writing by hand has the policy
steering sensibly. It is the argument for property-based testing in one bug.

**Stated limitations:** the shield chooses between the commanded steer and the held steer; it
never searches for an *evasive* one, so it brakes for obstacles it might have swerved around.
The footprint is a conservative disc cover, which inflates the car ~0.12 m per side and ~0.55 m
past each bumper.

---

## Stereo visual odometry

![estimated vs ground-truth trajectory](docs/trajectory.png)

Full 447-frame drive, against OXTS ground truth:

| metric | value |
| --- | --- |
| ATE RMSE | 6.83 m |
| final drift | 11.82 m over 332.8 m — **3.55%** |
| speed | 16.9 s for 447 frames — **26.4 fps**, single-threaded CPU |
| tracking | 1064 mean matches, 377 mean PnP inliers, **0 fallback steps** |

ORB → ratio-tested matching → back-project through stereo depth → `solvePnPRansac` → compose.

**What 3.55% honestly is:** respectable frame-to-frame VO, clearly *not* state of the art.
No bundle adjustment, no keyframing, no loop closure, so error accumulates monotonically —
exactly what the plot shows. Known levers: local bundle adjustment, keyframing, and a learned
front-end (SuperPoint+LightGlue beat ORB in gsplat-rt, 3.5 cm vs 5.7 cm ATE on TUM).

One trap worth naming: ground truth needs `T_cam2_imu` composed onto the OXTS `T_w_imu`. The
IMU and camera are ~1.1 m and a ~90° rotation apart, so comparing VO against raw OXTS
**manufactures a large fake error**. Cross-checked against integrated OXTS speed.

---

## BEV occupancy from lidar

![lidar BEV occupancy with the shield's verdict](docs/bev.png)

Velodyne scans (~122k points) rasterise into a top-down grid at **1.9 ms/frame**, everything
staying in the lidar frame (+x forward, +y left) — which is already the bicycle model's frame,
so the planner consumes the grid with no axis juggling.

### Ground removal, chosen by measurement

Removing road returns is the whole ballgame; without it the grid is uniformly occupied. Two
strategies are implemented and the default was picked by measuring, not assuming:

| mode | occupied | clearance 30 m ahead | time |
| --- | --- | --- | --- |
| `plane` — fixed height band above a global ground height | 6.23% | **2.58 m** | 2.4 ms |
| `height_diff` — local per-cell ground *(default)* | **1.82%** | **6.04 m** | 1.9 ms |

KITTI documents the lidar 1.73 m above the ground, but near-field returns on this drive
spread ~0.4 m in `z` and are visibly bimodal — sensor pitch plus real road slope — so no
single constant is right across the scan. The plane mode's extra cells are road surface
drifting out of the band: **phantom obstacles that would brake the car for open road.** The
per-cell estimate follows the road instead, and is both more accurate and faster.

Ground is estimated over a small *neighbourhood* rather than a single cell, because many
cells hold no ground return at all (lidar rings spread with range; nothing under an overhang
reaches the road) and would otherwise treat the structure above them as their own floor.
A stated limitation: an overhang wider than that window still reads as an obstacle — safe
direction, but wrong.

### The shield's speed limit vs a real human driver

![shield-permitted speed vs driven speed](docs/speed_profile.png)

`max_safe_speed` bisects `can_stop_safely` to express the shield as a speed limit, directly
comparable to what the driver actually did. Over the whole drive (speed cap raised to 35 m/s
so geometry rather than the cap is what binds):

- median permitted **35.0 m/s** vs driven 9.0 m/s — on open road the shield does not bind
- it would have slowed the driver on **3 of 443 frames**

Those 3 frames are informative. At frame 294 the car threads a gap between parked cars with
only **0.03 m** of modelled clearance — well inside the 0.30 m safety margin — so the shield
permits just 3.4 m/s where the human drove 10.0 m/s. That is not a bug; it is the price of a
hard guarantee computed with a conservative footprint and a straight-line braking path, while
the human steered through a curve they could see was fine.

**A geometry bug this exposed.** The Velodyne is roof-mounted **0.81 m ahead of the rear
axle**, and the bicycle model's pose *is* the rear axle. Placing the vehicle at the lidar
origin pushed a 4.77 m car most of a metre too far forward and corrupted every clearance
query. Fixing it (plus tightening the disc cover from 3 to 5 discs) moved median permitted
speed 19.8 → 35.0 m/s and dropped binding frames from 15 → 3.

---

## Layout

```
src/kitti_nav/vehicle.py     bicycle model, footprint geometry, braking shield
src/kitti_nav/kitti.py       KITTI raw access: calibration, stereo, OXTS ground truth
src/kitti_nav/stereo.py      SGBM disparity -> metric depth
src/kitti_nav/odometry.py    ORB + PnP visual odometry, trajectory evaluation
src/kitti_nav/bev.py         lidar -> BEV occupancy + distance field (ObstacleField)
scripts/fetch_kitti.py       dataset download (data is never committed)
scripts/eval_odometry.py     run VO over a drive, score it, plot it
scripts/render_bev.py        BEV/shield visualisations
tests/                       pure-NumPy unit tests; dataset tests skip when data is absent
```

## Attribution

Built on KITTI (Geiger et al., **CC BY-NC-SA 3.0 — non-commercial**, not redistributed here)
and pykitti (MIT). Full credits, licenses, and citations in [`ATTRIBUTION.md`](ATTRIBUTION.md).
Code in this repo is MIT; that covers the code only, not the dataset.
