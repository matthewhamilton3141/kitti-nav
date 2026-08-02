# kitti-nav — session handoff (2026-08-02)

Plain-English "pick up here." The README is the polished public account; this is the working
notes — what was decided and why, what broke, and what is actually left.

## What this is

AV navigation on real recorded drives. Replay a KITTI sequence, build the occupancy/BEV
representation a real AV stack plans on, and drive a vehicle through it behind a **braking-
aware safety shield** that provably cannot admit a collision it could have braked out of.

Public: <https://github.com/matthewhamilton3141/kitti-nav>. Runs entirely on the Mac — pure
NumPy/OpenCV core, CPU-only RL. **No Brev box needed for anything here**, which was a
deliberate constraint (A10G credits were out as of 2026-08-01 and remain unconfirmed).

## Where it came from

The "Option B" branch out of `gsplat-rt`'s 2026-08-01 handoff, after the pivot to *"I don't
want a portfolio piece, I just want to make something cool."* `gsplat-rt` itself is complete
and untouched (its own Option A — browser splat scan + WebGL physics — was **not rejected,
just not picked first**; details in that repo's HANDOFF.md, which still has uncommitted
edits from before this work started).

The seed idea was `gsplat-rt`'s nav capstone: a hard safety shield wrapping any policy. The
question was whether it survives contact with *driving*. Mostly it did — but almost nothing
ported unchanged, and one of its headline results did not reproduce (below).

## Status — all merged to `main`, 5 commits, **105 tests green**

| milestone | state | measured |
| --- | --- | --- |
| Bicycle (Ackermann) model + braking shield | done | — |
| KITTI raw loading (stereo + lidar + OXTS) | done | drive 0009: 447 frames, 332.8 m, ≤11.4 m/s |
| Stereo VO (ORB + PnP) | done | **3.55% drift** over 332.8 m, 26.4 fps CPU |
| Lidar → BEV occupancy + distance field | done | **1.8% occupied, 1.9 ms/frame** |
| Shield on real lidar | done | **6 ms/frame**; binds on 3/443 frames |
| Learned planner (PPO) behind the shield | done | **78% success, 0 collisions** on real KITTI |
| Shield-in-the-loop, 5-seed replication | done | **negative result** (see below) |

Full numbers: `README.md` and `scripts/RESULTS.md`.

## Decisions already made — don't re-litigate these

- **Occupancy/BEV is the live representation, not splats.** Real stacks plan on occupancy;
  splatting's AV role is offline digital twins for closed-loop testing.
- **The shield verifies a full braking rollout, not one step.** A one-step lookahead is
  *unsound* at driving speed (12 m/s needs ~16 m to stop; one 0.1 s step covers 1.2 m).
  `test_one_step_shield_crashes_where_braking_shield_stops` demonstrates the naive port
  driving into a wall.
- **No Sim(3) alignment in VO evaluation.** Correct for monocular, self-flattery for stereo —
  it would absorb real scale error. A test pins this.
- **Grid-native `ObstacleField`, never fitting circles to occupancy.** Circles would discard
  exactly the arbitrary shape occupancy is good at.
- **`height_diff` (local per-cell) ground removal**, chosen by measurement over a fixed plane:
  1.82% vs 6.23% occupied, 6.04 m vs 2.58 m clearance at 30 m. The plane mode's extra cells
  are *road* drifting out of the band — phantom obstacles that brake the car for open road.
- **Ray observations, not an occupancy crop** — `gsplat-rt` measured the crop as marginal.
- **CPU for RL.** Batches are far too small to amortise GPU launches; 6191 fps unshielded.
- **Trained policies are gitignored** — reproducible in minutes.

## Bugs found — all instructive, all fixed

1. **Shield passed steering through unconditionally → unsound.** The braking certificate is
   issued under the *current* wheel angle; a different commanded angle can curve full braking
   into an obstacle the certificate never covered. Now falls back to the last certified steer.
   **Found by a randomised rollout test, not by hand.**
2. **Rear axle ≠ lidar origin.** The Velodyne is roof-mounted **0.81 m ahead** of the rear
   axle, and the bicycle pose *is* the rear axle. Placing the car at the lidar origin pushed a
   4.77 m vehicle most of a metre too far forward and corrupted every clearance query. Fixing
   it (+ 3→5 footprint discs) moved median permitted speed 19.8 → 35.0 m/s and dropped frames
   where the shield overrides the driver from 15 → 3.
3. **443 lidar scans for 447 images.** Real datasets have holes; bound lidar loops on
   `n_velodyne`.
4. **36% of episode starts were inevitable-collision states** — spawned faster than any
   braking could save. This surfaced as *shield-on collisions* and was very easy to misread as
   the shield failing. `certifiable_start()` clamps spawn speed; collisions then go to exactly
   0. **Lesson: check start certifiability before blaming a shield.**
5. **A confidence interval that measured the wrong variance.** 600 eval episodes measure
   *scene* luck; the claim was about a *training method*, whose replicate is the training run.
   Fixed by the 5-seed sweep.

## The negative result (keep it honest)

**Training through the shield did not beat bolting it on at evaluation** — contradicting
`gsplat-rt`, where shield-in-the-loop strictly dominated (100%/0/56 vs 98%/4/58).

5 seeds × 200 episodes, 0 collisions in all 10 runs:

| scenes | at-eval | in-loop | difference |
| --- | --- | --- | --- |
| synthetic | 71.9% ± 1.5% | 72.0% ± 2.9% | +0.1%, CI [−3.5, +3.7], p = 0.95 |
| KITTI | 76.2% ± 1.0% | 77.3% ± 1.2% | +1.1%, CI [−0.5, +2.7], p = 0.14 |

**The precise claim:** not that in-loop training never helps, but that its *large* win in
`gsplat-rt` does not reproduce here. The CI caps any real effect at +2.7 points, and 4/5 seeds
do favour in-loop — so a small genuine benefit is **not** excluded. "Not significant" ≠ "no
effect." Why: eval-time shielding already costs this policy ~nothing (on synthetic it *helps*,
since a collision ends the episode as a failure), so there is no penalty left to recover.

Seed spread was small (1.0–2.9 pts), unusually low for deep RL. Sweep cost ~35 min for 10 runs.

## Known gaps — read this before picking next work

- **⚠ VO and BEV are not connected.** This is the biggest architectural hole. `odometry.py`
  produces poses; `bev.py` builds a grid from a *single frozen scan*. Nothing fuses scans
  across frames into a persistent map. Right now they are two good components that don't talk.
- **Everything is static.** No moving actors — the single largest gap versus real AV. KITTI
  raw has tracklet annotations for some drives.
- **The shield never steers evasively.** It picks between the commanded steer and the held
  steer and otherwise brakes; it will brake for something it could have swerved around.
- **The footprint disc cover is conservative** — inflates the car ~0.12 m per side and ~0.55 m
  past each bumper. Real cost: at frame 294 it permits 3.4 m/s where the human drove 10.0,
  threading parked cars with 0.03 m modelled clearance.
- **VO has no bundle adjustment, keyframing, or loop closure** — error accumulates
  monotonically. 3.55% is respectable, not SOTA.
- **One drive, one hyperparameter set.**
- **Unknown space is treated as free** (`outside_is_free=True`) — only defensible while the
  grid covers the braking envelope; `covers_stopping_distance()` exists to assert it.

## What next — options

**My recommendation: (1) then (2).**

1. **Fuse VO poses + lidar into an accumulated BEV map.** Closes the gap above and makes the
   repo one coherent pipeline instead of two halves. Concretely: transform each scan into a
   common frame using the VO trajectory, accumulate occupancy over a sliding window, and let
   the planner drive the *accumulated* map. It also creates a real experiment — accumulated
   maps inherit VO drift, so this measures how odometry error propagates into planning safety,
   which is a genuinely interesting AV question and needs no new dependencies.
2. **Dynamic obstacles.** Parse KITTI tracklets (or synthesise moving actors) and extend the
   shield to reason about a moving obstacle's reachable set rather than a static one. This is
   where the safety argument gets properly hard — and where "AV" actually lives.
3. Evasive steering in the shield (search over steer candidates, not just two).
4. Strengthen VO: local bundle adjustment or keyframing; SuperPoint+LightGlue front-end beat
   ORB in `gsplat-rt` (3.5 cm vs 5.7 cm ATE on TUM) but is box-gated.
5. More drives / seeds to firm up generalisation claims.

## Environment / repo facts

- Dev Mac runs **everything**: numpy, OpenCV, and the RL stack — **torch 2.13 + sb3 2.9 +
  gymnasium 1.3, CPU**. No GPU needed anywhere in this repo.
- Data: `data/kitti_raw/` is **gitignored, never committed** (KITTI is CC BY-NC-SA 3.0,
  non-commercial). Refetch: `python3 scripts/fetch_kitti.py`. Drive 0009 is ~1.7 GB.
- `models/` is gitignored; retrain via `scripts/train_ppo.py` (1.5 min raw, ~5 min shielded).
- Tests: `python3 -m pytest tests/ -q`. Dataset-backed tests skip cleanly without KITTI; the
  env core is pure NumPy and tests with no RL stack installed.
- Attribution is policy, not decoration: `ATTRIBUTION.md` lists every upstream with license;
  adapted files carry provenance headers saying what changed and why. **KITTI is
  non-commercial**; pykitti/sb3/gymnasium are MIT, OpenCV Apache-2.0, torch BSD-3.
- Workflow so far: direct commits to `main` (solo repo, no PRs yet).
