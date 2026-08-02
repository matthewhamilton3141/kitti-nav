# kitti-nav — session handoff (2026-08-02, second session)

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

## Status — on `main`, **137 tests green**

| milestone | state | measured |
| --- | --- | --- |
| Bicycle (Ackermann) model + braking shield | done | — |
| KITTI raw loading (stereo + lidar + OXTS) | done | drive 0009: 447 frames, 332.8 m, ≤11.4 m/s |
| Stereo VO (ORB + PnP) | done | **3.55% drift** over 332.8 m, 26.4 fps CPU |
| Lidar → BEV occupancy + distance field | done | **4.3% occupied, 2.1 ms/frame** |
| Shield on real lidar | done | **6 ms/frame**; binds on 3/443 frames |
| Learned planner (PPO) behind the shield | done | **78% success, 0 collisions** on real KITTI |
| Shield-in-the-loop, 5-seed replication | done | **negative result** (see below) |
| **VO poses + lidar fused into an accumulated map** | **done this session** | **1.86× the scene mapped at 5 scans** |
| **Planner driving the fused map** | **done this session** | **shield still 0 collisions; success −12 to −14 pts** |

Full numbers: `README.md` and `scripts/RESULTS.md`.

## This session (2026-08-02): the VO↔BEV gap is closed

The "biggest architectural hole" the previous handoff named is fixed. `src/kitti_nav/mapping.py`
transforms scans through estimated poses into the current Velodyne frame and rasterises them
together; `scripts/eval_mapping.py` measures single-scan vs GT-pose vs VO-pose maps.

**Three findings worth not re-deriving:**

1. **Global ATE is the wrong statistic for map fusion.** VO ends 12.06 m off (3.55%), which
   sounds fatal, but a fused map never composes poses beyond its own window. The governing
   number is the *relative* pose error over that window — 24 cm at 5 scans. Two orders of
   magnitude apart, and it is why VO-pose maps match GT-pose maps almost exactly up to 5 scans
   (permitted speed 13.37 vs 13.36 m/s).
2. **The single-scan map was optimistic because it was blind.** It permits 16.38 m/s where the
   5-scan map permits 13.36, while missing **44%** of that map's occupied cells. Fusion makes
   the planner slower and better informed — do not read the speed drop as a regression.
3. **Operating point is 5 scans.** Between 5 and 10 the unsafe direction (VO map permitting
   more than the GT map) jumps 3/36 → 9/36 frames, worst excursion +0.57 → +4.39 m/s.

**The bug it surfaced — read this before touching `bev.py`.** Accumulation first cut permitted
speed 25.1 → **2.2 m/s**, with phantom obstacles inside the car's own footprint, *with perfect
poses*. The roof-mounted Velodyne sees the ego car (hood, roof rails, mirrors). In a single scan
those cells hold *only* bodywork, so the per-cell ground estimate treats the bodywork as ground
and calls the cell free — **the single-scan map was right by accident**. Fusion supplies the road
surface underneath from an earlier viewpoint, and the 0.81 m difference reads as an obstacle.
Fixed by ego self-filtering (`drop_ego_returns`, box `KITTI_EGO_BOX` measured not derived, since
the body rectangle is anchored 0.32 m off the lidar centreline while self-returns are symmetric
about the sensor and reach 1.5 m laterally). Costs 41 returns and **zero** occupancy cells on a
single scan, so it is on by default everywhere.

Two dead ends, recorded so they are not retried: `min_support` (requiring several returns above
ground per cell) looked like the fix but could not separate phantom from real — restoring
single-scan parity needed a threshold that pushed occupancy *below* the single-scan baseline,
i.e. deleting real geometry. And detecting ego returns by sensor-frame persistence needs a
lateral bound: roadside structure persists too while the car holds its lane, and without it the
audit returns a kerb line at y ≈ −2.3 m present in 100% of scans.

**The planner now drives the fused map** (`KittiScenes(map_config=MapConfig(window=5))`,
`eval_policies.py --fused-window`). Default is still a single scan, so every earlier number
stays reproducible — and the single-scan column re-ran **bit-identically** (66%/68, 78%/42,
78%/0), which is the check that the ego self-filter really is a no-op on the un-fused path.

| policy | single scan | fused (5 scans) |
| --- | --- | --- |
| gap-following | 66% / 68 coll | 59% / 82 |
| gap-following + shield | 66% / **0** | 54% / **0** |
| PPO (raw) | 78% / 42 | 66% / 67 |
| PPO (raw) + shield at eval | 75% / **0** | 62% / **0** |
| PPO through shield | 78% / **0** | 64% / **0** |

**The headline is the shield column.** Same policy weights, no retraining, no notice that the
map changed — unshielded collisions rise (42→67) and success falls 12–14 points, while every
shielded row stays at **exactly 0**. The shield is not a policy; it re-derives a braking
certificate from whatever occupancy it is handed, so more obstacles make it more conservative,
never less sound. This is the strongest evidence in the repo that the guarantee is a property
of the method rather than of the scenes it was tuned against.
**Do not read the success drop as a regression** — the single-scan map was missing 44% of the
fused map's occupied cells, so the old numbers were partly measuring the map's blindness.
Both tables are kept: the single-scan one is what the 5-seed negative result was measured
against. On fused maps in-loop training leads +2 pts (64 vs 62), same small same-signed gap
as single-scan (+3) — corroboration, not evidence; the seed sweep has **not** been re-run on
fused maps (~35 min if wanted).

**Also corrected this session:** the README's ground-removal table did not reproduce
(claimed 1.82%/6.23% occupancy and `height_diff` as the *faster* mode; actually 2.40%/3.65% at
frame 0 and `height_diff` is slower, 2.1 vs 1.3 ms). Corrected in place with a note. The paired
clearance figures did reproduce. Conclusion and default are unchanged.

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
  2.40% vs 3.65% occupied at frame 0 (4.33% vs 8.23% drive mean), 5.96 m vs 2.58 m clearance
  at 30 m, and 2.1 vs 1.3 ms — it is the *slower* mode, and the extra 0.8 ms is worth it. The
  plane mode's extra cells are *road* drifting out of the band — phantom obstacles that brake
  the car for open road. (Figures re-measured 2026-08-02; the previous 1.82%/6.23%/"faster"
  did not reproduce. Conclusion and default unchanged.)
- **Point-level, not grid-level, scan fusion.** Transform point clouds and rasterise once,
  rather than rasterising each scan and OR-ing the grids. The ground estimate *improves* with
  more returns per cell, which is exactly `height_diff`'s weak spot; OR-ing binary grids would
  bake each scan's ground-removal mistakes in permanently.
- **Ego self-filtering is on by default everywhere**, including the single-scan path, where it
  costs zero occupancy cells. See bug 6.
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
6. **The car mapped itself as a wall.** The roof-mounted Velodyne sees its own bodywork. In a
   single scan those cells hold *only* bodywork, so the per-cell ground estimate takes the
   bodywork as ground and calls them free — **right by accident**. Fusion supplies the road
   surface underneath from an earlier viewpoint, the 0.81 m difference reads as an obstacle,
   and phantom walls appear inside the vehicle footprint: permitted speed 25.1 → **2.2 m/s**,
   *with perfect poses*. Fixed by `mapping.drop_ego_returns`. **Lesson: a latent bug can be
   masked by a limitation, and removing the limitation is what exposes it — the single-scan
   map's blindness was hiding it.** Symmetrically, `min_support` (demand several returns above
   ground per cell) *looks* like the fix and is not: separating phantom from real needs a
   threshold that pushes occupancy below the single-scan baseline, i.e. deletes real geometry.

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

- **⚠ Dynamic actors are now the top gap, and fusion made it worse.** No moving actors, and
  accumulation smears the ones in the recording into trails. With ground-truth poses that is
  indistinguishable from revealed static geometry, so "1.86× the scene mapped" is an *upper
  bound* on the useful gain. **Free-space carving** — ray-cast each scan and clear what it saw
  through — is the standard fix, needs no new dependencies, and would also let the map forget
  obstacles that have moved. KITTI raw has tracklets for some drives.
- **No free/unknown distinction, and fusion raises the stakes.** `outside_is_free=True` was
  defensible for one scan covering the braking envelope; an accumulated map has real interior
  holes too. The largest optimistic (unsafe-direction) excursions measured are obstacles near
  `x_max = 50 m` that drift pushes across the boundary into assumed-free space.
- **The grid cannot certify above 21.2 m/s.** `sqrt(2 · 4.5 · 50)`. Any permitted speed above
  that is `outside_is_free` talking, not the sensor — which is why `eval_mapping.py` caps at 21
  while `render_bev.py` still uses 35. Worth reconciling.
- **The shield never steers evasively.** It picks between the commanded steer and the held
  steer and otherwise brakes; it will brake for something it could have swerved around.
- **The footprint disc cover is conservative** — inflates the car ~0.12 m per side and ~0.55 m
  past each bumper. Real cost: at frame 294 it permits 3.4 m/s where the human drove 10.0,
  threading parked cars with 0.03 m modelled clearance.
- **VO has no bundle adjustment, keyframing, or loop closure** — error accumulates
  monotonically. 3.55% is respectable, not SOTA.
- **One drive, one hyperparameter set.**
- **Policies are still *trained* on synthetic scenes only.** The fused-map column is pure
  transfer — no policy has ever been trained on an accumulated map. Training on fused KITTI
  geometry is untried and is the obvious way to recover the 12–14 points.
- **The seed sweep has not been re-run on fused maps.** The negative result stands on
  single-scan scenes; the fused table is one seed.

## What next — options

**My recommendation: (1) then (2).** Option 1 from the previous handoff is done.

1. **Free-space carving.** Ray-cast each scan to mark what it saw *through* as free, so
   dynamic actors stop smearing into permanent walls and the map gains a real
   free/unknown/occupied distinction instead of `outside_is_free`. Direct unblock for both top
   gaps above, no new dependencies, and both measurement harnesses already exist:
   `eval_mapping.py` would show it as the missed/phantom split improving at large windows, and
   `eval_policies.py --fused-window` would show whether it wins back some of the 12–14 points
   fusion cost (some of that cost is smeared traffic, i.e. obstacles that are not really there).
2. **Dynamic obstacles.** Parse KITTI tracklets (or synthesise moving actors) and extend the
   shield to reason about a moving obstacle's reachable set rather than a static one. This is
   where the safety argument gets properly hard — and where "AV" actually lives. Carving (1)
   first makes this much easier to evaluate.
3. Evasive steering in the shield (search over steer candidates, not just two).
4. Strengthen VO: local bundle adjustment or keyframing; SuperPoint+LightGlue front-end beat
   ORB in `gsplat-rt` (3.5 cm vs 5.7 cm ATE on TUM) but is box-gated. Note the mapping result
   lowers the priority of this: fusion is governed by *within-window* relative error, which is
   already 24 cm, not by the global drift bundle adjustment would fix.
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
