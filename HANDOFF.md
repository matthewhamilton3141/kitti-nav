# kitti-nav — session handoff

Concise "pick up here." The long, detailed session-by-session record lives in
[`docs/HANDOFF-archive.md`](docs/HANDOFF-archive.md) — dip into it for the *why* behind any
decision, bug, or measured non-result. This file is the current state, the next step, and a
runbook for running/watching everything yourself.

---

## ▶ Where things stand

- **Branch `feat/map-fusion` = `main`** (they coincide; both pushed to origin). Latest commit
  `e1a8958`. Working tree clean.
- **204 tests green**: `python3 -m pytest tests/ -q` (dataset-backed tests skip cleanly without
  KITTI; the env core is pure NumPy). The CARLA bridge's pure parts test with no `carla`/GPU.
- **Two KITTI drives on disk** (gitignored): 0009 (default, city) and 0093 (cross-drive, busy).
- **Direction has shifted to closed-loop.** The offline arc is a good stopping point; the new
  thread is running the BEV + shield in a real closed loop via **CARLA + NuRec** — see the next
  step. A client-side bridge (`src/kitti_nav/carla_bridge.py`) already exists; the rest needs a
  GPU box. Paused here.

> **To run or watch the tests/evals → jump to [▶ Run & watch — runbook](#-run--watch--runbook)
> below.** It has every command, its runtime, what "good" looks like, and how to `tail -f` a long
> eval so you can watch it stream without blocking your terminal.

**What this is.** AV navigation on real recorded KITTI drives: replay a sequence, build the
lidar→BEV occupancy a real stack plans on, and drive a bicycle-model vehicle through it behind a
**braking-aware safety shield** that provably cannot admit a collision it could have braked out
of. Pure NumPy/OpenCV core + CPU RL; runs entirely on the Mac. Public:
<https://github.com/matthewhamilton3141/kitti-nav>.

## What's done (the arc)

| area | state | headline |
| --- | --- | --- |
| Bicycle model + braking shield | done | verifies a full braking rollout, not one step (a one-step port drives into a wall) |
| Stereo VO (ORB+PnP) | done | 3.55% drift / 332 m, metric by construction (no Sim(3) self-flattery) |
| Lidar→BEV occupancy + distance field | done | `height_diff` ground removal, chosen by measurement |
| Shield on real lidar + PPO behind it | done | **78% success, 0 collisions** on real KITTI |
| Shield-in-the-loop, 5-seed sweep | done | **negative result**: gsplat-rt's big in-loop win does *not* reproduce |
| VO+lidar fused accumulated map | done | 1.9× the scene mapped at 5 scans; shield still 0 collisions |
| Free-space carving (occ/free/unknown) | done | de-smears movers 9% at the safe point; honest map metric non-result |
| Cautious speed cap + hole-closing | done | honest unknown map drivable: 4% → ~37%, shield 0 collisions |
| Object tracklets | done | 0009 has 12 real movers (not "too static") |
| Dynamic shield (brakes for where a car is *going*) | done | binds on 8% of real mover-frames |
| Closed-loop dynamic env | done | static shield drives into a crossing car 51×, dynamic 4× (0009) |
| Evasive steering (opt-in) | done | swerves out of an open-road ICS; marginal on cluttered real traffic |
| Training on real KITTI geometry | done | held-out shielded success 71% → **78% (+7 pts)** |
| **Cross-drive validation (drive 0093)** | done | **shield transfers (0 collisions on an unseen drive); trained policy does *not* (59%≈58%)** |
| BEV render mirror fix | done | render/animation BEV was L/R mirrored (`imshow` col-0 vs `extent`); un-mirrored, occ/trajectory overlays realigned |
| **Closed-loop CARLA + NuRec bridge** | scaffold | **BEV + shield as a CARLA ego controller; handedness + "brakes for a wall" tests green; PPO hook stubbed** |

Full numbers: [`README.md`](README.md) and [`scripts/RESULTS.md`](scripts/RESULTS.md).

## ▶ The next step

**Close the loop in CARLA + NuRec.** Everything measured so far is *open-loop on frozen geometry*:
the ego solves a nav problem posed on a recorded scan, but its steering never changes what the
sensor sees next — and a recorded log can't render the viewpoints the driver didn't visit. The
move is to a simulator that renders novel viewpoints: **CARLA** (v0.9.16) with NVIDIA **NuRec**
neural reconstruction (3D Gaussian splatting of real drives), so the exact BEV + shield code drives
in a genuine closed loop where the policy's own steering shapes the next observation.

The client-side bridge already exists — `src/kitti_nav/carla_bridge.py` (CARLA lidar → `BEVGrid` →
`safety_shield` → CARLA control), with its pure parts tested on the Mac (no `carla`, no GPU). What
remains needs a **GPU box** (Linux, NVIDIA RTX — CARLA + NuRec do *not* run on the Mac; the plan is
a cloud instance):

1. Provision the box; install CARLA 0.9.16 + NuRec + the NVIDIA Container Toolkit; pull the ~200 GB
   NVIDIA Physical AI NuRec dataset. **Use the ready dataset first** to prove the loop; reconstruct
   a KITTI drive (COLMAP → 3DGUT) later so closed-loop numbers match the offline ones.
2. Calibrate the two approximations flagged in the code: CARLA's throttle→accel map (currently
   proportional) in `shield_to_carla_control`, and `--rear-axle-x` to the sensor mount.
3. Swap `ForwardGoalPlanner` for the trained PPO policy behind the `BasePlanner` seam and reproduce
   the offline **"0 collisions"** guarantee in closed loop.

**Still-open correctness gap (secondary — was the previous "next step").** Fusion spawn-safety on
fast/busy drives: on 0093 the **fused** 5-scan map spawns the ego already in collision (27/30
frames; single-scan 0/30) — the wide window smears *elevated* geometry (z median 1.12 m, not ground
removal) into the ego's spawn footprint. Carving can't retire it (height gate spares elevated
returns), nor a post-fusion ego-box clear (the smear surrounds the ego, isn't its body). Principled
fix: a **dedicated dynamic mover-forgetting channel** (tracklet labels, or a two-frame occupancy
diff) that leaves static geometry untouched. Interim workaround in place:
`eval_kitti_trained.py --fused-window 1` scores single-scan, which is spawn-safe.

Other options: seed-sweep the KITTI-trained policies (+7 pts is one seed); train on
`KittiDynamicScenes`; a third drive. See the archive's "What next" for the full menu.

---

## ▶ Run & watch — runbook

All commands from the repo root. `PYTHONPATH` is not needed for scripts (they add `src/`
themselves); it *is* needed for ad-hoc `python3 -c` against the package.

### Tests (fast — watch them stream)

```bash
python3 -m pytest tests/ -q            # ~35 s, 197 pass; quiet dots
python3 -m pytest tests/ -v            # same, but prints every test name as it runs
python3 -m pytest tests/test_vehicle.py -v -k evasive   # just the evasive-steering tests, ~0.1 s
```

Good = `197 passed`. Dataset tests self-skip if KITTI isn't downloaded — not a failure.

### The evals (slow — minutes; these are what to "watch running")

Each prints a table at the end. They build BEV grids per frame (fused = distance transform), so
first frames are slow, then cache-warm. Run in the foreground to watch; expected runtimes on the
Mac in comments.

```bash
# static shield vs learned policy, synthetic + KITTI single-scan + fused (the core tables)   ~15 min
python3 scripts/eval_policies.py --episodes 200

# dynamic shield closed-loop: static vs dynamic on 0009's crossing traffic                    ~1 min
python3 scripts/eval_dynamic_policies.py --episodes 200
#   good: static shield "drove in" ~51, dynamic ~4, both far below unshielded
#   add --evasive 15 for the swerve row (slow: ~8 min; gain is marginal, that's the finding)

# train on real KITTI geometry, then score on the held-out stretch of the drive        ~3+10 min train
python3 scripts/train_ppo.py --scenes kitti-fused --steps 600000 --out models/ppo_kitti_raw
python3 scripts/train_ppo.py --scenes kitti-fused --shield --steps 600000 --out models/ppo_kitti_shielded
python3 scripts/eval_kitti_trained.py --episodes 200 \
  --model synthetic=models/ppo_shielded.zip --model kitti=models/ppo_kitti_shielded.zip     # ~4 min
#   good: kitti-shielded 78% vs synthetic 71%, 0 collisions in every shielded column

# cross-drive: fetch the second drive, then score 0009-trained policies on all of it (single-scan)
python3 scripts/fetch_kitti.py --drive 0093 --tracklets   # ~1.6 GB download (needs network)
python3 scripts/eval_kitti_trained.py --drive 0093 --all-frames --fused-window 1 --episodes 200 \
  --model synthetic=models/ppo_shielded.zip --model kitti-0009=models/ppo_kitti_shielded.zip  # ~4 min
python3 scripts/eval_dynamic_policies.py --drive 0093 --episodes 200                            # ~1 min
#   good: 0 collisions shielded on 0093; dynamic shield still cuts "drove in" (14 -> 4)
```

**Watching a long eval without blocking your terminal:** run it redirected and `tail -f` it:

```bash
python3 scripts/eval_policies.py --episodes 200 > /tmp/eval.log 2>&1 &
tail -f /tmp/eval.log        # Ctrl-C to stop watching; the job keeps running
```

### See the shield actually drive (pictures & video)

```bash
python3 scripts/render_bev.py --frame 294 --speed-profile docs/speed_profile.png
python3 scripts/eval_odometry.py --plot docs/trajectory.png
python3 scripts/animate_drive.py --drive 0009 --start 40 --stop 200 --out docs/drive_scene.mp4
python3 scripts/animate_rollout.py --frame 60 --out docs/drive.mp4   # shielded PPO rollout movie
```

The BEV mini-map in these was left-right **mirrored** until `2db8a7c` (`imshow` draws array col-0
at `extent[0]`, but the rasteriser puts world-right there) — now un-mirrored; the camera's left is
the map's left. If a render ever looks flipped again, that's the extent/axis coupling in the render
scripts, not the perception.

### Closed-loop in CARLA (needs a GPU box — not runnable on the Mac)

The bridge's pure parts test locally; the loop itself needs CARLA + NuRec on a Linux NVIDIA-RTX box.

```bash
python3 -m pytest tests/test_carla_bridge.py -q   # sign conventions + "brakes for a wall", no carla/GPU
# on the box, with CARLA 0.9.16 + a NuRec scene running:
python3 -m kitti_nav.carla_bridge --host <box-ip> --port 2000 --target-speed 8
```

---

## Key facts & gotchas (compact)

- **`main` and `feat/map-fusion` coincide.** History was consolidated by a fast-forward; commit
  straight ahead. End commit messages with the `Co-Authored-By` / `Claude-Session` trailers.
- **Models (`models/*.zip`) and data (`data/kitti_raw/`) are gitignored.** Retrain in minutes;
  KITTI is CC BY-NC-SA (non-commercial), never committed. Drive 0009 ≈ 1.7 GB, 0093 ≈ 1.6 GB.
- **RL stack:** torch 2.13 + sb3 2.9 + gymnasium 1.3, **CPU only** (batches too small to amortise
  GPU launches). Env core needs none of it.
- **Everything additive is off by default and measured** (carving, unknown-blocks, cautious cap,
  evasive steering, fused maps) so every prior number reproduces. Keep that discipline.
- **Numbers are re-measured, not inherited.** If a doc figure disagrees with what you measure,
  trust the measurement and correct the doc. Several claims were corrected downward this way.
- **Two open correctness/perception gaps:** fusion spawn-safety on fast/busy drives (now the
  *secondary* next step), and label-free obstacle velocity is ~95% false-positive (so the dynamic
  shield runs on tracklet motion, not label-free — measured, not assumed).
- **CARLA + NuRec is a GPU-box job.** The bridge (`carla_bridge.py`) is deliberately split so the
  shield/BEV/handedness/actuator logic is pure and Mac-testable; `import carla` is lazy, only inside
  `CarlaBridge`. CARLA (Unreal, left-handed, +y right, +steer right) ↔ kitti-nav (right-handed,
  +y left, +steer left): the two sign flips live in `carla_lidar_to_velodyne` and
  `shield_to_carla_control`, pinned by tests. The PPO policy plugs in behind the `BasePlanner` seam.

For anything not covered here — the bug write-ups, the rejected approaches, the per-session
reasoning — see [`docs/HANDOFF-archive.md`](docs/HANDOFF-archive.md).
