# Learned planning behind a braking safety shield — results

All numbers produced by `scripts/eval_policies.py` and `scripts/eval_seeds.py` on this
machine (CPU only). Every PPO policy was trained for 600k steps on **synthetic** obstacle
fields only — the KITTI tables are therefore a transfer test onto real recorded street
geometry, never trained on.

Training cost, measured across the 10 runs of the seed sweep: raw PPO **1.5–1.7 min**,
shield-in-the-loop PPO **4.8–5.4 min** (8 parallel envs, CPU).

## Synthetic scenes (200 episodes)

| policy | success | collisions | reward | steps |
| --- | ---: | ---: | ---: | ---: |
| gap-following heuristic | 24% | 151 | 3.2 | 31 |
| gap-following + shield | 39% | **0** | 21.9 | 34 |
| PPO (raw) | 66% | 50 | 29.7 | 37 |
| PPO (raw) + shield at eval | 70% | **0** | 35.5 | 41 |
| PPO trained *through* the shield | 68% | **0** | 34.7 | 38 |

## Real KITTI scenes — transfer, never trained on (200 episodes)

| policy | success | collisions | reward | steps |
| --- | ---: | ---: | ---: | ---: |
| gap-following heuristic | 66% | 68 | 26.0 | 27 |
| gap-following + shield | 66% | **0** | 31.5 | 29 |
| PPO (raw) | 78% | 42 | 32.3 | 28 |
| PPO (raw) + shield at eval | 75% | **0** | 34.5 | 29 |
| PPO trained *through* the shield | 78% | **0** | 35.4 | 29 |

## What holds

**The shield's guarantee is absolute across every run: 0 collisions, always.** That covers a
weak heuristic that crashes 151 times unaided, a learned policy that crashes 50 times
unaided, and (in the unit tests) uniformly random actions. The guarantee does not depend on
the policy being any good, which is the entire point of a runtime shield.

**Learning clearly beats the heuristic.** 24% → 66% success on synthetic scenes, 66% → 78%
on KITTI.

**The transfer to real geometry works.** A policy that has only ever seen random circles
handles recorded streets. Note the KITTI numbers are *higher* than synthetic, which is not
evidence the policy is better there — a recorded street has a genuinely drivable corridor
(a car drove down it), whereas the synthetic generator scatters obstacles arbitrarily,
including directly across the path. Synthetic is the harder distribution.

**Shielding a learned policy is nearly free**, and on synthetic scenes it *improves* success
(66% → 70%). That reads oddly until you notice collisions terminate an episode as a failure:
the shield converts would-be crashes into continued driving that sometimes still reaches the
goal. Safety and success are not in tension here the way they are for the heuristic.

## What does not hold — a negative result, tested across 5 training seeds

**Training through the shield did not beat bolting it on at evaluation, contradicting what
`gsplat-rt` found.** There, the shield-in-the-loop policy strictly dominated the raw one on
every axis (100%/0 collisions/56 steps vs 98%/4/58).

This claim is about a *training method*, so the unit of replication has to be the training
run, not the episode. Five independently seeded policies were trained per configuration
(`scripts/eval_seeds.py`), each scored on the **same** 200 evaluation scenes so scene
difficulty cannot contribute to the difference.

### Synthetic scenes

| policy | per-seed success | mean ± sd | collisions |
| --- | --- | --- | ---: |
| PPO (raw) + shield at eval | 70.5 / 71.0 / 73.5 / 71.0 / 73.5 | **71.9% ± 1.5%** | 0 |
| PPO trained through shield | 67.5 / 73.5 / 75.0 / 71.0 / 73.0 | **72.0% ± 2.9%** | 0 |

Difference **+0.1%**, 95% CI **[−3.5%, +3.7%]**, Welch *t* = 0.07, **p = 0.95**.

### Real KITTI scenes

| policy | per-seed success | mean ± sd | collisions |
| --- | --- | --- | ---: |
| PPO (raw) + shield at eval | 75.0 / 77.5 / 76.5 / 75.5 / 76.5 | **76.2% ± 1.0%** | 0 |
| PPO trained through shield | 77.5 / 78.5 / 75.5 / 78.0 / 77.0 | **77.3% ± 1.2%** | 0 |

Difference **+1.1%**, 95% CI **[−0.5%, +2.7%]**, Welch *t* = 1.63, **p = 0.14**.

### Reading this honestly

**The negative result survives replication.** Neither comparison is significant at 0.05.

**But the confidence interval says more than the p-value does.** On KITTI the interval caps
any real effect at **+2.7 percentage points**, and 4 of 5 seeds do favour in-loop training —
so a *small* genuine benefit is not excluded, and "not significant" is not "no effect." What
*is* excluded is an effect anywhere near the magnitude `gsplat-rt` reported (95% → 100%
success, 79 → 56 steps). That is the defensible claim: **not that in-loop training never
helps, but that its large win there does not reproduce here.**

**Seed-to-seed spread turned out to be small** — 1.0 to 2.9 points, well below the deep-RL
norm where seed variance routinely swamps algorithmic differences (Henderson et al. 2018).
That makes this comparison better powered than the seed count alone suggests, and it is why
5 seeds were enough to bound the effect usefully.

### Why the difference from gsplat-rt

Visible in the shielding rows above: eval-time shielding already costs this policy nothing
(on synthetic scenes it *helps*), so there is no penalty left for in-loop training to
recover. In `gsplat-rt` the shield *did* cost real performance when bolted on (98%/4 →
95%/0, and 58 → 79 steps), and closing that gap is exactly what training through it achieved.
A shield that is already free leaves nothing on the table.

That is itself explainable: this shield modulates a *continuous* throttle/brake channel and
usually intervenes by scaling acceleration slightly, whereas the diff-drive shield frequently
forced a discrete "stop and rotate in place" that the unshielded policy had never planned for.

**Remaining caveats.** One drive, one hyperparameter set, 600k steps, 5 seeds. Absolute
success rates also shift with the evaluation scene set (a 600-episode set scored the same
seed-0 policies ~5 points lower than the 200-episode set), which is why only within-table
comparisons on identical scenes are meaningful.

## Reproduce

```bash
python3 scripts/train_ppo.py --steps 600000 --out models/ppo_raw
python3 scripts/train_ppo.py --steps 600000 --shield --out models/ppo_shielded
python3 scripts/eval_policies.py --episodes 200

# the 5-seed comparison behind the negative result
for s in 0 1 2 3 4; do
  python3 scripts/train_ppo.py --steps 600000 --seed $s --out models/ppo_raw_s$s
  python3 scripts/train_ppo.py --steps 600000 --seed $s --shield --out models/ppo_shielded_s$s
done
python3 scripts/eval_seeds.py --seeds 5 --episodes 200
```
