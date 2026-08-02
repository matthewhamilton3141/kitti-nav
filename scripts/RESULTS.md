# Learned planning behind a braking safety shield — results

All numbers produced by `scripts/eval_policies.py` on this machine (CPU only). Both PPO
policies were trained for 600k steps on **synthetic** obstacle fields only — the KITTI table
is therefore a transfer test onto real recorded street geometry, never trained on.

Training cost: raw PPO **1.5 min**, shield-in-the-loop PPO **5.2 min** (8 parallel envs, CPU).

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

## What does not hold — a negative result

**Training through the shield did not dominate eval-time shielding, contradicting what
`gsplat-rt` found.** There, the shield-in-the-loop policy strictly dominated the raw one on
every axis (100%/0 collisions/56 steps vs 98%/4/58). Here the two are statistically
indistinguishable — at 600 episodes on synthetic scenes:

| policy | success (95% CI) | collisions | steps |
| --- | --- | ---: | ---: |
| PPO (raw) + shield at eval | 66.0% ± 3.8% | 0 | 40 |
| PPO trained through the shield | 67.7% ± 3.7% | 0 | 39 |

The confidence intervals overlap heavily, so the ~1.7-point gap is noise. Reporting this as
a win would have been easy and wrong.

The likely reason is visible in the row above it: eval-time shielding already costs the
learned policy nothing, so there is no penalty left for in-loop training to recover. In
`gsplat-rt` the shield cost the policy real performance when bolted on at evaluation
(98%/4 → 95%/0, and 58 → 79 steps), which is exactly the gap training through it closed.
A shield that is already nearly free leaves nothing on the table.

That difference is itself explainable: this shield modulates a *continuous* throttle/brake
channel and usually intervenes by scaling acceleration slightly, whereas the diff-drive
shield frequently forced a discrete "stop and rotate in place" that the unshielded policy had
never planned for.

**Caveats.** Single seed per configuration, one KITTI drive, 600k steps. The claim being made
is only the negative one — that in-loop training showed no measurable advantage *here* —
which is much weaker than a claim that it never helps.

## Reproduce

```bash
python3 scripts/train_ppo.py --steps 600000 --out models/ppo_raw
python3 scripts/train_ppo.py --steps 600000 --shield --out models/ppo_shielded
python3 scripts/eval_policies.py --episodes 200
```
