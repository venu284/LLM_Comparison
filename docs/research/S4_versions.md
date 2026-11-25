# S4 router versions log (S0.7)

Every router version tried on the development split, with its development numbers, including
versions that are discarded. Development figures, not results. Rules and definitions:
`docs/research/S4_spec.md`. Source: `router/experiments/s4_v1.py`, outputs
`router/experiments/out/s4_v1*.{csv,json}` (byte-reproducible; two runs compared).

Common to every row: complete case (O2); the scored populations of spec §2 under the recommended
P1 (b): headline 195 tasks / 180 problems / 3 families; all-eligible with an X13 record 459 / 355 /
13; sensitivity with an X13 record 279 / 244 / 6. Folds `stats.dev_folds(tasks, seed)`, fitted on
four folds and scored on the fifth. Gate seed 0; paired cluster-bootstrap 95% CIs over problems
(1,000 resamples). Gap in acceptance points, saving in percent of the reference's cost.

## v1 (2026-11-24): `router_v1`

Spec §3 with the recommendations P2-P6: predicted family refit per fold; beta-binomial shrinkage
with one shared kappa (ANOVA estimator); Rule A on the shrunk values with margin 0.02; the menu is
the eight models plus one two-rung cascade with no-signal cells escalated.

Fitted kappa per fold (seed 0): headline 0.82-1.02, all-eligible 1.22-1.34, sensitivity 1.00-1.15
(never infinite). The cascade chosen in every fold of every population at seed 0:
`Qwen3-Max > claude-opus-4-6` on the headline and sensitivity; on all-eligible
`kimi-k2.5 > Opus`, `gpt-5.4 > Opus` or `Qwen3-Max > Opus` by fold. Each reached the training best
single model's acceptance.

Gate seed 0:

| Population | Acceptance | Cost (m$/task) | vs best single: gap / saving | vs CellBest-pred: gap / saving | vs Rule A escalated: gap / saving | vs fixed cascade (esc.): gap / saving |
|---|---|---|---|---|---|---|
| headline | 84.62 [78.57, 89.64] | 5.28 | −0.51 [−1.58, 0.00] / 11.11 [5.70, 16.98] | 0.00 [0.00, 0.00] / −1.99 [−7.13, 1.98] | +0.51 [0.00, 1.55] / −2.60 [−4.86, −0.83] | −0.51 / 4.08 [2.79, 6.23] |
| all-eligible | 72.77 [67.90, 77.24] | 3.62 | +1.96 [−1.84, 5.95] / 29.15 [23.82, 34.41] | +0.22 [0.00, 0.68] / 2.97 [−0.96, 6.41] | +0.44 [−0.68, 1.72] / −9.01 [−15.79, −3.95] | +2.40 / 23.80 [18.07, 29.25] |
| sensitivity | 86.38 [82.27, 90.22] | 4.14 | +0.72 [−1.06, 2.54] / 23.77 [17.51, 30.00] | +0.36 [0.00, 1.10] / 3.38 [−4.60, 9.86] | +1.08 [0.00, 2.42] / −8.83 [−17.21, −2.85] | +0.72 / 8.46 [5.52, 11.80] |

- Cost note: the headline and sensitivity cascade starts with Qwen3-Max, whose released cost is a
  per-dimension constant (S1.7 data note; $0.000208 per headline task), so its per-task cost
  carries no within-dimension information.
- Verifier runs: 22 (headline), 81, 80. CPAT at the floor (best single's acceptance): not
  defined on the headline (84.62 < 85.13); 4.97 and 4.79 m$ per accepted task elsewhere.
- Regret against the cost-aware oracle: 12.31, 17.21 and 10.04 points.
- Seeds 1-4 (point estimates; CIs in `s4_v1_seeds.csv`): headline saving vs best single 9.93 to
  12.32 at a gap of −0.51 every seed; vs Rule A escalated saving −5.25 to −2.14. All-eligible
  saving vs best single 26.01 to 29.91; vs Rule A escalated −14.56 to −9.57. Sensitivity 20.89
  to 24.04; vs Rule A escalated −8.62 to −6.39.
- CRUXEval ≥ 0.8 (seed 0): the decisions are fitted under the main rule, so cost is unchanged;
  acceptance 98.97 (headline), 78.87, 96.42.
- P1 (a) variant (seed 0, no-record tasks included, labelled): all-eligible 966 tasks,
  acceptance 34.47, saving vs best single 50.74; sensitivity 294 tasks, 81.97, 12.36.
- Shuffled-label check (headline, S2.6 protocol, 20 shuffles): shuffled mean 46.56
  [43.40, 49.73], Random's expected 50.38 outside it. Diagnostic: shuffle seeds 20-119 give 51.33
  [49.75, 52.92] (inside). The test is marked `xfail(strict)` and left for the reviewer.
- Status: built; pending the reviewer's rulings on P1-P6.

## v1 accept (2026-11-24): `router_v1_accept`

v1 with no-signal cells accepted inside the cascade (spec §3.4). Logged as its own version.

| Population | Acceptance | Cost (m$/task) | vs best single: gap / saving | vs CellBest-pred: gap / saving | vs Rule A escalated: gap / saving |
|---|---|---|---|---|---|
| headline | 84.62 | 5.18 | −0.51 [−1.58, 0.00] / 12.84 [7.36, 18.63] | 0.00 / 0.00 | +0.51 / −0.60 [−3.53, 2.44] |
| all-eligible | 72.55 | 3.73 | +1.74 [−1.99, 5.71] / 26.98 [22.12, 31.80] | 0.00 / 0.00 | +0.22 / −12.35 [−18.14, −7.27] |
| sensitivity | 86.02 | 4.28 | +0.36 [−1.44, 2.40] / 21.11 [15.56, 26.58] | 0.00 / 0.00 | +0.72 / −12.64 [−17.22, −8.55] |

- At seed 0 it matches CellBest-pred on all three populations: the gap is 0 and the saving rounds
  to 0.00. On all-eligible the accept-mode cascade (`Opus > glm-5`) is chosen for a group in some
  folds (Opus first, verified: 3 verifier runs in all). Seeds 1-4:
  the same, except sensitivity seed 2 (gap +2.51, saving −5.63) and seed 4 (gap 0, saving −0.13).
- Verifier runs: 0, 3, 0.
- Shuffled-label check (headline, S2.6 protocol): shuffled mean 46.67 [43.47, 49.87], Random's
  expected 50.38 outside it; marked `xfail(strict)` like v1.
- Status: built; reported beside v1.

## Diagnostic (given more information, not a version): `v1_true_groups`

v1 reading the true family from the task handle. Seed 0: headline 84.62, cost 5.32, saving vs best
single 10.46; all-eligible 73.20, 3.58, 29.84; sensitivity 86.38, 4.11, 24.17.

## Diagnostic, shuffled labels only (2026-11-24): `router_v1_m0`

v1 with margin 0, run only while investigating the shuffled-label miss: headline development split,
seed 0, shuffle seeds 0-19, shuffled mean 49.46 (SD over shuffles 7.06). Never scored on unshuffled
labels; not a candidate version.

## v2 under P9 (a) (2026-11-25): `router_v2_p9a_ground_truth`, `router_v2_p9a_verifier`

Superseded by `router_v2` (P9 (b), spec §4.2); kept as its own logged version. It was run as
`router_v2_ground_truth` / `router_v2_verifier`: the label value `ground_truth` in its outputs is
the pairing now called `gt_probes`. Outputs renamed with a `_p9a` suffix (content unchanged);
they reproduce from commit `fca65e8` (`python -m router.experiments.s4_v2`, then the rename).

Spec §4 and §4.1 as ruled (P7 (a), P8 (a)), plus the §4.1 choice the reviewer later ruled against
(P9 (a): M not in the menu while its group explores). v1 is fitted without the new model M, M is visible from position 0, and the
calibration probes (≤ 5 per training group, stream 21) are labelled by ground truth or by the
verifier. Exploration is every tenth request per predicted group, a Beta posterior is kept per
group, a group stops when its logit SD is ≤ 0.5, and the call cap is 100. Leave-one-model-out, all 8
models, seeds 0–4, the three populations. Source: `router/experiments/s4_v2.py` at `fca65e8`;
outputs `out/s4_v2_seeds_p9a.csv` (every row), `s4_v2_seed0_p9a.csv` (all S2 metrics, gate seed),
`s4_v2_groups_p9a.csv` (one row per group and fold) and `s4_v2_p9a.json`. Two runs gave
byte-identical outputs. Per model, never pooled (R8).

**Calls to M that k did not count.** Under P9 (a) the stopped groups with M in their plan routed
tasks to M (direct dispatch, unverified), and `k_total` counts only probes and exploration calls.
Counted afterwards by rerunning the `fca65e8` code (every row's acceptance and cost reproduce the
committed outputs): **21 plan calls to M in 5 of the 240 rows**. All-eligible, Qwen3-Max, seed 0,
ground-truth pairing: 13 (k counted 311). Sensitivity, kimi-k2.5, seeds 2 and 4, both pairings: 2
each (k counted 168 and 167). None on the headline. k-matched DimensionBest
in those rows therefore got up to 13 fewer labels than the calls v2 made to M.

**What v2 did with M (all seeds, models, label sources and populations: 8,768 group-fold records).**
- **Stops.** 2,988 records stopped, with median k = 5 (the probes); none ran out of budget, and the
  call cap was never reached. In 2,490 of the 2,988 the prior alone already met the tolerance: the
  median prior strength α + β is 51.6 in stopped groups against 4.17 in groups still exploring.
- **Headline.** 43 of 1,200 records stopped.
- **M in the menu.** M is in a stopped group's plan in 86 of the 2,988. While a group explores, M is
  not in its menu (§4.1, P9).
- **Acceptance against v1 without M (k = 0).** At seed 0 the gap is 0.00 in 44 of the 48 rows, and
  ±0.36 in the other 4 (all sensitivity: Qwen3-Max +0.36 and kimi-k2.5 −0.36, both label sources).
  Across all seeds it lies in [−0.36, +0.36]. On the tasks routed after M entered the group's menu,
  the paired gap against k = 0 is 0.00 in every gate-seed row that has such tasks.
- **Cost against k = 0, seed 0.** Read the break-even length and the per-task saving first; the
  percentage saving is driven mostly by the fixed probe cost spread over about 39 scored tasks per fold.
  - **Break-even stream length** (S0.2): 306 to 283,796 tasks at seed 0 (headline 306 to 162,989;
    all-eligible 1,653 to 178,366; sensitivity 2,101 to 283,796). In 4 of 240 rows (headline,
    kimi-k2.5, seeds 3 and 4, both pairings) there is no positive per-task saving, so no break-even.
  - **Per-task saving, onboarding excluded:** $0.001 to $0.062 per thousand tasks at seed 0,
    against k = 0's median cost of $3.64 to $5.28 per thousand tasks. It credits explored tasks
    where M passed, because the rest of the plan was skipped there.
  - **Percentage saving and onboarding cost:**

| Population | Saving vs k = 0 | Onboarding cost (probes + exploration) | Calls to M per fold (mean) |
|---|---|---|---|
| headline | −356% to −0.6% | $0.019 to $0.53 | 18.0 to 18.2 |
| all-eligible | −261% to −3.5% | $0.086 to $1.78 | 61.4 to 62.8 |
| sensitivity | −362% to −3.2% | $0.044 to $0.96 | 33.8 |


**Against k-matched DimensionBest at v2's k** (gate seed; paired cluster-bootstrap CIs; k-matched
reads the dimension, given more information):
- **New model Opus (the best single model):**

| Population | Labels | v2 acceptance | k-matched acceptance | Gap |
|---|---|---|---|---|
| headline | ground truth | 71.28 | 84.62 | −13.33 [−21.05, −5.67] |
| headline | verifier | 71.28 | 84.62 | −13.33 [−21.05, −5.67] |
| all-eligible | ground truth | 66.88 | 72.33 | −5.45 [−9.07, −1.94] |
| all-eligible | verifier | 66.88 | 67.76 | −0.87 [−2.98, 1.13] |
| sensitivity | ground truth | 77.42 | 84.59 | −7.17 [−13.00, −1.39] |
| sensitivity | verifier | 77.42 | 78.85 | −1.43 [−5.42, 2.38] |

  v1 with all 8 models (k = ∞) accepts 84.62, 72.77 and 86.38.
- **Verifier-only pairing, other models.** k-matched DimensionBest often falls far below v2: on the
  headline, gaps of +30.8 to +47.7 points for MiniMax, Qwen3-Max, Sonnet, glm-5 and qwen3.5-plus.
  With mostly-UNKNOWN probes it keeps few labels and can pick the new model on a handful of PASSes.
  Percent savings against its near-zero cost (for example $0.35 per thousand tasks for Qwen3-Max on
  the headline) are not meaningful; read the gap and the absolute cost in `s4_v2_seeds.csv`.
- **Not S0.5-paired.** In the ground-truth pairing, v2's k mixes ground-truth probes with
  verifier-only exploration, while k-matched DimensionBest got all k from the ground truth (spec
  §9). The Opus gap of −13.33 [−21.05, −5.67] on the headline and the other ground-truth rows above
  are therefore not S0.5-paired comparisons. `router_v2` fixes this on the comparator side (§4.2).

**S2 metrics, gate seed** (`s4_v2_seed0.csv`): acceptance with its CI, cost, CPAT at the best single
model's acceptance (not defined where v2 is below it, for example the headline except with
Qwen3-Max held out), and regret against the cost-aware oracle with probe cost included (headline:
11.79 to 25.64 points).

Status: superseded by `router_v2` after the checkpoint 2 rulings (P9 (b); P10 (a); the comparator
label mix). Kept as a logged version.

## S4 gate check (checkpoint 2, 2026-11-25)

Plan §4 S4 gate: "router v1 and v2 run end to end on the development split with all S2 metrics; v3
and v4 components have unit tests."

| Item | Evidence | Status |
|---|---|---|
| v1 end to end, development split | `s4_v1.py`: three populations (P1 (b)), seeds 0–4, five problem-grouped folds; `out/s4_v1*` reproduce byte for byte (rerun after the v2 refactor as well) | met |
| v1, S2 metrics | acceptance with CI, cost (verifier runs included), CPAT at the best single model's acceptance, regret against the cost-aware oracle, paired cluster-bootstrap CIs against the best single model, CellBest-pred, Rule A escalated and the fixed cascade; CRUXEval ≥ 0.8 and P1 (a) variants (`s4_v1_seed0.csv`, `s4_v1_seeds.csv`, `s4_v1_variants.csv`) | met, except cost at matched acceptance |
| v2 end to end, development split | `s4_v2.py`: leave-one-model-out for all 8 models, both label sources, three populations, seeds 0–4; `out/s4_v2*` reproduce byte for byte | met |
| v2, S2 metrics and onboarding report | `s4_v2_seed0.csv`: every S2 metric for v2 with paired CIs against k = 0, k-matched DimensionBest, k = ∞ and the best single model, regret including probe cost. `s4_v2_seeds.csv` and `s4_v2_groups.csv`: k per fold and per group, each group's end state, probe and exploration cost, break-even length, acceptance before and after onboarding, paired after-onboarding gap | met, except cost at matched acceptance |
| Cost at matched acceptance | deferred to S5 E2 by the 2026-11-25 ruling, traced over the margin m with each m logged | deferred (ruled) |
| Router tests | v1 and v2: garbage scores and costs, handles whose `info` raises, held-out model read only through probes (v2), shuffled-label check one-sided (toy and real headline); exactly 10% exploration, probe limits, UNKNOWN never a label, probe cost charged, call cap (v2) | met |
| S4.6 | `router/core/guards.py`; the scan covers the new v2 modules | met (`irt.py` exempt until S4.3) |
| v3 and v4 components have unit tests | checkpoint 3 (S4.3–S4.5) | not started |

The v1 and v2 half of the gate is met on the evidence above. The gate as a whole waits for
checkpoint 3.
