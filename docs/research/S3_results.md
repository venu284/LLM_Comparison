# S3 results (checkpoint 2)

_Written 2026-11-23 on branch `s3-analysis`, under the spec frozen at `b29db49` plus the dated
implementation entry in its §7 (committed in `ca22ef8`, before any DP number). For review; the
verdicts are confirmed by the reviewer, and the user decides what follows._

Every number names its view and population. "Development split" numbers come from
`stats.dev_folds` cross-fitting and are development figures. Seed 0 decides; seeds 1 to 4 are in the
`*_seeds.csv` files. Qwen3-Max's released cost is a per-dimension constant (S1.7 data note), so
its per-task cost carries no information within a dimension.

## DP1a (S3.1): accuracy headroom

**Measurement.** View: filtered = headline, complete case, 1,046 tasks on 954 problems. Method:
cross-fitted over the six committed groups (the development split and the five outer folds, D1 (i)),
through the closed evaluation path `stats.fit_free_headroom` (D3), which read 851 evaluation tasks.

| Quantity (seed 0) | Point | 95% CI (cluster, 2,000) |
|---|---|---|
| DimensionBest − best single, acceptance | 0.00 points | [−0.29, +0.28] |
| Oracle − best single, acceptance (ceiling, O18) | +15.97 points | [+13.74, +18.35] |

**Against the threshold.** DP1a needs the CI lower bound to be ≥ 3.0 points.
- Measured lower bound: −0.29. Margin: −3.29.
- Label: "does not clear".
- Lower bounds for seeds 1 to 4: −0.47, −0.48, −0.47, −0.47.

**Why the headroom is zero.** The best single model is Opus in every group. DimensionBest picks:
- Opus for `code_completion`;
- Sonnet (glm-5 in one group) for `multi_language`.

On the held-out `multi_language` tasks, the chosen model succeeds exactly where Opus does.

**Sensitivity checks** (information only; `s3_1_dp1a.csv`):

| Variant (filtered = headline) | DimensionBest headroom | Oracle headroom |
|---|---|---|
| CRUXEval ≥ 0.8 | 0.00 [−0.29, 0.28] | 2.68 |
| Empty responses unavailable | 0.00 [−0.29, 0.28] | 15.97 |
| All tasks | −0.28 [−0.65, 0.00] | 15.82 |
| Mean released score (O10) | 0.00 [−0.29, 0.28] | 5.33 |

The other views are reported on their development splits only (D3 (α)), complete case:

| View | DimensionBest headroom | Oracle headroom |
|---|---|---|
| `all_eligible` (966 tasks) | +0.52 [−1.59, +2.39] | +9.32 |
| `sensitivity` (294 tasks) | −1.36 [−2.77, −0.33] | +10.20 |

## DP1b (S3.2): the oracle-on-no-signal bound

**Population and statistic.**
- Population: the headline development split, complete case (195 tasks, 180 problems, 3 cells).
- Statistic: per-cell Rule A in bound mode with predicted cells (D4 (a)), seed 0, point estimates,
  verifier cost ×1, base prices.

| Criterion | Measured | Threshold | Margin | CI (information) | Label |
|---|---|---|---|---|---|
| (a) saving vs best single | 65.87% | ≥ 20% | +45.87 | [59.0, 72.4] | clears |
| (a) gap vs best single | −1.54 points | ≥ −2 | +0.46 | [−8.2, +5.1] | at the threshold |
| (c) saving vs CellBest-pred | 60.84% | > 0 | +60.84 | [53.6, 67.9] | clears |
| (c) gap vs CellBest-pred | −1.03 points | ≥ −2 | +0.97 | [−7.7, +5.6] | at the threshold |

**Verdict.** All four point estimates meet their thresholds, so the bound meets DP1b's decision
rule. The two gaps are labelled "at the threshold" because their CIs reach below −2.

Gaps for seeds 1 to 4:
- against the best single model: −1.03, 0.00, +0.51, +1.03;
- against CellBest-pred: −0.51, +0.51, +1.03, +1.54.

The bound is not buildable: it reads the truth on no-signal cells. In fold 0 its tables are:
- CRUXEval: Qwen3-Max > gpt-5.4 > MiniMax;
- HumanEval completion: Qwen3-Max > qwen3.5-plus;
- HumanEval-X Python: Qwen3-Max > kimi.

**Fixed ladders.** The 84 ladders, scored in-sample (labelled as in-sample maxima):

| No-signal cells | Ladders meeting (a) | Ladders meeting (c) | Best gap at ≥ 20% saving vs best single |
|---|---|---|---|
| All accepted | 0 | 0 | −12.8 |
| All escalated | 0 | 0 | −13.3 |
| Oracle decides | 14 | 18 | +10.3 |

**The gap the learned router must close (D5 (a)).** The bound is compared with Rule A escalated,
the bar C2 must beat:

| Reference | Rule A escalated (saving, gap) | Bound (saving, gap) | Δsaving | Δgap |
|---|---|---|---|---|
| Best single | 13.4%, −1.03 | 65.9%, −1.54 | +52.5 pp | −0.51 |
| CellBest-pred | 0.6%, −0.51 | 60.8%, −1.03 | +60.2 pp | −0.51 |

Paired, the bound against Rule A escalated saves 60.6% [53.0, 67.6] at a gap of −0.51 [−7.1, +6.1].

**Sweeps (D11 (a), refit at each point; seed 0).**
- The bound meets both (a) and (c) at 15 of 19 points.
- Every miss is on the gap, never on the saving:
  - MiniMax price ×0.5: −2.56 against the best single model, −2.05 against CellBest-pred;
  - Qwen3-Max price ×2: −2.56 and −2.05;
  - glm-5 price ×0.5: −3.08 and −2.56;
  - gpt-5.4 price ×0.5: −2.05 against the best single model only.
- Savings stay between 45.3% and 80.3% against the best single model.
- The verifier cost ×1, ×10 and ×100 moves the saving by under 1.5 points.

**Other rows** (information only), bound against best single and against CellBest-pred:

| Row | vs best single (gap, saving) | vs CellBest-pred (gap, saving) |
|---|---|---|
| Headline development, CRUXEval ≥ 0.8 | −0.51, 67.4% | 0.00, 62.6% |
| Headline development, all tasks (196) | +0.51, 63.0% | +1.02, 58.0% |
| Headline development, empty responses unavailable | same as main | same as main |
| `all_eligible` development X13, complete case (459 tasks, 13 cells) | +4.79, 63.2% | +3.05, 49.6% |
| `sensitivity` development X13, complete case (279 tasks, 6 cells) | +0.72, 70.2% | +0.36, 62.2% |

kNN is not run in S3: its embedding pin has not moved into `fetch_data.py`.

## S3.3 fitted: give-up classifiers against their perfect-information bound

Population: development split, complete case. Seed 0 is shown, with cluster-bootstrap CIs
(1,000 resamples). The ladder is the training folds' `FixedCascade("escalate")`. Each threshold t
is set on out-of-fold training predictions so that the training acceptance loss is at most 2
points.

| Population | Classifier | Acceptance loss (held out) | Saving / perfect saving | Capture | AUC |
|---|---|---|---|---|---|
| Headline (195) | post-failure | 4.1 [1.5, 7.3] | 5.1% / 17.1% | 29.9% [10.1, 66.1] | 0.53 [0.43, 0.63] |
| Headline (195) | pre-attempt | 4.1 [1.6, 7.3] | 5.7% / 15.9% | 35.8% [13.6, 83.8] | 0.63 [0.54, 0.73] |
| `all_eligible` X13 (459) | post-failure | 1.7 [0.7, 3.0] | 5.2% / 29.4% | 17.8% [9.9, 26.7] | 0.77 [0.72, 0.83] |
| `all_eligible` X13 (459) | pre-attempt | 2.4 [1.1, 3.9] | 9.4% / 22.1% | 42.7% [27.8, 60.3] | 0.79 [0.74, 0.84] |

**Headline.** The post-failure AUC is near chance (0.53). The held-out acceptance loss (4.1 points)
exceeds the 2-point budget the threshold was set to on the training folds. Capture across seeds 0
to 4: 29.9, 26.6, 4.6, 5.6 and 16.0%.

**Route Anything C.5.** It measures pre-attempt signals within a cell (−0.013 to +0.078). That
definition differs from this one, so the comparison is qualitative only.

## S3.4: k-matched DimensionBest (development split, complete case, seed 0)

Full curves for the 8 new models × k × label source are in `s3_4_kmatched.csv`, with paired CIs
against full DimensionBest and against k = 0.

**Unreachable k.** k = 250 is unreachable on the headline development split: the smallest
training-fold availability is under 250, so nothing was run for it.

**Headline (195 tasks):**
- Full DimensionBest reaches 84.6%.
- Without Opus, the 7-model table (k = 0) reaches 71.8%.
- Ground-truth labels, onboarding Opus: +7.7 points over k = 0 at k = 10, and +12.8 at k ≥ 25,
  which equals full DimensionBest.
- Ground-truth labels, every other model: within −1.0 to +0.5 points of k = 0 at k ≥ 25. The
  exception is gpt-5.4 at k = 10, at −7.7.
- Verifier labels: each training fold carries 1 to 7 PASS/FAIL labels, whatever the k.
  - For six of the eight models, the k-matched table loses 7.7 to 47.7 points against full DimensionBest.
  - A few optimistic labels switch a dimension to a cheaper new model.
  - Opus and kimi are within 0.5 points.

**`all_eligible` (966 tasks):**
- Full DimensionBest reaches 34.4%.
- Ground-truth labels: within −5.4 to +0.4 points of full DimensionBest.
- Verifier labels: within −6.6 to +0.6 points of full DimensionBest; there are 1 to 32 labels per
  fold.

## S3.5: expected value of onboarding in this pool (from S3.4 only)

On the headline development split, onboarding a new model with ground-truth probes changes
acceptance by −1.0 to +0.5 points for every model except Opus. That holds at every reachable k from
25 up. The pool's tables already route both dimensions to models that match Opus.

Only a model that is better than the pool, as Opus is better than the other seven, adds anything
measurable: +12.8 points at k = 25.

With the X13 verifier as the only label source, the headline gives 1 to 7 labels per fold. The
k-matched table then does worse than ignoring the new model, by 7.7 to 47.7 points for six of eight
models.

So in this pool, the expected value of onboarding a model that is not better than Opus is about
zero with ground-truth probes. It is negative with verifier-only probes on the headline. On
`all_eligible`, the measured effects against k = 0 are within −6.6 to +2.6 points.

## S3.7: power of the gate at −2

**Setup.**
- Population: the headline development split, complete case (195 tasks, 180 problems), scaled to
  the evaluation size, 774 problems in the complete case.
- Stand-ins: the best single model, CellBest-pred and Rule A escalated. The bound is left out by
  the D4 import fence (spec §7).
- A draw passes when the gate interval's lower bound of the gap is ≥ −2.

**(c) Resampling.** Each seed draws 100 resamples of 774 problems from stream 13. Power is
reported at the development gap and at gap 0, as seed 0's value followed by seeds 1–4 in brackets.
The Monte Carlo standard error is at most 0.05.

| Comparison | Development gap | Power at development gap | Power at gap 0 | Median half-width |
|---|---|---|---|---|
| CellBest-pred vs best single | −0.51 | 0.81 (0.86–0.89) | 0.92 (0.93–0.96) | 0.54 |
| Rule A escalated vs best single | −1.03 | 0.16 (0.17–0.23) | 0.39 (0.47–0.51) | 1.18 |
| Best single vs CellBest-pred | +0.51 | 1.00 | 1.00 | 0.54 |
| Rule A escalated vs CellBest-pred | −0.51 | 0.37 (0.42–0.53) | 0.53 (0.59–0.63) | 1.03 |

**(b) Projection (cross-check).** The development half-width is scaled by √(180/774), then the E0
formula is applied:

| Comparison | Projected half-width | Power at development gap | Power at gap 0 |
|---|---|---|---|
| CellBest-pred vs best single | 0.38 | 1.00 | 1.00 |
| Rule A escalated vs best single | 0.63 | 0.86 | 1.00 |
| Rule A escalated vs CellBest-pred | 0.38 | 1.00 | 1.00 |

**Why (b) and (c) disagree for the ladder stand-in.**

The resampled CIs are wider than projected: about 1.1 against 0.63 for Rule A escalated.

The resampled cross-fitted gaps are also lower than the development gaps:

| Comparison | Development gap | Mean resampled gap |
|---|---|---|
| Rule A escalated vs best single | −1.03 | −2.00 |
| Rule A escalated vs CellBest-pred | −0.51 | −1.16 |
| CellBest-pred vs best single | −0.51 | −0.84 |

A fit on a resample, with its duplicated problems, picks a different ladder from the one the
development folds pick. So the stated true-gap assumption (true gap = development gap) does not
hold inside the resamples. "Power at gap 0", shifted by the development gap as D10 defines it, is
therefore conservative for these stand-ins.

Read literally, the gate's power at −2 for a ladder policy whose true gap is near 0 is about
0.4–0.6 under (c) and about 1.0 under (b). The plan expected 60–70%.

## S3 gate check (plan §4 S3: "S3.1–S3.7 reported with CIs in all three data views")

| Item | Evidence | Views with CIs | Gap against the gate wording |
|---|---|---|---|
| S3.1 | `s3_1_dp1a*.csv`, `.json` | filtered = headline (whole view); `all_eligible`, `sensitivity` (development splits, D3 (α)) | none |
| S3.2 | `s3_2_*` | headline (decides, all variants); `all_eligible` and `sensitivity` development X13 tasks | none |
| S3.3 descriptive | `s3_3_unsolved.csv`, `s3_3_rescue_pairs.csv`, `s3_3_historical*.csv` | all three | none |
| S3.3 fitted | `s3_3_classifier*.csv`, `.json` | headline, `all_eligible` | `sensitivity` not run (spec D8 names two populations) |
| S3.4 | `s3_4_kmatched*.csv` | headline, `all_eligible` | `sensitivity` not run (spec D9 names two populations) |
| S3.5 | this document | from S3.4 | as S3.4 |
| S3.6 | `s3_6_*.csv` | all three | none |
| S3.7 | `s3_7_power*.{json,csv}` | headline development split (D10) | power is a headline quantity; no other view |

The three gaps follow the frozen spec's populations. The reviewer decides whether the gate wording
needs the `sensitivity` view for S3.3 fitted and S3.4.
