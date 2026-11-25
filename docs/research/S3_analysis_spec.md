# S3 analysis spec (pre-registration)

_Written 2026-11-22 on branch `s3-analysis` (from `main` at `f5e1750`), before any DP1a or DP1b
number was computed. Status: **rulings final (reviewer, user approval 2026-11-23); population
counts restated after CLAIMS A4 (`main` at `06c21f7`); frozen for checkpoint 2, pending the
checkpoint 1b review.** No DP number exists. A change after a DP number exists needs a dated entry
in §7 giving the reason._

This document fixes, for every S3 item (plan §4 S3, S3.1 to S3.7) and for DP1a and DP1b, the
population, view, metric, estimator, seeds, CI, sensitivity checks and output file. It follows
`docs/CLAIMS.md` (the contract) and the plan; where it settles something they leave open, it lists
the options and a recommendation, and the reviewer picks.

Sources read: plan §1, §4 S2 and S3, §6; CLAIMS §1 to §3, O1 to O19, A1 to A4;
`docs/research/P0.4_power.md`, `P0.5_recount.md`, `P0.6_cellbest_bound.md`, `S1.7_audit.md`;
`router/sim/`, `router/baselines/`, `router/experiments/s2_baselines.py`.

---

## 1. Rules common to every S3 item

**Views** (after CLAIMS A4: S1.7 Amendment 11 moved `mbpp` out of `filtered`, `headline` and
`sensitivity`). Every number is reported in the four `audit.VIEW_NAMES`. `filtered` and `headline`
are the same 1,062 tasks (964 problems; complete case 1,046; S1.7, the code asserts that their task
ids are identical), so they appear as one column, "filtered = headline". `sensitivity` (1,935
tasks) and `all_eligible` (5,626 tasks, unchanged by A4) are their own columns. Where an item is
defined only on tasks with X13 verdicts (cascade claims, S0.11), the `all_eligible` and
`sensitivity` columns use their X13 tasks (2,911 and 1,855, all tasks) and say so.

**Populations.** Every table and report names its population: "complete case" (O2) or "all tasks
(available cells)". A count with no population named is not reported.

**Success** (S0.1, A3). Released score = 1.0 for every family (`scoring.success`).

**Unavailable cells** (S0.11, O2). Empty X13 responses and the Qwen3-Max missing token records
(`audit.load_release`). The main population is the complete case; the counts dropped are reported
per family.

**Sensitivity checks on every headline number.**
- CRUXEval at 0.8 or more counts as success (`cruxeval_lenient=True`).
- Empty `response_matrix.json` cells treated as unavailable (S1.7; `audit.response_stats`).
- All tasks, with unavailable rungs skipped, in place of the complete case (O2).
- DP1a only: the mean released score in place of acceptance (O10).

**Cost** (S0.2). Released `cost_usd` (tokens × `models.json` prices), plus `VERIFIER_RUN_USD`
($6.9 × 10⁻⁶, A1) per verifier run actually made, plus probe cost spread over the tasks scored.
Qwen3-Max's released cost is a per-dimension constant (S1.7 data note), so its per-task cost carries
no within-dimension information; every table that shows per-task cost says so in a note. On every
cost number that feeds DP1b or a headline table: the verifier sweep ×1, ×10, ×100
(`VERIFIER_COST_SWEEP`) and the price sweep (each model ×0.5 and ×2, one at a time; the range is
reported, O13).

**Statistics** (S0.8). 95% percentile CIs. Anything a DP verdict or gate reads uses
`stats.gate_interval` (cluster bootstrap over problems, 2,000 resamples, seed 0). Everything else
uses `stats.bootstrap` with `clusters=problem_id`, `N_BOOT` = 1,000, seed 0. Per-cell rates on a
single family × model use the Clopper-Pearson exact interval. Comparisons are paired (both arms in
one frame). Five seeds 0 to 4 where a seed changes anything (folds, stream order, probe draw); seed
0 decides and seeds 1 to 4 are reported next to it, never averaged into a verdict (R12). Every
threshold comparison carries `stats.threshold_label` (O12).

**Determinism.** Outputs go to `router/experiments/out/s3_*`. They contain no timestamps, run
times or absolute paths; CSVs are written with `lineterminator="\n"` and sorted rows; JSON with
sorted keys. A rerun must reproduce every output byte for byte.

**Data and safety.** All data is read-only. No model output and no dataset test code is run. The
Phase 5 database and `pipeline/.env` are not touched.

## 2. Stream ids

All draws go through `stats.rng(seed, stream)`. S3 reuses S2's streams only for the purpose S2
defined them for, and takes new ids from 12 to 19 only for new kinds of draw. A new id is added to
`stats.STREAM_IDS` and to the table in `stats.py`'s docstring in the commit that first uses it.

| id | Name | Purpose | Used at |
|---:|---|---|---|
| 3 | `probe_draw` (S2) | S3.4 probe draw inside `KMatchedDimensionBest` | checkpoint 2 |
| 6, 7 | `bootstrap_task`, `bootstrap_cluster` (S2) | every bootstrap CI, through `stats.bootstrap` | both |
| 9 | `dev_folds` (S2) | problem-grouped folds inside the development split (DP1b, S3.3 fitted, S3.4, S3.7) | checkpoint 2 |
| 12 | reserved | not used: D1 ruled (i), so DP1a makes no new fold draw | |
| 13 | `s3_power_resample` | S3.7 resampling of development problems (D10 ruled (c)) | checkpoint 2 |
| 14 to 19 | reserved | none planned | |

Checkpoint 1 makes no new draw: S3.6 and the descriptive S3.3 use only the bootstrap streams 6
and 7.

## 3. Definitions (ruled 2026-11-23)

The options and recommendations below are kept as written for checkpoint 1; each item ends with
the ruling, which is what checkpoint 2 implements.

### D1. DP1a folds on the full filtered pool

DP1a is scored on the whole `filtered` view (CLAIMS §3): 1,062 tasks, 964 problems, complete case
1,046. Each task needs a prediction from a best single model and a DimensionBest table fitted
without its problem.

- **(i) The committed splits as six groups.** The development split (196 tasks; complete case 195)
  is a sixth group next to S1.6's five outer folds (`fold_seed{s}`; under seed 0, 164 to 186
  evaluation tasks per fold, all tasks). Each group
  is scored by tables fitted on the other five. No new draw; the R13 test already covers these
  tables. Weakness: the development group is the same on every seed, so seeds 1 to 4 vary only the
  evaluation folds and the spread across seeds understates the fold-draw spread.
- **(ii) A new problem-grouped 5-fold split over the whole pool,** drawn from stream 12 per seed:
  sorted problem ids, seeded permutation, round-robin into five folds (the method of
  `stats.dev_folds`). Equal folds, full re-randomisation per seed. Weakness: a new split table that
  must also pass the R13 check, and a new draw after the data has been seen (the draw does not
  depend on outcomes).
- **What P0.4 used.** `p0_5_recount.folds`: `np.random.default_rng(seed).permutation` of the
  pool's sorted problem ids, round-robin into five folds, seeds 0 to 19, no SeedSequence stream. In
  effect option (ii) without stream discipline, on the provisional pool (3,380 tasks).
- **Best single model alongside DimensionBest** (O1). In each training set: best single =
  `static.best_single_model` (highest training acceptance, ties to the lower training mean cost).
  DimensionBest = `lookup.DimensionBest()` (per dimension, the direct-dispatch model with the highest
  training acceptance, ties to the cheaper; an unseen dimension falls back to the training best
  single model). Both are fitted on exactly the same training tasks and score the same held-out
  group.

**Recommendation: (i).** It uses only committed, tested split tables and adds no draw. The
narrower seed spread is stated next to the seed table.

**Ruling (2026-11-23): (i).** The six committed groups (the development split plus S1.6's five
outer folds); no new draw, stream 12 unused. The seed table states that the development group is
the same on every seed, so seeds 1 to 4 vary only the evaluation folds and their spread is narrower
than a full fold redraw would give.

### D2. DP1a statistic

- Headroom = acceptance of cross-fitted DimensionBest minus acceptance of the cross-fitted best
  single model, paired per task, in points, on the `filtered` complete case (1,046 tasks).
- CI: `stats.gate_interval` (cluster bootstrap over problems, 2,000 resamples, seed 0).
- DP1a holds when the gate seed's CI lower bound is ≥ 3.0 (`DP1A_MIN_LOWER`). The margin (lower
  bound minus 3.0) and the at-threshold label are reported.
- The oracle's headroom (any available model succeeds, minus the cross-fitted best single model,
  paired) is reported as the ceiling with the same CI. It decides nothing (O18).
- "Dimension" is CodeRouterBench's `dimension` field (`OutcomeMatrix.dimension`), as in `m2` and S2.
  On `filtered` after A4 there are two: `code_completion` (CRUXEval and HumanEval completion; 888
  complete-case tasks) and `multi_language` (HumanEval-X Python; 158 complete-case tasks). Before A4
  a third, MBPP's dimension, was also present.
- Reported in every view; the verdict reads `filtered` only.
- Sensitivities (information only): CRUXEval ≥ 0.8; mean released score (O10; tables still chosen
  by acceptance, the metric is the chosen model's mean released score, the oracle takes the maximum
  score); empty `response_matrix.json` cells unavailable; all tasks with unavailable rungs skipped
  (a direct dispatch to an unavailable model gives up, as `PlanRouter` does, and the count is
  reported).

**Ruling (2026-11-23): as specified.** `dimension` is CodeRouterBench's field.

### D3. The evaluation-population path for DP1a

`stats.cross_fit` refuses evaluation tasks unless `frozen=True` and a git tag that exists. The tag
belongs to the frozen router candidate list (S0.6), and creating one would unlock the guard for
every caller. No tag is created.

- **(a) A separate, closed function (recommended).** `stats.fit_free_headroom(view, groups, seed)`
  takes no candidate argument. It builds the three fit-free routes itself (best single model,
  DimensionBest, oracle), from classes it imports, and checks each route's class by identity
  (`type(x) is BestSingle`, `type(x) is GroupBest` with `key == "dimension"` and the default
  settings), never by name, since any router can call itself "dimension_best". It runs plain
  group-wise cross-fitting (no inner folds: nothing is selected), and records in its result
  `evaluation_population_path = "S3 spec D3 whitelist"`, the route classes, the groups and the
  number of evaluation tasks read. `cross_fit` is not changed.
  Tests: (1) `cross_fit` still refuses evaluation tasks without the tag; (2) the new function has no
  parameter that accepts a router or candidate, and a subclass or a renamed object is refused;
  (3) no scored task's problem is in its training set; (4) the result carries the path record.
- **(b) A one-off script that reads the matrix directly**, without any guard. Simpler, but nothing
  stops the same code from being reused for a router.
- **(c) A git tag.** Ruled out by the brief.

**Views the path may read.** CLAIMS' DP1a exception names only the artefact-filtered pool. Two
readings:
- **(α) `filtered` only (recommended).** The path reads evaluation tasks of `filtered` (= the
  headline) and nothing else. DP1a's `all_eligible` and `sensitivity` columns are then reported on
  their development splits only (1,104 and 374 tasks; complete case 966 and 294), with
  `stats.dev_folds`, and labelled so.
- **(β) every view.** The path also reads the evaluation tasks of `all_eligible` and `sensitivity`,
  so all three columns use the same six-group cross-fit. Wider than CLAIMS' wording.

The function is written at checkpoint 2, after this ruling.

**Ruling (2026-11-23): (a) with (α).** The closed fit-free function reads `filtered` evaluation
tasks only; the other views' DP1a columns are on their development splits. `cross_fit` is
unchanged and no tag is created.

### D4. DP1b: the oracle-on-no-signal bound

In the bound, a ladder rung with visible tests uses its real X13 verdict; a rung without visible
tests is accepted exactly when the ground truth succeeds (the oracle), at no verifier cost, as in
P0.5 and P0.6. The last available rung is always accepted. The bound is not buildable and is
labelled so.

- **Policy class.** Per-cell choice over P0.6's 92 candidate columns (8 direct dispatches and the
  84 ladders whose training mean cost is non-decreasing), with no-signal rungs decided by the
  oracle.
- **Options for the single statistic that decides DP1b:**
  - **(a) Per-cell Rule A in bound mode (recommended).** In each training fold and true cell, the
    cheapest column whose bound-mode training acceptance is at least CellBest's; held-out tasks are
    routed by their predicted cell. Fitted on training folds only, so there is no selection on the
    scored tasks. It is the same class as C2's
    bar (Rule A with no-signal cells escalated, O19), so the difference between them is exactly the
    value of the no-signal decision. In P0.6 it gave the largest point saving against CellBest at a
    gap of −2 or better.
  - **(b) Rule A or Rule B, passing if either does.** Two looks at the data; mildly optimistic.
  - **(c) P0.5's reading: the maximum over the 84 fixed ladders in bound mode, on the scored
    data.** An in-sample maximum, and fixed ladders cannot meet (c) robustly (P0.6: the best one is
    at the threshold). Not recommended.
- **Decision rule (any option).** One policy, gate seed 0, point estimates, ×1 verifier cost, base
  prices. DP1b holds when that policy has, at the same time: saving ≥ 20% and gap ≥ −2 points
  against the best single model, and saving > 0 and gap ≥ −2 points against CellBest with predicted
  cells. Savings are `stats.saving_pct` on cost totals; gaps are paired acceptance differences.
  Margins and the at-threshold label are reported for all four quantities. CIs (`gate_interval`)
  are information only; so are seeds 1 to 4, the sweeps and every other row.
- **Which cell predictor.** On the development split the predictor is refit in each training fold
  (S2.5's settings, `cells.fit_predictor`), as S2's `run_dev` does, for the bound, the rules and
  CellBest-pred alike. The frozen S2.5 predictor was trained on every headline development task's
  cell, so using it here would let each scored task's own cell label into its route. The frozen
  predictor is kept for the evaluation run (O3).
- **Reported beside it:** Rule B in bound mode; Rules A and B with no-signal cells accepted and
  escalated; the 84 fixed ladders in accept, escalate and bound mode (count meeting (a) and (b),
  count meeting (c), best gap among ladders saving ≥ 20%, best gap among ladders with saving > 0
  against CellBest, each labelled "in-sample maximum"); the fixed cascade; CellBest with true
  cells as a reference.
- **Implementation (checkpoint 2).** A third ladder mode `oracle` next to `accept` and `escalate`
  in `baselines.common` (`policy_outcomes` and `PlanRouter`), served by an `OracleOnNoSignal`
  verifier wrapper that returns the truth on no-signal cells with `ran=False`. Its results carry
  `verifier_mode = "oracle_on_no_signal"`, and the S2 test that routers never see the released score
  gets an explicit exemption for this mode only.

**Ruling (2026-11-23): (a), with the decision rule above.** Per-cell Rule A in bound mode decides;
the cell predictor is refit per training fold on the development split. **Added by the ruling:** a
test that only the S3.2 experiment module can import the `OracleOnNoSignal` wrapper or use the
`oracle` ladder mode, enforced the same way as the import restriction on `sim/paper.py`
(checkpoint 2 deliverable).

### D5. The gap the learned router must close

Rule A escalated and the bound are single operating points at different acceptance rates, so a
difference in saving alone mixes cost and acceptance.

- **(a) The pair (recommended).** Against CellBest with predicted cells: Δsaving = saving(bound) −
  saving(Rule A escalated), in percentage points of CellBest's cost, and Δgap = gap(bound) −
  gap(Rule A escalated), in acceptance points. The same pair against the best single model. Both
  pairs on the DP1b population, seed 0, with a paired CI of the bound against Rule A escalated
  directly (saving and gap). Seeds 1 to 4 beside.
- **(b) Cost at matched acceptance** on curves built by mixing each policy with the best single
  model. A single number, but the curve construction is a new definition and the mixing is not a
  policy anyone would run.

**Recommendation: (a).**

**Ruling (2026-11-23): (a).** The pair (Δsaving, Δgap), against CellBest with predicted cells and
against the best single model.

### D6. DP1b population (user decision, passed on by the reviewer)

Options as written at checkpoint 1 (the default then was the headline development split, complete
case, 247 tasks with `mbpp`):
- **(a) Development split only** (the plan's S3 header).
- **(b) The full headline pool**, through an extension of the D3 path to the fitted baselines
  (CellBest-pred, Rules A and B, best single). This widens the whitelist to fitted lookups and
  would score them on evaluation tasks before S7.
- **(c) Both**, with the development split deciding.

Other views for S3.2: the development split's X13 tasks of `all_eligible` and `sensitivity`, with
the cell predictor fitted on that view's training folds (the headline predictor cannot predict
cells it never saw).

**Ruling (user, 2026-11-22; restated after A4 on 2026-11-23): (a).** The headline development
split, complete case: 195 tasks, 180 problems, 3 cells (plan, S2 data notes after Amendment 11),
with `stats.dev_folds` per seed. DP1a's full-pool exception does not extend to DP1b.

### D7. S3.3: what is descriptive and what is fitted

- **Descriptive (full views, no fitting; checkpoint 1):** the unsolved share; the pairwise rescue
  rates and the rates observable from verifier histories; the historical cost figures. Defined in
  §5.
- **Fitted (development split only; checkpoint 2):** the post-failure classifier and its capture
  against the perfect-information bound (D8).

Open sub-items of the descriptive part:
- **D7.1 Perfect verifier cost in the recomputed $41.0 / $6.0 figures.** (a) No verifier charge,
  as in the 2026-11-10 figures (recommended as the main row, for comparability); (b) one verifier
  run charged per rung reached at `VERIFIER_RUN_USD` (A1), swept ×1, ×10, ×100. Both are computed
  and reported; the ruling only says which row is the headline. **Ruling (2026-11-23):** (a) is
  the headline row; (b) is reported beside it.
- **D7.2 Rung order of the cheapest-first ladder.** The 2026-11-10 figure ordered rungs by mean
  released cost on the population. That order differs from the published price order
  (`prices.price_order`): on the `filtered` complete case (after A4 as before it), gpt-5.4 is the
  third cheapest by observed cost and the sixth by price. Recommendation: keep the observed mean
  cost order on each population (the old definition) and report the price-order ladder as a
  sensitivity. **Ruling (2026-11-23):** observed mean-cost order; the price order is a
  sensitivity.

### D8. S3.3 fitted part: the post-failure classifier (checkpoint 2)

- **Population.** Headline development split, complete case; `stats.dev_folds`, five seeds. The
  `all_eligible` development split's X13 tasks as a second population.
- **Ladder.** In each training fold, S2's `FixedCascade("escalate")` (the cheapest P0.5 ladder whose
  training acceptance is at least the training best single model's).
- **Target.** After rung j fails or is escalated (verdict FAIL, or UNKNOWN), y = 1 when any later
  rung of that ladder succeeds (ground truth). Training labels come from the training folds only.
- **Features** (observable at decision time, S0.13): the predicted cell, the verdict history of the
  rungs tried (FAIL or UNKNOWN per rung) and the task's `content_text` (TF-IDF as in
  `cells.make_classifier`). Logistic regression; no released score, family or dimension.
- **Policy.** Stop (give up) when the predicted P(y = 1) < t. t is the largest value whose
  acceptance loss is at most 2 points on **out-of-fold** predictions inside the training folds
  (each training fold predicted by a classifier fitted on the other training folds). In-sample
  predictions of a TF-IDF logistic regression are close to perfect on its own training data, so a
  t set on them would be badly calibrated.
- **Perfect-information bound.** Stop exactly when y = 0: same acceptance as the full ladder,
  largest cost saving.
- **Capture.** Held-out cost saving of the classifier policy divided by the perfect-information
  saving, with the classifier's acceptance loss reported beside it; the held-out curve over t and
  the AUC with a cluster-bootstrap CI. Route Anything C.5 measures pre-attempt signals within a cell
  (−0.013 to +0.078); its definition differs and the comparison is qualitative.
- Option: add a pre-attempt variant (before rung 1) to match C.5 more closely.

**Ruling (2026-11-23): as specified, plus the pre-attempt variant as a secondary result.** The
pre-attempt variant (stop before rung 1 when the predicted P(any rung succeeds) < t, features as
above without a verdict history) is required and reported beside the post-failure classifier; the
post-failure classifier stays the primary result. Nothing else in D8 changes because of the
observability figure from checkpoint 1 (§5.3): changing the spec in response to a descriptive
result is what the pre-registration exists to prevent.

### D9. S3.4: k-matched DimensionBest curves (checkpoint 2)

- **k grid.** `DP2_K_GRID` = {10, 25, 50, 100}, plus 250.
- **Label sources.** Ground truth and verifier-only (the two E1 pairings, S0.5). Verifier-only
  labels are PASS = 1, FAIL = 0, UNKNOWN dropped; k still counts every probe call, and the
  effective number of labels (PASS plus FAIL) is reported per fold. On the headline, where about 90%
  of cells have no signal, the verifier pairing will have very few labels.
- **Models.** Leave-one-model-out: each of the 8 models in turn is the new model; the other 7 keep
  their full training tables (`lookup.KMatchedDimensionBest`). Results are per model and never
  pooled (R8).
- **Populations.** The headline development split and the `all_eligible` development split,
  complete case, `stats.dev_folds`, seeds 0 to 4.
- **Comparators** (paired, same folds): DimensionBest on all 8 models with full tables (the k = ∞
  end) and the table without the new model (k = 0, the pool prior, R11).
- **Unreachable k.** `KMatchedDimensionBest.fit` caps k at the number of training tasks where the
  new model is available, without saying so. A wrapper refuses that: a (model, k) is reported as
  "unreachable" with the smallest training-set availability, and nothing is run for it. On the
  headline development split (195 complete-case tasks, so about 156 training tasks per fold) k = 250
  is unreachable.
- Probe draws use stream 3 (`probe_draw`) under the seed, as S2 defined it.

**Ruling (2026-11-23): as specified.**

### D10. S3.7: the power check

The plan asks for `e0_power.py` "with the real verifier and nested cross-fitting on the final
headline population". The gate's population is the evaluation part of the headline (866 tasks;
complete case 851 tasks on 774 problems), so read literally the check scores baselines on
evaluation tasks. Flagged for the reviewer.

- **(a) Literal.** Baselines only (best single, CellBest-pred, Rule A escalated, the bound
  stand-ins) on the headline evaluation population with its outer folds, through an extension of
  the D3 path. Touches evaluation outcomes before S7, which R14's spirit argues against.
- **(b) Development split, projected.** Half-widths of the gap against the best single model and
  against CellBest-pred measured on the headline development split with `dev_folds`, then scaled by
  √(n_dev problems / n_eval problems) to the evaluation size. Power at −2 for a policy with true
  gap 0 = Φ(z·2/h − z), the E0 formula. Assumes the development split is representative.
- **(c) Development split, resampled (recommended, with (b) as a cross-check).** Draw evaluation-
  sized cluster resamples of the development problems (stream 13); copies of one problem share a
  fold. In each, cross-fit the stand-ins with five folds and record whether the gap CI lower bound
  is ≥ −2. Power is the share of resamples that pass for stand-ins with a true gap near 0. This is
  sampling power, which P0.4 could not measure (P0.4 §6 item 3).

Nesting: the stand-ins select nothing, so nested cross-fitting reduces to the five outer folds;
stated in the output.

**Ruling (2026-11-23): (c), with (b) as a cross-check.** Fixed details:
- **Sizes (complete case).** Development: 195 tasks on 180 problems. Evaluation target: 851 tasks
  on 774 problems. Resamples draw 774 problems with replacement from the 180 development problems
  (stream 13); (b) scales half-widths by √(180 / 774).
- **True-gap assumption, stated in the output.** Power depends on each stand-in's true gap against
  the best single model and against CellBest-pred, which is unknown. The resampling treats the
  development split as the population, so each stand-in's true gap is taken to be its observed
  development gap. This is an assumption, not an estimate of the evaluation gap.
- **Two reported points per stand-in.** (1) Power at the stand-in's development gap, as resampled.
  (2) Power with the gap shifted to 0: in each resample the stand-in's paired per-task differences
  are shifted by minus the development mean gap before the CI is formed, so the true gap is 0 and
  the CI width is unchanged. (b) gives the same two points from the E0 formula with the projected
  half-width.

### D11. Price and verifier sweeps for fitted policies

`metrics.task_outcomes(price_scale=)` rescores fixed choices. The DP1b policies choose by training
cost (Rule A, CellBest's tie-break, the ladder order), so their choices can change with prices.
- **(a) Refit (recommended for DP1b and DP1a's cost-aware rows).** Each sweep point refits every
  policy and comparator on training costs scaled the same way, then scores; seed 0 only.
- **(b) Rescore only.** Cheaper; answers "what if prices moved after deployment".

Descriptive cost figures (S3.3) are not fitted, so they are rescored.

**Ruling (2026-11-23): (a).** Refit at each sweep point; descriptive figures stay rescored.

## 4. Per-item specification

| Item | Population | View(s) | Metric | Estimator | Seeds | CI | Sensitivities | Output |
|---|---|---|---|---|---|---|---|---|
| S3.1 / DP1a | Whole view (dev + eval) via D3, complete case | all; verdict on `filtered` | Headroom of cross-fitted DimensionBest and of the oracle over the cross-fitted best single model, points | D1 groups, D2 | 0 to 4 | `gate_interval` | CRUXEval 0.8; mean score (O10); empty `response_matrix` cells; all tasks | `s3_1_dp1a.csv`, `s3_1_dp1a_seeds.csv` |
| S3.2 / DP1b | D6: headline dev, complete case, 195 tasks | headline decides; others' X13 dev tasks | D4 statistic; D5 gap | `run_dev` over `dev_folds` | 0 to 4 | `gate_interval` (information) | CRUXEval 0.8; empty `response_matrix` cells; all tasks; verifier ×1/10/100; prices ×0.5/×2 (D11) | `s3_2_dp1b.csv`, `s3_2_ladders.csv`, `s3_2_sweeps.csv` |
| S3.3 descriptive | Whole view, complete case | all | §5.2 to §5.4 | counts | none | `bootstrap` (`N_BOOT`) | CRUXEval 0.8; empty `response_matrix` cells; all tasks; prices (rescored) | `s3_3_unsolved.csv`, `s3_3_rescue_pairs.csv`, `s3_3_historical.csv` |
| S3.3 fitted | Headline dev; all-eligible dev X13 | headline; all_eligible | D8 capture, AUC | D8 | 0 to 4 | `bootstrap` | CRUXEval 0.8 | `s3_3_classifier.csv` |
| S3.4 | Headline dev; all-eligible dev | headline; all_eligible | Acceptance and cost per (model, k, labels) | D9 | 0 to 4 | `bootstrap`, paired | CRUXEval 0.8 | `s3_4_kmatched.csv` |
| S3.5 | From S3.4 only | as S3.4 | Written statement | none | | | | `S3_results.md` |
| S3.6 | Whole view; X13 cells | all | §5.1 | counts | none | Clopper-Pearson per family × model; `bootstrap` pooled | CRUXEval 0.8; empty `response_matrix` cells | `s3_6_*.csv` |
| S3.7 | D10 | headline | Power at −2 | D10 | 0 to 4 | as E0 | | `s3_7_power.json` |

## 5. Checkpoint 1 definitions (fixed now; descriptive, no fitting)

These read outcomes in every view, including the evaluation population, as S1.7 did. They feed no
decision (plan S3 header; brief, "Populations").

### 5.1 S3.6 verifier and population profile

**Cells.** A cell is one (task, model) of a view's X13 tasks. It is available when it is neither
an empty X13 response nor a missing token record. It has a signal when its task has visible tests
(`n_visible_tests > 0`). The verdict is PASS when all visible tests pass, otherwise FAIL
(`verdicts.load_verdicts`).

**Rates on available cells with a signal,** against S0.1 success:
- false-pass rate = #(PASS and not success) / #(not success);
- false-reject rate = #(FAIL and success) / #(success).
The four counts (PASS/FAIL × success/not) are written next to every rate, so any rate can be
recomputed. A denominator of 0 is written as "no signal" (no cells) or "none" (cells, but no
failures or no successes), never as 0.

**Populations of the verifier tables.** The primary rows count the available cells of every task
in the view ("all tasks (available cells)"). The same tables repeat every count on the view's
complete case ("complete case"), in a `population` column. On the headline, main variant: 3 / 63
false pass and 36 / 893 false reject on all 1,062 tasks; 3 / 57 and 26 / 871 on the 1,046
complete-case tasks (956 and 928 signal cells). A4 does not change these counts, because `mbpp`
had no visible tests.

**Bridge to the plan's figures** (plan §1 C3, log 2026-11-19): 15.0% false-pass and 27.0%
false-reject on 8,420 cells, and 66.5% heuristic false-pass on 12,472 no-signal cells. Those used
every X13 cell with a binary released score (`truth` ∈ {0, 1}), any family, empty cells included,
and the same two conditionals. The script reproduces them and stops if they do not match.

**Tables.**
- Per view × family × model: counts, both rates, Clopper-Pearson 95% intervals.
- Per view × family (pooled over models), per view × model (pooled over families) and per view:
  counts, both rates, cluster-bootstrap CIs over problems.
- No-signal share per view × family: cells with no signal / available cells; tasks with no signal
  for any available model / tasks; the same on the complete case.
- Population profile per view: tasks, problems, development and evaluation tasks, complete-case
  tasks, family mix (count and share), per-model acceptance on the complete case with CIs, the best
  single model, the share solved by at least one model, and the unsolved share with CIs. The
  headline column sits next to `all_eligible` (S0.11).
- Sensitivities: CRUXEval ≥ 0.8 (changes only the truth); empty `response_matrix.json` cells
  unavailable.

### 5.2 S3.3 unsolved share

Per view and per view × family: the share of complete-case tasks where no available model
succeeds, with a cluster-bootstrap CI. Checks against `artefacts.yaml` (after A4): on the
`filtered` complete case, 42 of 1,046 (5 under CRUXEval ≥ 0.8). Sensitivities: CRUXEval ≥ 0.8; all tasks (a task with an
unavailable cell counts as unsolved when no available model succeeds); empty `response_matrix.json`
cells unavailable.

### 5.3 S3.3 rescue rates for each pair of rungs

For every ordered pair of distinct models (l, h), 56 pairs, on a view's complete-case tasks:
- **true rescue** = #(l fails and h succeeds) / #(l fails), ground truth (S0.1);
- **observable rescue** = #(l's verdict is FAIL and h succeeds) / #(l's verdict is FAIL), on tasks
  where l's cell has a signal. This conditions on what a router can see (S0.13); false rejections of
  l are included, as they would be in a real history;
- **verifier-only rescue** = #(l's verdict FAIL and h's verdict PASS) / #(l's verdict FAIL), on
  tasks where both cells have a signal;
- **observability** = #(l's verdict FAIL and l fails) / #(l fails): the share of l's true failures
  that the verifier shows.

Every rate carries its counts. Two flag columns say whether l is cheaper than h by published price
(`prices.price_order`) and by observed mean cost on the view. The true and observable rescue rates
get cluster-bootstrap CIs per view; per view × family only counts and rates are written. On
`filtered`, where 88.9% of complete-case tasks (930 of 1,046) have no signal for any model, the
observable rates
rest on few tasks; the counts show how few.

### 5.4 S3.3 historical cost figures

**Old definition** (plan log 2026-11-10, reproduced exactly on this data): CodeRouterBench ID-test
split, complete view, all dimensions (2,919 tasks, eligible or not); success = score ≥ 0.5; the 8
models ordered by mean released cost on that split; a perfect verifier at no cost. Full ladder: each
task pays every rung up to its first success, and unsolved tasks pay all 8 rungs. Always-Opus: the
sum of Opus's cost. Perfect stop: unsolved tasks pay only the first rung. Values: $41.05, $34.02,
$5.99, with 39.3% of tasks unsolved. Missing token records counted as $0 cost and score 0.

**New definition.** For each view, both CodeRouterBench ID splits pooled (the S0.6 pool), success
under S0.1, complete case; rungs in observed mean-cost order on the view (D7.2); unavailable rungs
skipped, never charged; perfect verifier at no cost (D7.1 (a)), with the A1-charged variant (one
run per rung reached, ×1, ×10, ×100) as the second row. Always-Opus is reported, and the view's best
single model if it is not Opus. Reported as totals and as m$ per task, with N, and with the share
of the full ladder's cost spent on unsolved tasks (RQ-B).

**Bridge rows.** (1) The old definition; (2) the old population with success = 1.0 and unavailable
rungs skipped; (3) the new definition per view. Each step changes one thing.

Sensitivities: CRUXEval ≥ 0.8; all tasks; price-order ladder; prices ×0.5 and ×2 per model,
rescored (range reported).

## 6. What checkpoint 1 delivers

- This spec, committed before any DP1a or DP1b number.
- `router/sim/profile.py`: the §5 definitions as functions, with toy tests whose values are
  checked by hand (`tests/router/test_s3_profile.py`), plus data tests pinning the bridges (15.0% /
  27.0% on 8,420 cells; $41.05 / $34.02 / $5.99; after A4, on the headline complete case, 42 and 5
  unsolved of 1,046 and 930 of 1,046 with no signal; before A4 these were 198, 161 and 1,148 of
  1,264).
- `router/experiments/s3_descriptive.py` and its outputs `router/experiments/out/s3_3_*` and
  `s3_6_*`, reproducible byte for byte.
- No DP number, no fitted quantity, no evaluation-population path code.

## 7. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | Drafted for checkpoint 1 | |
| 2026-11-22 | D3: which views the path may read, options (α) and (β). D4: the cell predictor is refit per training fold on the development split; the frozen S2.5 predictor is kept for the evaluation run. D8: t is set on out-of-fold predictions inside the training folds. §5.1: `rm_empty_unavailable` complete-case counts checked against `artefacts.yaml` | Pre-review tightening before any DP number exists (commit order unchanged: no DP number has been computed) |
| 2026-11-23 | Rulings D1 to D11 written in (reviewer, user approval 2026-11-23): D1 (i), stream 12 unused; D2 as specified; D3 (a) with (α); D4 (a) plus the import-restriction test for the oracle wrapper and mode; D5 (a); D6 headline development split (user); D7.1 (a) headline, (b) beside; D7.2 cost order, price order as sensitivity; D8 as specified plus the pre-attempt variant as a secondary result, no change for the observability figure; D9 as specified; D10 (c) with (b) as a cross-check, the true-gap assumption and power at the development gap and at gap 0; D11 (a) | Checkpoint 1 review; no DP number has been computed |
| 2026-11-23 | Population counts restated after CLAIMS A4 (`mbpp` leaves `filtered`, `headline` and `sensitivity`). Headline = filtered: 1,287 → 1,062 tasks, 1,188 → 964 problems, complete case 1,264 → 1,046; development split 248 → 196 tasks (complete case 195, 180 problems, 3 cells; was 247, 232, 4); evaluation part 1,039 → 866 tasks (complete case 851, 774 problems); outer folds under seed 0: 164 to 186 evaluation tasks (was 200 to 215); dimensions 3 → 2. Sensitivity: 2,160 → 1,935 tasks, X13 tasks 2,080 → 1,855, development split 426 → 374. `all_eligible` unchanged (5,626; X13 2,911; development 1,104). Unsolved (complete case) 198 / 161 of 1,264 → 42 / 5 of 1,046; no signal for any model 1,148 of 1,264 (90.8%) → 930 of 1,046 (88.9%) | CLAIMS A4, S1.7 Amendment 11 (`main` at `06c21f7`, merged into this branch). No definition changes |
| 2026-11-23 | §1 and §5.1: every table names its population; the verifier tables carry both "all tasks (available cells)" and "complete case" rows | Checkpoint 1 review: the interim report put all-task verifier counts under a complete-case heading. The definition of the primary rows is unchanged |
| 2026-11-23 | Checkpoint 2 implementation, fixed before any DP number: (1) D4: the `oracle` ladder mode, the `OracleOnNoSignal` wrapper and the bound rules live in `router/baselines/oracle_bound.py`, not in `baselines.common`; `common.policy_outcomes` and `PlanRouter` are unchanged (they already escalate on anything but PASS outside mode `accept`, so with the wrapper a no-signal rung stops exactly when the truth succeeds). The wrapper refuses tasks with no X13 record. (2) D3: `stats.fit_free_headroom(seed, *, cruxeval_lenient, empty_unavailable, all_tasks)` builds the `filtered` view itself and refuses any other task set (checked against `artefacts.yaml`'s task-id hash), so it takes no view, matrix or group argument; the six groups are D1's. (3) D11: `Context.verifier_usd` (default A1) carries the scaled verifier cost into the ladder fits of Rules A and B, the fixed cascade and the bound, so the verifier sweep refits; scored costs are rescaled from the verifier runs made. S2's outputs are unchanged. (4) CRUXEval ≥ 0.8 sensitivity, in DP1a and DP1b alike: tables and rules are chosen under S0.1, as `run_dev` does, and scored with CRUXEval ≥ 0.8; the bound's truth is lenient whenever the scoring is. (5) DP1a's at-threshold label is `stats.threshold_label` applied to the CI lower bounds of seeds 0 to 4 (the quantity the verdict reads), with seed 0's lower bound as the gate value. (6) S3.7: the bound stand-in is left out, because the D4 ruling lets only the S3.2 module import the oracle; the stand-ins are the best single model, CellBest-pred and Rule A escalated. Resamples: 100 per seed for seeds 0 to 4 (stream 13), with the Monte Carlo standard error reported | The D4 import restriction can only be enforced at module level; the D3 (α) ruling is enforced in code rather than only documented; the D11 ruling needs the verifier cost in the refit; reviewer's checkpoint 1b go-ahead (item 1) |
| 2026-11-24 | D8 features. The checkpoint 2 classifier added a rung-position feature (`pos=j`; `pos=-1` before rung 1) that D8 does not list. The primary result is now D8's features exactly (the predicted cell, the verdict history, `content_text`); the rung-position run is kept as a labelled post-hoc variant (`features` column of `s3_3_classifier.csv`). Before → after, headline, seed 0: post-failure capture 29.9% → 29.9%, acceptance loss 4.1 → 4.1, AUC 0.5292 → 0.5295; pre-attempt capture 35.8% → 35.8%, loss 4.1 → 4.1, AUC 0.6343 → 0.6345; `all_eligible` unchanged at the precision reported (17.8% / 1.7 / 0.77; 42.7% / 2.4 / 0.79). Over all 20 earlier rows (2 populations × 2 classifiers × 5 seeds) the policy outcome is identical and the AUC moves by at most 0.0025: with one history token per rung tried, the position is a linear function of the history (a constant before rung 1) | Reviewer check of checkpoint 2, item 1 (run D8 exactly as specified) |
| 2026-11-24 | D8 implementation detail, recorded: each fold's cell predictor is fitted on the training fold and applied to every task, so training rows (and the out-of-fold rows used to set t inside the training folds) carry the predictor's in-sample predicted cell, while held-out rows carry an out-of-sample one. Timing: the code was written and committed in `ca22ef8` (2026-11-23 21:08 UTC) and first run on data at 21:12 UTC, so the choice was made before any fitted S3.3 number existed; it was made after S3.3's descriptive numbers from checkpoint 1 (including the §5.3 observability figure), which do not involve the classifier. Not changed | Reviewer check of checkpoint 2, item 2 (deviation 3) |
| 2026-11-24 | D10 implementation details, recorded: resample r of seed s draws from `SeedSequence([1000·s + r, 13])` (stream 13), and the gap CI clusters on draws (each copy of a drawn problem is its own cluster, as each evaluation problem would be). Timing: `s3_power.py` was first written at 2026-11-23 21:11 UTC, after the DP1a output (21:09 UTC) and before the DP1b run (21:12 UTC). A two-resample smoke run at 21:11 UTC printed mean CI bounds for seed 0; a later patch changed only `CopyVerifier.empty_cells` (speed) and the docstring. Seeding and clustering are unchanged since the first write. Neither changes a definition | Reviewer check of checkpoint 2, item 2 (deviation 4) |
| 2026-11-24 | `sensitivity` added to S3.3 fitted (D8) and S3.4 (D9): its development split, complete case (S3.3: X13 tasks, 279 tasks on 244 problems, as DP1b's `sensitivity` column; S3.4: 294 tasks on 259 problems, 7 cells). Additive: the headline and `all_eligible` rows are unchanged (S3.4 checked by key; S3.3's earlier rows reappear as the variant). Before: not run. After: see `S3_results.md` §S3.3 and §S3.4. S3.7 stays headline-only by definition | Reviewer check of checkpoint 2, item 4 (plan §4 S3 gate: all three views); no DP impact |
| 2026-11-24 | D10 reporting additions, computed from the same draws (no new draw, no new stream): (1) (c) and (b) side by side per comparison with half-widths (`summary` in `s3_7_power.json`; (c) on seed 0 and on all 500 draws); (2) a labelled post-hoc, data-dependent variant of "power at gap 0" that shifts each draw by minus the resample-world mean gap (the mean resampled gap over all 500 draws of that comparison) instead of the development gap; (3) a width check, the E0 formula at the median resampled half-width. Before → after: D10's two points are unchanged (Rule A escalated vs best single, seed 0: 0.16 at the development gap, 0.39 at gap 0; all draws 0.19 and 0.47). New: post-hoc gap-0 power 0.68 (seed 0) / 0.76 (all draws); width check 0.40 at the development gap and 0.94 at gap 0, against (b)'s 0.86 and 1.00. The half-width (1.11 resampled against 0.63 projected) explains about two thirds of the (b)–(c) difference at the development gap but only 0.06 of it at gap 0, where the resampled mean gap (−2.00 against the development −1.03) dominates | Reviewer check of checkpoint 2, item 3; the resampled mean gaps differ from the development gaps that D10's shift uses |
