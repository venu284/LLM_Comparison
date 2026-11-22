# S3 analysis spec (pre-registration)

_Written 2026-11-22 on branch `s3-analysis` (from `main` at `f5e1750`), before any DP1a or DP1b
number was computed. Status: **draft for the reviewer at S3 checkpoint 1.** Every item marked
"Ruling" is open until the reviewer rules; the ruled text is then frozen and checkpoint 2 runs on
it. A change after a DP number exists needs a dated entry in §7 giving the reason._

This document fixes, for every S3 item (plan §4 S3, S3.1 to S3.7) and for DP1a and DP1b, the
population, view, metric, estimator, seeds, CI, sensitivity checks and output file. It follows
`docs/CLAIMS.md` (the contract) and the plan; where it settles something they leave open, it lists
the options and a recommendation, and the reviewer picks.

Sources read: plan §1, §4 S2 and S3, §6; CLAIMS §1 to §3, O1 to O19, A1 to A3;
`docs/research/P0.4_power.md`, `P0.5_recount.md`, `P0.6_cellbest_bound.md`, `S1.7_audit.md`;
`router/sim/`, `router/baselines/`, `router/experiments/s2_baselines.py`.

---

## 1. Rules common to every S3 item

**Views.** Every number is reported in the four `audit.VIEW_NAMES`. `filtered` and `headline` are
the same 1,287 tasks (S1.7; the code asserts that their task ids are identical), so they appear as
one column, "filtered = headline". `sensitivity` (2,160 tasks) and `all_eligible` (5,626 tasks) are
their own columns. Where an item is defined only on tasks with X13 verdicts (cascade claims,
S0.11), the `all_eligible` and `sensitivity` columns use their X13 tasks (2,911 and 2,080) and say
so.

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
| 12 | `s3_dp1a_folds` | DP1a folds over the whole filtered pool, **only if D1 option (ii) is ruled** | checkpoint 2 |
| 13 | `s3_power_resample` | S3.7 resampling of development problems, **only if D10 option (c) is ruled** | checkpoint 2 |
| 14 to 19 | reserved | none planned | |

Checkpoint 1 makes no new draw: S3.6 and the descriptive S3.3 use only the bootstrap streams 6
and 7.

## 3. Open definitions (the reviewer rules on each)

### D1. DP1a folds on the full filtered pool

DP1a is scored on the whole `filtered` view (CLAIMS §3): 1,287 tasks, 1,188 problems, complete case
1,264. Each task needs a prediction from a best single model and a DimensionBest table fitted
without its problem.

- **(i) The committed splits as six groups.** The development split (248 tasks) is a sixth group
  next to S1.6's five outer folds (`fold_seed{s}`, 200 to 215 tasks each under seed 0). Each group
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
narrower seed spread is stated next to the seed table. **Ruling:** _open_.

### D2. DP1a statistic

- Headroom = acceptance of cross-fitted DimensionBest minus acceptance of the cross-fitted best
  single model, paired per task, in points, on the `filtered` complete case (1,264 tasks).
- CI: `stats.gate_interval` (cluster bootstrap over problems, 2,000 resamples, seed 0).
- DP1a holds when the gate seed's CI lower bound is ≥ 3.0 (`DP1A_MIN_LOWER`). The margin (lower
  bound minus 3.0) and the at-threshold label are reported.
- The oracle's headroom (any available model succeeds, minus the cross-fitted best single model,
  paired) is reported as the ceiling with the same CI. It decides nothing (O18).
- "Dimension" is CodeRouterBench's `dimension` field (`OutcomeMatrix.dimension`), as in `m2` and S2.
  On `filtered` there are three: `code_completion` (CRUXEval and HumanEval completion),
  `code_generation` (MBPP) and `multi_language` (HumanEval-X Python).
- Reported in every view; the verdict reads `filtered` only.
- Sensitivities (information only): CRUXEval ≥ 0.8; mean released score (O10; tables still chosen
  by acceptance, the metric is the chosen model's mean released score, the oracle takes the maximum
  score); empty `response_matrix.json` cells unavailable; all tasks with unavailable rungs skipped
  (a direct dispatch to an unavailable model gives up, as `PlanRouter` does, and the count is
  reported).

**Ruling:** _open_ (confirm the statistic and the dimension field).

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

The function is written at checkpoint 2, after this ruling. **Ruling:** _open_.

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
    routed by their predicted cell (the S2.5 predictor, fitted per fold as S2 does). Fitted on
    training folds only, so there is no selection on the scored tasks. It is the same class as C2's
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

**Ruling:** _open_ (option and decision rule).

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

**Recommendation: (a).** **Ruling:** _open_.

### D6. DP1b population (user decision, passed on by the reviewer)

Default until the user decides: the headline development split, complete case (247 tasks, 232
problems, 4 cells; S2 data note), with `stats.dev_folds` per seed. Options:
- **(a) Development split only** (the plan's S3 header).
- **(b) The full headline pool**, through an extension of the D3 path to the fitted baselines
  (CellBest-pred, Rules A and B, best single). This widens the whitelist to fitted lookups and
  would score them on evaluation tasks before S7.
- **(c) Both**, with the development split deciding.

Other views for S3.2: the development split's X13 tasks of `all_eligible` and `sensitivity`, with
the cell predictor fitted on that view's training folds (the headline predictor cannot predict
cells it never saw). **Ruling:** _user_.

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
  and reported; the ruling only says which row is the headline. **Ruling:** _open_.
- **D7.2 Rung order of the cheapest-first ladder.** The 2026-11-10 figure ordered rungs by mean
  released cost on the population. That order differs from the published price order
  (`prices.price_order`): on `filtered`, gpt-5.4 is the third cheapest by observed cost and the
  sixth by price. Recommendation: keep the observed mean cost order on each population (the old
  definition) and report the price-order ladder as a sensitivity. **Ruling:** _open_.

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
- **Policy.** Stop (give up) when the predicted P(y = 1) < t; t is set on the training folds as the
  largest value whose training acceptance loss is at most 2 points.
- **Perfect-information bound.** Stop exactly when y = 0: same acceptance as the full ladder,
  largest cost saving.
- **Capture.** Held-out cost saving of the classifier policy divided by the perfect-information
  saving, with the classifier's acceptance loss reported beside it; the held-out curve over t and
  the AUC with a cluster-bootstrap CI. Route Anything C.5 measures pre-attempt signals within a cell
  (−0.013 to +0.078); its definition differs and the comparison is qualitative.
- Option: add a pre-attempt variant (before rung 1) to match C.5 more closely.

**Ruling:** _open_.

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
  headline development split (about 200 training tasks per fold) k = 250 is unreachable.
- Probe draws use stream 3 (`probe_draw`) under the seed, as S2 defined it.

**Ruling:** _open_.

### D10. S3.7: the power check

The plan asks for `e0_power.py` "with the real verifier and nested cross-fitting on the final
headline population". The gate's population is the evaluation part of the headline (1,039 tasks),
so read literally the check scores baselines on evaluation tasks. Flagged for the reviewer.

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
stated in the output. **Ruling:** _open_.

### D11. Price and verifier sweeps for fitted policies

`metrics.task_outcomes(price_scale=)` rescores fixed choices. The DP1b policies choose by training
cost (Rule A, CellBest's tie-break, the ladder order), so their choices can change with prices.
- **(a) Refit (recommended for DP1b and DP1a's cost-aware rows).** Each sweep point refits every
  policy and comparator on training costs scaled the same way, then scores; seed 0 only.
- **(b) Rescore only.** Cheaper; answers "what if prices moved after deployment".

Descriptive cost figures (S3.3) are not fitted, so they are rescored. **Ruling:** _open_.

## 4. Per-item specification

| Item | Population | View(s) | Metric | Estimator | Seeds | CI | Sensitivities | Output |
|---|---|---|---|---|---|---|---|---|
| S3.1 / DP1a | Whole view (dev + eval) via D3, complete case | all; verdict on `filtered` | Headroom of cross-fitted DimensionBest and of the oracle over the cross-fitted best single model, points | D1 groups, D2 | 0 to 4 | `gate_interval` | CRUXEval 0.8; mean score (O10); empty `response_matrix` cells; all tasks | `s3_1_dp1a.csv`, `s3_1_dp1a_seeds.csv` |
| S3.2 / DP1b | D6 (default: headline dev, complete case, 247 tasks) | headline decides; others' X13 dev tasks | D4 statistic; D5 gap | `run_dev` over `dev_folds` | 0 to 4 | `gate_interval` (information) | CRUXEval 0.8; empty `response_matrix` cells; all tasks; verifier ×1/10/100; prices ×0.5/×2 (D11) | `s3_2_dp1b.csv`, `s3_2_ladders.csv`, `s3_2_sweeps.csv` |
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
succeeds, with a cluster-bootstrap CI. Checks against `artefacts.yaml`: on `filtered`, 198 of
1,264 (161 under CRUXEval ≥ 0.8). Sensitivities: CRUXEval ≥ 0.8; all tasks (a task with an
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
`filtered`, where 90.8% of complete-case tasks have no signal for any model, the observable rates
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
  27.0% on 8,420 cells; $41.05 / $34.02 / $5.99; 198 and 161 unsolved; 1,148 of 1,264 with no
  signal).
- `router/experiments/s3_descriptive.py` and its outputs `router/experiments/out/s3_3_*` and
  `s3_6_*`, reproducible byte for byte.
- No DP number, no fitted quantity, no evaluation-population path code.

## 7. Change log

| Date | Change | Reason |
|---|---|---|
| 2026-11-22 | Drafted for checkpoint 1 | |
