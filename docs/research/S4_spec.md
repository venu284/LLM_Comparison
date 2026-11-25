# S4 spec: router v1, v2 and the v3/v4 components (pre-registration)

_Written 2026-11-24 on branch `s4-router` (from `main` at `a324b89`), before any S4 router number
was computed. Status: **recommendations, awaiting the reviewer's rulings.** Router v1 is built on
the recommendations below so that checkpoint 1 can be reviewed end to end; if a ruling differs, the
change gets a dated entry in §9 and v1 is rerun. A change after a ruling also needs a dated §9
entry giving the reason._

Sources read: plan §1 (C1, C2), §2, §4 S2–S5, §6; CLAIMS S0.1, S0.2, S0.5–S0.8, S0.11, S0.13,
O1–O3, O8, O9, O19, A1–A4; `docs/research/S3_analysis_spec.md`, `S3_results.md`;
`router/sim/`, `router/baselines/`, `router/experiments/s2_baselines.py`, `s3_dp1b.py`,
`s3_classifier.py`, `m3_irt.py`.

User decisions already made (2026-11-24) are marked **(decided)**. Every other point lists the
options and a recommendation (**rec.**).

---

## 1. Rules common to all of S4

- **Development split only** (S0.6, S0.7, R14). Folds come from `stats.dev_folds(tasks, seed)`
  (problem-grouped, five folds). Every router is fitted on four folds and scores the fifth through
  `baselines.run.run_dev`, so every task is scored once per seed by a fit that never saw its
  problem. `stats.cross_fit`'s guard and `stats.fit_free_headroom` are not touched.
- **What a router sees** (S0.13, S2): an opaque task handle with `content_text`, the available
  models, and its own verdicts (PASS, FAIL, UNKNOWN). No released score, no family or dimension
  label and no cost at decision time. Fitting may read the training folds' scores, costs and
  labels (O1). Anything that reads a scored task's true label is a diagnostic labelled **given more
  information** and is never a router version.
- **Success** = released score 1.0 (S0.1, A3); CRUXEval ≥ 0.8 as a sensitivity check.
- **Cost** (S0.2): released `cost_usd` + `VERIFIER_RUN_USD` ($6.9 × 10⁻⁶, A1) per verifier run
  made + probe cost spread over the tasks scored.
- **Statistics** (S0.8): seeds 0–4, seed 0 the gate seed; seeds 1–4 reported beside it, never
  averaged into a verdict (R12). Paired cluster-bootstrap CIs over problems (`stats.bootstrap`,
  `N_BOOT` = 1,000, bootstrap seed 0) on every comparison. These are development figures, not
  results.
- **Outputs** in `router/experiments/out/s4_*` reproduce byte for byte; no timestamps, no absolute
  paths. Every router version tried is logged in `docs/research/S4_versions.md` with its
  development numbers, discarded versions included (S0.7).

## 2. Populations (decided, with one open point)

Three development populations, complete case (O2):

| Population | Dev tasks, complete case | Problems | Families | With an X13 record | With `content_text` |
|---|---|---|---|---|---|
| headline | 195 | 180 | 3 | 195 | 195 |
| all-eligible | 966 | 567 | 20 | 459 | 956 |
| sensitivity | 294 | 259 | 7 | 279 | 284 |

(Counted 2026-11-24 from `audit.build_views`, `splits.task_splits` and `ReplayVerifier.from_x13`.)

**Open point P1: tasks without an X13 record.** On those tasks the replay verifier returns UNKNOWN
with state `no_record`, and S0.11 says cascade claims never score them. Router v1 contains a
cascade.
- (a) Score all complete-case tasks, the no-record ones included (966 and 294).
- (b) Score only the tasks with an X13 record (459 and 279), as S3 did for these two views
  (S3 spec §1). Report (a) as a labelled variant.
- **Rec. (b).** It follows S0.11 and S3, and every comparator is scored on the same tasks. The
  headline is unaffected (all 195 have a record).

A task without `content_text` cannot be given a predicted group; it is routed by the pool row
(§3.3), and the count is reported.

## 3. Router v1

### 3.1 Groups (decided)

The router's group is the **predicted S1.1 family** of the task, read from `content_text` by the
S2.5 predictor (`cells.CellPredictor`: TF-IDF word 1–2 and char 2–5 grams, logistic regression,
C = 10), refit on each training fold.

- In all three development populations each family lies in exactly one dimension (measured: no
  family spans two dimensions), so the dimension × family cell and the family are the same
  partition. v1 therefore reuses `cells.fit_predictor` on `true_cells` unchanged, and a predicted
  cell is a predicted family.
- **Open point P2: the predictor's training pool.** (a) The scored population's own training
  folds; (b) S2's fixed pool (the headline development split), which knows only 3 families and
  would put every all-eligible task into one of them; (c) the full development pool's training
  folds (O1 allows the router more data). **Rec. (a)**, with the comparators that use predicted
  cells (CellBest-pred, Rules A and B pred, zero-loss pred) fitted on the same pool and the same
  folds. On the headline, (a) differs from S2's pool by at most the one development task dropped
  for an unavailable cell.
- **Open point P3: which labels the table is built on.** (a) The training tasks' true families,
  with the router reading the predicted family at decision time (as CellBest-pred does); (b) the
  training tasks' in-sample predicted families. **Rec. (a)**: it matches the comparator, and the
  training tasks' families are training labels (O1).
- **A predicted family with no training row.** The predictor only outputs families it was trained
  on, and with P2 (a) and P3 (a) every such family has training rows. The case can still arise
  under other rulings or with a missing text, so it is defined: the task goes to the **pool row**
  (§3.3), the all-groups estimate, which is the pool prior (R11). A family with few training rows
  is shrunk toward the pool row by §3.2 (for example swebench has 10 development tasks).
- True families appear only in the diagnostic `v1_true_groups` (given more information).

### 3.2 The shrunk lookup table

**Candidates per group.** The eight models by direct dispatch (no verifier), plus the one fixed
cascade of §3.4 (rungs verified). Candidate j's training outcome on task i is its acceptance a_ij
(0/1) and its cost c_ij, computed by `common.policy_outcomes`, which a test already matches to
replay.

**Shrinkage, open point P4.**
- (a) **Beta-binomial partial pooling with one shared prior strength κ** (rec.). For candidate j in
  group g with n_g training tasks and s_gj accepted:
  `â_gj = (s_gj + κ · p_j) / (n_g + κ)`, where p_j is j's pool-wide training acceptance.
  κ is estimated on the training folds only, by the ANOVA (method-of-moments) estimator of the
  intra-class correlation ρ, with the numerator and denominator summed over candidates:
  - N = Σ_g n_g, G = number of groups with training rows, n₀ = (N − Σ_g n_g² / N) / (G − 1);
  - per candidate: MSB_j = Σ_g n_g (r_gj − p_j)² / (G − 1), MSW_j = Σ_g n_g r_gj (1 − r_gj) / (N − G),
    with r_gj = s_gj / n_g;
  - ρ = Σ_j (MSB_j − MSW_j) / Σ_j (MSB_j + (n₀ − 1) MSW_j); κ = (1 − ρ) / ρ.
  - Degenerate cases: G ≤ 1, a zero denominator or ρ ≤ 0 means κ = ∞ (full pooling: every group
    uses the pool row); ρ ≥ 1 means κ = 0 (no pooling).
  One shared κ because per-candidate κ is close to unidentified with three groups (headline).
- (b) Logit-normal empirical Bayes per candidate (random intercept per group), fitted by marginal
  likelihood.
- (c) A fixed κ grid (for example {0, 5, 20, 80}); each value would be a separate logged version.

**Costs** are the group's training mean cost per candidate, not shrunk (costs are observed, not
inferred). A group with no training row takes the pool-wide mean cost.

**The pool row** (no training row, or no text): â_j = p_j and the pool-wide mean cost.

**Decision rule, open point P5.**
- (a) **Rule A on shrunk values** (rec.): in each group, let k be the direct-dispatch candidate
  with the highest shrunk acceptance (ties to the cheaper, then the lower column; P0.6's CellBest
  rule). Choose the cheapest candidate whose shrunk acceptance is at least â_gk − m; ties to the
  higher shrunk acceptance, then the lower column. Column order: the models in matrix order, then
  the cascade.
- (b) The same with the group's best over all candidates, the cascade included, as the reference.
- (c) No margin, the highest shrunk acceptance outright (ties to the cheaper).
- **The margin m is fixed before any run at 0.02** (2 acceptance points), the value P0.5's
  cost-aware DimensionBest uses. Any other value tried is its own entry in `S4_versions.md`.
- Under (a), with κ = 0 (no shrinkage), m = 0 and P0.6's full candidate list (8 direct + 84
  ladders), v1 **is** per-cell Rule A. A test checks this reduction, and it gives a hand-checkable
  value.

**How v1 relates to Rule A escalated (C2's bar, O19).** Rule A escalated chooses among 92
candidates with unshrunk per-cell acceptance and no margin. v1 chooses among 9 candidates (one
cascade), shrinks the per-group acceptance toward the pool, and accepts a 2-point margin below the
group's best direct model. v1 is the simpler, pooled member of the same family of rules. How the
two compare is measured on the development split and reported beside each other.

### 3.3 Fallbacks
- Predicted group absent from the table, or no text: the pool row (§3.2).
- A chosen candidate whose model is unavailable for a task is handled by `PlanRouter` as in S2
  (unavailable rungs skipped; a task with no available rung gets GIVE_UP). In the complete case
  this does not occur.

### 3.4 The fixed cascade (decided)
- One two-rung ladder for all groups, chosen on the training folds among P0.5's ladders with two
  rungs (`common.ladders`, non-decreasing training mean cost): the cheapest whose training
  acceptance is at least the training best single model's; ties to the higher acceptance, then the
  lower index. If none reaches it, the two-rung ladder with the highest training acceptance (ties
  to the cheaper). This is `baselines.cascade.FixedCascade`'s rule restricted to two rungs.
- **Mode: no-signal cells escalated** (the buildable mode of C2's bar, O19) is router v1. The same
  router with no-signal cells **accepted** is logged as its own version, `v1_accept`, and reported
  beside it.
- **Combination, open point P6.**
  - (a) The cascade is one more candidate column in every group's menu, and the lookup chooses it
    only where it wins under §3.2's rule on the training folds (rec.).
  - (b) The lookup picks rung 1 per group and the cascade's top rung is always rung 2.
  - (c) The cascade runs everywhere and the lookup is used only to choose rung 1 inside it.
  - **Rec. (a)**: it uses one decision rule for everything, it reduces to Rule A as stated in
    §3.2, and a group where the cascade does not pay keeps direct dispatch with no verifier cost.

### 3.5 Reported for v1
- Development populations: §2 (P1 ruling), seeds 0–4.
- Every S2 metric: acceptance (S0.1) with its CI, cost per task (S0.2) with the verifier runs
  and dollars, CPAT at the best single model's acceptance as the floor (R3), cost at matched
  acceptance, and regret against the cost-aware oracle (S2 definition; probe cost included).
- Paired cluster-bootstrap CIs (gap in points, saving in percent) against: the best single model;
  CellBest-pred (S0.3(c)); per-cell Rule A escalated with predicted cells (O19); the fixed cascade
  (escalated), all on the same folds and the same predictor pool.
- Also reported (R2): DimensionBest and cost-aware DimensionBest (given more information),
  CellBest-true (reference), Rules A and B in both modes, zero-loss (pred and true), the fixed
  cascade accepted, Always-best.
- Fit records per fold: κ, the chosen candidate per group, the cascade chosen, the count of tasks
  routed by the pool row.
- Sensitivity: CRUXEval ≥ 0.8 (seed 0).
- Diagnostic (given more information): `v1_true_groups`, the same v1 reading the true family.

### 3.6 Tests for v1
- Toy cases with hand-checked values: the shrunk estimate, κ from the ANOVA estimator, the κ = ∞
  and κ = 0 cases, the margin rule and its ties, the pool row.
- The reduction to per-cell Rule A (κ = 0, m = 0, full candidate list).
- The S2 garbage tests (scores and costs of scored tasks replaced; decisions unchanged) and the
  shuffled-label test (toy and real headline development split, as S2.6).
- No true label: v1 replays with task handles whose `info` raises on any access.

## 4. Router v2: bounded onboarding in replay (draft for checkpoint 2)

Fixed here so checkpoint 2 has a rule to build to; the reviewer may rule now or at checkpoint 2.

- **Arrivals.** Leave-one-model-out: for each of the eight models, v1 is fitted on the training
  folds without that model, and the model arrives at the start of the scored stream
  (`replay(arrivals=...)`). Reported per model, never pooled (R8).
- **Label sources** (S0.5): ground truth for probes (pairing 1); verifier-only for probes
  (pairing 2). Exploration traffic is always verifier-only.
- **Exploration, open point P7.** Exactly 10% of each predicted group's requests
  (`EXPLORATION_SHARE`) until the group stops or the model's total calls reach k.
  - (a) Deterministic: the j-th request of predicted group g (counted from the arrival, 1-based)
    is exploration iff j is a multiple of 10 (rec.: exactly ⌊n_g / 10⌋, no draw);
  - (b) a Bernoulli(0.1) draw per request from stream `s4_explore` (id 20): 10% in expectation
    only, which does not meet "exactly".
  - An exploration request calls the new model first, verified. PASS is accepted; FAIL or UNKNOWN
    continues with v1's plan for that task. The new model's call is charged.
- **Calibration probes.** Charged calls on training-pool tasks of the group (training folds only,
  S0.6), at most 5 per group and 100 per model (`MAX_PROBES_PER_GROUP`, `MAX_PROBES_PER_MODEL`),
  drawn without replacement from stream `s4_probe_draw` (id 21). Cost = the new model's released
  cost on the probe task, plus a verifier run when the verifier-only pairing runs one.
- **The posterior the stopping rule reads (without the S4.3 ability model), open point P8.**
  Per (new model, predicted group): Beta(α, β) on the pass rate.
  - Prior mean = the pool prior: the known models' shrunk group acceptance under v1's κ
    (R11: never θ = 0). Prior strength, options: (a) the method-of-moments spread of the known
    models' rates around that mean in the group (rec.: the prior expresses how much models differ,
    which is the uncertainty about a new one); (b) a weak fixed strength α + β = 2.
  - Update: PASS adds 1 to α, FAIL adds 1 to β; **UNKNOWN never updates** (S0.13) but counts as a
    call toward k and the budget.
  - SD on the logit scale: the exact standard deviation of logit(p) under Beta(α, β),
    √(ψ₁(α) + ψ₁(β)) (ψ₁ the trigamma function). The offset's SD equals it, since the pool mean
    is fixed. Stop when it is ≤ `STOP_POSTERIOR_SD` (0.5).
- **After stopping:** the new model enters the group's menu as a direct candidate at its posterior
  mean, with the mean cost of its calls in that group (or its pool-wide observed mean cost).
  **Out of budget first:** the group uses the pool prior for the new model (S0.13, R11).
- **Cost** (S0.2): probe and exploration cost is charged to the stream and spread over the tasks
  scored (`probe_usd`, `n_probes`, read by `run_dev`); the break-even stream length is reported.
- **Comparators:** v1 without the new model (k = 0, pool prior) and k-matched DimensionBest at the
  same k (`lookup.KMatchedDimensionBest`, same label source).

## 5. S4.6: the non-transductive rule

- **Rule.** A scored task's difficulty, or any other item parameter, is never fitted from any
  model's outcomes on that task or on any task of its problem. Item parameters are fitted on
  training-fold tasks only; a scored task's difficulty comes only from a predictor applied to its
  `content_text` (S4.4).
- **The bug it fixes.** `router/experiments/m3_irt.py` `a6_a7` fits `irt.fit` on the probe items
  and the test items together (`items = probe.tasks.append(test.tasks)`), so the test items'
  difficulties are fitted from the other seven models' outcomes on them.
- **Enforcement.** `router/core/guards.py`:
  `check_non_transductive(fit_item_ids, scored_ids, problem_of)` raises
  `TransductiveFitError` when a fitted item is a scored task or shares a problem with one. Every
  item-parameter fit in `router/core/` calls it. Tests:
  1. a reproduction of the `a6_a7` pattern (fit on probe + test items) is refused, and the
     problem-sharing case is refused;
  2. a fit on training items only passes;
  3. an AST scan of `router/core/` lists every function named `fit` or `fit_*`, and every function
     that calls a fitting routine (`.fit`, `partial_fit`, `fit_transform`, `minimize`,
     `minimize_scalar`, `curve_fit`, `least_squares`), and asserts that each calls the guard or is
     listed in `guards.NOT_ITEM_FITS` with the reason it fits no item parameter.
     `router/core/irt.py` (quick-check reference code, 2026-11-10) is exempt by name until S4.3
     rebuilds it; the exemption list is in the test and in this section.

## 6. v3 and v4 components (checkpoint 3)

Not built at checkpoint 1. Their rules are in the handover (S4.3–S4.5) and plan §2; this spec gets
a section for each before its code, as for v1.

## 7. Stream ids

| Id | Stream | Used by | Status |
|---|---|---|---|
| — | none | v1 (deterministic: no draw anywhere) | — |
| 20 | `s4_explore` | v2, only if P7 (b) is ruled | reserved |
| 21 | `s4_probe_draw` | v2 calibration probes | reserved |
| 22 | `s4_rescue_split` | S4.5 held-out split for t inside the training folds | reserved |
| 23 | `s4_sim` | S4.3–S4.5 simulated data in unit tests | reserved |
| 24–29 | — | spare | reserved |

Each id is registered in `stats.STREAM_IDS` in the commit that first uses it.

## 8. Open points for the reviewer

| # | Point | Rec. |
|---|---|---|
| P1 | All-eligible and sensitivity: tasks without an X13 record | (b) X13 tasks only, all tasks as a labelled variant |
| P2 | Group predictor training pool | (a) each population's own training folds, comparators the same |
| P3 | Labels the table is built on | (a) training tasks' true families |
| P4 | Shrinkage | (a) Beta-binomial, one shared κ by the ANOVA estimator |
| P5 | Decision rule and margin | (a) Rule A on shrunk values, m = 0.02 fixed |
| P6 | Lookup and cascade combination | (a) the cascade as one more candidate column |
| P7 | Exploration selection (v2) | (a) deterministic every tenth request per predicted group |
| P8 | v2 prior strength | (a) method-of-moments spread of the known models |

## 9. Change log

- 2026-11-24: written; v1 built on the recommendations above, pending rulings.
- 2026-11-24 (after the first v1 run, before any ruling):
  - §2: the scored counts under P1 (b) are all-eligible 459 tasks / 355 problems / 13 families and
    sensitivity 279 / 244 / 6 (the table above gives the complete case before the X13 restriction).
    No task scored under P1 (b) lacks `content_text`, so the pool row routes none of them.
  - §3.5: "cost at matched acceptance" is not computed at checkpoint 1. S2's tables never reported
    it, and one policy is one operating point, not a curve. Proposal: trace v1's curve over the
    margin m in S5 E2, each m logged as its own version.
  - §3.6: the real-data shuffled-label test (S2.6 protocol, shuffle seeds 0-19) fails for v1:
    shuffled mean 46.56 [43.40, 49.73] against Random's expected 50.38. Shuffle seeds 20-119 give
    51.33 [49.75, 52.92]. The test was not changed after the failure; it is marked
    `xfail(strict=True)` with these numbers for the reviewer to rule on.
    `router_v1_accept` misses the same way (46.67 [43.47, 49.87]) and is marked the same way.
  - §2, evidence for P1: of the 507 all-eligible complete-case development tasks with no X13
    record, 4 have a success for any model (15 successful cells); of the 15 such sensitivity
    tasks, none has. Under P1 (a) these tasks add almost no acceptance to any policy, and the
    savings they add come from cost spent on tasks no model solves.
  - §5: the scan is by name (`fit`, `fit_*`) and also by call (a function that calls a fitting
    routine); the "registered as an item fit" clause is dropped, as no registry exists.
