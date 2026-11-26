# PROBE-CASCADE claim contract

| Field | Value |
|---|---|
| Status | **Signed off by the user on 2026-11-20** (S0 Gate). In force; amended 2026-11-20 (A1), 2026-11-21 (A2, reversed for the headline by A3), 2026-11-23 (A4) and 2026-11-25 (A5) ([Amendments](#amendments)) |
| Date | Drafted 2026-11-20 (P0.4); signed off 2026-11-20 with the decisions in [Sign-off decisions](#sign-off-decisions-2026-11-20) |
| Written from | `docs/IMPLEMENTATION_PLAN.md` at commit `1921063` (§1, §2, S0.1–S0.14, §6) |
| Constants | `router/settings.py`, section "Claim constants (docs/CLAIMS.md)"; checked by `tests/router/test_claims_constants.py` |
| Change rule | After sign-off, a change needs a progress-log entry giving the reason, and every result already computed is re-scored (S0 Gate) |

This document states what is measured, how, against what, and what counts as success or failure,
so that no result can be reinterpreted after it is seen. It records decisions already in the plan;
it does not make new ones. Each rule cites its source. Where the plan was silent, vague or
inconsistent, the draft listed an open item (O1–O19); the user decided each at sign-off, and the
decisions are written into the rules below and listed under
[Sign-off decisions](#sign-off-decisions-2026-11-20).

All task counts are provisional (S0): they come from a prototype that has not passed S1, and every
count that depends on eligibility (S0.12) or the audit (S0.10) is frozen in `data/artefacts.yaml`
only after S1.4 and S1.7.

---

## 1. Goal and the two gates

**Goal 1** (§1): reach the best single model's acceptance rate at clearly lower cost, and beat the
best lookup table (CellBest). The metric is cost per accepted task (CPAT), always with an
acceptance floor (R3). Goals 2 (onboard a new LLM from few outcomes without retraining) and 3
(usable CLI and static report) are judged through C1 and SC6.

### 1.1 Performance gate (S0.3)

Scored on the nested cross-fitted evaluation scores (S0.6), on the headline set for cascade claims
(S0.11), with cluster-bootstrap CIs over problems (S0.8). All three conditions must hold.

| | Comparator | Quantity | Rule on the 95% CI lower bound |
|---|---|---|---|
| (a) | Best single model | Cost saving, % (paired; O17) | **≥ 20%** (`GATE_SAVING`) |
| (b) | Best single model | Acceptance gap, points (paired) | **≥ −2 points** (`GATE_GAP`) |
| (c) | CellBest with predicted cells, fitted in the same training folds (R2) | Cost saving, % (paired) | **> 0** (`C_SAVING`, strict) |
| (c) | CellBest with predicted cells, fitted in the same training folds (R2) | Acceptance gap, points (paired) | **≥ −2 points** (`C_GAP`) |

- Cost and acceptance are both judged on CI lower bounds (S0.3; user decision 2026-11-19), on
  **2,000** cluster-bootstrap resamples over problems (`N_BOOT_GATE`; O4).
- **Scoring population:** the complete case of the headline set. A task with any unavailable cell
  (S0.11) is dropped for every policy and counted; all headline tasks with unavailable rungs
  skipped are a sensitivity check (O2).
- **Training population:** the best single model and every fitted baseline (CellBest,
  DimensionBest, the per-cell rules) are fitted on the training folds of the population being
  scored. The router may also learn from the full S0.6 pool's training folds; no problem of the
  scored fold enters either (O1).
- **Best single model** is chosen inside each training set, on the same eligibility and population
  as the scored fold (for cascade claims: the training folds' S0.11 headline tasks), never on the
  scored fold (S0.3).
- **The S0.3(c) comparator is fixed now and is not changed after results are seen** (S0.3).
  CellBest with true source cells is reported next to it as a reference, not as a gate (S0.3, R2).
  The cell predictor is built in S2.5 on the development split and frozen with the candidate list
  before the evaluation population is touched; CellBest keeps its argmax rule (highest training
  acceptance, ties to the cheaper model) with no near-tie rule. Its strength depends on the
  development-split draw (15.4% to 32.9% saving against the best single model across draws,
  P0.4); this is recorded, not corrected (O3).
- The gate uses one pre-registered seed, **0** (`GATE_SEED`; O5); the other four seeds are reported
  next to it (S0.8). No gate on seed-averaged estimates (R12).
- Power is reported with the result; the −2 margin is kept even if power is about 60–70% (S0.3,
  S3.7; user decision 2026-11-19).

### 1.2 Research gate (S0.4)

Passes when every decision point (DP1a, DP1b, DP2, DP3) and every contribution (C1–C4) ends in a
verdict — **holds**, **fails** or **not reached** — with quoted evidence. A negative verdict passes
this gate (S0.4, §1).

---

## 2. Definitions

### S0.1 Acceptance and ground truth
- A task is accepted only when the ground truth passes it.
- **Replay:** the ground truth is the benchmark's released score. Success means all tests pass:
  **score = 1.0** (`SUCCESS_SCORE`), including execution-scored families with partial credit.
- **CRUXEval (amendments A2 and A3):** the headline success rule is 1.0, as for every family.
  - CRUXEval's released values are only 0, 0.8 and 1.0. A2 made 0.8 a success; A3 reversed that
    for the headline.
  - Counting a CRUXEval score ≥ 0.8 (`CRUXEVAL_SUCCESS_SCORE`) as success is reported as a
    **sensitivity check**: "correct value, format penalty".
- The X13 hidden-test outcome is reported only as a sensitivity check.
- **Live runs:** the ground truth is the reserved hidden test pool.
- The router's own verifier decides escalation; it never counts as acceptance.

### S0.2 Cost
- Cost = model tokens × published price (source URL and access date in `router/data/prices.py`)
  + verifier runs + retries + probe calls, spread over the stream. Actual spend is not the metric
  (R7, R15).
- **Verifier runs** are reported as a count and in dollars: the X13 mean visible-test run time
  over the cells where a verifier ran, **0.200 s per run** (`VERIFIER_RUN_S`), priced on Route
  Anything's serverless basis (AWS Lambda x86, 2 GB, us-east-1, plus the per-request charge;
  2611.05275 C.4), about **$6.9 × 10⁻⁶ per run** (`VERIFIER_RUN_USD`), swept at **×1, ×10,
  ×100** (`VERIFIER_COST_SWEEP`). The URL, access date and the recomputation are in
  `router/data/prices.py` (S1.10, O11). Amended 2026-11-20 (amendment A1): the signed-off
  0.074 s and $10⁻⁶ averaged in the cells with no visible tests as 0 s.
- Probe cost is spread over the number of tasks scored; the break-even stream length is reported.
- A price-sensitivity sweep is run on every headline number: each model's price ×0.5 and ×2, one
  model at a time, and the range is reported (`PRICE_SWEEP`; O13).

### S0.3 Performance gate
See §1.1.

### S0.4 Research gate
See §1.2.

### S0.5 k and label source
- k = the number of calls made to the new model.
- Every onboarding arm and baseline with the same k, including k-matched DimensionBest, receives
  exactly the same k outcomes from the same label source.
- E1 is reported in two pairings: ground-truth labels for all arms; verifier-only labels
  (PASS/FAIL/UNKNOWN) for all arms.
- In E1 the held-out model's outcomes are absent from every fitting and selection step, except the
  probes and online feedback it is allowed.
- E1 is scored on a single pass over the stream with no repeated problem (R13 problem id, not task
  id). A test checks this.

### S0.6 Development split and nested cross-fitting (option (a))
- **Task pool:** CodeRouterBench ID probing and ID-test splits pooled, eligible tasks only (S0.12).
  Problems (variants of one problem kept together) are the unit of every split.
- **Development split:** **20% of problems** (`DEV_FRACTION`), drawn once with seed **0** (`DEV_SEED`;
  O6), fixed before any router result exists. All open-ended, human-directed iteration happens here and nowhere else (S0.7).
- **Evaluation population:** the other 80% of problems, in **5 outer folds** (`N_FOLDS`).
- Before the evaluation population is touched, the candidate list (router versions v1–v4 and their
  hyperparameter grids) and the selection rule are frozen and tagged in git.
- For each outer fold: the configuration is chosen automatically on inner folds formed within the
  other four outer folds (**4** inner folds, one per remaining outer fold, `N_INNER_FOLDS`; O7),
  fitted on those four, and the
  held-out fold is scored once.
- Every evaluation task is scored once, by a configuration that never saw its problem and was
  chosen without it. Onboarding probes are drawn from the training folds only.
- The original ACRouter split is used only to reproduce the paper's numbers (R13).

### S0.7 Iteration rule
- Router changes are made and judged on the development split only.
- The nested evaluation and the live hidden tests are run once, with the frozen candidate list and
  selection rule (R14).
- Every version tried on the development split is logged and reported.

### S0.8 Statistics
- 95% CIs (`CI_LEVEL`): paired bootstrap over tasks and cluster bootstrap over problems; **the
  cluster bootstrap is the one used for the gates.** Repeated live outputs of one task are resampled
  together (cluster by task).
- A seed = fold assignment + stream order + probe draw. **5 seeds** (`N_SEEDS`): one
  pre-registered gate seed (**0**, `GATE_SEED`; O5) plus four reported next to it.
- Leave-one-model-out results are shown per model, never pooled (R8).
- **At the threshold:** a point estimate that clears its threshold on the gate seed is reported as
  "at the threshold" when it falls to the threshold or below on another seed, when its CI lower
  bound is at or below the threshold, or when its margin is under **0.1 points**
  (`AT_THRESHOLD_MARGIN`; percentage points for savings). It is a **reporting label only** and
  never changes a verdict: a lower bound exactly at a threshold that S0.3 accepts ("at least")
  passes (O12).
- Bootstrap resamples: **2,000** for gate scoring (`N_BOOT_GATE`), **1,000** elsewhere (`N_BOOT`)
  (O4).

### S0.9 Live pool and budget
| Tier | Size / type | Expected web-dev pass rate | Role |
|---|---|---|---|
| Small | about 7–17B dense, or small MoE | 20–35% | Cheapest rung |
| Mid | about 30–70B dense, or MoE about 100B total | 40–50% | Middle rung |
| Large open | ≥ 100B MoE, or a large code-specialised model | 50–60% | Strongest free rung |
| Paid frontier | `claude-opus-4-6` (`PAID_TOP_MODEL`), $5 / $25 per M tokens (`PAID_TOP_PRICE_USD_PER_MTOK`) | best | Top rung and Always-best baseline; links the two corpora |

- At least **2 model families** from at least **2 providers** (`MIN_LIVE_FAMILIES`,
  `MIN_LIVE_PROVIDERS`); one reasoning model; one code-specialised model.
- **1–2 models** held out of calibration, arriving mid-stream as an onboarding demonstration
  (`LIVE_HELD_OUT_MODELS`); exploratory, since 65 tasks cannot support the k grid, which is used in
  replay only.
- At least a **3× price gap** between neighbouring rungs (`MIN_RUNG_PRICE_RATIO`); a published paid
  price for every model.
- Each free model: pinnable id, enough quota for one collection window, terms that allow benchmark
  use.
- **Budget $20** (`LIVE_BUDGET_USD`) with a hard cap in the provider console. Every response is
  cached; model id and date recorded on every call.

### S0.10 Headline data view
- The headline is the **artefact-filtered** view; the all-tasks view is always reported next to it
  (R10: every headline number in all three data views).
- The S1.7 audit decides each suspected family on evidence from the raw responses: proven
  defective → excluded; proven sound → kept; undecided → out of the headline and in the
  sensitivity view.
- The headline is a **restricted population** (runnable, audited, test-checkable tasks); its
  results do not establish performance across all executable coding tasks.

### S0.11 Replay verifier and the headline set for cascade claims
- **Verdict source:** the real visible-test verdicts from route-anything-data X13 (2,911 tasks × 8
  models, both splits), not simulated noise (R1).
- **Headline set for cascade claims (E2, C2):** eligible, filtered tasks with X13 verdicts. Training
  uses the full S0.6 pool for the router; baselines and the best single model are fitted on the
  scored population's training folds; only cascade scoring is restricted (O1, §1.1).
  Tasks without verdicts are not scored for cascade claims.
- Cascade results are worded as holding for runnable tasks; S3 reports the headline set's family
  mix, pass rates and unsolved share next to the pool's.
- **No visible tests means no verifier signal**, decided per cell. Where a cell has visible tests,
  the recorded verdict is used. The X13 syntax/quality heuristic is **not** used; the
  heuristic-accept variant is a sensitivity check.
- **Missing responses are missing, not failures.** Empty X13 responses and the zero-score,
  zero-cost Qwen3-Max rows mark the model as unavailable for that task: the rung is skipped in
  replay, the cell is not scored, and the counts are reported. Gate scoring uses the complete
  case (§1.1; O2). Route Anything's derived X13 tables, which impute the
  released truth on empty cells, are not reused.
- Simulated noise (E3) is a sensitivity analysis only.

### S0.12 Scope and eligibility
- Router claims cover execution-scored, test-checkable tasks only (R4).
- Eligibility is decided by each source benchmark's documented scoring rule (S1.4), **never by the
  scores observed** (R16).
- Partial-credit execution families are kept, with success defined by S0.1; a family whose scoring
  is not execution-based is excluded as a whole. Similarity- and judge-scored tasks are reported
  separately.

### S0.13 Observations, probes and pinned thresholds
- **In-stream observations** are PASS, FAIL or UNKNOWN, from the visible-test verifier only.
  UNKNOWN is never turned into a training label. Replay never passes the released score to the
  router (a test checks this; R16).
- **Calibration probes:** separate, charged calls on training-pool tasks whose answers are known;
  the only observations labelled with ground truth. Limits: **5 per group** and **100 per new
  model** (`MAX_PROBES_PER_GROUP`, `MAX_PROBES_PER_MODEL`). A **group is an S1.1 family** (O9).
- **Exploration traffic:** live requests routed to the new model, **exactly 10%** of each group's
  requests (`EXPLORATION_SHARE`), until the group's stopping rule fires or the new model's total
  calls (probes plus exploration) reach k (S0.5, DP2 grid) (O8). It returns PASS, FAIL or UNKNOWN
  from the verifier only. A
  live request never becomes a ground-truth-labelled probe.
- Sensitivity check: probe labels from the visible-test verifier only (also the second E1 pairing).
- **Stopping:** stop exploring a group when the posterior SD of its offset is **≤ 0.5 on the logit
  scale** (`STOP_POSTERIOR_SD`), or when its probe and exploration budget is used. A group that runs
  out of budget first uses the pool prior. The k = 0 baseline is the pool prior, never θ = 0 (R11).
- **Rescue decisions** are trained on the observable histories (PASS, FAIL, UNKNOWN, including
  false rejections), not on the true failures.
- **Families** (S1.1): no family under **30 tasks** (`MIN_FAMILY_TASKS`); the final list with counts
  is frozen in `sources.py`.

### S0.14 Fallback lead claim
- If C2 and C3 are already covered, C1 leads. If C1 is also covered, the lead is C4 together with
  the real-verifier economics. P0.1 (2026-11-19) found C1–C4 each partly covered and none covered;
  the order C1–C4 stands (§1).

---

## 3. Decision points (§1)

| DP | When | Question and threshold | On "no" |
|---|---|---|---|
| DP1a | End of S3 | Headroom of the oracle and of cross-fitted DimensionBest over the best single model, on the **full** artefact-filtered eligible pool. Passes if the 95% cluster-bootstrap CI lower bound is **≥ 3.0 AvgPerf points** (`DP1A_MIN_LOWER`). **Cross-fitted DimensionBest must clear it; the oracle is reported as the ceiling** (O18). AvgPerf = acceptance rate under S0.1 (score = 1.0), in points; the mean released score is a sensitivity check (O10) | C4 leads the accuracy story |
| DP1b | End of S3 (S3.2) | Does the **oracle-on-no-signal bound** meet S0.3 on **point estimates** on the headline set, verifier cost included: saving ≥ 20% and gap ≥ −2 points against the best single model, **and** saving > 0 and gap ≥ −2 points against CellBest with predicted cells? The bound uses the real verifier where visible tests exist and an oracle accept-or-escalate decision on no-signal cells. Fixed ladders are not the test. Also report the gap the learned router must close | Stop and bring it to the user before S4 (LLMRouterBench becomes a candidate primary corpus for C1 and C2). A reassessment, not a verdict that a learned policy cannot succeed. This half decides the economic claim |
| DP2 | During S5 | Does the hierarchical model beat k-matched DimensionBest in routing units (AvgPerf, cost) at a k in **{10, 25, 50, 100}** (`DP2_K_GRID`), on the development split? | The routing model is the shrunk lookup table; IRT is reported as an ablation |
| DP3 | S6 pilot | Is the routing gap on the live pool (free models plus the paid top model) **more than 2 ×** the run-to-run SD (`DP3_GAP_SD_MULTIPLE`), measured on **≥ 2 runs per task** (`DP3_MIN_RUNS_PER_TASK`)? The routing gap is the noise-corrected oracle headroom over the best single model: choose on run 1, score on run 2 (O14) | Live work is reported as a verifier study and a transfer test, not as a routing win |

---

## 4. Contributions (§1)

| # | Claim | Minimal evidence | If it fails |
|---|---|---|---|
| C1 | How many verified outcomes a new model needs before routing to it matches a lookup table built from the same k outcomes (k-matched DimensionBest, same label source), and whether a pooled hierarchical prior, in-stream outcomes from a partial verifier and a per-group stopping rule shorten that; single pass, no repeated problem (S0.5) | E1: four onboarding arms against k-matched DimensionBest on both corpora, with CIs | Report the number of shared probes a lookup table needs as the measured answer to RQ-A |
| C2 | A calibrated per-task classifier of P(a later rung passes \| earlier rungs failed), with a give-up action, under the real visible-test verifier (S0.11) | E2: cost at matched acceptance on exec-scored tasks, against cost-aware DimensionBest, the fixed cascade, Route Anything's zero-loss direct-dispatch route and **per-cell Rule A with no-signal cells escalated** (R2), which C2 must beat (O19) | C2 counts as failed if it does not beat per-cell Rule A escalated at matched acceptance, whatever the S0.3 result. Report how far a realistic predictor falls short of the perfect-truncation bound |
| C3 | False-pass rate of a verifier that sees a fraction f of each web-dev task's tests, against a disjoint hidden grader, measured live (E4), and its effect on cascade cost and acceptance | E4: visible vs held-out tests on web-dev, false-pass curve over f with a task-level Clopper-Pearson bound; CodeRouterBench replay with the X13 verdicts as context | Report the web-dev curve as a characterisation; if E4 cannot be run, C3 counts as covered and S0.14 is re-applied |
| C4 | How much of the oracle's and DimensionBest's gain over the best single model survives the raw-response audit of the suspected families and run noise (fallback). Suspected anomalies without execution evidence are sensitivity analyses, not confirmed defects | S1.7 audit plus the three data views; noise-corrected oracle on web-dev | If headroom survives, CodeRouterBench is cleared as primary data |

---

## 5. Success criteria (§1)

| # | Criterion | Measured by |
|---|---|---|
| SC1 | E1 onboarding curves in routing units (AvgPerf, cost) for the onboarding arms against k-matched DimensionBest, with CIs | S5, S7 |
| SC2 | Cascade with stop-or-escalate against cost-aware DimensionBest, the fixed cascade and per-cell Rule A escalated (O19) at matched acceptance, exec-scored tasks, paired CIs; or an honest negative | S5, S7 |
| SC3 | Verifier noise results: simulated noise (E3) and the measured false-pass curve on web-dev (E4), with an upper bound when no false pass is seen | S5, S7 |
| SC4 | Headroom audit in three data views | S1, S3 |
| SC5 | At least **3** free-tier models (`SC5_MIN_FREE_MODELS`) plus the paid top model × **65** tasks (`WEBDEV_N_TASKS`), stored with per-test results; transfer test exploratory | S6, S7 |
| SC6 | CLI and static-report demo; proxy optional | S7 |
| SC7 | Findings report with a verdict per DP and contribution (research gate), and the performance-gate result | S8 |

---

## 6. Baselines always reported (R2)

A result without these is not a result (R2):

1. DimensionBest, full and k-matched;
2. cost-aware DimensionBest;
3. CellBest (Route Anything's 21-cell lookup) with **predicted cells — the S0.3(c) comparator**;
4. CellBest with true source cells — reference only, not a gate;
5. Route Anything's zero-loss direct-dispatch route (2611.05275 C.2, Table 10);
6. per-cell choice without an oracle: P0.6 Rules A and B, no-signal cells accepted or escalated;
7. fixed cascade;
8. always-best.

Other binding rules: CPAT always carries an acceptance floor (R3); cascade claims use exec-scored
tasks only (R4); multiple seeds and bootstrap CIs on every headline number (R8); the artefact list
is frozen before results (R10); problem-disjoint splits, enforced by a test (R13); the evaluation
population is scored once (R14).

---

## Constants

Every number above, its name in `router/settings.py` and its source. Values the plan did not set
were decided at sign-off (O-numbers in the note column).

| Name | Value | Source | Note |
|---|---|---|---|
| `SUCCESS_SCORE` | 1.0 | S0.1 | Success = all tests pass |
| `CRUXEVAL_SUCCESS_SCORE` | 0.8 | S0.1 (A2, A3) | CRUXEval sensitivity threshold only; the headline uses 1.0 (amendment A3) |
| `VERIFIER_RUN_S` | 0.200 s | S0.2 | X13 mean visible-test run time over cells where a verifier ran (amendment A1) |
| `VERIFIER_RUN_USD` | $6.9 × 10⁻⁶ | S0.2 | Lambda x86, 2 GB, plus per-request charge (O11; amendment A1) |
| `VERIFIER_COST_SWEEP` | ×1, ×10, ×100 | S0.2 | |
| `PRICE_SWEEP` | ×0.5, ×2 | S0.2 | Each model's price, one at a time (O13) |
| `GATE_SAVING` | 20% | S0.3(a) | Lower bound ≥ |
| `GATE_GAP` | −2 points | S0.3(b) | Lower bound ≥ |
| `C_SAVING` | 0% | S0.3(c) | Lower bound > (strict) |
| `C_GAP` | −2 points | S0.3(c) | Lower bound ≥ |
| `CI_LEVEL` | 0.95 | S0.3, S0.8 | |
| `DEV_FRACTION` | 0.20 of problems | S0.6 | |
| `DEV_SEED` | 0 | S0.6 | Development-split draw (O6) |
| `N_FOLDS` | 5 | S0.6 | Outer folds, by problem |
| `N_INNER_FOLDS` | 4 | S0.6 | One per remaining outer fold (O7) |
| `N_SEEDS` | 5 | S0.8 | Gate seed + four |
| `AT_THRESHOLD_MARGIN` | 0.1 points | S0.8 | Percentage points for savings; label only (O12) |
| `GATE_SEED` | 0 | S0.8 | O5 |
| `N_BOOT` | 1,000 | S0.8 | Outside gate scoring (O4) |
| `N_BOOT_GATE` | 2,000 | S0.3, S0.8 | Gate scoring (O4) |
| `LIVE_BUDGET_USD` | $20 | S0.9 | Hard cap in the provider console |
| `PAID_TOP_MODEL` | `claude-opus-4-6` | S0.9 | |
| `PAID_TOP_PRICE_USD_PER_MTOK` | $5 / $25 | S0.9 | Input / output, per M tokens |
| `MIN_RUNG_PRICE_RATIO` | 3× | S0.9 | Between neighbouring rungs |
| `MIN_LIVE_FAMILIES` | 2 | S0.9 | |
| `MIN_LIVE_PROVIDERS` | 2 | S0.9 | |
| `LIVE_HELD_OUT_MODELS` | 1–2 | S0.9 | |
| `MAX_PROBES_PER_GROUP` | 5 | S0.13 | Group = S1.1 family (O9) |
| `MAX_PROBES_PER_MODEL` | 100 | S0.13 | |
| `EXPLORATION_SHARE` | 0.10 | S0.13 | Exactly 10% of each group's requests (O8) |
| `STOP_POSTERIOR_SD` | 0.5 (logit) | S0.13 | |
| `MIN_FAMILY_TASKS` | 30 | S0.13, S1.1 | |
| `DP1A_MIN_LOWER` | 3.0 points | DP1a | Cross-fitted DimensionBest must clear it; AvgPerf = acceptance (O10, O18) |
| `DP2_K_GRID` | 10, 25, 50, 100 | DP2 | |
| `DP3_GAP_SD_MULTIPLE` | 2 | DP3 | |
| `DP3_MIN_RUNS_PER_TASK` | 2 | DP3 | |
| `SC5_MIN_FREE_MODELS` | 3 | SC5 | |
| `WEBDEV_N_TASKS` | 65 | SC5, S0.9 | |

---

## Provisional numbers

These motivate the rules above. They are not claims; S1–S3 recompute them on the final population.

| Figure | Value | Source |
|---|---|---|
| Headline set (filtered, X13 tasks, success = 1.0) | 1,971 tasks; complete case 1,907 | [P0.5](research/P0.5_recount.md) |
| No visible-test signal, complete case | 80.0% of tasks (1,525) and of cells | P0.5 |
| **Frozen (S1.7, A4):** headline set = artefact-filtered view | 1,062 tasks (CRUXEval 800, HumanEval-X Python 164, HumanEval completion 98); 964 problems; complete case 1,046 | `router/data/artefacts.yaml` |
| **Frozen (S1.7, A4):** best single model, complete case | Opus 80.0% (96.9% under the CRUXEval 0.8 sensitivity) | `artefacts.yaml` |
| **Frozen (S1.7, A4):** headline tasks solved by no model, complete case | 42 of 1,046 (4.0%; 5 under the CRUXEval 0.8 sensitivity) | `artefacts.yaml` |
| **Frozen (S1.7, A4):** no visible-test signal, complete case | 88.9% of tasks (930 of 1,046) | `artefacts.yaml` |
| **Measured (S3, DP1a):** cross-fitted DimensionBest headroom over the best single model, filtered = headline complete case (1,046 tasks) | 0.00 points [−0.29, +0.28]; oracle ceiling +16.0 [13.7, 18.4]. **DP1a: no** (lower bound −0.29 < 3.0): C4 leads the accuracy story | `router/experiments/out/s3_1_dp1a.csv` |
| **Measured (S3, DP1b):** oracle-on-no-signal bound (per-cell Rule A, bound mode), headline development split complete case (195 tasks), gate seed 0, point estimates | Against the best single model: saving 65.9%, gap −1.54; against CellBest-pred: saving 60.8%, gap −1.03. **DP1b: yes**, both gaps at the threshold (O12). Gap to close against Rule A escalated: +52.5 saving points at −0.51 gap | `router/experiments/out/s3_2_dp1b.csv` |
| Superseded by A4 (with `mbpp`; transparency only, not a sensitivity view) | 1,287 tasks (complete case 1,264); Opus 70.0%; no model 198 of 1,264; no signal 90.8% | `artefacts.yaml` at `faa8025` |
| Best single model, complete case | Opus 67.3% in every training fold (gpt-5.4 53.2%, qwen3.5-plus 42.2%) | P0.5 |
| CellBest, true cells, against Opus | saving 22.9% [20.8, 25.2], gap +3.0 [+1.5, +4.5]: passes (a)–(b) | P0.5 |
| CellBest, predicted cells (the comparator), against Opus | saving 18.2% [16.4, 20.2], gap +2.9 [+1.5, +4.3] | [P0.6](research/P0.6_cellbest_bound.md) |
| Cell predictor held-out accuracy (TF-IDF prototype) | 96.2% | P0.6 |
| Oracle-on-no-signal bound against CellBest (seed 0, point) | Rule A 36.6% / +1.6; Rule B 16.4% / +8.1 | P0.6 |
| Per-cell choice without an oracle against CellBest (point) | 1.7% to 3.3% saving, gap −0.3 to +0.3 | P0.6 |
| Gap CI half-width against CellBest | 0.3 to 2.3 points, by policy | P0.6 |
| CI half-widths and gate pass rates of stand-in policies | see the power doc | [P0.4 power](research/P0.4_power.md) |

---

## Sign-off decisions (2026-11-20)

The draft (P0.4, commit `d87e8b7`) listed open items O1–O18, each quoting the plan text it came
from; the reviewer added O19 from the P0.4 power results. The user accepted the reviewer's
recommendation on every item on 2026-11-20. The draft's quotes are in git history.

| # | Item | Decision |
|---|---|---|
| O1 | Training population | Baselines and the best single model: training folds of the scored population. The router may also use the full pool's training folds |
| O2 | Scoring population | Complete case; all headline tasks with unavailable rungs skipped as a sensitivity check |
| O3 | Comparator predictor and stability | S2.5 predictor built on the development split, frozen with the candidate list; argmax rule kept, no near-tie rule; draw dependence recorded, not corrected |
| O4 | Bootstrap resamples | 2,000 for gate scoring, 1,000 elsewhere |
| O5 | Gate seed | 0 |
| O6 | Development-split seed | 0, fixed before any router result |
| O7 | Inner folds | 4, one per remaining outer fold |
| O8 | Exploration | Exactly 10% of each group's requests, until the stopping rule fires or the model's total calls reach k |
| O9 | Group | The S1.1 family |
| O10 | AvgPerf | Acceptance rate under S0.1, in points; mean released score as a sensitivity check |
| O11 | Verifier rate | Route Anything's serverless pricing basis (AWS Lambda); URL, date and per-run value recorded in S1.10 |
| O12 | "At the threshold" | Reporting label only; percentage points for savings |
| O13 | Price sweep | Each model's price ×0.5 and ×2, one at a time; range reported |
| O14 | DP3 routing gap | Noise-corrected oracle headroom over the best single model (choose on run 1, score on run 2) |
| O15 | Stale "score ≥ 0.5" figure in §1 | Marked historical in the plan; this document follows S0.1 |
| O16 | (c) half-width wording | "0.3–1.8 points for per-cell policies, 2.3 for fixed ladders" in the plan |
| O17 | Paired (a) | Yes |
| O18 | DP1a headroom | Cross-fitted DimensionBest must clear 3.0; the oracle is the ceiling. Provisionally (P0.4) DimensionBest is +1.45 [0.68, 2.23], so DP1a would fail and C4 would lead the accuracy story; the verdict is taken at the end of S3 |
| O19 | Bar for C2 | S0.3 unchanged. C2 and SC2 must beat per-cell Rule A with no-signal cells escalated at matched acceptance, because that buildable policy passes all of S0.3 on the evaluation population (P0.4 §3.2) |

## Amendments

Changes after sign-off, each approved by the user and logged in the plan's progress log (Change
rule above).

| # | Date | Change | Reason | Re-scoring |
|---|---|---|---|---|
| A1 | 2026-11-20 | `VERIFIER_RUN_S` 0.074 s → 0.200 s; `VERIFIER_RUN_USD` $10⁻⁶ → $6.9 × 10⁻⁶ (S0.2) | S1.10 (batch A2) found that 0.074 s is the X13 `w1_sec` mean over all cells, counting the 80% of cells with no visible tests as 0 s; no verifier runs there and none is charged. Over the cells where a verifier ran the mean is 0.200 s, which costs $6.87 × 10⁻⁶ on the O11 basis. The old value also came from a vCPU-hour rate, not the O11 basis ($2.67 × 10⁻⁶ at 0.074 s) | The P0.5, P0.6 and E0 records keep $10⁻⁶ and are labelled so. Every result from S3 on uses the amended value. A verifier run is about 3% of the cheapest model's mean cost per task and 0.1% of Opus's; the ×10 sweep already covered the amended value |
| A2 | 2026-11-21 | CRUXEval success: released score = 1.0 → ≥ 0.8 (S0.1; `CRUXEVAL_SUCCESS_SCORE`) | S1.7 audit, checkpoint 2a. All 128 sampled 0.8 cells contain the correct output from CRUXEval's `output` field; all 35 sampled 1.0 cells give it as a bare answer. So 0.8 is a format penalty that CRUXEval's documented metric (arXiv 2401.03065v1 §3) does not impose. It is also strongly model-dependent: 0.8 is 72% of Sonnet's CRUXEval cells and 12% of gpt-5.4's, so under the strict rule the router would learn answer formatting, not correctness | The P0.5, P0.6 and E0 records keep the strict rule and are labelled so. Every result from S3 on uses A2. The strict rule is reported as a sensitivity check. No S1-or-later result existed, so nothing else is re-scored |
| A3 | 2026-11-21 | A2 reversed for the headline: CRUXEval success is again released score = 1.0 (S0.1). Counting ≥ 0.8 (`CRUXEVAL_SUCCESS_SCORE`) is a sensitivity check only | A2's evidence was a substring check: the expected value appears somewhere in every sampled 0.8 cell. But it also appears in 17% of 0-scored cells, and a fixed extraction rule confirms the value is stated as the answer in only 39 of 97 sampled 0.8 cells (S1.7 audit, checkpoint 2b; a lower bound). What 0.8 means is only partly established, and under C4 an unproven reading is a sensitivity analysis. Under the strict rule, the audit's re-run reproduces the release on 88 of 89 cells | No result had been computed under A2. The P0 records, the headline and every result from S3 on use 1.0. The 0.8 reading is reported next to the headline as a sensitivity check |
| A4 | 2026-11-23 | S1.7 label for `mbpp`: cleared → confirmed defect (S1.7 criteria Amendment 11, mechanism `extraction` / `entry_point_unstated`). The headline set loses `mbpp`'s 225 tasks: 1,287 → 1,062 (S0.11) | CodeRouterBench's `mbpp` prompts omit the entry-point name the MBPP+ harness calls (219 of 225; Route Anything's task notes say the same). The release and our re-run both fail with `NameError` and agree, so the pre-registered mechanism could not be assigned, and `mbpp` was cleared. Amendment 11, general (every family with a named entry point) and committed before relabelling, assigns it on agreeing cells: d = 78 of 97, Wilson lower bound of r 0.714. Found at the S3 checkpoint-1 review on 2026-11-22, before any DP number existed. Success stays at released score 1.0 | The S2 development records and the frozen S2.5 predictor (O3) are rerun on the new headline. S3 checkpoint 1 (unmerged) ran on the old headline and is rerun. `s1_7_gain_shares` (C4 inputs) is on the old view and is superseded by S3.1. With-`mbpp` figures are reported for transparency only |
| A5 | 2026-11-25 | S0.5 and S0.13 clarified for arms that make live calls to the new model. (1) E1's first pairing is **`gt_probes`**: calibration probes are labelled by the ground truth and every live call (exploration and routed calls) by the verifier. Every arm with the same k, k-matched DimensionBest included, receives the same mix: as many ground-truth outcomes as probes, the rest from the verifier. The verifier-only pairing is unchanged. (2) k counts every call to the new model, including calls made because the router chose it. The S0.13 budget (the cap) counts probes and exploration calls only. A verified routed call updates the posterior; an unverified one returns no verdict | S0.5's "ground-truth labels for all arms" conflicts with S0.13's "a live request never becomes a ground-truth-labelled probe" once an arm makes live calls. S0.13's "total calls (probes plus exploration)" did not foresee routed calls (S4 checkpoint 2 rulings, approved 2026-11-25). Counting routed calls in k gives the comparator at least as many labels as the router learns from | None. No result is decided on it yet; S4's development runs follow it, and E1 (S5.2, DP2) is the first decision that uses it |
