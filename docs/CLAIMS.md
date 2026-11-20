# PROBE-CASCADE claim contract

| Field | Value |
|---|---|
| Status | **Draft for user sign-off** (P0.4). Not in force until the user signs it off (S0 Gate) |
| Date | 2026-11-20 |
| Written from | `docs/IMPLEMENTATION_PLAN.md` at commit `1921063` (§1, §2, S0.1–S0.14, §6) |
| Constants | `router/settings.py`, section "Claim constants (docs/CLAIMS.md)"; checked by `tests/router/test_claims_constants.py` |
| Change rule | After sign-off, a change needs a progress-log entry giving the reason, and every result already computed is re-scored (S0 Gate) |

This document states what is measured, how, against what, and what counts as success or failure,
so that no result can be reinterpreted after it is seen. It records decisions already in the plan;
it does not make new ones. Each rule cites its source. Where the plan is silent, vague or
inconsistent, the item is listed under [Open items for sign-off](#open-items-for-sign-off) and is
not decided here.

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
| (a) | Best single model | Cost saving, % | **≥ 20%** (`GATE_SAVING`) |
| (b) | Best single model | Acceptance gap, points (paired) | **≥ −2 points** (`GATE_GAP`) |
| (c) | CellBest with predicted cells, fitted in the same training folds (R2) | Cost saving, % (paired) | **> 0** (`C_SAVING`, strict) |
| (c) | CellBest with predicted cells, fitted in the same training folds (R2) | Acceptance gap, points (paired) | **≥ −2 points** (`C_GAP`) |

- Cost and acceptance are both judged on CI lower bounds (S0.3; user decision 2026-11-19).
- **Best single model** is chosen inside each training set, on the same eligibility and population
  as the scored fold (for cascade claims: the training folds' S0.11 headline tasks), never on the
  scored fold (S0.3).
- **The S0.3(c) comparator is fixed now and is not changed after results are seen** (S0.3).
  CellBest with true source cells is reported next to it as a reference, not as a gate (S0.3, R2).
- The gate uses one pre-registered seed; the other four seeds are reported next to it (S0.8). No
  gate on seed-averaged estimates (R12).
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
- The X13 hidden-test outcome is reported only as a sensitivity check.
- **Live runs:** the ground truth is the reserved hidden test pool.
- The router's own verifier decides escalation; it never counts as acceptance.

### S0.2 Cost
- Cost = model tokens × published price (source URL and access date in `router/data/prices.py`)
  + verifier runs + retries + probe calls, spread over the stream. Actual spend is not the metric
  (R7, R15).
- **Verifier runs** are reported as a count and in dollars: the X13 mean visible-test run time,
  **0.074 s per run** (`VERIFIER_RUN_S`), priced at a published on-demand cloud vCPU-hour rate,
  about **$10⁻⁶ per run** (`VERIFIER_RUN_USD`), swept at **×1, ×10, ×100**
  (`VERIFIER_COST_SWEEP`). The rate's source and date are an open item (O11).
- Probe cost is spread over the number of tasks scored; the break-even stream length is reported.
- A price-sensitivity sweep is run on every headline number (multipliers: open item O13).

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
- **Development split:** **20% of problems** (`DEV_FRACTION`), drawn once with a recorded seed (seed:
  open item O6). All open-ended, human-directed iteration happens here and nowhere else (S0.7).
- **Evaluation population:** the other 80% of problems, in **5 outer folds** (`N_FOLDS`).
- Before the evaluation population is touched, the candidate list (router versions v1–v4 and their
  hyperparameter grids) and the selection rule are frozen and tagged in git.
- For each outer fold: the configuration is chosen automatically on inner folds formed within the
  other four outer folds (number of inner folds: open item O7), fitted on those four, and the
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
  pre-registered gate seed (value: open item O5) plus four reported next to it.
- Leave-one-model-out results are shown per model, never pooled (R8).
- **At the threshold:** a point estimate that clears its threshold on the gate seed is reported as
  "at the threshold" when it falls to the threshold or below on another seed, when its CI lower
  bound is at or below the threshold, or when its margin is under **0.1 points**
  (`AT_THRESHOLD_MARGIN`; units and interaction with S0.3: open item O12).
- Number of bootstrap resamples: open item O4.

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
  uses the full S0.6 pool; only cascade scoring is restricted (conflict with S0.3: open item O1).
  Tasks without verdicts are not scored for cascade claims.
- Cascade results are worded as holding for runnable tasks; S3 reports the headline set's family
  mix, pass rates and unsolved share next to the pool's.
- **No visible tests means no verifier signal**, decided per cell. Where a cell has visible tests,
  the recorded verdict is used. The X13 syntax/quality heuristic is **not** used; the
  heuristic-accept variant is a sensitivity check.
- **Missing responses are missing, not failures.** Empty X13 responses and the zero-score,
  zero-cost Qwen3-Max rows mark the model as unavailable for that task: the rung is skipped in
  replay, the cell is not scored, and the counts are reported. S2 fixes how a baseline that needs
  such a cell is scored (open item O2). Route Anything's derived X13 tables, which impute the
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
  model** (`MAX_PROBES_PER_GROUP`, `MAX_PROBES_PER_MODEL`; "group": open item O9).
- **Exploration traffic:** live requests routed to the new model, "about 10% per group"
  (`EXPLORATION_SHARE`, open item O8), returning PASS, FAIL or UNKNOWN from the verifier only. A
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
| DP1a | End of S3 | Headroom of the oracle and of cross-fitted DimensionBest over the best single model, on the **full** artefact-filtered eligible pool. Passes if the 95% cluster-bootstrap CI lower bound is **≥ 3.0 AvgPerf points** (`DP1A_MIN_LOWER`; "AvgPerf": open item O10) | C4 leads the accuracy story |
| DP1b | End of S3 (S3.2) | Does the **oracle-on-no-signal bound** meet S0.3 on **point estimates** on the headline set, verifier cost included: saving ≥ 20% and gap ≥ −2 points against the best single model, **and** saving > 0 and gap ≥ −2 points against CellBest with predicted cells? The bound uses the real verifier where visible tests exist and an oracle accept-or-escalate decision on no-signal cells. Fixed ladders are not the test. Also report the gap the learned router must close | Stop and bring it to the user before S4 (LLMRouterBench becomes a candidate primary corpus for C1 and C2). A reassessment, not a verdict that a learned policy cannot succeed. This half decides the economic claim |
| DP2 | During S5 | Does the hierarchical model beat k-matched DimensionBest in routing units (AvgPerf, cost) at a k in **{10, 25, 50, 100}** (`DP2_K_GRID`), on the development split? | The routing model is the shrunk lookup table; IRT is reported as an ablation |
| DP3 | S6 pilot | Is the routing gap on the live pool (free models plus the paid top model) **more than 2 ×** the run-to-run SD (`DP3_GAP_SD_MULTIPLE`), measured on **≥ 2 runs per task** (`DP3_MIN_RUNS_PER_TASK`)? ("routing gap": open item O14) | Live work is reported as a verifier study and a transfer test, not as a routing win |

---

## 4. Contributions (§1)

| # | Claim | Minimal evidence | If it fails |
|---|---|---|---|
| C1 | How many verified outcomes a new model needs before routing to it matches a lookup table built from the same k outcomes (k-matched DimensionBest, same label source), and whether a pooled hierarchical prior, in-stream outcomes from a partial verifier and a per-group stopping rule shorten that; single pass, no repeated problem (S0.5) | E1: four onboarding arms against k-matched DimensionBest on both corpora, with CIs | Report the number of shared probes a lookup table needs as the measured answer to RQ-A |
| C2 | A calibrated per-task classifier of P(a later rung passes \| earlier rungs failed), with a give-up action, under the real visible-test verifier (S0.11) | E2: cost at matched acceptance on exec-scored tasks, against cost-aware DimensionBest, the fixed cascade and Route Anything's zero-loss direct-dispatch route (R2) | Report how far a realistic predictor falls short of the perfect-truncation bound |
| C3 | False-pass rate of a verifier that sees a fraction f of each web-dev task's tests, against a disjoint hidden grader, measured live (E4), and its effect on cascade cost and acceptance | E4: visible vs held-out tests on web-dev, false-pass curve over f with a task-level Clopper-Pearson bound; CodeRouterBench replay with the X13 verdicts as context | Report the web-dev curve as a characterisation; if E4 cannot be run, C3 counts as covered and S0.14 is re-applied |
| C4 | How much of the oracle's and DimensionBest's gain over the best single model survives the raw-response audit of the suspected families and run noise (fallback). Suspected anomalies without execution evidence are sensitivity analyses, not confirmed defects | S1.7 audit plus the three data views; noise-corrected oracle on web-dev | If headroom survives, CodeRouterBench is cleared as primary data |

---

## 5. Success criteria (§1)

| # | Criterion | Measured by |
|---|---|---|
| SC1 | E1 onboarding curves in routing units (AvgPerf, cost) for the onboarding arms against k-matched DimensionBest, with CIs | S5, S7 |
| SC2 | Cascade with stop-or-escalate against cost-aware DimensionBest and the fixed cascade at matched acceptance, exec-scored tasks, paired CIs; or an honest negative | S5, S7 |
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

Every number above, its name in `router/settings.py` and its source. "Pending sign-off" marks a
value the plan does not set; P0.5 and P0.6 used it, and it is listed as an open item.

| Name | Value | Source | Note |
|---|---|---|---|
| `SUCCESS_SCORE` | 1.0 | S0.1 | Success = all tests pass |
| `VERIFIER_RUN_S` | 0.074 s | S0.2 | X13 mean visible-test run time |
| `VERIFIER_RUN_USD` | $10⁻⁶ | S0.2 | Rate source and date: O11 |
| `VERIFIER_COST_SWEEP` | ×1, ×10, ×100 | S0.2 | |
| `GATE_SAVING` | 20% | S0.3(a) | Lower bound ≥ |
| `GATE_GAP` | −2 points | S0.3(b) | Lower bound ≥ |
| `C_SAVING` | 0% | S0.3(c) | Lower bound > (strict) |
| `C_GAP` | −2 points | S0.3(c) | Lower bound ≥ |
| `CI_LEVEL` | 0.95 | S0.3, S0.8 | |
| `DEV_FRACTION` | 0.20 of problems | S0.6 | |
| `N_FOLDS` | 5 | S0.6 | Outer folds, by problem |
| `N_SEEDS` | 5 | S0.8 | Gate seed + four |
| `AT_THRESHOLD_MARGIN` | 0.1 points | S0.8 | O12 |
| `GATE_SEED` | 0 | S0.8 | Pending sign-off (O5) |
| `N_BOOT` | 1,000 | S0.8 | Pending sign-off (O4) |
| `LIVE_BUDGET_USD` | $20 | S0.9 | Hard cap in the provider console |
| `PAID_TOP_MODEL` | `claude-opus-4-6` | S0.9 | |
| `PAID_TOP_PRICE_USD_PER_MTOK` | $5 / $25 | S0.9 | Input / output, per M tokens |
| `MIN_RUNG_PRICE_RATIO` | 3× | S0.9 | Between neighbouring rungs |
| `MIN_LIVE_FAMILIES` | 2 | S0.9 | |
| `MIN_LIVE_PROVIDERS` | 2 | S0.9 | |
| `LIVE_HELD_OUT_MODELS` | 1–2 | S0.9 | |
| `MAX_PROBES_PER_GROUP` | 5 | S0.13 | "Group": O9 |
| `MAX_PROBES_PER_MODEL` | 100 | S0.13 | |
| `EXPLORATION_SHARE` | 0.10 | S0.13 | Plan says "about 10%": pending sign-off (O8) |
| `STOP_POSTERIOR_SD` | 0.5 (logit) | S0.13 | |
| `MIN_FAMILY_TASKS` | 30 | S0.13, S1.1 | |
| `DP1A_MIN_LOWER` | 3.0 points | DP1a | "AvgPerf": O10 |
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
| Best single model, complete case | Opus 67.3% in every training fold (gpt-5.4 53.2%, qwen3.5-plus 42.2%) | P0.5 |
| CellBest, true cells, against Opus | saving 22.9% [20.8, 25.2], gap +3.0 [+1.5, +4.5]: passes (a)–(b) | P0.5 |
| CellBest, predicted cells (the comparator), against Opus | saving 18.2% [16.4, 20.2], gap +2.9 [+1.5, +4.3] | [P0.6](research/P0.6_cellbest_bound.md) |
| Cell predictor held-out accuracy (TF-IDF prototype) | 96.2% | P0.6 |
| Oracle-on-no-signal bound against CellBest (seed 0, point) | Rule A 36.6% / +1.6; Rule B 16.4% / +8.1 | P0.6 |
| Per-cell choice without an oracle against CellBest (point) | 1.7% to 3.3% saving, gap −0.3 to +0.3 | P0.6 |
| Gap CI half-width against CellBest | 0.3 to 2.3 points, by policy | P0.6 |
| CI half-widths and gate pass rates of stand-in policies | see the power doc | [P0.4 power](research/P0.4_power.md) |

---

## Open items for sign-off

Each item quotes the plan text (commit `1921063`). None is decided here.

**O1. Training population.** S0.3: "'Best single model' is chosen inside each training set, on the
same eligibility and population as the scored fold (for cascade claims: the training folds' S0.11
headline tasks)". S0.11: "Folds still cover the full S0.6 pool and training uses all of it; only
cascade scoring is restricted." P0.5 and P0.6 fitted everything on the headline set's training
folds (P0.5 open question 3; P0.6 open question 4). Which population trains routes, baselines and
the best-single choice?

**O2. Scoring population for unavailable cells.** S0.11: unavailable cells mean "the rung is
skipped in replay, the cell is not scored, and the counts are reported. S2 fixes how a baseline
that needs such a cell is scored." The provisional set has 1,971 tasks; P0.5, P0.6 and E0 use the
1,907-task complete case. Is the gate scored on all headline tasks with rungs skipped, or on the
complete case?

**O3. The S0.3(c) comparator's predictor.** S0.3(c): "Against CellBest with predicted cells (R2),
fitted in the same training folds". R2: "predicted cells as the S0.3(c) comparator". The comparator
is defined: a CellBest table fitted on the true cells of the training folds, with each task routed
by its predicted cell. The cell predictor is built in S2.5; P0.6's TF-IDF predictor is a prototype.
To decide: is the predictor's specification frozen before the evaluation population is touched,
and is it the P0.6 prototype or the S2.5 build? Related evidence (E0, [P0.4 power](research/P0.4_power.md),
provisional): CellBest's own saving against the best single model is 18.2% on the 1,907-task
complete case, but 15.4% to 32.9% on the five evaluation populations left after removing a 20%
development split (gate seed, development seeds 0–4). On development seed 0 it passes S0.3 (a) and
(b) by itself: its `code_generation|bcb` cell flips from Opus to qwen3.5-plus, which are close in
that cell. CellBest picks the highest training acceptance per cell, ties to the cheaper model
(P0.5); there is no rule for near-ties. The comparator's definition is unchanged by this; the
item records how much its strength depends on the population draw.

**O4. Number of bootstrap resamples.** S0.8 fixes the CI type and level but not the number of
resamples. P0.5, P0.6 and E0 used 1,000 (`N_BOOT`).

**O5. Gate seed.** S0.8: "gates use a single pre-registered seed". No value is given. P0.5 and P0.6
used seed 0 (`GATE_SEED`).

**O6. Development-split seed and pool.** S0.6: "20% of problems, drawn once with a recorded seed."
No seed is recorded. E0 uses a provisional seed (0) and draws 20% of the problems of the
artefact-filtered pool.

**O7. Number of inner folds.** S0.6: "the configuration is chosen automatically on inner folds
formed within the other four outer folds". The number of inner folds is not set.

**O8. Exploration share and budget.** S0.13: "about 10% per group" and "when its probe and
exploration budget is used". The share is approximate and the exploration budget has no number.

**O9. Meaning of "group".** S0.13: "up to 5 probes per group"; "stop exploring a group when the
posterior SD of its offset is ≤ 0.5". §2: "g is the task's group (dimension or family)". Which one
is the group for the probe limits and the stopping rule?

**O10. AvgPerf.** DP1a: "≥ 3.0 AvgPerf points"; DP2 and SC1: "routing units (AvgPerf, cost)". S2.3
lists "AvgPerf, cost (S0.2), acceptance rate (S0.1)" as separate metrics. AvgPerf is not defined.
If it is the mean released score, partial credit counts, unlike S0.1 (success = score 1.0). E0
reports both readings on the DP1a pool.

**O11. Verifier rate source.** S0.2: "priced at a published on-demand cloud vCPU-hour rate, with its
source and date in `CLAIMS.md` (about $10⁻⁶ per run…)". No source or date is recorded in the plan,
P0.5 or P0.6. $10⁻⁶ per 0.074 s run implies about $0.049 per vCPU-hour. A source and access date
are needed.

**O12. "At the threshold" units and effect.** S0.8: "when its CI lower bound is at or below the
threshold, or when its margin is under 0.1 points". S0.3(a): "the lower end … is at least 20%".
(i) Savings are in percent, so "0.1 points" is ambiguous for (a) and (c). (ii) A lower bound
exactly at a threshold passes S0.3 ("at least") but is labelled "at the threshold" by S0.8. Is
"at the threshold" a reporting label only, or does it change the verdict?

**O13. Price-sensitivity sweep.** S0.2: "A price-sensitivity sweep is run on every headline
number." The multipliers or price scenarios are not set (the verifier sweep, ×1/×10/×100, is).

**O14. DP3 "routing gap".** DP3: "Is the routing gap on the live pool … more than twice the
run-to-run standard deviation". Which gap is meant (oracle over the best single model, or the
router over the best single model) is not stated.

**O15. Stale pass rule in §1.** §1, "Why v1's framing changed": "39.3% of ID-test tasks (all
dimensions, pass = score ≥ 0.5) that no model solves". S0.1 sets success = score 1.0. This document
follows S0.1; the §1 figure is historical.

**O16. Half-width wording for (c).** S0.3: "gap CI half-width 0.3–1.8 points against CellBest".
The P0.6 checkbox and log entry: "0.3 to 2.3 points". The difference is whether the fixed ladders
(2.33) are included. Wording only.

**O17. "Paired" in S0.3(a).** S0.3(a): "the lower end of the 95% cluster-bootstrap CI of its cost
saving"; (b) and (c) say "paired". P0.5, P0.6 and E0 compute the saving on the same paired cluster
resamples as the gap. Confirm that (a) is paired too.
