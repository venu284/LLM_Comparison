# Final Progress Report (Fall 2026): PROBE-CASCADE Design, Claims and Validated Data Layer

**Multi-LLM Comparison Platform: From Model Comparison to a Cost-Aware Router That Generalises to New Models**

**Student:** Venu Dattathreya Vemuru (811776500)<br>
**Course:** CSCI 7200, Master's Project<br>
**Covers:** August to November 2026 (Reports 7 and 8, and the work since)<br>
**GitHub:** https://github.com/venu284/LLM_Comparison.git

---

## 1. Summary

This semester the project moved from comparing five models on our own 65 web-development tasks to
building a cost-aware router for code generation, PROBE-CASCADE, on the public CodeRouterBench
release (8 priced frontier models, 9,999 in-distribution tasks). The work had three parts:

1. **August to September (Report 7):** direction reset, literature review and method design.
2. **October (Report 8):** a prototype on CodeRouterBench (data layer, baselines, IRT onboarding).
3. **November:** a stricter plan, a signed claims contract with numeric gates, and a validated
   data layer. A raw-response audit of the benchmark is in progress.

The router itself (stages S2 to S4) is designed but not yet built on the validated data layer.
No result in this report comes from the router.

Main results:

1. **The benchmark paper's numbers reproduce exactly** from the released data (oracle 57.00,
   Always-Opus 43.83, ACRouter 50.14).
2. **Simple lookup tables are the real competitor.** Picking the best model per benchmark cell
   (CellBest), with the cell predicted from the task text, saves 18.2% [16.4, 20.2] of cost
   against Claude Opus at +2.9 points of acceptance. The router must beat this.
3. **Stricter evaluation reversed the prototype's cascade result.** Report 8's cheap-first
   cascades saved 29% to 74%. With the real verifier, full-pass success and verifier cost
   counted, no fixed cascade meets the cost gate; the best is 13 points below Opus.
4. **There is room for a learned router.** If an oracle decided only the tasks the verifier
   cannot check, per-cell choice would save 48.1% at +4.5 points; without the oracle it saves
   20.9%. A learned router has to close that gap.
5. **The data layer is validated:** 5,626 eligible tasks in 20 families, task text for 98.7% of
   them (48% in Report 8), 6,997 problem ids with leak-tested splits, and 384 tests (19 in
   Report 8).

No model APIs were called this semester. All November numbers are provisional until the audit
labels are final.

---

## 2. Project Arc

| Period | Report | Question | Outcome |
|---|---|---|---|
| Nov 2025 to Jul 2026 | 3 to 6 | Which LLM is best per web-development category? | 65-task benchmark, 325 runs, 5 models. Best single model 53.8%; routing ceiling only +8.6 points, so H4 (> 15%) was rejected |
| Aug to Sep 2026 | 7 | Can a cheap model be good enough, and how do we know? | Move from selection to sufficiency routing; review of about 30 routing papers; PROBE-CASCADE design |
| Oct 2026 | 8 | Does the design work on a priced frontier pool? | Prototype on CodeRouterBench; published numbers reproduced |
| Nov 2026 | this report | Are the claims, data and evaluation sound enough to build on? | Claims contract, validated data layer, benchmark audit in progress |

**Why the direction changed.** On our five-model pool a static pair of models already reached the
oracle, so better model selection had little left to gain. The project now asks whether a cheap
model is good enough for a task, checks its output with tests, and escalates to a stronger model
when it fails. CodeRouterBench supplies eight priced frontier models with per-task outcomes, so
cost claims can be measured. Our 65 tasks remain the live showcase domain.

---

## 3. Method and Contributions

PROBE-CASCADE routes each task cheap-first. It estimates each model's chance of passing from a
per-group ability model, verifies the output with run-time tests, and escalates or stops after a
failure. A new model is brought in from a small number of verified outcomes, without retraining.
The router is built in four versions, and each part is kept only if it beats the previous version
on the development split:

| Version | Adds |
|---|---|
| v1 | Shrunk per-group lookup table and a fixed two-model cascade (the floor) |
| v2 | Bounded onboarding of a new model |
| v3 | Hierarchical ability model |
| v4 | Stop-or-escalate classifier |

A novelty check found that each planned contribution is partly covered by prior work (UniRoute,
ICL-Router, IRT-Router, Route Anything), so the contributions were narrowed:

| # | Contribution |
|---|---|
| C1 | How many verified outcomes a new model needs before routing to it matches a lookup table built from the same outcomes |
| C2 | Per-task give-up or escalate decisions under a partial verifier |
| C3 | False-pass rate of a verifier that sees only part of the tests, measured live on web development |
| C4 | Headroom audit of CodeRouterBench: how much routing gain survives a check of the raw responses |

---

## 4. Workflow

The plan has ten stages. Each ends in a gate that must pass before the next starts.

| Stage | Content | Status |
|---|---|---|
| Pre-S0 | Novelty read, data pinning, provisional feasibility checks | Done |
| S0 | Claims contract (`docs/CLAIMS.md`) | Done, signed off |
| S1 | Validate data | S1.1 to S1.6 done; S1.7 audit in progress |
| S2 to S4 | Evaluation harness and baselines; analysis and go/no-go decisions; build the router | Designed |
| S5 to S8 | Offline tuning, live pilot, final run scored once, results report | Designed |

Working rules:

- **Claims before code.** Success is a released score of 1.0. The router must save at least 20% of
  cost against the best single model with an acceptance gap of at least −2 points, both judged on
  95% CI lower bounds, and must also beat CellBest.
- **No leakage.** Splits are by problem, not by task. 20% of problems form a development split,
  and the rest are scored once with 5-fold cross-fitting.
- **Independent checking.** Separate sessions implement each step; a reviewer re-runs every
  script and re-derives the numbers before merging. Audit criteria are committed before any
  response is read.
- **Reproducibility.** 194 data files are pinned by checksum. Model outputs run only in an
  isolated Docker sandbox with no network.

---

## 5. What Was Built and Checked

| Measure | Report 8 | Now |
|---|---|---|
| Evaluation basis | Simulated verifier, success at score ≥ 0.5, verifier cost excluded | Real verifier verdicts, success at score 1.0, verifier cost included, CIs |
| Task text recovered | 48% of tasks | 98.7% of the eligible pool (5,555 of 5,626) |
| Task pool | All tasks | 20 families, 5,626 tasks with documented execution-based scoring |
| Splits | Folds grouped by prompt | 6,997 problem ids; development split of 697 problems; injected leaks caught |
| Claims | Informal | Contract with numeric gates and 19 recorded decisions |
| Tests | 19 | 384 |
| Pinned data files | 0 | 194 |

---

## 6. Results

### 6.1 Prototype (Report 8, quick checks)

| Check | Result |
|---|---|
| Reproduction: oracle / Always-Opus / ACRouter | 57.00 / 43.83 / 50.14, exact |
| DimensionBest against ACRouter (paper view) | 50.10 against 50.14, at 56% of ACRouter's cost |
| Cheap-first cascades against Always-Opus | 29% to 74% cheaper |

These came from code that the plan treats as quick checks. Section 6.2 supersedes the cascade
result; Section 7 corrects the onboarding results.

### 6.2 Cost against the best single model (provisional)

1,907 execution-scored tasks with real verifier verdicts. Opus is the best single model (67.3%
acceptance). Values are cross-fitted (seed 0), with paired 95% cluster-bootstrap CIs. The gate
needs a saving lower bound ≥ 20% and a gap lower bound ≥ −2 points.

| Policy | Saving vs Opus | Acceptance gap (points) | Gate |
|---|---|---|---|
| Fixed cascades, real verifier (best at ≥ 20% saving) | ≥ 20% | −13.1 | fail (gap) |
| CellBest, true cells (reference; uses labels the router cannot see) | 22.9% [20.8, 25.2] | +3.0 [+1.5, +4.5] | pass |
| CellBest, cells predicted from task text (the comparator) | 18.2% [16.4, 20.2] | +2.9 [+1.5, +4.3] | fail (saving) |
| Per-cell choice, unverifiable tasks escalated | 20.9% [18.8, 23.3] | +3.1 [+1.7, +4.6] | fail (saving) |
| Bound: fixed cascades, oracle decides unverifiable tasks | 23.2% (point) | +10.9 (point) | bound only |
| Bound: per-cell choice, oracle decides unverifiable tasks | 48.1% [45.1, 50.8] | +4.5 [+2.2, +6.6] | bound only |

80.0% of these tasks give the verifier no signal (no visible tests), which is why fixed cascades
fail. The learned router's target is the gap between per-cell choice without the oracle (20.9%)
and with it (48.1%).

### 6.3 Accuracy headroom (provisional)

On 3,380 pool tasks, the oracle is +12.66 points [11.55, 13.91] above the best single model and
cross-fitted DimensionBest +1.45 [0.68, 2.23]. The plan requires DimensionBest's lower bound to
reach 3.0. If this holds after the audit, the accuracy story rests on C4.

### 6.4 Benchmark audit (interim, under review)

The audit re-runs a sample of released responses in the sandbox and compares the outcome with the
released score. Causes are the audit's interim attribution and are still under review.

| Family | Cells | Agreement with release | Main cause of disagreement (interim) |
|---|---|---|---|
| BigCodeBench (5 families) | 97 each | 0.64 to 0.72 | Release environment: on tasks using matplotlib, sklearn, seaborn, scipy or wordcloud the release passed 0 of 251 sampled cells; our re-run passes 125 |
| DS-1000 | 97 | 0.58 | Code extraction |
| LiveCodeBench, LeetCode tasks | 41 | 0.46 | Harness never calls the solution function |
| LiveCodeBench, stdin tasks | 56 | 0.98 | None confirmed |
| HumanEval | 97 | 0.95 | Under investigation |
| MBPP | 97 | 1.00 | None |

Under the pre-set rule, a family is cleared only at agreement ≥ 0.95 with no unexplained
disagreement, and a family with a confirmed defect leaves the headline view. Labels are not yet
decided. The other six languages and SWE-bench are next.

---

## 7. Corrections to Report 8

- The 29% to 74% cascade savings used a simulated verifier, success at score ≥ 0.5 and no
  verifier cost. They are superseded by Section 6.2.
- k = 0 was scored at ability 0. A prior at the pool's mean ability matches 50 random probes on
  ability error (0.268 against 0.270).
- The ranking gate (Spearman 0.81) averaged over 10 seeds. A single 50-probe draw averages 0.51.
- With 7 known models, a 1PL item bank has only 8 distinct difficulty levels, so Fisher anchor
  selection cannot be tested on CodeRouterBench. The anchor claim is dropped.
- The visible-test verifier false-passes 15.0% and false-rejects 27.0% of cells with visible
  tests (Route Anything's released verdicts).

## 8. Next Steps

1. Finish the S1.7 audit and the S1 gate.
2. S2: evaluation harness and baselines. S3: analysis and the accuracy and cost go/no-go
   decisions.
3. S4: router v1 and v2, then v3 and v4 components.
4. S5 to S8 and the thesis follow next semester.

## 9. Limitations

- The router is not built; Section 6 reports baselines and bounds only.
- The November numbers predate the audit labels and will be recomputed.
- Each task has one recorded outcome per model, so the oracle includes run-to-run noise.
- Five of the eight models' prices were converted from CNY by the dataset authors.

## References

As in Report 8, plus:
UniRoute, arXiv:2502.08773; ICL-Router, arXiv:2510.09719; IRT-Router, arXiv:2506.01048;
ACRouter (CodeRouterBench), arXiv:2606.22902; Route Anything, arXiv:2611.05275;
ORACLE, arXiv:2607.22465; The Routing Plateau, arXiv:2606.07587.
Data: Hugging Face `Lance1573/CodeRouterBench` and `Lance1573/route-anything-data`, pinned by
checksum (MIT licence).
