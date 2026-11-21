# PROBE-CASCADE — Implementation Plan (living document)

> **Read this first in every new session.** It is the single source of truth for what is being
> built, what is done, and what comes next. Update the status block, the checkboxes and the
> progress log at the end of each session (see §9, Session protocol).

> **Plan v3 (2026-11-19).** Milestones M1–M8 of plan v2 are replaced by stages S0–S8 (§4). Stage S0
> fixes the claim definitions before any code is written. Changes come from a second independent
> review (progress log, 2026-11-19, plan v3). Plan v2 is at commit `f9877bf` and plan v1 at `b8edb5f`.
> The M1–M3 code and numbers from 2026-11-10 were quick checks written for Report 8, not validated
> infrastructure; cite them only after the S1–S4 rebuild reproduces them.

## Status

| Field | Value |
|---|---|
| Current stage | **S1 — Validate data** (§4). Pre-S0 and S0 done; `docs/CLAIMS.md` signed off 2026-11-20 |
| Next action | **S1 gate passed (2026-11-21).** Next: S2 (evaluation infrastructure and baselines) on the S1.7 views, with S1.8 (LLMRouterBench) as a hedge in parallel, because a DP1b failure on CodeRouterBench is now likely. `docs/CLAIMS.md` is the contract: where it is more specific than S0 below, it wins |
| Overall progress | Design v3 100% · Quick checks for M1–M3 done (Report 8) · Validated stages: Pre-S0, S0 · Write-up: Report 8 |
| Last updated | 2026-11-20 |
| Schedule | Not planned: the user asked on 2026-11-19 to ignore the report due date and the semester timeline. Order of work: Pre-S0, then S0–S8, then the thesis |
| Budget | $20 for the paid top-rung model in live runs (S0.9) |

---

## 1. Goal

Build a router for code-generation tasks that:

1. **reaches the best single model's acceptance rate at clearly lower cost, and beats the best
   lookup table (CellBest).** The metric is cost
   per accepted task (CPAT), always with an acceptance floor (R3). "Clearly lower" and "same
   acceptance" are defined as numbers in S0.3;
2. **brings in a newly added LLM from a small number of measured outcomes, without retraining;**
3. **is usable** as a CLI and static report, with an OpenAI-compatible proxy as an optional extra.
   Web development (the 65 Docker-tested tasks) is the showcase domain.

**Research questions (v3).**

- **RQ-A (main).** How many verified outcomes does a new model need before routing to it is at
  least as good as a per-group lookup table built from the same outcomes? Do a pooled prior
  and in-stream learning shorten that?
- **RQ-B.** How much of a cascade's cost comes from tasks that no model solves, and how much of
  it can a "stop or escalate" prediction recover?
- **RQ-C.** How often does a run-time verifier that sees only part of a task's tests accept wrong
  code, and what does that do to a cascade's cost and acceptance rate?

**Why v1's framing changed.** These findings were confirmed on 2026-11-19 (log entry):
- k = 0 was scored at θ = 0. A zero-probe prior at the pool's mean ability already matches 50
  random probes on ability error (0.268 vs 0.270).
- The M3 ranking gate averaged θ over 10 seeds. A single 50-probe draw ranks the models at
  Spearman 0.51 on average, with 1 of 10 draws reaching 0.8.
- With 7 known models and complete binary data, a 1PL item bank has only 8 distinct difficulty
  levels, so Fisher anchor selection cannot be tested on CodeRouterBench.
- With a perfect verifier, trying Qwen3-Max before Opus pays off at P(pass) > 4.5%, so a
  predicted-pass threshold rarely binds. The real cost lever is the 39.3% of ID-test tasks (all dimensions, pass = score ≥ 0.5) that no
  model solves. (Historical figure under the old pass rule; S0.1 now sets success = score 1.0, and
  S3.3 recomputes it.)

**Two separate gates.** Finishing the research and reaching the goal are judged separately:
- the **research gate** (S0.4) passes when every decision point and contribution has a verdict
  with quoted evidence. A negative verdict passes it;
- the **performance gate** (S0.3) passes only when goal 1 is met on the cross-fitted evaluation scores (S0.6),
  including S0.3(c): the router must beat the best lookup table, not only the best single model.

### Contributions (ranked) and what the thesis says if each fails

| # | Contribution | Minimal evidence | If it fails |
|---|---|---|---|
| C1 | **Probes-to-decision for a new model, from verified outcomes in one pass.** How many outcomes a new model needs before routing to it matches a lookup table built from the same k outcomes (k-matched DimensionBest, same label source), and whether a pooled hierarchical prior, in-stream outcomes from a partial verifier (PASS/FAIL/UNKNOWN) and a per-group stopping rule shorten that. Scored on a single pass over the stream with no repeated problem (S0.5). Quality against onboarding-set size is already published (UniRoute Fig. 2); the in-stream, stopping-rule and same-label-source comparison is not | E1: four onboarding arms against k-matched DimensionBest on both corpora, with CIs | Report the number of shared probes a lookup table needs (currently about 250 to reach 50.7) as the measured answer to RQ-A |
| C2 | **Give up or escalate, per task, under partial verification.** A calibrated per-task classifier of P(a later rung passes \| earlier rungs failed), with a give-up action for tasks predicted to be unsolvable, under the real visible-test verifier, where about four-fifths of headline tasks (80.0% of the 1,907 complete-case tasks, provisional, P0.5) give no signal (S0.11). The stop-or-escalate rule on post-failure histories (Route Anything Prop. 8) and cell-level truncation under an oracle verifier (Route Anything X10) are prior work that C2 uses | E2: cost at matched acceptance on exec-scored tasks, against cost-aware DimensionBest, the fixed cascade, Route Anything's 21-cell zero-loss direct-dispatch route and **per-cell Rule A with no-signal cells escalated** (R2), which C2 must beat (`CLAIMS.md` O19) | C2 counts as failed if it does not beat per-cell Rule A escalated at matched acceptance, whatever the S0.3 result. Report how far a realistic predictor falls short of the perfect-truncation bound ($6.0 vs $41.0, Always-Opus $34.0) |
| C3 | **Verifier strength on web-dev, by visible fraction.** False-pass rate of a verifier that sees a fraction f of each web-dev task's tests, against a disjoint hidden grader, measured live (E4), and its effect on cascade cost and acceptance. On CodeRouterBench the verifier profile is a re-analysis of Route Anything's released X13 verdicts (visible-test verifier: 15% false-pass and 27% false-reject on cells with visible tests; cells without visible tests give no signal under S0.11, and the X13 syntax/quality heuristic, which S0.11 does not use, false-passes 66.5% there; measured 2026-11-19), and the with/without-examples split and its cascade effect are already in Route Anything's X7b and X8 | E4: visible vs held-out tests on web-dev, false-pass curve over f with a task-level Clopper-Pearson bound; CodeRouterBench replay with the real X13 verdicts as context | Report the web-dev false-pass curve as a characterisation; the CodeRouterBench profile is cited as Route Anything's data. If E4 cannot be run, C3 counts as covered and S0.14 is re-applied |
| C4 | **Headroom audit of CodeRouterBench (fallback contribution).** How much of the oracle's and DimensionBest's gain over the best single model survives the raw-response audit of the suspected families and run noise. Builds on Route Anything's BigCodeBench grading diagnostic and on the run-noise methods of 2608.23023 and the Routing Plateau; suspected anomalies without execution evidence are reported as sensitivity analyses, not as confirmed defects | S1.7 audit plus the three data views; noise-corrected oracle on web-dev | If headroom survives, CodeRouterBench is cleared as primary data, which strengthens C1 and C2 |

**Novelty status (P0.1, 2026-11-19; details in `docs/research/P0.1_novelty_read.md`).**
C1–C4 are each **partly covered**; none is fully covered. Under S0.14 the order stands.
- **C1.** UniRoute (2502.08773) reports routing quality on unseen LLMs against validation-set size
  (area under the deferral curve, K-NN comparator). ICL-Router (2510.09719) onboards from a fixed
  query set. IRT-Router (2506.01048) onboards a new model from a text profile with no outcomes
  (§4.1) and reports new-model prediction accuracy of 0.67 (Table 4), naming few-shot onboarding as
  future work (§6.3). ACRouter (2606.22902) uses a fixed pool. None of the
  papers read measures in-stream verified outcomes with a pooled prior and a stopping rule against
  a k-matched lookup table. Two published cases show in-stream feedback failing in one pass:
  ORACLE (2607.22465) Appendix E.8 matches random mixing in one cold pass, and its gain needs eight
  passes; Route Anything C.6 has a verified-write memory scoring 0.116 below DimensionBest.
- **C2.** Route Anything (2611.05275; code `InfRec-Cardinal/route-anything`) publishes the
  stop-or-escalate rule on post-failure histories (Prop. 8, Finding 7) and cell-level truncation
  under an oracle verifier (X10). It has no give-up action and no per-task classifier, and it does
  not work under a partial verifier.
- **C3.** Route Anything's repository publishes the visible-test verifier's false accepts with and
  without prompt examples and their effect on cascade cost (X7b C5, X8, X13). C3's own evidence is
  therefore E4 on web-dev.
- **C4.** Route Anything documents BigCodeBench grader failures (`experiments/grading_diagnostic`).
  2608.23023 and "The Routing Plateau" (2606.07587) measure run noise against routing headroom
  (both read in part). C4 is new only as a raw-response audit of the CodeRouterBench
  test-generation and LCB families, plus the noise-corrected web-dev oracle.

### Decision points

| DP | Where | Test | Consequence |
|---|---|---|---|
| DP1a | End of S3 | **Accuracy:** headroom of the oracle and of cross-fitted DimensionBest over the best single model, on the **full** artefact-filtered eligible pool (not the development split: the oracle needs no fitting and a cross-fitted lookup table is not router development, so no leak; the development split alone, 697 problems and 1,104 eligible tasks before the artefact filter, would leave the test weak). Passes if the lower end of the 95% cluster-bootstrap CI of **cross-fitted DimensionBest's** headroom is ≥ 3.0 AvgPerf points; the oracle is reported as the ceiling (`CLAIMS.md` O18). AvgPerf = acceptance rate under S0.1, in points (O10). Provisional (P0.4): oracle +12.66 [11.55, 13.91], DimensionBest +1.45 [0.68, 2.23] | No: C4 leads the accuracy story |
| DP1b | End of S3 (S3.2) | **Cost (feasibility upper bound):** does the **oracle-on-no-signal bound** meet S0.3 on point estimates on the headline set, with verifier cost included: saving ≥ 20% and acceptance gap ≥ −2 points against the best single model, **and** saving > 0 and gap ≥ −2 points against CellBest with predicted cells (S0.3(c))? In the bound, cascades use the real verifier where visible tests exist and an oracle accept-or-escalate decision on cells with no verifier signal. Fixed ladders cannot pass by construction (S3.2), so they are not the test. Also report the gap the learned router must close (provisional: about 11 points against the best single model, P0.5; against CellBest with predicted cells (P0.6, seed 0): the bound meets (c) on point estimates (per-cell Rule A: 36.6% saving, +1.6 points; Rule B: 16.4%, +8.1). The best policy without an oracle (per-cell Rule A, no-signal cells escalated) reaches 3.3%, +0.3. The learned no-signal decision must recover the difference, about 33 points of saving at similar acceptance) | No: the bound misses the gate, so stop and bring it to the user before S4 (LLMRouterBench becomes a candidate primary corpus for C1 and C2). A failed DP1b means the tested policies have not shown economic feasibility; it does not prove a learned policy cannot succeed, so the pause is a reassessment, not a verdict. This half decides the economic claim |
| DP2 | During S5 | Does the hierarchical model beat k-matched DimensionBest **in routing units** (AvgPerf, cost) at a k in the fixed grid {10, 25, 50, 100}, on the development split? | No: the routing model is the shrunk lookup table, and IRT is reported as an ablation |
| DP3 | S6 pilot | Is the routing gap on the live pool (free models plus the paid top model) more than twice the run-to-run standard deviation, measured on the pilot's ≥ 2 runs per task? The routing gap is the noise-corrected oracle headroom over the best single model: choose on run 1, score on run 2 (`CLAIMS.md` O14) | No: live work is reported as a verifier study and a transfer test, not as a routing win |

### Success criteria

| # | Criterion | Measured by |
|---|---|---|
| SC1 | E1 onboarding curves in routing units (AvgPerf, cost) for the onboarding arms against k-matched DimensionBest, with CIs | S5, S7 |
| SC2 | Cascade with stop-or-escalate against cost-aware DimensionBest, the fixed cascade and per-cell Rule A escalated (`CLAIMS.md` O19) at matched acceptance rate, exec-scored tasks, paired CIs; or an honest negative | S5, S7 |
| SC3 | Verifier noise results: simulated noise (E3) and the measured false-pass curve on web-dev (E4), with an upper bound when no false pass is seen | S5, S7 |
| SC4 | Headroom audit in three data views | S1, S3 |
| SC5 | At least 3 free-tier models plus the paid top model × 65 tasks, stored with per-test results; transfer test reported as exploratory | S6, S7 |
| SC6 | CLI and static-report demo; proxy optional | S7 |
| SC7 | Findings report with a verdict per DP and contribution (research gate), and the performance-gate result | S8 |

(Success criteria are numbered SC1–SC7 so they do not clash with the stage names S0–S8.)

---

## 2. The method, briefly (v3)

1. **Ability model.** P(model m passes task q) = σ(μ_g + δ_m + γ_{m,g} − b_q), where g is the
   task's group (dimension or family), μ_g is the pool's group mean, δ_m a model shift, and
   γ_{m,g} a model-by-group offset shrunk toward 0 with variances estimated from the known
   models. The plain 1PL is the special case with μ_g and γ removed.
2. **New model.** It starts at δ_m = 0, γ = 0, i.e. the pool prior. This is the k = 0 baseline,
   never θ = 0 (R11).
3. **Evidence.** Either k dedicated probes (random or stratified by group and difficulty band) or
   verified outcomes from the live stream. A greedy router may never call a new model, so
   exploration is explicit: a forced share of traffic per group, a maximum number of probe calls,
   and a stop once the posterior spread in every group is below a set tolerance or the budget is
   used (numbers fixed in S0.13). Verifier outcomes are treated as noisy evidence.
4. **Difficulty.** b_q is predicted from the prompt embedding. The source-benchmark label is not
   available at decision time, so it is not a feature; the group may be predicted from the prompt.
5. **Decide.** Cheap-first, using each step's break-even ratio (cost of the cheaper model over the
   next one). After a failed rung, a calibrated classifier predicts P(a later rung passes | the
   rungs tried so far failed). Single-model pass rates cannot give this, because models fail on the
   same tasks. Stop the ladder when it falls below t; the classifier is fitted on the fit split
   and t is set on held-out data inside the training set (S0.6).
6. **Verify and escalate.** Run-time signals only (tests, sandbox), on exec-scored tasks only.
   Each verified outcome updates the model's posterior.

**Build order (S4–S5).** Each part is kept only if it beats the version before it on the development split (S0.6, S0.7):
- router v1: shrunk per-group lookup table plus a fixed two-model cascade (always built; the floor);
- router v2: v1 plus bounded onboarding (always built);
- router v3: v2 plus the hierarchical ability model (kept only if DP2 says yes);
- router v4: v3 (or v2) plus the stop-or-escalate classifier (kept only if it beats the fixed cascade).

Deferred until there is evidence they help: Thompson sampling (only if the pass threshold binds)
and the kNN memory residual (only once prompts cover ≥ 95% of tasks). There is no LLM in the
routing decision.

---

## 3. Architecture and repo layout

The new code lives in `router/` at the repo root. It reuses `pipeline/components/` where
possible. Items marked *(quick check)* exist from 2026-11-10 and must be rebuilt or validated in
their stage before any result from them is cited.

```
router/
  __init__.py
  settings.py            # paths, price tables, seeds (not `config.py`: pipeline/ owns bare `config`)
  data/
    schema.py            # OutcomeMatrix: score, cost, dimension, family, score_type, run, eligible  (quick check, extend)
    sources.py           # explicit task id -> source family table (replaces the regex source_of)
    coderouterbench.py   # two views: complete release, paper matrix with missing = 0  (quick check)
    llmrouterbench.py    # code subsets (livecodebench, swe-bench verified, humaneval, mbpp)
    phase5.py            # read-only loader for data/phase5_results.csv
    prices.py            # single source of published $/M-token prices  (quick check)
    verdicts.py          # real visible/hidden-test verdicts (route-anything X13) for replay
    prompts.py           # task -> prompt join for all 11 families  (quick check, 48%)
    scoring.py           # score type per family from the source benchmark's rule  (quick check)
    splits.py            # problem ids and 5-fold problem-disjoint cross-fitting
    audit.py             # per family x model anomaly table; writes the frozen artefact list
    artefacts.yaml       # frozen artefact list, committed before any router result
  core/
    irt.py               # 1PL/2PL fit and damped ability solver  (quick check, keep)
    ability.py           # hierarchical ability model with shrinkage
    difficulty.py        # prompt embedding -> b_q, no source label
    onboarding.py        # four onboarding arms
    stopping.py          # per-group posterior stopping rule
    exploration.py       # bounded forced exploration for a new model
    rescue.py            # calibrated P(later rung passes | earlier rungs failed)
    policy.py            # cheap-first by break-even ratio, ladder truncation, budget cap
    anchors.py           # random / stratified / Fisher selection  (quick check)
  baselines/             # always-X, random, DimensionBest (full, k-matched, cost-aware), SourceBest,
                         # fixed cascade, released ACRouter decisions, LinUCB, kNN on prompt embeddings
  sim/
    stream.py            # seeded replay, model arrivals, bandit feedback only  (quick check)
    metrics.py           # AvgPerf, cost, accept rate, CPAT@floor, cost at matched accept, regret incl. probe cost
    stats.py             # paired bootstrap over tasks, cluster bootstrap over problems
    verifier.py          # real-verdict replay; simulated noise for sensitivity  (quick check, extend)
  live/
    registry.py          # free-tier model registry (SQLite, keys from env)
    gateway.py           # thin wrapper over pipeline/components/llm_gateway.py
    sandbox.py           # wraps TestRunner; records per-test results from a structured reporter
    testsplit.py         # hidden grader pool reserved first, then the visible pool, per web-dev task
    cache.py             # every live response cached; model id and date recorded per call
  experiments/           # e0_power.py, e0_audit.py, e1_onboarding.py, e2_cost.py, e3_verifier.py, e4_webdev.py
  report/                # static HTML report generator
  cli.py                 # router audit | replay | probe | run | report  (proxy optional, S7.4)
tests/router/            # pytest, mirrors router/
scripts/fetch_data.py    # pinned download of CodeRouterBench and route-anything-data into external/
external/CHECKSUMS       # SHA-256 of every fetched file (external/ itself is git-ignored)
```

**Import rule:** `pyproject.toml` puts the repo root, `pipeline/` and `human_eval/` on the test path,
matching their bare imports. Router code imports pipeline modules by those bare names
(`from components.llm_gateway import LLMGateway`). It must never define a top-level module called
`config`, `models`, `analysis` or `db`. Run tests from the repo root with
`pipeline/venv/bin/python -m pytest`.

**Worktrees:** the venv lives only in the main checkout. A worktree session symlinks it in, with
`ln -s ../../../../pipeline/venv pipeline/venv` run from the worktree root. The `venv` ignore
pattern covers that symlink.

**Existing code to reuse; do not rewrite it:**
- `pipeline/components/llm_gateway.py` (`LLMGateway`, an async LiteLLM wrapper with retry)
- `pipeline/components/test_runner.py` (`TestRunner`, the Docker test execution; it already parses
  per-test assertions)
- `pipeline/components/code_extractor.py` (`CodeExtractor`)
- `pipeline/components/task_loader.py` (`TaskLoader`, for the 65 tasks in `benchmark/tasks/`)
- `pipeline/analysis/data.py` (`load_experiment_data(csv_path="data/phase5_results.csv",
  prefer_database=False, zero_test_treatment=...)`)
- `pipeline/analysis/failure_taxonomy.py` (derived failure labels; on `main` since 2026-10-05)
- `data/phase5_results.csv`, the frozen 325-run snapshot (SHA-256 in `data/README.md`). Never edit it.

---

## 4. Stages

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in the log).
Every stage has an **exit gate**. Do not start the next stage's core work until the gate passes, or
until a waiver is written in the log with its reason. Plan v2 ids (M1.1 etc.) are given
in brackets where a task carries over.

**Terms used below (S0.6).** The **development split** is 20% of problems, used for all
open-ended iteration. The **evaluation population** is the other 80%, split into five **outer
folds**; each task is scored once, by a configuration fitted on the other four folds and chosen
automatically on **inner folds** inside them, from a candidate list frozen before evaluation.

### Setup (done)
- [x] Done 2026-10-05: Docker verified, `router/` skeleton, worktree branches merged, novelty
      check (contribution reframed), reference ids checked. See the log.
- [x] Done 2026-11-19: `worktree-progress-report-8` merged into `main` (`6065c3d`); 91 tests pass,
      1 skipped (needs the external data).
- [ ] Repo-audit backlog: see §4a.

### Pre-S0 — Requirements to complete before S0

Found in the 2026-11-19 repo and data audit. Work through them in this order (P0.5 and P0.6 run
before P0.4); S0 cannot be signed off until P0.1–P0.6 are done.

- [x] **P0.1 Novelty read: "Route Anything" (arXiv 2611.05275), ORACLE (2607.22465, Appendix
      E.8), "The Routing Plateau" (2606.07587) and "Most of the LLM Routing Gap Is Task Type"
      (2608.23023).** Done 2026-11-19: `docs/research/P0.1_novelty_read.md` (merged `0a4f40b`).
      C1–C4 are each partly covered and none is covered; under S0.14 the order stands and only the
      §1 wording changed. IRT-Router (2506.01048) was read in P0.5 (§7 of the novelty doc): it
      does not cover C1. The Routing Plateau and 2608.23023 were read in part, which is enough
      while C4 is the fallback.
- [x] **P0.2 Branch state.** Commit plan v3 on `worktree-revised-plan`, rebase it onto the new
      `main`, and merge it, so that one branch carries the plan and the code. Done 2026-11-19:
      merged to `main` through PR #2 (`8a2be1e`).
- [x] **P0.3 Data fetch with pinned versions.** `scripts/fetch_data.py` downloads
      `Lance1573/CodeRouterBench` (revision `e567d89b`) and `Lance1573/route-anything-data`
      (revision `bd361209`) into `external/`, and checks SHA-256 hashes written in
      `external/CHECKSUMS`. Files needed from
      route-anything-data: `response_matrix.json` (raw responses), `experiments_x/common/tasks.jsonl`
      (prompts), `experiments_x/X13_memory/cache/verdicts.jsonl` (real verifier verdicts), and the
      X7b, X8 and X12 records for comparison. Done 2026-11-19 (merged `b933171`): both repos
      pinned to full commit SHAs; 187 files hashed, all checked against the Hub's own hashes;
      `cd external && shasum -a 256 -c CHECKSUMS` works without the script; an offline re-run
      fetches nothing and a corrupted file is re-fetched alone; only `external/CHECKSUMS` is
      tracked. The data-file paths in `router/data/coderouterbench.py` and three `m1_profile`
      scripts were corrected to the repo layout. Tests: 100 passed.
- [x] **P0.4 Claim contract, constants and power port.** Done 2026-11-20 (merged `88a425f`):
      `docs/CLAIMS.md` (drafted with 18 open items, plus O19 from the reviewer; signed off by the
      user on 2026-11-20), the claim constants in `router/settings.py`
      (`tests/router/test_claims_constants.py`), and `router/experiments/e0_power.py` (output
      `router/experiments/out/e0_power.json`; `docs/research/P0.4_power.md`). The original
      `power4.py` reproduces the plan's 0.85–1.4-point half-widths. On the current definitions,
      the no-oracle per-cell policies' gap CI half-width against the best single model is 1.4–1.5
      points on the 1,907-task complete case and 1.8–1.9 on the evaluation population (development
      split removed), giving 55–77% power at −2 for a true gap of 0. Against CellBest it is 0.3–0.6
      points. CellBest's own saving depends on the development-split draw (15.4% to 32.9%).
- [x] **P0.5 Provisional recount (before P0.4).** Done 2026-11-19 (merged `dcf50ad`):
      `docs/research/P0.5_recount.md`, script `router/experiments/p0_5_recount.py` (output
      `router/experiments/out/p0_5_recount.json`). The 2026-11-19 numbers reproduce exactly.
      Only 66 empty X13 cells (64 tasks; glm-5 61) and none of the 148 missing Qwen3-Max runs
      fall in the 1,971-task headline set: 1,479 of the 1,551 empty rows are LCB, already
      excluded, and the missing runs are bug-fixing and test-generation tasks. Complete case:
      1,907 tasks, Opus best at 67.3% in every training fold, 80.0% of tasks and cells with no
      signal. The S3.2 table is unchanged in substance. Route Anything's zero-loss
      direct-dispatch route fails S0.3 (a)–(b) on the acceptance CI (saving 50.5% [47.9, 53.1],
      gap −1.3 [−3.2, +0.5]); **CellBest passes (a)–(b)** on all five seeds (saving 22.9% [20.8,
      25.2], gap +3.0 [+1.5, +4.5]), with the saving's lower bound less than one point above 20%.
      This led to S0.3(c). IRT-Router read: it onboards from a text profile with no outcomes, so
      it does not cover C1 (P0.1 §7).
- [x] **P0.6 Feasibility against CellBest (before P0.4).** Done 2026-11-19 (merged `36c3371`):
      `docs/research/P0.6_cellbest_bound.md`, script `router/experiments/p0_6_cellbest_bound.py`
      (output `router/experiments/out/p0_6_cellbest_bound.json`). P0.5 reproduces exactly. Cell
      predictor from prompt text only (TF-IDF and logistic regression inside the cross-fit):
      96.2% held-out accuracy. All errors are in the three small bcb cells, whose prompts are
      byte-identical to prompts in other cells for 63 of 67 tasks, and between the HumanEval and
      MBPP cells. CellBest with predicted cells saves 18.2% [16.4, 20.2] at a gap of +2.9
      [+1.5, +4.3] against Opus (true cells: 22.9%, +3.0). DP1b on point estimates, seed 0: the
      oracle-on-no-signal bound meets S0.3(c) with fixed ladders (7 of 84; the best, an
      in-sample pick, saves 24.7% at −0.6 against CellBest) and with per-cell choice fitted in
      training folds (Rule A: 36.6%, +1.6; Rule B: 16.4%, +8.1), on all five seeds. Per-cell
      choice without an oracle also meets (c) on point estimates (1.7% to 3.3%, gap −0.3 to
      +0.3) and is now an R2 baseline. Gap CI half-width against CellBest: 0.3–1.8 points for
      per-cell policies, 2.3 for fixed ladders.

The remaining requirements are placed in the stage that needs them, in priority order:

| Priority | Requirement | Stage |
|---|---|---|
| P1 | Problem ids from source ids, linking `lcb` to `lcb_testgen` through `tasks.jsonl`; split-integrity test | S1.6 |
| P1 | Long-format `OutcomeMatrix` (task, model, run, problem_id, family, score_type, eligible) | S1.2 |
| P1 | Prompt join extended with `tasks.jsonl` (3,297 prompts, including 888 with `source = lcb`: 776 `lcb`, 112 `lcb_testgen`) | S1.5 |
| P1 | Audit of the suspected families using the raw responses; `artefacts.yaml` frozen | S1.7 |
| P1 | Real-verifier loader (X13 verdicts) for replay | S1.11 |
| P1 | `prices.py` with source URLs and dates; reference prices for the live models | S1.10 |
| P1 | Eligibility from documented scoring rules; partial-credit success rule; recompute the pool | S1.4 |
| P2 | Evaluation infrastructure: PASS/FAIL/UNKNOWN replay, real-verdict verifier, metrics, `sim/stats.py`, development split and nested cross-fitting driver; k-matched and cost-aware DimensionBest; validation of the quick-check replay and baselines | S2 |
| P2 | Feasibility analysis with no-signal bracketing; power rerun on the final population | S3.2, S3.7 |
| P3 | Router modules (`ability`, `difficulty`, `exploration`, `stopping`, `rescue`, `policy`, `onboarding`) and E1–E3 | S4, S5 |
| P4 | Docker daemon; structured test reporters with stable ids; sandbox limits; new run writer | S6.1, S6.2 |
| P4 | Per-provider `temperature` handling (the 2026-10-21 Anthropic API version rejects it); `<think>` stripping; per-attempt temp files | S6.1 |
| P4 | Re-run the 324 stored Phase 5 responses through the fixed runner for per-test data at no API cost | S6.6 |
| P4 | API keys for the chosen tiers (S0.9) and Anthropic | S6.4 |
| P5 | CLI, static report, Report 9 | S7, S8 |

### S0 — Claim definitions

Purpose: fix what is measured and what counts as success before any S1 code is written, so that no
result can be reinterpreted after it is seen. **Signed off by the user on 2026-11-20 as
`docs/CLAIMS.md`, the contract; where `CLAIMS.md` is more specific than the items below (its
sign-off decisions O1–O19: populations, seeds, resamples, inner folds, exploration, groups,
AvgPerf, DP1a, DP3, the C2 bar), `CLAIMS.md` wins.** S0.3, S0.6 and S0.10 were approved by the user on
2026-11-19 on the evidence of the power calculation; the third and fourth reviews of that day
added the population, observation and nesting rules below (progress log).

**All task counts in S0 are provisional.** They come from a prototype that has not passed S1. The
counts that cannot change (release row counts, the X13 coverage of 2,911 tasks) are tested in S1;
every count that depends on eligibility (S0.12) or on the audit (S0.10) is frozen in
`data/artefacts.yaml` only after S1.4 and S1.7 run.

- [x] **S0.1 Acceptance.** A task is accepted only when the ground truth passes it. **In replay the
      ground truth is the benchmark's released score**: it covers every task and is what the
      baselines and the paper use. For execution-scored families with partial credit (scores
      such as 0.2–0.8 are fractions of tests passed), **success means all tests pass (score =
      1.0)**, the standard pass@1 rule. The X13 hidden-test outcome agrees with the released score
      on 87.7% of cells and has 2,051 harness-mismatch and 2,718 extraction-failure cells, so it is
      reported only as a sensitivity check. In live runs the ground truth is the reserved hidden
      test pool. The router's own verifier decides escalation but never counts as acceptance.
      **Amendments A2 and A3 (2026-11-21):** A2 made CRUXEval's 0.8 a success. A3 reversed that
      for the headline, because the evidence that 0.8 means a stated correct value is only
      partial (`CLAIMS.md` A3). The headline is 1.0 for every family. CRUXEval ≥ 0.8
      (`CRUXEVAL_SUCCESS_SCORE`) is a sensitivity check:
      `scoring.success(score, family, cruxeval_lenient=True)`.
- [x] **S0.2 Cost.** Cost = model tokens × published price (source URL and access date in
      `router/data/prices.py`) + verifier runs + retries + probe calls, spread over the stream.
      Actual spend is not the metric. Verifier runs are reported as a count and in dollars at a
      stated per-run rate, swept at ×1, ×10 and ×100. **Amended 2026-11-20 (`CLAIMS.md` A1):**
      0.200 s per visible-test run (the X13 mean over cells where a verifier ran) on the AWS Lambda
      basis (O11), about $6.9 × 10⁻⁶ per run; the original 0.074 s and $10⁻⁶ counted cells with no
      visible tests as 0 s. The P0.5, P0.6 and E0 records used $10⁻⁶. The stream length over which probe cost is spread is the number of tasks scored
      (reported with the break-even stream length). A price-sensitivity sweep is run on every
      headline number.
- [x] **S0.3 Performance gate.** The router passes if, over the nested cross-fitted evaluation
      scores (S0.6):
      - (a) the lower end of the paired 95% cluster-bootstrap CI of its cost saving against the
        best single model is at least **20%**;
      - (b) the lower end of the paired 95% cluster-bootstrap CI of its acceptance gap against the
        best single model is **at least −2 points**;
      - (c) **it beats the strongest lookup baseline.** Against CellBest with predicted cells
        (R2), fitted in the same training folds, the lower end of the paired 95% CI of its cost
        saving is **above 0**, and the lower end of its acceptance gap is **at least −2 points**.

      Condition (c) was added by the user on 2026-11-19, because P0.5 showed that CellBest alone
      passes (a) and (b): saving 22.9% [20.8, 25.2], gap +3.0 [+1.5, +4.5]. CellBest with true
      source cells is reported next to it as a reference, not as a gate, since the router cannot
      see the source benchmark (§2). The comparator is fixed now and is not changed after results
      are seen.

      **"Best single model" is chosen inside each training set, on the same eligibility and
      population as the scored fold** (for cascade claims: the training folds' S0.11 headline
      tasks), never on the scored fold. Provisional example (P0.5, success = 1.0): on the
      complete-case headline set Opus scores 67.3%, gpt-5.4 53.2% and qwen3.5-plus 42.2%; Opus is
      the best model in every training fold. Power (P0.4, provisional): on the evaluation
      population, a router routing like the no-oracle per-cell policies is measured against the
      best single model to about ±1.8–1.9 points, which gives about 55–60% power at −2 for a true
      gap of 0 (±1.4–1.5 and 72–77% on the full complete-case headline set). Routers that change
      more cells are measured less precisely (±1.9–2.6 for the oracle-bound policies). Against
      CellBest with predicted cells the half-width is 0.3–0.6 points for policies close to
      CellBest and 1.3–2.0 for policies that depart from it as much as the bound. The user chose
      on 2026-11-19 to keep −2 and report the power with the result; S3.7 recomputes it on the
      final population with `e0_power.py`.
- [x] **S0.4 Research gate.** Every decision point and contribution ends in a verdict (holds,
      fails, not reached) with quoted evidence. A negative verdict passes this gate.
- [x] **S0.5 k and label source.** k is the number of calls made to the new model. Every
      onboarding arm and baseline with the same k, including k-matched DimensionBest, receives
      exactly the same k outcomes **from the same label source**. E1 is reported in two pairings:
      ground-truth labels for all arms, and verifier-only labels (PASS/FAIL/UNKNOWN) for all arms,
      so that a probe-versus-in-stream difference cannot come from label quality alone. In E1 the held-out model's outcomes are absent from
      every fitting and selection step, except the probes and online feedback it is allowed.
      E1 is scored on a **single pass** over the stream with **no repeated problem** (the R13
      problem id, not the task id), so in-stream learning cannot recall its own or a sibling
      task's verdict (ORACLE E.6 and E.8); a test checks this.
- [x] **S0.6 Development split and nested cross-fitting** (option (a), chosen by the user on
      2026-11-19).
      - **Task pool:** CodeRouterBench ID probing and ID-test splits pooled, eligible tasks only
        (S0.12). Problems (variants of one problem kept together) are the unit of every split.
      - **Development split:** 20% of problems, drawn once with a recorded seed. All open-ended,
        human-directed iteration (S5) happens here and nowhere else.
      - **Evaluation population:** the other 80% of problems, split into five outer folds. Before it
        is touched, the **candidate list** (router versions v1–v4 and their hyperparameter grids)
        and the **selection rule** are frozen and tagged in git. For each outer fold, the
        configuration is chosen automatically on inner folds formed **within the other four outer
        folds**, fitted on those four, and then the held-out fold is scored once. The held-out fold
        plays no part in choosing or fitting its configuration.
      - Every evaluation task is scored once, by a configuration that never saw its problem and
        was chosen without it. Onboarding probes are drawn from the training folds only.
      - The original ACRouter split is used only to reproduce the paper's numbers.
      - Cost of the development split: about 20% of the headline tasks (provisional: about 1,970
        to about 1,580).
- [x] **S0.7 Iteration rule.** Router changes are made and judged on the development split only.
      The nested evaluation and the live hidden tests are run once, with the frozen candidate
      list and selection rule. Every version tried on the development split is logged and
      reported.
- [x] **S0.8 Statistics.** Paired bootstrap over tasks and cluster bootstrap over problems, 95%
      CIs; the cluster bootstrap is the one used for the gates. Repeated live outputs of one task
      are resampled together (cluster by task). A seed means the fold assignment plus the stream
      order plus the probe draw; gates use a single pre-registered seed, and the other four seeds
      are reported next to it. Leave-one-model-out results are shown per model, never pooled.
      A point estimate that clears its threshold on the gate seed is reported as **at the
      threshold** when it falls to the threshold or below on another seed, when its CI lower
      bound is at or below the threshold, or when its margin is under 0.1 points (adopted from
      P0.6 on 2026-11-19).
- [x] **S0.9 Live pool and budget.** The live ladder needs steps in both price and ability, and
      models the router has not seen:

      | Tier | Size / type | Expected web-dev pass rate | Role |
      |---|---|---|---|
      | Small | about 7–17B dense, or small MoE | 20–35% | Cheapest rung |
      | Mid | about 30–70B dense, or MoE about 100B total | 40–50% | Middle rung |
      | Large open | ≥ 100B MoE, or a large code-specialised model | 50–60% | Strongest free rung |
      | Paid frontier | `claude-opus-4-6` ($5 / $25 per M tokens) | best | Top rung and Always-best baseline; also in CodeRouterBench, so it links the two corpora |

      Also required: at least 2 model families from at least 2 providers; one reasoning model; one
      code-specialised model; 1–2 models held out of calibration that arrive mid-stream as an
      onboarding **demonstration** (exploratory: 65 tasks cannot support the k grid, which is used
      in replay only); at least a 3× price gap between neighbouring rungs; a published paid price
      for every model. Each free model must have a pinnable id, enough quota for one collection
      window, and terms that allow benchmark use. Budget **$20** with a hard cap in the provider
      console (Opus 4.6 estimate: about $4 for 65 tasks × 3 runs without extended thinking). Every
      response is cached; the model id and date are recorded on every call.
- [x] **S0.10 Headline data view.** The headline is the **artefact-filtered** view; the all-tasks
      view is always reported next to it. Rule, fixed before any result: the S1.7 audit decides
      each suspected family on evidence from the raw responses. A family proven defective is
      excluded; a family proven sound is kept; a family the audit cannot decide stays out of the
      headline and appears in the sensitivity view. The headline is therefore a **restricted
      population** (runnable, audited, test-checkable tasks); its results do not establish
      performance across all executable coding tasks.
- [x] **S0.11 Replay verifier and the headline set for cascade claims.**
      - **Verdict source:** the real visible-test verdicts from route-anything-data X13, not
        simulated noise. X13 covers 2,911 tasks × 8 models drawn from both splits (2,044 probing,
        867 ID-test). They are the tasks Route Anything could make runnable (2,911 of the 2,912
        marked runnable in its `tasks.jsonl`), not a random sample.
      - **Headline set for cascade claims (E2, C2):** eligible, filtered tasks with X13 verdicts.
        Provisional size under the S0.1 success rule: 1,971 tasks, of which 1,907 have no
        unavailable cell (complete case, P0.5), before the development split.
        Folds still cover the full S0.6 pool and training uses all of it; only cascade scoring is
        restricted. Tasks without verdicts are not scored for cascade claims.
      - **The headline set is easier than the pool.** Provisional (filtered): best single model
        66.8% on the headline set against 38.5% on the pool; 10.8% of headline tasks are solved by
        no model, against 48.4%. Its family mix also differs (more MBPP, less BigCodeBench and
        DS-1000). Cascade results are worded as holding for runnable tasks, and S3 reports the
        headline set's family mix, pass rates and unsolved share next to the pool's.
      - **No visible tests means no verifier signal**, decided per cell. Provisional share on the
        complete-case headline set (1,907 tasks; P0.5): 80.0% of tasks (1,525) have no signal for
        any model, and 80.0% of cells. With empty responses removed, every model on a task has the
        same visible-test count, so the two shares coincide. Empty responses were not the cause:
        on all 1,971 tasks, removing them moves the cell share from 79.4% to 79.3%. C2 therefore
        relies on prediction for about four-fifths of the headline tasks, with a thin verified
        layer on the rest. Where a cell has visible tests the
        recorded verdict is used (false-pass 15%, false-reject 27%). The X13 syntax/quality
        heuristic (false-pass 66.5%) is **not** used; the heuristic-accept variant is reported as
        a sensitivity check.
      - **Missing responses are missing, not failures.** 1,551 X13 rows are empty responses
        (glm-5 554, qwen3.5-plus 538, MiniMax-M2.7 379, kimi-k2.5 79, gpt-5.4 1), all with
        `w1_n = 0`; 877 of them carry a passing released score. They, and the 148 zero-score,
        zero-cost Qwen3-Max rows, mark the model as unavailable for that task: the rung is skipped
        in replay, the cell is not scored, and the counts are reported. S2 fixes how a baseline
        that needs such a cell is scored. The earlier note that the visible-test count differs
        between models on 611 tasks is caused entirely by these empty rows. Route Anything's
        derived X13 tables impute the released truth on empty cells and are not reused. In the
        headline set this affects 66 empty cells in 64 tasks (glm-5 61, MiniMax-M2.7 3,
        qwen3.5-plus 2) and none of the Qwen3-Max rows; 1,479 of the 1,551 empty rows are LCB
        (P0.5).
      - Simulated noise (E3) is a sensitivity analysis only.
- [x] **S0.12 Scope and eligibility.** Router claims cover execution-scored, test-checkable tasks
      only (consistent with R4). **Eligibility is decided by each source benchmark's documented
      scoring rule (S1.4), never by the scores observed** — an unseen model's outcomes must not
      decide which tasks enter its evaluation. Partial-credit execution families are kept, with
      success defined by S0.1; a family whose scoring is not execution-based is excluded as a
      whole. Similarity- and judge-scored tasks are reported separately.
- [x] **S0.13 Observations, probes and pinned thresholds.**
      - **In-stream observations** are PASS, FAIL or UNKNOWN, from the visible-test verifier only.
        UNKNOWN (no verifier signal) is never turned into a training label. Replay must never pass
        the released score to the router (a test checks this; the quick-check
        `router/sim/stream.py:35-36` does pass it and is replaced in S2).
      - **Two separate onboarding mechanisms.**
        - *Calibration probes:* separate, charged calls on training-pool tasks whose answers are
          known. They are the **only** observations labelled with ground truth, which is also how
          onboarding works in practice; R1 forbids the answer key only for escalation on a live
          request. Limits: up to 5 probes per group and 100 per new model.
        - *Exploration traffic:* live requests routed to the new model, about 10% per group, which
          return PASS, FAIL or UNKNOWN from the verifier only. A live request never becomes a
          ground-truth-labelled probe because it was exploratory.
        - Sensitivity check: probe labels from the visible-test verifier only (also the second E1
          pairing, S0.5).
      - **Stopping:** stop exploring a group when the posterior SD of its offset is ≤ 0.5 on the
        logit scale, or when its probe and exploration budget is used. If the budget runs out
        before a group reaches the threshold, that group uses the pool prior.
      - **Rescue decisions** are trained on the histories the router can observe (PASS, FAIL,
        UNKNOWN, including false rejections), not on the true failures.
      - Families (S1.1): no family under 30 tasks; the final list with counts is frozen in
        `sources.py`. DP1a, DP1b and DP3 thresholds as stated in §1.
- [x] **S0.14 Fallback lead claim, fixed in advance.** If P0.1 finds C2 and C3 already covered, C1
      becomes the lead contribution. If C1 is also covered, the lead becomes C4 together with the
      real-verifier economics (cost of a verified cascade under a measured weak verifier).
- **Deliverable:** `docs/CLAIMS.md` with S0.1–S0.14, plus their constants in `router/settings.py`.
- **Gate:** the user signs off on `docs/CLAIMS.md` and it is committed before any S1 code. A later
  change needs a log entry giving the reason, and every result already computed is re-scored.
  **Passed 2026-11-20.**

### S1 — Validate data

**Order of work (reviewer, 2026-11-20).** Batches A1 and A2 run in parallel; B starts after A1;
C starts after everything else. S1.8 is a side task.

| Batch | Items | Depends on |
|---|---|---|
| A1 | S1.1, S1.2, S1.3 | — |
| A2 | S1.11, S1.10, S1.9 | — (S1.11 returns verdicts keyed by task and model, ready to join to the S1.2 schema) |
| B | S1.4 and S1.5 in parallel, then S1.6 | A1; S1.6 also needs S1.5 (prompts for the identical-prompt merge and the `lcb` link) |
| C | S1.7 and `artefacts.yaml`, then the S1 gate check | all of the above (`artefacts.yaml` freezes headline counts, which need S1.11) |
| Side | S1.8 | `scripts/fetch_data.py` must pin LLMRouterBench first; low priority unless DP1b fails |

The P0.5, P0.6 and E0 scripts keep their provisional definitions; they are records and S1 does not
edit them.

- [x] S1.1 `data/sources.py` [M1.1]: an explicit task-id-to-family table covering every id in the
      probing, ID-test and OOD splits. It replaces `coderouterbench.source_of`, which yields 600
      "families" in the probing split, 572 of them with a single task: every
      `dim7_multiple_<lang>_HumanEval_<n>_<function>` id keeps its function name, and each of the
      176 OOD ids (`new64::featurebench::…`) becomes its own family. Unit test: every task maps to
      one of a fixed list of families, no ID family under 30 tasks (S0.13). OOD families are listed
      with their counts but exempt from the minimum, because the OOD tasks are outside the router's
      evaluation population (user, 2026-11-20). If an ID family falls under 30, the agent stops and
      reports candidate merges instead of merging, because the family is the exploration group (O9)
      and its granularity is the user's call. **Done 2026-11-20 (A1, merged `1ffed9c`):** 34
      families (30 ID, 4 OOD: `featurebench` 49, `longcli` 14, `old112` 112, `swe_ci` 1), one
      parse rule per id pattern (an unmatched id raises), per-id table
      `router/data/task_families.csv` with a drift test against the pinned release; every family
      has one dimension. `old112` keeps the release's batch label (97 django, 15 astropy; OOD is
      not routed in evaluation). **`canitedit`** (code_refactoring, 20 tasks) is the one ID
      family under 30; it stays in `PENDING_USER_DECISION` until S1.4 rules on it (user,
      2026-11-20): if CanItEdit is not execution-scored it leaves the eligible pool, the 30-task
      minimum applies to eligible families only, and it leaves the pending set; if it is
      execution-scored, the merge options go back to the user. **Resolved by S1.4:**
      `canitedit` is ineligible (Proxy+), the 30-task minimum applies to eligible ID families
      only, and `PENDING_USER_DECISION` is empty.
- [x] S1.2 `data/schema.py` [M1.2]: long-format storage (task, model, run, score, cost, tokens,
      problem_id, family, score_type, eligible, **available, unavailable_reason**), with the wide
      task × model views derived from it. `unavailable_reason` is a missing token record (the 148
      zero-score, zero-cost Qwen3-Max rows, `cost_source == "missing_token_record_zero_total"`) or
      an empty X13 response (CLAIMS S0.11). The existing `OutcomeMatrix` callers (quick-check
      baselines, `sim/`, m2/m3, their tests) keep working. Test that every field survives `subset`
      and `without`. **Done 2026-11-20 (A1):** `OutcomeMatrix.from_long`; the wide constructor
      is unchanged and the new fields are keyword-only; `mark_unavailable(cells, reason)` takes
      A2's empty-response cells (wired in batch C). `OutcomeMatrix.source` is now the S1.1
      family, so the 2026-11-10 quick-check LinUCB/kNN features and per-source M2/M3 numbers
      would differ if re-run; they are rebuilt in S2.
- [x] S1.3 `data/coderouterbench.py` [M1.3]: two views, complete and paper (missing = 0). Tests:
      56,640 probing cells (7,080 × 8), 23,352 ID-test cells (2,919 × 8), 881 missing paper-view
      test cells, no NaN scores. The 148 missing-token-record rows are flagged unavailable, not
      scored 0 at cost 0 (the current `fillna(0.0)` on cost does nothing, because no cost is NaN);
      the score stays as released, so the no-NaN test still holds. **Done 2026-11-20 (A1):**
      all count tests pass; 148 unavailable cells (106 probing, 42 ID-test), cell for cell the
      ones the S0 scripts treated as missing; score and cost matrices identical to the pre-A1
      loader in both views (reviewer check). The paper view zero-fills 2,417 probing and 881
      ID-test cells and leaves them available: it reproduces the paper only and is never used for
      claims (S1.7 records this in `artefacts.yaml`).
- [x] S1.4 `data/scoring.py` [M1.4]: score type per family from the source benchmark's documented
      rule (exec, similarity or judge), not from the observed score distribution, with the citation
      for each rule. Record which families are eligible (S0.12) and which carry partial credit
      (provisional: `lcb` 1,000 tasks, CRUXEval 800, `lcb_testgen` 869, MBPP completion 124 have
      non-binary scores; an observed-score filter kept only 871, 93, 362 and 122 of them).
      Recompute the eligible pool from these rules. CRUXEval is 792 of the 1,907 provisional
      headline tasks (42%), so its ruling needs a citation and before-and-after counts. Success
      goes through `settings.SUCCESS_SCORE` (S0.1): `ACCEPT_THRESHOLD = 0.5` and `accepted()` are
      removed from the live path, kept only as a legacy constant for the 2026-11-10 m2/m3 records
      (`sim/metrics.py` and `sim/verifier.py` import it today). Rule on `canitedit` (see S1.1).
      **Done 2026-11-20 (merged from `s1-b-scoring` @ `4de4264`; note
      `docs/research/S1.4_scoring_rules.md`, report `router/experiments/s1_4_pool.py`):**
      CodeRouterBench scores per dimension (arXiv 2606.22902v3 Table 5, verified verbatim by
      the reviewer): Exec (pass@1) for code generation, algorithm design, bug fixing, code
      completion, data science and multi-language; Proxy+ (proxy metric plus LLM judge) for code
      refactoring, code understanding and test generation. Eligible: 20 ID families, **5,626
      tasks** (3,968 probing / 1,658 ID-test). Ineligible: the Proxy+ families (including
      `canitedit` and all three testgen families) and DebugBench / `debugbench_sup` (Exec, but
      the test suites need a LeetCode account, 2401.04621v3 §3 fn. 3; the released scores take
      3,351 distinct values). Partial credit is unstated for every ID family; S0.1 applies.
      Against the provisional `EXEC_DIMS`: in, multi_language (1,111) and swebench (71); out,
      test generation (1,111). Every task in the 1,971 / 1,907 provisional sets is eligible
      (CRUXEval 800 / 792). `lcb` is eligible by rule and stays artefact-suspected, pending S1.7
      (776 tasks with X13 verdicts). Success is `scoring.success` (S0.1); the 0.5 threshold is
      `LEGACY_ACCEPT_THRESHOLD`, confined to m3 and `sim/*` (checked by parsing imports). User
      decisions (2026-11-20): `bcb_completion` and `mbpp_completion` stay eligible on the
      dimension-level rule; DebugBench stays ineligible; the hex and all-zero findings go to
      S1.7, not to an amendment.
- [x] S1.5 `data/prompts.py` [M1.5]: add route-anything `tasks.jsonl` (3,297 prompts, including 888
      with `source = lcb`: 776 `lcb` and 112 `lcb_testgen`) to the existing join, then adapters for
      families still missing. Report the join rate per family. **Done 2026-11-20 (merged
      `0474330`; report `router/experiments/s1_5_prompt_join.py`):**
      - **Sources:** `tasks.jsonl` first, then six pinned upstream extracts (DS-1000,
        BigCodeBench v0.1.4, HumanEval-X, HumanEval+, MultiPL-E rs/ts, LiveCodeBench), streamed
        from pinned revisions keeping only task-text fields, hashed in `external/CHECKSUMS` (193
        files). No solution, test or expected-output field is read.
      - **The `tasks.jsonl` prompts are Route Anything reconstructions** (`inventory.json`); the
        CodeRouterBench ID prompt template is unpublished. `prompt` is `raw_prompt` plus a
        per-family instruction wrapper; the builders rebuild all 3,297 prompts byte for byte.
      - **Router field (user, 2026-11-20): `content_text`**, the raw task text plus its task
        payload (CRUXEval's `f(...)` call, LiveCodeBench's starter code, MBPP completion's `def`
        header), with no instruction sentence or label. `join()` returns it. `raw_text` and
        `wrapped_prompt` are kept as columns; the wrapped prompt is used only to reproduce P0.6,
        whose 96.2% cell-predictor accuracy may be inflated by the wrapper. No task-kind field
        (it would be the dimension label).
      - **Join (gate measured on the eligible pool, user 2026-11-20):** 5,555 / 5,626 eligible
        tasks (98.7%; every eligible family 100% except `swebench`, 0 / 71, no source). All ID
        tasks: 6,555 / 9,999 (65.6%); the gaps are dimensions 3, 5 and 8 (3,333 tasks) and
        `mbpp_testgen`, each with the evidence that would close it.
      - **Shared content:** across eligible families, 830 tasks share content, all of them
        BigCodeBench cross-dimension prompts or HumanEval variants of the same problem (a test
        fails on any other group); within families, 5 pairs (P0.6's three plus
        `ds1000_515`/`516` and `ds1000_621`/`622`). P0.6's pairs and 63 / 67 bcb result reproduce
        on all three fields; all 1,907 P0.6 tasks join.
      - Open notes: which MultiPL-E prompt variant CodeRouterBench used cannot be checked (no
        overlap); all six LiveCodeBench release files are pinned (no minimal set computed).
- [x] S1.6 `data/splits.py` (new): `problem_id` taken from the source-benchmark ids in the S1.1
      table (HumanEval, MBPP and BigCodeBench variants linked; `lcb` linked to `lcb_testgen` through
      `tasks.jsonl`), with prompt matching as a cross-check only. Tasks with byte-identical
      `content_sha256` (S1.5) are merged into one problem even when their source ids differ, **for
      splits only; they are never deduplicated for scoring** (user, 2026-11-20). S1.5 found five
      within-family pairs (`ds1000_138`/`180`, `ds1000_137`/`179`, `mbpp_141`/`71`,
      `ds1000_515`/`516`, `ds1000_621`/`622`) and 830 eligible tasks in cross-family groups, all
      BigCodeBench cross-dimension prompts or HumanEval variants of one problem. Translations are linked to
      their HumanEval problem: `hex_<lang>_<Lang>/<n>` (HumanEval-X; the provisional regex misses
      it) and `multiple_<lang>_HumanEval_<n>_<function>`. Every `_sup` family is checked against
      its base family for shared problems. The development split and the five outer folds by
      problem (S0.6), with recorded seeds, and an automated split-integrity test that fails if any
      problem id appears in two of them (R13). The development split is committed as a membership
      file of problem ids, pinned by a test: seed 0 alone is not enough, because changed problem
      ids make seed 0 draw a different split, and CellBest's saving ranged from 15.4% to 32.9%
      across development draws (P0.4). Recount the second review's figure of 643 of 1,430 joined
      test tasks sharing prompt text with the probing split.
      **Done 2026-11-20** (`8e0c4cf`).
      - **Problem ids** (`router/data/problems.csv`) come from three kinds of link:
        - source keys: HumanEval `n` across humaneval, he_*, hex_* and multiple_*; MBPP `n`;
          BigCodeBench `n`; the LiveCodeBench question id;
        - DS-1000 variants, linked to their original through `perturbation_origin_id`, which
          counts within `library` (pinned extract `ds1000_origin`; user, 2026-11-20);
        - identical `content_sha256`, for splits only.

        This gives 6,997 problems (ds1000: 1,000 tasks in 450 problems), 3,487 of them in the
        eligible pool. The only identical-text links beyond source keys are S1.5's five pairs.
      - **Development split:** 697 problems, `round_half_up(0.20 × 3,487)`, holding 1,104
        eligible tasks. Drawn with `SeedSequence([DEV_SEED, 0])`. Not stratified (user,
        2026-11-20); by family it holds 13.5% to 23.1% of eligible tasks.
      - **Outer folds:** fold seeds 0–4 (seed 0 = `GATE_SEED`; user, 2026-11-20). Each seed is a
        `SeedSequence([seed, 1])` permutation of the 2,790 evaluation problems, assigned
        round-robin: 558 problems per fold, 996–1,135 tasks.
      - **Labels:** every task takes its problem's label. Problems with no eligible task are
        `outside_pool`.
      - **Files and tests:** the development split and all five fold assignments are committed in
        `router/data/splits.csv` and pinned by a test. The R13 test checks problem ids, content
        hashes and source keys for every seed: 0 violations.
      - **Eligibility:** the split is drawn on the S1.4 eligible pool. S1.7's artefact filter
        never redraws it. A change to eligibility fails the pin test and needs the user's
        decision.
      - **Second-review recount:** 618 of 1,914 joined ID-test tasks share content with a probing
        task. Eligible ID-test tasks sharing a problem with a probing task: 791 of 1,658 against
        eligible probing tasks, 978 against any probing task.
- [x] S1.7 `data/audit.py` and `experiments/e0_audit.py` [M1.6]: a family × model table flagging
      all-zero families and flagship models below the pool median. Each family's share of the
      oracle and DimensionBest gain is computed only after the labels in `artefacts.yaml` are
      committed (blind labelling; reviewer, 2026-11-20). For each suspected family, inspect the raw responses
      (`response_matrix.json`) and, where tests are available, re-execute them, then label it
      **confirmed defect**, **cleared** or **undecided** (S0.10). Write `data/artefacts.yaml`,
      including every audit-dependent count, and commit it **before** any router result is computed
      (R10). Suspected families (all ID tasks, 2026-11-19): `lcb_testgen` (n=869; Opus 0.48,
      Sonnet 0.51, Qwen3-Max 0.95), `lcb` (Opus 0.37 vs Qwen3-Max 0.63), `mbpp_testgen` (n=111;
      every model 0.00), `he_testgen` (n=131; Opus 0.17, Sonnet 0.00, Qwen3-Max 0.46). Opus and
      Sonnet responses exist for every one of these tasks. Also audit the all-zero tasks Route
      Anything excludes (X12b: BigCodeBench 750, DS-1000 703, LCB 224, MBPP 281, HumanEval 16).
      For BigCodeBench there is execution evidence already: Route Anything's local re-run passed
      91 of 318 recorded responses (`experiments/grading_diagnostic`). Also audit
      `code_generation|mbpp` (218 headline tasks): under success = 1.0 Opus passes 22.0% and every
      other model 11.0% or less, against 75–81% on the MBPP completion cell (P0.5). Also check
      whether the 63 bcb tasks whose prompt is identical to a task in another dimension score
      differently for the same model (P0.6). The labelling criteria (confirmed defect, cleared,
      undecided) are written and committed before any response is inspected, and labelling does
      not look at P0.5, P0.6 or E0 outcomes. The audit states which families were actually
      re-executed and in what sandbox; BigCodeBench may rest on Route Anything's diagnostic.
      **Added from S1.4 (user, 2026-11-20):** audit `hex_cpp`, `hex_go`, `hex_java` and
      `hex_js` (Exec by rule, but 999–1,135 distinct released values per family and almost none
      at 1.0; Route Anything marks all their tasks not runnable); `multiple_rs`, `multiple_ts`
      and `swebench` (every released score is 0); and CRUXEval's 0.8 value (707 of 800 tasks have
      a 0.8 cell; under S0.1 it is a failure, and CRUXEval is 42% of the provisional complete
      case; find what 0.8 means; any change to the success rule is the user's).
      `artefacts.yaml` also records that the paper view is zero-filled and for reproduction only,
      and the empty X13 cells (1,551, of which 877 score 1.0 in the release: X13 lost the
      response, the model did not fail; S0.11 keeps them unavailable, user 2026-11-20).
      **Checkpoint 1 decisions (user, 2026-11-20).**
      - **Empty responses** in CodeRouterBench's `response_matrix.json` keep their released
        scores. There are 3,330 such cells across the 8 models; 895 of them score 1.0, and all
        but the 148 Qwen3-Max rows carry output tokens. They are left out of the audit's
        samples only. S3 reports a sensitivity check with them treated as unavailable. No
        CLAIMS amendment.
      - **Model mapping:** glm-5, MiniMax-M2.7 and qwen3.5-plus are matched to their responses
        by name, marked unconfirmed (their released token counts include hidden reasoning).
      - **Scope:** the 19 flagged eligible families stay in scope as pre-registered.
        Checkpoint 2 is split by toolchain.

      **Done 2026-11-21** (merged `b5d5bc0`). Criteria: `docs/research/S1.7_audit_criteria.md`
      (pre-registered, Amendments 1–10). Report: `docs/research/S1.7_audit.md`.
      - **Labels** (`audit.LABELS`, `router/data/artefacts.yaml`, committed `471ddb5` before any
        router result):
        - **confirmed defect** (3,466 eligible tasks):
          - BigCodeBench ×5 (`toolchain`: tasks using matplotlib, sklearn, seaborn, scipy or
            wordcloud pass 0.18% against 41.3%);
          - DS-1000 (`extraction`: fences and `</code>` not stripped);
          - `lcb`'s 422 LeetCode tasks (functional tasks never called; subset rule on `platform`);
          - HumanEval-X C++/Go/Java/JS and MultiPL-E Rust/TypeScript (`wrong_field`).
        - **cleared** (1,189): `mbpp`, `hex_python`, `cruxeval` (strict 1.0).
        - **undecided** (873): `humaneval`, `mbpp_completion`, `lcb`'s 578 AtCoder/Codeforces
          tasks and `swebench` (the last two by the capped-response rule).
        - **never suspected:** `he_completion` (98).
        - The test-generation families are undecided, ineligible, and in no view.
      - **Views:**
        - all eligible: 5,626 tasks / 3,487 problems;
        - artefact-filtered = headline set: 1,287 / 1,188 (complete case 1,264);
        - sensitivity: 2,160.

        The headline set is CRUXEval 800, MBPP 225, HumanEval-X Python 164 and HumanEval
        completion 98.
      - **Gain shares** on the filtered view, computed after the yaml:
        - the oracle is +14.3 points over Opus, and 91% of that comes from CRUXEval; under the
          CRUXEval 0.8 sensitivity it is +3.3;
        - in-sample DimensionBest beats Opus by 1 task in 1,264;
        - 90.8% of complete-case headline tasks have no visible-test signal for any model.
      - **Reviewer checks:**
        - 48 sandbox re-runs matched;
        - the mechanisms were re-derived from the released scores;
        - the two single-cell reassignments were read and confirmed;
        - the views were recounted.
      - 428 tests pass.
- [~] S1.8 `data/llmrouterbench.py` [M1.7]: code subsets; drop zero-cost models from economic
      metrics. Validate the loader early, so it is ready if DP1b makes it the primary corpus.
      **Started 2026-11-21 as a hedge (user):** S1.7's headline makes a DP1b failure on
      CodeRouterBench likely. In-sample DimensionBest is +1 task, the oracle gain is mostly
      CRUXEval format, and 90.8% of tasks have no verifier signal.
- [x] S1.9 `data/phase5.py` [M1.8]: read-only loader (R5), including the stored raw responses
      (324 non-empty of 325) read from the database without writing to it. **Done 2026-11-20 (A2,
      merged `e2a572f`):** CSV refused on a hash mismatch; one read-only fetch (connection option,
      read-only session, checked before the query; no retries) cached 325 responses to
      git-ignored `external/phase5/`, 324 non-empty; the empty one is FE-010 / Llama-3.3-70B /
      run 1.
- [x] S1.10 `data/prices.py` [M1.9] as the single source of prices, with source URL and access date,
      including reference prices for the live models (S0.2, S0.9) and the verifier rate basis
      (AWS Lambda, O11). Every released `cost_usd` equals tokens × the `models.json` prices
      (reviewer check, 2026-11-20); a test keeps that true. Check `kimi-k2.5`'s input price
      ($0.01/M in `models.json`) against Moonshot's published price and record the finding; the
      released costs are not changed (the ×0.5 / ×2 price sweep covers it). **Done 2026-11-20
      (A2):** cost identity holds on all 79,844 token-priced rows; Opus 4.6 still published at
      $5 / $25; Lambda method reproduces Route Anything's $5.2 × 10⁻⁶ example, and the
      recompute led to amendment A1 (S0.2). kimi-k2.5: Moonshot's own price could not be read
      (JavaScript-rendered page); OpenRouter providers list $0.49–0.60/M input, which raises
      kimi's total cost ×1.15–1.18 with no change in price order. Left open (user, 2026-11-20).
      Live candidates are recorded as `candidate` only; the pool is chosen in S6.4.
- [x] S1.11 `data/verdicts.py` (new): loader for the X13 visible-test and hidden-test verdicts
      (S0.11), joined to CodeRouterBench task ids. Tests cover only counts that cannot change: the
      23,288-row count, 2,911 tasks, and the 2,044/867 split coverage. Headline counts are derived
      after S1.4 and S1.7 and frozen in `artefacts.yaml`. Cells with `w1_n = 0` are marked "no
      signal" (S0.11); `empty` cells are counted separately and marked unavailable (S0.11).
      **Done 2026-11-20 (A2):** 8,577 signal, 13,160 no-signal and 1,551 empty cells; observations
      PASS / FAIL / UNKNOWN; the released score, the X13 heuristic and the hidden-test (`w2`)
      fields are behind separate, labelled functions. Agrees with the P0.5 pivots on every cell (0
      mismatches); X13 `truth` equals the released score on all 23,288 cells.
- **Gate:** the S1.3 count tests pass; zero singleton ID families (OOD families are exempt, as
  from the 30-task minimum; user, 2026-11-20); the split-integrity test passes (S1.6: 0
  violations, all five seeds);
  eligibility comes from S1.4 rules only; prompt join ≥ 95% on the eligible pool (S1.5: 98.7%), or the gap is documented per family
  and every embedding result is restricted to joined tasks; both matrix views load; the verdict
  loader passes its count tests; `artefacts.yaml` committed with every audit-dependent count; the
  three data views (all eligible tasks, artefact-filtered, headline set) can be built from
  `OutcomeMatrix`.
  **Passed 2026-11-21 (reviewer, on `main` at `b5d5bc0`):**
  - the count tests pass (428 in all);
  - no singleton ID family (smallest: `canitedit`, 20, ineligible);
  - 0 R13 violations;
  - 20 eligible families, 5,626 tasks, by rule;
  - prompt join 5,555 / 5,626 (98.7%; the swebench gap is documented);
  - probing and ID-test load in both views;
  - verdicts: 23,288 rows, 2,911 tasks;
  - `artefacts.yaml` is tracked;
  - `audit.build_views` builds all eligible 5,626, filtered 1,287, sensitivity 2,160 and
    headline 1,287 from `OutcomeMatrix`.

### S2 — Evaluation infrastructure and baselines
Validated before any analysis that feeds a decision point, so that DP1 does not rest on the
quick-check code.
- [ ] S2.1 `sim/stream.py` [M2.1]: seeded task order, model-arrival events, bandit feedback only,
      observations as PASS / FAIL / UNKNOWN from the verifier (S0.13). Test: the router never
      receives the released score.
- [ ] S2.2 `sim/verifier.py`: real-verdict replay (S0.11) with per-cell "no signal"; simulated noise
      kept for the E3 sensitivity analysis.
- [ ] S2.3 `sim/metrics.py` [M2.2]: AvgPerf, cost (S0.2), acceptance rate (S0.1), CPAT@floor, cost at
      matched acceptance rate, and regret that includes probe cost in dollars and in probe count.
- [ ] S2.4 `sim/stats.py` [M2.3]: S0.8, plus the development split and nested cross-fitting driver
      of S0.6, including automatic per-fold selection from a frozen candidate list. Reuse
      `splits.load_splits` and `splits.r13_violations`. Stream order and the probe draw under a
      seed use their own SeedSequence stream ids (`[seed, 2]`, `[seed, 3]`, …, at most 99; 100
      and up belong to the S1.7 audit samples); ids 0 and 1
      belong to the development split and the folds (S1.6).
- [ ] S2.5 Baselines [M2.4]: Always-X, Random, DimensionBest (full, k-matched, cost-aware),
      CellBest with true source cells (reference) and with predicted cells (the S0.3(c)
      comparator), using a cell predictor that infers dimension × source from the task content
      (`content_text`, the router's field; S1.5) (Route Anything Table 11) with its accuracy
      reported, and its accuracy on the wrapped prompt reported as a sensitivity (P0.6's 96.2%
      used the wrapped prompt), Route Anything's zero-loss direct-dispatch
      route,
      SourceBest (on the S1.1 families), fixed cascade, released ACRouter decisions, LinUCB with
      stated features, kNN on prompt embeddings (joined tasks only). Baselines that use labels the
      router may not use (source, dimension) are labelled as given more information.
- [ ] S2.6 Tests [M2.5]: oracle 57.00, Always-Opus 43.83 and released ACRouter 50.14 reproduced on
      the paper view; a shuffled-label test in which each learned baseline falls to Random.
- **Gate:** S2.6 passes; the no-ground-truth test passes; every comparison table carries a paired
  CI; baselines that would need the paper's unavailable code are labelled "our definition", with
  the definition difference documented. Reference: DimensionBest vs ACRouter is +1.54 on the
  complete view and −0.05 [−0.31, 0.23] on the paper view.

### S3 — Analyse data and feasibility
All analyses run on the development split and the S2 infrastructure; the evaluation population is
not used for any S3 number that feeds a decision.
**Data notes from S1.7 (2026-11-20):**
- Qwen3-Max's released `output_tokens` take one fixed value per dimension, so its released
  cost is a per-dimension constant, not a per-task cost. State this wherever per-task cost
  matters (S0.2).
- Every headline number also gets the sensitivity check with empty `response_matrix.json`
  cells treated as unavailable (S1.7).

- [ ] S3.1 Q1 accuracy headroom: oracle and DimensionBest against the best single model, in all
      three data views.
- [ ] S3.2 Q2 cost headroom and feasibility bound (**DP1b**): cost at matched acceptance against the
      best single model, verifier cost included (S0.2), with the real verifier (S0.11), on the
      headline set. DP1b is decided on the **oracle-on-no-signal bound** (§1); fixed cascades
      with the no-signal cells bracketed (all accepted / all escalated) are reported next to it.
      Provisional check on 2026-11-19 (P0.5; complete-case headline set, 1,907 tasks, success =
      all tests pass, point estimates, verifier cost at S0.2's rate, Opus best at 67.3% and chosen
      in every training fold of a 5-fold cross-fit, five seeds):

      | No-signal cells | Ladders meeting both thresholds | Best gap at ≥ 20% saving |
      |---|---|---|
      | All accepted | 0 | −13.1 points |
      | All escalated | 0 | −13.9 points |
      | Oracle decides (bound) | 12 | +10.9 points (23.2% saving) |

      The bound passes (a) and (b), so they are reachable in principle; the learned prediction must
      recover about 11 points over the best fixed cascade. Against CellBest with predicted
      cells (P0.6, seed 0, point estimates): fixed ladders, 0 / 0 / 7 ladders meet (c) with
      no-signal cells accepted / escalated / oracle-decided; the best gap at saving > 0 is −16.0 /
      −16.8 / +10.0 points (the last at a saving of 1.6%, at the threshold). Per-cell choice fitted
      on training folds meets (c) in all three modes; in the bound, Rule A saves 36.6% at +1.6 and
      Rule B 16.4% at +8.1. CellBest with predicted cells itself saves 18.2% against the best
      single model (true cells: 22.9%). Rerun after S1.4 fixes the population. The 1,971-task
      figures with unavailable cells scored as recorded (0 / −13.4, 0 / −14.0, 12 / +10.8) and
      the earlier −1.4 to −1.7 (X13 heuristic) and −6.3 / −8 (old 1,262-task set) figures are
      superseded.
- [ ] S3.3 Q3 unsolved and rescue: share of tasks no model solves; for each pair of rungs, the rate
      at which the higher rung passes when the lower one failed, and the rate observable from
      verifier histories (S0.13). Recompute the $41.0 full-ladder, $34.0 Always-Opus and $6.0
      perfect-stop figures (all ID-test tasks, pass = score ≥ 0.5) on the eligible population.
      Report how much a post-failure classifier captures against its perfect-information bound,
      so the gap is visible before S4 (Route Anything C.5: pre-attempt signals capture −0.013 to
      +0.078 within a cell).
- [ ] S3.4 Q4 onboarding budget: k-matched DimensionBest curve (reference 2026-11-19, original
      split: 44.4 ± 2.6 at k = 25, 47.9 ± 2.1 at 50, 49.3 ± 0.9 at 100, 50.7 ± 0.6 at 250).
- [ ] S3.5 State the expected value of onboarding before running E1: how much a new model could
      add in a pool that Opus dominates.
- [ ] S3.6 Verifier and population profile: false-pass and false-reject rates of the X13
      visible-test verifier per family and model; the no-signal share per family; the headline
      set's family mix, pass rates and unsolved share next to the full pool's (S0.11).
- [ ] S3.7 Power check: rerun `e0_power.py` with the real verifier and nested cross-fitting on the
      final headline population; report the gate's power at −2 (kept by the user even if about
      60–70%).
- **Gate:** S3.1–S3.7 reported with CIs in all three data views.
- **DP1a and DP1b** (see §1). If DP1b says no, stop and bring it to the user before S4.

### S4 — Build the router
All fitting uses the training folds only (S0.6).
- [ ] S4.1 Router v1: shrunk per-group lookup table plus fixed two-model cascade, running end to end
      through the S2 harness.
- [ ] S4.2 `core/exploration.py` and `core/stopping.py` [M4.1, M4.2]: bounded forced exploration
      (share per group, maximum probe calls, stopping tolerance, noisy-verifier handling). Router v2.
- [ ] S4.3 `core/ability.py` [M3.1]: the hierarchical model of §2; `core/irt.py` kept as the 1PL case.
- [ ] S4.4 `core/difficulty.py` [M3.2]: prompt-embedding predictor without the source label,
      on `content_text` (S1.5; never the wrapped prompt, whose instruction wording identifies the
      family);
      optional group classifier from the prompt; CV grouped by problem.
- [ ] S4.5 `core/rescue.py` (new): calibrated classifier for P(later rung passes | earlier rungs
      failed); `core/policy.py` [M5.1]: cheap-first by break-even ratio, ladder truncation, budget cap.
- [ ] S4.6 Non-transductive evaluation [M4.3]: test-item difficulty is never fitted from the other
      models' outcomes on those same items (fixes `router/experiments/m3_irt.py:67-74`).
- **Gate:** router v1 and v2 run end to end on the development split with all S2 metrics; v3 and v4
  components have unit tests.

### S5 — Offline test and improve (development split only)
- [ ] S5.1 Evaluation of the ability model [M3.3]: leave-one-model-out held-out log-likelihood; k = 0
      scored at the pool prior; rankings reported per single draw. Reported, but DP2 is decided in
      routing units.
- [ ] S5.2 E1 onboarding [M4.4]: arms zero-probe prior, k random probes, k stratified probes,
      in-stream learning with stopping; against k-matched DimensionBest; fixed k grid
      {10, 25, 50, 100}, replay only; report the distribution of probes the stopping rule uses. **DP2.**
- [ ] S5.3 E2 cost [M5.3]: eligible headline tasks under the S0.1 success rule, real verifier
      (S0.11); v1–v4 against
      cost-aware DimensionBest, fixed cascade, Always-best and released ACRouter; cost at matched
      acceptance.
- [ ] S5.4 E3 verifier sensitivity [M5.4]: simulated ε ∈ {0, 0.1, 0.2, 0.3} for uniform, one-sided and
      difficulty-linked noise, reported against the real-verifier result.
- [ ] S5.5 Improve and re-test, on the development split only (S0.7). Log every router version
      tried with its development-split numbers.
- [ ] S5.6 Optional [M4.5, M5.5]: Fisher anchors on LLMRouterBench; Thompson sampling and the kNN
      memory residual, only under the conditions in §2.
- **Gate:** DP2 decided; the candidate list (v1–v4 and hyperparameter grids) and the per-fold
  selection rule are frozen and tagged in git before S7 (S0.6).

### S6 — Live pilot
- [ ] S6.1 The §4a "before S6" fixes (sandbox limits, `<think>` stripping, true `tests_total`,
      `failure_mode`, per-attempt temp files, TestRunner fixes), plus per-provider handling of
      `temperature` (`pipeline/config.yaml` sets 0.2; the 2026-10-21 Anthropic API version rejects
      it). Start the Docker daemon (not running on 2026-11-19).
- [ ] S6.2 Structured test results [M6.2]: Jest and Playwright JSON reporters, stable test ids,
      explicit skipped and error states. Today's parser invents names such as `passed_1`
      (`pipeline/components/test_runner.py:174-177`), which cannot support a test split. A new run
      writer in `router/live/` replaces `results_storage.py`, which writes to the Phase 5 `runs`
      table (R5).
- [ ] S6.3 `live/testsplit.py`: run the full suite once per output, then split hidden and visible
      tests afterwards; reserve the hidden grader pool first. The split manifest and its random
      seed are committed before any stored result is inspected. Record the achievable visible
      fractions per task.
- [ ] S6.4 `live/registry.py`, `live/gateway.py`, `live/cache.py` [M6.3]: the S0.9 pool. Record which
      models are siblings and their reference prices. Collect a full model × task output matrix
      once, cache it, and replay routing over it, so spend does not depend on the policy.
- [ ] S6.5 Pilot on about 20 tasks with at least 2 runs per task, with the frozen procedure, then
      **DP3**.
- [ ] S6.6 Re-run the 324 stored Phase 5 responses through the fixed runner to get per-test
      results at no API cost; store them in new files (R5).
- **Gate:** per-test results stored with stable ids; pilot results logged; DP3 decided; paid spend
  so far logged against the $20 budget.

### S7 — Final run, scored once
- [ ] S7.1 Run the nested evaluation once on the evaluation population: per-fold automatic
      selection from the frozen candidate list, scoring of the frozen procedure and all baselines
      (E1, E2, E3).
- [ ] S7.2 E4 [M6.5]: the S0.9 pool × 65 tasks (runs per task as the budget and rate limits allow),
      stored in new tables or files (R5).
- [ ] S7.3 Analysis [M6.6]: false-pass rate against visible fraction by category, replaying each
      output's real verifier decision. False-pass uncertainty uses the **task** as the unit: for
      each task with at least one wrong output, record whether any wrong output was falsely
      passed, and give a Clopper-Pearson bound over those tasks (about 3/n when none is seen). The
      bootstrap is not used for this bound, because resampling zero events always returns zero;
      noise-corrected oracle (choose on run 1, score on run 2), which requires at least 2 runs
      per task and is marked unavailable otherwise; uncertainty by cluster bootstrap over tasks,
      since repeated outputs of one task are not independent; transfer test through Opus 4.6,
      reported as exploratory.
- [ ] S7.4 Demo [M7]: `cli.py` (`router audit | replay | probe | run | report`) and the static HTML
      report. Proxy optional, only after S7.1–S7.3.
- **Gate:** nested-evaluation and live hidden-test results stored; the performance gate (S0.3) computed;
  the CLI and report run end to end from a clean start.

### S8 — Results report
- [ ] S8.1 `Reports/Research_Progress_Report_9.md`: a verdict table with one row per decision
      point (DP1a, DP1b, DP2, DP3) and per contribution (C1–C4), each with outcome, evidence and
      consequence; the performance-gate result.
- [ ] S8.2 Report every simple baseline, every negative result, every router version tried and all
      three data views (R2).
- [ ] S8.3 Figures from the static report; a short CLI demo.
- **Gate:** the research gate (S0.4) passes.

Thesis chapters, slides and the defence follow S8. Fix the seven reference titles listed in the
2026-10-05 log then.

### 4a. Repo-audit backlog (from the 2026-10-05 pre-implementation audit)

These items were found while auditing the reused components. Each is a **prerequisite of the
stage named**, so do it at the start of that stage, not earlier.

**Before S6 (live layer):**
- [ ] Find out why every Phase 5 model output for API-013 and CSS-012 collected 0 tests
      (`extraction_success=True`, `tests_total=0`), when the reference solutions pass 12/12 and 8/8.
      The `defective_task` label for these two is unsupported. Recheck whether `drop_artifactual`
      should still drop them, and whether the RQ1 numbers change.
- [ ] Derive and persist `failure_mode`, `runner_error`, `llm_error` and `extraction_method` when
      writing runs (`results_storage.py:213-221`, `run_eval.py` failure builders). Stage S7.2
      depends on this.
- [ ] Failure builders in `run_eval.py` record `tests_total=task.test_count` even when no code was
      extracted. Record 0, or the real collected count, for new runs.
- [ ] Strip `<think>…</think>` in `CodeExtractor` before code extraction (reasoning models are in the live pool).
- [ ] Give each attempt its own temp file and folder in `CodeExtractor` (`code_extractor.py:76`), and
      have `cleanup()` delete only its own files.
- [ ] `TestRunner`:
  - take `model_name` as a parameter (it is always `"unknown"`);
  - wrap its call in `asyncio.to_thread`;
  - run with `docker run --name` and `docker kill` on timeout;
  - build the image before timing starts.
- [ ] Rebuild the Docker image when `tests/` or `tasks/` change (content-hash tag). It currently
      only builds if the image is missing.
- [ ] Sandbox untrusted model code: `--network none --memory --cpus --pids-limit` in `docker-run.sh:70`.

**Before the optional proxy (S7.4):**
- [ ] Add `LLMGateway.query_messages()` for chat message lists, make the retry backoff
      configurable, and narrow the `"rate"` substring check (`llm_gateway.py:73,123,127`).

**During S1:**
- [ ] `router/data/prices.py` becomes the single source of prices. The hardcoded Groq prices in
      `pipeline/analysis/exploratory/*.py` and the all-zero `config.yaml` `cost_rates` stay as
      Phase 5 provenance only.

**Backlog (not blocking):**
- [ ] `TestRunner._count_any_usage` counts `any` inside comments and strings. Count type positions only.
- [ ] Remove the dead `.env.example` keys (`MAX_TOKENS`, `TEMPERATURE`, `WARMUP_REQUESTS`, `RUNS_PER_TASK`).
- [ ] Remove the unused requirements (`sqlalchemy`, `aiohttp`, `asyncio-pool`, `pydantic`) and pin a lockfile.
- [ ] Move `configure_logging()` in `run_eval.py` out of import time.
- [ ] `config.yaml` `runs_per_task: 3` disagrees with the data (Phase 5 ran 1); add a comment.
- [ ] Add a LICENSE before vendoring third-party code under `external/`.

---

## 5. Timeline

Dates are not planned: on 2026-11-17 the user asked for a plan without a timeline, and confirmed
this on 2026-11-19. Scope is Pre-S0 and S0–S8 (implementation plus a findings report); the
thesis is next semester. If time runs short, protect in this order (first item is protected
longest): S0 > S1 and the C4 audit (S1.7) > S2 > S3 and DP1 > router v1 and v2 (C1) > router v4 (C2) > S6 pilot
> full E4 (C3) > static report > proxy. A decision point may reorder this; record any reordering
in the log.

---

## 6. Rules (do not break)

- **R1 — No clairvoyant escalation.** During replay, never escalate on the benchmark's ground
  truth; use the real visible-test verdicts (S0.11), with simulated noise only as a sensitivity
  analysis. In live runs, the verifier's tests and the
  grader's tests must be disjoint.
- **R2 — Always report the simple baselines:** DimensionBest (full and k-matched), cost-aware
  DimensionBest, Route Anything's 21-cell lookup (CellBest) in two variants (true source cells
  as a reference; predicted cells as the S0.3(c) comparator), its zero-loss direct-dispatch route
  (2611.05275 C.2, Table 10), per-cell choice without an oracle (P0.6 Rules A and B, no-signal
  cells accepted or escalated), fixed cascade and always-best. A result without them is not a result.
- **R3 — CPAT always carries an acceptance floor.** Never quote unconstrained CPAT; it picks a
  16.8%-accuracy model.
- **R4 — Primary evidence comes from execution-scored data only.** Cascade claims use
  exec-scored binary tasks only. Other dimensions are reported separately.
- **R5 (D10) — Never write to or repair the Phase 5 `runs` table.** New live runs go to new tables or files.
- **R6 (D5) — No hypothesis may depend on human raters.**
- **R7 — Cost comes from modelled published prices,** with the source and date recorded, plus a
  price-sensitivity sweep (D7).
- **R8 — Multiple seeds and bootstrap CIs on every headline number.** Use paired bootstrap over
  tasks and cluster bootstrap over problems. Never pool leave-one-model-out results into one CI.
- **R9 — Keys and secrets live in `.env` only.** Never commit them.
- **R10 — The artefact list is frozen before results.** `data/artefacts.yaml` is committed before
  any router result is computed, and every headline number is reported in all three data views.
- **R11 — The k = 0 baseline is the pool prior,** never θ = 0.
- **R12 — No gate on seed-averaged estimates.** Gates use single-draw results; averages are
  reported only alongside them.
- **R13 — Problem-disjoint splits for every claim.** The development split and the cross-fitting
  folds share no problem id, enforced by an automated test (S1.6). Priors, difficulty models, thresholds and
  hyperparameters are fitted without evaluation outcomes. The original ACRouter split is used only
  to reproduce published numbers.
- **R14 — The evaluation population is scored once.** Router changes are made and judged on the
  development split. The nested evaluation and the live hidden tests are run once, with the frozen
  candidate list and selection rule (S0.6, S0.7).
- **R15 — Cost is modelled, not spent.** Every cost figure uses S0.2. Actual API spend is tracked
  only against the $20 budget (S0.9).
- **R16 — Eligibility never comes from observed scores.** Which tasks are evaluated is decided by
  documented scoring rules (S0.12, S1.4), and replay never passes the released score to the
  router (S0.13).

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| CodeRouterBench headroom is mostly harness artefacts | DP1a/DP1b; C4 leads the accuracy story; LLMRouterBench as primary corpus if DP1b fails |
| C1 is incremental over UniRoute / ICL-Router | Measure in routing units against k-matched DimensionBest; in-stream learning with stopping is the differentiator; novelty reads in Pre-S0 P0.1 |
| Prior work covers C1–C4 (P0.1: each partly covered) | §1 reworded; C3 rests on E4, so if E4 cannot be run C3 counts as covered and S0.14 is re-applied; IRT-Router read in P0.5 (does not cover C1) |
| A lookup table passes S0.3 (a)–(b) on its own (P0.5: CellBest saving 22.9% [20.8, 25.2], gap +3.0 [+1.5, +4.5]; Route Anything's zero-loss route fails on the gap) | S0.3(c): the router must also beat CellBest with predicted cells. P0.6: the bound meets (c) (per-cell Rule A 36.6%, +1.6); per-cell choice without an oracle meets (c) on point estimates and fails (a) on its CI, and is reported under R2 |
| Little predictive signal within a cell (Route Anything C.5) | S3.3 reports capture against the perfect-information bound before S4; DP1b |
| In-stream learning has lost to static baselines in one pass (ORACLE E.8, Route Anything C.6) | E1 scored on a single pass with no repeated problem (S0.5); calibration probes as a second mechanism (S0.13) |
| The real visible-test verifier is weak (15% false-pass where visible tests exist) and absent for about four-fifths of headline cells (80.0% of the complete-case set, provisional, P0.5); the learned router must recover about 11 points over fixed cascades and also beat CellBest (S0.3(c)) | Replay with real verdicts (S0.11); feasibility bound S3.2; power check S3.7; verifier profile S3.6 |
| Suspected families decide which model is "best single" | S0.10 audit rule fixed before results; raw responses used in S1.7 |
| Fit and evaluation sets share problems (second review: 643 of 1,430 joined test tasks) | R13 and the S1.6 split-integrity test |
| Router changes overfit the tasks they are judged on | R14: development split for changes; nested evaluation with frozen candidates scored once |
| The free-tier pool has no strong top model | One paid strong model as top rung and Always-best baseline, within the $20 budget (S0.9) |
| Paid spend exceeds $20 | Hard cap in the provider console; cache every response; pilot before the full run |
| Free endpoints change or are quantised | Record model id and date per call; finish each run in one short window |
| The hierarchical model does not beat the lookup table | DP2; report the lookup table as the routing model and IRT as an ablation |
| The prompt join stays below 95% | Restrict embedding results to joined tasks and state the coverage |
| About 7.5 tests per web-dev task makes the visible-test split coarse | Report the achievable f per task; pool by category |
| The free-tier pool is nested again (Phase 6 found +8.6%) | DP3 pilot (S6.5) before the full E4 run |
| Free-tier rate limits slow E4 | Batch runs overnight, cache every response, drop to 2 runs if needed |
| Docker breaks again | Subprocess sandbox fallback, flagged as less isolated |

---

## 8. Key references

| Paper | Where it's used |
|---|---|
| Agent-as-a-Router / ACRouter (2606.22902) | the base system, data and baselines |
| UniRoute (2502.08773), EmbedLLM (2410.02223), RouteProfile (2605.00180) | new-model representation (S4, E1) |
| IRT-Router (2506.01048), tinyBenchmarks (2402.14992) | IRT and anchors (S4) |
| Dekoninck et al. (2410.10347) | cascade optimality (S4.5) |
| Route Anything (2611.05275; code `InfRec-Cardinal/route-anything` at `e137f3f`; data `Lance1573/route-anything-data` at `bd361209`) | real verifier verdicts (S0.11), stop-or-escalate rule (C2), lookup baselines (R2), BigCodeBench diagnostic (S1.7) |
| ORACLE (2607.22465) | online verifier feedback; the cold-start result in Appendix E.8 frames C1 |
| PILOT (2508.21141), BaRP (2510.07429), LLM Bandit (2502.02743) | exploration and online learning (S4.2) |
| LLMRouterBench (2601.07206), kNN baseline (2505.12601), Routing Plateau (2606.07587), 2608.23023 (confirmed as the task-type paper) | evaluation honesty (S3, S5) |

The full list is in `Reports/Research_Progress_Report_7.md` §10.

---

## 9. Session protocol (for every agent or session)

1. **At the start:** read §Status, §6 Rules, and the current stage's section. Check the
   progress log for open issues.
2. **During work:** tick the checkboxes as you go (`[~]` when started, `[x]` when the gate or task is verified).
3. **At the end of the session:**
   - update the Status table: current stage, next action, progress, date;
   - append one dated entry to the progress log: what was done, any numbers produced, any
     problems, any decisions;
   - if a decision changes this plan, edit the plan and note the change in the log. Do not
     silently diverge from it.
4. **Never mark a gate passed without the evidence** (test output, reproduced number) quoted in the log.

---

## 10. Progress log

Newest entries go at the bottom. One `###` heading per session.

### 2026-08-16

Direction reset to a sufficiency router; charter D1–D10 (branch `worktree-phase8-goal-lock`).

### 2026-08-18

Adversarial review: H-C is prior art, CPAT needs a floor, nesting holds, −33 to −50% CPAT prize.

### Sep 2026

Literature review; PROBE-CASCADE designed; Report 7 written and submitted (branch `worktree-progress-report-7`).

### 2026-10-04

This plan created. User decisions: code in a `router/` package in this repo; free-tier APIs only; demo as an OpenAI-compatible proxy plus dashboard; Docker available. Next: M0.

### 2026-10-05

Repo cleanup before implementation (branch `worktree-repo-cleanup`, pushed to `main`).

- **Git:**
  - added a root `.gitignore` (`.claude/` ignored and untracked, `.DS_Store` untracked, `.env*` ignored);
  - merged data-sufficiency, goal-lock and progress-report-7 with `--no-ff`;
  - moved the 5 handover notes to `docs/history/handover/`.
- **Data and tests:**
  - froze the Phase 5 data at `data/phase5_results.csv`;
  - repointed the exploratory scripts to it;
  - added root `pyproject.toml` pytest config (73 tests pass);
  - made DB writes opt-in (`--db`).
- **Fix and erratum:** `failure_taxonomy` now labels extraction failures whatever the test total, which affects FE-010 / Llama-3.3-70B only. `drop_artifactual` drops 21 rows, not 20.
  - RQ1 under `drop_artifactual` moves from p = 2.6e-4, V = 0.265 to p = 2.4e-4, V = 0.267.
  - `keep` is unchanged (p = 1.9e-5), and the conclusion holds.
  - Report 7 (submitted) quotes the old taxonomy counts. The thesis must use the corrected ones.
- **Analysis reproduced:** oracle 58.5%, best single 53.8%, recommender 36.9%.
- **Blocker:** the Docker daemon is not running.

### 2026-10-05 (M0 session)

M0 completed on branch `worktree-m0-setup`. Context came from the peer session that wrote this plan.

- **M0.1 Docker.**
  - Docker 29.2.0 is running. The image was force-rebuilt with `./docker-run.sh build`.
  - The reference solutions pass: API-013 12/12 (jest), CSS-012 8/8 (playwright).
  - In Phase 5, all 10 runs on these two tasks had `extraction_success=True` and `tests_total=0`,
    so the suite never collected tests on *model* code. The tasks are not defective. A §4a item
    tracks the real cause and its effect on `drop_artifactual`.
- **M0.2 skeleton** (commit `5cb8635`):
  - `router/` has its 8 subpackages and `settings.py`;
  - `tests/router/test_skeleton.py` checks the import rule (`router.data` is not shadowed by the
    repo's `data/`; pipeline modules import by bare name);
  - `tests/router` is added to `testpaths`;
  - router dependencies are declared in `pipeline/requirements.txt`, and `huggingface_hub`
    1.9.0 is installed;
  - **84 tests pass** (73 before).
- **M0.5 references.** Every id checked resolves on the arXiv API to the cited paper: [3], [5],
  [6], [13]–[17], [19], [23], [25], [26] 2410.03834, [28] 2409.19886, [29] 2502.18482,
  [30] 2305.05176, [31] 2310.12963, plus EmbedLLM 2410.02223.
  - The arXiv comments confirm the venues of [28] (NeurIPS 2024), [29] (NAACL 2025) and
    [31] (NeurIPS 2024). The ICLR 2025 venue of [26] is not in the arXiv metadata.
  - Seven titles are wrong. Fix them in the thesis (M8.3), not in the submitted Report 7:
    - [3] "RouterEval: A Comprehensive Benchmark for Routing LLMs to Explore Model-level Scaling Up in LLMs";
    - [14] "Learning to Route LLMs from Bandit Feedback: One Policy, Many Trade-offs";
    - [15] "Learning to Route and Schedule LLMs from User Retrials via Contextual Queueing Bandits";
    - [16] "The Routing Plateau: Understanding the Accuracy Limits of LLM Routers";
    - [17] "Most of the LLM Routing Gap Is Task Type";
    - [19] "RouteProfile: Graph-Based Profiling for Cold-Start LLM Routing";
    - [25] (Avengers-Pro) "Beyond GPT-5: Making LLMs Cheaper and Better via Performance-Efficiency Optimized Routing".
- **M0.4 novelty verdict: the gate tripped partially, so the contribution is reframed** (§1).
  The sources are abstracts, plus the cited full-text sections for UniRoute and ICL-Router.
  - UniRoute (2502.08773, Fig. 2) plots deferral AUC for unseen LLMs against the number of
    validation samples. It is cost-aware and includes LiveCodeBench.
  - ICL-Router (2510.09719, §4.6) reports routing accuracy for 100/300/500/1000 profile exemplars:
    74.94% at 100, peaking at 76.30% at 500. It covers MBPP and HumanEval, with no cost, no
    cascade and no online update.
  - These are not threats:
    - RouteProfile 2605.00180 builds metadata profiles without probing; it is now the E1
      text-only comparator;
    - FlyRoute 2605.22057 is an agent-profiling flywheel with an LLM router and proprietary data;
    - RGD/CASCAL 2601.09692 uses generated data and has no k curve;
    - none of ACRouter's 11 Semantic Scholar citers measures a k curve; they cover agentic,
      harness and VLM routing.
  - MonoRouter was not found on arXiv or the web.
  - E1 now adds random-vs-Fisher anchors and a UniRoute-style cluster representation.
  - Follow-up checks on the raw arXiv HTML:
    - the ICL-Router numbers (74.94%, 76.30%) are confirmed in the text;
    - UniRoute's validation sizes appear only in the Fig. 2 image. Its K-NN grid starts at
      {5,10,25,…} and it runs 100–1000 trials, so small k is probably covered. "Small k" is
      therefore dropped as a differentiator;
    - IRT-Router 2506.01048 §6.3 onboards a new LLM (Claude 3.5 Haiku) from its text profile
      only, reaching ACC 0.67, and names few-shot LLM cold-start as future work. This supports
      the IRT-anchor claim;
    - the abstracts of all 11 ACRouter citers were read, and none measures a k curve. ORACLE
      (2607.22465, verifier-feedback routing) is related work for the verified cascade.
  - The Phase 5 tests for API-013 and CSS-012 were last changed on 2026-04-05, and Phase 5 ran
    on 2026-04-27. So "not defective" refers to the same tests the models were scored on.
- **Decision needed from the user:** confirm the reframe. Submission and defence dates are still TBD.

### 2026-11-10 (Report 8 session)

No implementation work happened between 2026-10-05 and today. This session wrote Report 8
(`Reports/Research_Progress_Report_8.md`): requirements, repo design and planned analyses through
M3, with no router results. Branch `worktree-progress-report-8`, built on `worktree-m0-setup`.

- **CodeRouterBench profile** (HF revision 2026-06-23; scripts and outputs in
  `router/experiments/m1_profile/`). No model calls.
  - 9 ID dimensions (Report 7 said 10); 11 source families. ID test model means range from
    35.62 (MiniMax-M2.7) to 43.83 (Opus). Qwen3-Max scores 39.67 at $0.00052/task vs Opus $0.01165.
  - Clairvoyant bounds (R1): oracle 60.02 on the complete matrix; 24.3% of tasks unsolved by any
    model; cheapest-max-score cost −82% vs Always-Opus. Nesting: 78–90% of each model's solves
    (score ≥ 0.5) are also Opus solves.
- **Requirement changes:**
  - ID tasks have no prompt text (new M1.7 prompt join);
  - 41% of scores are continuous (M1.2 scoring tags, M3.1 rule);
  - the paper's tables use the ragged `obs_matrix.json`, with 881 missing ID-test cells and
    GLM-5 missing 404 tasks (M1.2 two views);
  - the ACRouter GitHub repo returns 404 (M2.4 and its gate rewritten).
- **Reproduced from released files:**
  - Always-Opus 43.83 AvgPerf, Perf/$ 1.29;
  - released ACRouter ID decisions 50.14, $22.31, 2.25. The policy is labelled
    `hierarchical_group_model_map`: a source / first3 / dimension lookup, with Opus on 61.3% of tasks;
  - oracle 57.00 on the ragged matrix with missing cells scored 0. The paper's AvgPerf numbers fit
    this view (Always-Opus 43.83 needs missing = 0; skipping gives 43.85). Its Perf/$ numbers use the
    release CSV costs; the costs inside `obs_matrix.json` give Opus 0.43, not 1.29
    (`m1_profile/ragged_zero.py`).
  - **Open:** DimensionBest gives 50.10 on that view and 51.68 on the complete matrix, vs the
    paper's 47.50. Do not cite this as a finding until M2 resolves it.
- **Report 7 corrections recorded in Report 8 §4:**
  - the released OOD run *does* escalate, with a fixed-order cascade;
  - the release's OOD176 table gives ACRouter 73.30 and Opus 63.64, not 62.50 and 57.14.
- **Assumed, not confirmed by the user:** the narrowed contribution in §1. Report 8 §5 states it.
  Submission and defence dates are still TBD.

### 2026-11-10 (M1–M3 analyses session)

Implemented and ran M1–M3 on CodeRouterBench (offline, no API calls). Code is in `router/data`,
`router/sim`, `router/baselines` and `router/core`. Scripts are in `router/experiments/`
(`m2_reproduce.py`, `m3_irt.py`, `m3_difficulty.py`), with outputs in `router/experiments/out/`.
19 router tests pass, including the reproduced published numbers and a Fisher-on-synthetic-data regression test. Report 8 was rewritten as a short
results report.

- **M1:**
  - row counts, NaN checks and zero-token rows all match the release;
  - the paper matrix has 881 missing test cells;
  - prompt join reached **48%** (DS-1000, CRUXEval, MBPP, HumanEval, BigCodeBench; 4,797 tasks after fixing the `multiple_*` id parse), against the 95% target;
  - the gate is partial.
- **M2 (paper view, missing = 0):** three numbers reproduce exactly:
  - oracle 57.00;
  - Always-Opus 43.83 / 1.29;
  - ACRouter release 50.14.

  Random gives 38.35 against 38.75, within ±1. Three do not:
  - DimensionBest 50.10 against 47.50;
  - kNN 48.45 against 47.18 (metadata features);
  - LinUCB 50.32 against 46.84 (metadata features, warm-started).

  Gate partial: the paper's code is unavailable and we use metadata features.
- **A3 (complete matrix, floor = Always-Opus accept rate 43.5%):**
  - Qwen3-Max→Opus scores 49.4 at −44% cost, and 46.2 at ε = 0.2;
  - Qwen3-Max→GPT-5.4 scores 46.3 at −74%;
  - DimensionBest scores 51.7 at −62%;
  - verifier cost is not counted.
- **A4:** held-out log-likelihood per cell:
  - 1PL −0.373;
  - 2PL −0.366;
  - **per-dimension θ 1PL −0.307**.
- **A5 (5-fold CV grouped by prompt, 2,990 distinct problems):** MiniLM + source one-hot ridge reaches Spearman 0.77 with fitted b and pass AUC 0.89. Source mean gives 0.58 / 0.81, and the embedding alone 0.57 / 0.78. The first run used random folds, which leaked shared prompts; it was fixed.
- **A6/A7:** random anchors **beat** adaptive Fisher anchors at every k:
  - at k = 50, ability error is 0.27 vs 0.35 and the log-likelihood gain recovered is 84% vs 71%;
  - Spearman is 0.81 (1PL) / 0.88 (2PL) with random anchors, passing the gate, and 0.62 / 0.50 with Fisher.
  - The Fisher-anchor contribution is unsupported as built. M4 tests stratified anchors and per-dimension θ.
  - The test-task AUC in `a6_*.csv` is uninformative for the 1PL, because item ranking does not depend on θ. Use `ll`.
- **Fisher check (`m3_fisher_synthetic.py`):**
  - Building the check exposed a bug: the undamped Newton step in `irt.ability_posterior` diverged with Fisher anchors on true 2PL parameters. It is now damped and clipped, with a regression test.
  - After the fix, on synthetic 1-D data at k = 50, Fisher beats random: 0.09 vs 0.28 with true parameters, and 0.24 vs 0.32 with parameters fitted from 7 respondents.
  - The real-data A6/A7 numbers are unchanged by the fix, so the real-data loss is genuine. The likely cause is multidimensionality (A4).
  - Next: per-dimension Fisher with per-dimension θ.

### 2026-11-19 (plan v2)

The plan was rewritten as v2 (this file; v1 is at `b8edb5f`). The user clarified that the M1 and M2
work of 2026-11-10 was a set of quick checks for Report 8, not finished infrastructure, and asked
for a plan without a timeline. An independent reviewer (Fable) audited the plan and code in two
rounds; its numeric claims were rerun with the repo's own loaders
(`router.data.coderouterbench.load`, `router.baselines.static.group_best`) before adopting it.

- **Confirmed:**
  - zero-probe pool-mean prior ability error 0.268 vs random k = 50 0.270 (θ = 0 prior: 0.791);
  - single-draw Spearman at k = 50: mean 0.51, min −0.31, 1 of 10 draws ≥ 0.8; at k = 10: 0.02;
  - 8 distinct pass counts across items with 7 models, so a 1PL bank has 8 difficulty levels;
  - `source_of` yields 600 probing-split families, 572 of them singletons;
  - DimensionBest's 7.85-point gain over Always-Opus (complete view): test_generation +4.20,
    algorithm +2.80, code_refactoring +0.45, data_science +0.45, the rest −0.05;
  - within test_generation, Opus scores 0.47 on lcb_testgen (n=236), 0.18 on he_testgen (n=39) and
    0.00 on mbpp_testgen (n=31, all models 0.00); within algorithm, Opus 0.27 on lcb (n=279) vs
    qwen3.5-plus 0.57. (A check earlier in the same session reported Opus 0.86 on lcb_testgen;
    it used the broken `source_of` and was wrong.);
  - cheap-first break-even P(pass) for Qwen3-Max before Opus: 4.50% (mean cost $0.00052 vs
    $0.01165 per task);
  - full cheapest-first ladder with a perfect verifier (pass = score ≥ 0.5) costs $41.0 on the
    ID test split vs $34.0 for Always-Opus; 39.3% of tasks are solved by no model; stopping after
    the first rung on those tasks would cost $6.0 (perfect-predictor bound);
  - k-matched DimensionBest (20 draws): 44.4 ± 2.6 (k=25), 47.9 ± 2.1 (50), 49.3 ± 0.9 (100),
    50.7 ± 0.6 (250), 51.4 ± 0.3 (1000); full table 51.7.
- **Partly confirmed:** DimensionBest vs released ACRouter is +1.54 on the complete view but
  −0.05 [−0.31, 0.23] on the paper view (paired bootstrap, n = 2,919). The share of test tasks
  without a run-time verifier is larger than the reviewer's 45%: 41% of cells are continuous and
  67% of tasks sit in dimensions with continuous scores.
- **Web-dev per-test data:** the frozen Phase 5 CSV stores test counts only; the database schema
  has a `test_results` JSONB column and `TestRunner` parses assertions; 515 tests over 69 test
  files (about 7.5 per task).
- **Decisions:** Fisher anchors dropped as the headline (untestable with 7 respondents); goal
  reframed around RQ-A to RQ-C and contributions C1 to C4 (§1); decision points DP1 to DP3 added;
  rules R1, R2, R4 and R8 tightened and R10 to R12 added; Thompson sampling, the memory residual
  and the proxy deferred; metadata kNN dropped. C2 (stop or escalate) was added by the main
  session on the strength of the $41.0 vs $6.0 result; the reviewer listed it as an alternative
  approach.
- **The 2026-11-10 numbers above this entry are quick checks.** Cite them only after the M1–M3
  rebuild reproduces them.
- **Open questions for the user:** submission and defence dates; whether the committee accepts the
  audit (C4) as the lead result if DP1 trips; whether the scope is formally limited to
  test-verifiable tasks; which free-tier models are available now; whether raw outputs for
  `lcb` and `lcb_testgen` exist anywhere (the release holds scores and tokens only).

### 2026-11-19 (plan v3)

A second independent review of plan v2 was discussed with the user. Its claim that the test runner
invents test names was confirmed in code (`pipeline/components/test_runner.py:177-182`). Its
figure of 643 of 1,430 joined test tasks sharing prompt text with the probing split was not
recounted; S1.6 does that.

- **User decisions:**
  - this semester delivers the full implementation plus a findings report; the thesis and defence
    move to next semester; no timeline in the plan;
  - the claim definitions become their own stage, S0, approved as drafted: acceptance from the
    hidden grader; modelled cost including verifier runs, retries and probes; at least 20% cheaper
    with an acceptance gap of at least −1 point (lower end of the paired 95% CI); k = calls to the
    new model; 60/20/20 problem-disjoint splits; tune-only iteration; $20 budget for one paid
    strong model as top rung and Always-best baseline;
  - the live pool is free-tier models plus that one paid model.
- **Plan changes:**
  - milestones M1–M8 replaced by stages S0–S8 (§4), with plan v2 ids kept in brackets;
  - success criteria renamed SC1–SC7 to avoid clashing with the stage names;
  - research gate and performance gate separated (§1);
  - DP1 split into DP1a (accuracy) and DP1b (cost); DP2 decided in routing units on the tune split;
  - method (§2): explicit bounded exploration, a calibrated rescue classifier for stop-or-escalate,
    and a build order v1–v4 where each part must beat the previous version;
  - C4 renamed a fallback contribution; suspected anomalies are sensitivity analyses only;
  - novelty reads moved to S2.6, before the router is built;
  - rules R13 (problem-disjoint splits), R14 (evaluation scored once) and R15 (cost modelled, not
    spent) added; risks updated.
- **Still open:** the per-run rate for verifier cost (S0.2); which free-tier models are available;
  which paid model is the top rung; the report due date.

### 2026-11-19 (pre-S0 audit and S0 decisions)

- **Merge:** `worktree-progress-report-8` merged into `main` with `--no-ff` (`6065c3d`), on the
  user's instruction. `pipeline/venv/bin/python -m pytest`: 91 passed, 1 skipped (the skipped test
  needs `external/`). Not pushed. Plan v3 is still uncommitted on `worktree-revised-plan` (P0.2).
- **Hugging Face check** (read-only download to a scratch directory):
  - `Lance1573/route-anything-data` (rev `bd361209`) holds `response_matrix.json`: raw response
    text for all 9,999 CodeRouterBench ID tasks (Opus 4.6 missing 1, Sonnet 4.6 0, GPT-5.4 14,
    glm-5 1,458, qwen3.5-plus 1,074 empty). Opus and Sonnet responses exist for every task in the
    four suspected families, so S1.7 can confirm or clear them.
  - `experiments_x/common/tasks.jsonl`: 3,297 prompts with entry points, including 888 `lcb`.
  - `experiments_x/X13_memory/cache/verdicts.jsonl`: 23,288 rows (2,911 tasks × 8 models; corrected
    in the next entry: the tasks come from both splits, not the ID-test split only)
    with visible-test acceptance and hidden-test outcomes. On binary-truth cells the visible-test
    verifier passed 47.3% of wrong answers and rejected 21.6% of correct ones; 63% of cells had no
    visible tests; hidden-test outcome agreed with the released score on 87.7% of cells.
    False-pass by family: MBPP 99%, DS-1000 95%, HumanEval-X 68%, CRUXEval 32%, HumanEval 33%,
    LCB 14%, BCB 12%, HE 9%.
  - The data comes from "Route Anything: A Theory of Inference Routers", which studies verifier
    feedback, operating points and paid forecasts on the same benchmark: a novelty risk for C2 and
    C3 (P0.1).
  - `Lance1573/CodeRouterBench` (rev `e567d89b`) also carries `response_matrix.json` and
    `obs_matrix_clean.json`.
- **Phase 5 database** (read-only count, approved by the user): 325 runs, 325 `raw_response` (324
  non-empty), 325 `extracted_code`, 325 `test_results`. The stored `test_results` came from the
  parser that invents test names and are not used.
- **Power calculation** (CodeRouterBench, 4,210 exec-binary ID tasks; 2,735 with the four suspected
  families removed; cluster bootstrap over problems; simulated 10% one-sided verifier; fixed
  ladders as stand-ins). Scripts to be moved into `router/experiments/e0_power.py` (P0.4).
  - All tasks: best single model is qwen3.5-plus (33.9% acceptance, $1.77 per 1,000 tasks); Opus
    30.0% at $14.28. 47% of tasks are solved by no model; the oracle saves 45.7% against the best
    single model.
  - Suspected families removed: Opus becomes the best single model.
  - 60/20/20 split: acceptance-gap CI half-width 1.9–3.1 points (filtered view); a −1 margin
    passes a working cheap ladder in 17% of splits.
  - 5-fold cross-fitting by problem: half-width 0.85–1.4 points; margin needed for 80% power at a
    true gap of 0 is about 1.2–2 points.
- **User decisions:** S0.3 margin −2 points with cost and acceptance both tested on CI lower bounds;
  S0.6 5-fold cross-fitting by problem; S0.10 artefact-filtered headline view with the evidence
  rule for suspected families. Also adopted into S0: S0.11 real-verdict replay verifier, S0.12
  scope limited to test-checkable tasks (R4), and the S0.9 live-pool tiers with `claude-opus-4-6`
  as the paid top rung (still served at $5 / $25 per M tokens; about $4 for 65 tasks × 3 runs
  from Phase 5 token means). The user asked to ignore the report due date and the semester
  timeline; the Status table no longer plans by semester.
- **Plan changes:** new Pre-S0 stage with P0.1–P0.4 and a priority table of the remaining missing
  requirements mapped to stages; S0 rewritten (S0.1–S0.12); S1.2, S1.5, S1.6, S1.7, S1.9 updated
  and S1.11 added; S2.6–S2.7 added (novelty reads moved to P0.1); S5 and S7 moved to inner and
  outer folds; S6.3–S6.6 updated; DP2 uses a fixed k grid; R1, R13, R14 updated; §3 and risks
  updated.
- **Repo state found:** Docker daemon not running; `.env` holds only the Groq key and the database
  URL; `fastapi` and `statsmodels` not installed; `external/` exists only in the
  `progress-report-8` worktree.

### 2026-11-19 (third review and S0 fixes)

A third review of plan v3 was checked against the data before adoption.

- **Correction:** the previous entry and S0.11 said the X13 verdicts cover "2,911 ID-test tasks".
  They cover 2,911 tasks from both splits: 2,044 probing and 867 ID-test. 1,909 of them are
  exec-scored binary and 1,262 remain in the filtered view. The review's "2,911 vs 2,919, missing
  8" item followed from this error and is dropped.
- **Measured** (X13, binary-truth cells): with visible tests (8,420 cells) the verifier false-passes
  15.0% and false-rejects 27.0%; with no visible tests (12,472 cells, accepted by a
  syntax/quality heuristic) it false-passes 66.5% and false-rejects 17.6%. No-visible-test share:
  MBPP and CRUXEval 100%, DS-1000 97%, HumanEval about 55%, LCB 24%, BCB 6%. Mean visible-test run
  time 0.074 s; hidden-test run 0.73 s.
- **Feasibility check** (real verifier, X13 heuristic accepts included, point estimates): on the
  1,909 all-view tasks the best single model is qwen3.5-plus (62.4%) and two ladders meet both
  thresholds (Qwen3-Max then qwen3.5-plus: +1.5 points, 45% saving). On the 1,262 filtered tasks
  the best single model is Opus (62.8%); the best ladders reach −1.4 to −1.7 points at 31–57%
  saving. With a CI of about ±1.7, no fixed ladder would pass the −2 gate.
- **User decisions:** all review fixes approved; the −2 margin is kept at about 60–70% power on
  the headline set, and the power is reported with the result.
- **Plan changes:** S0.1 ground truth = released score (X13 hidden tests as sensitivity); S0.2
  verifier cost rate from measured run time with a ×1/×10/×100 sweep; S0.3 power note; S0.6 task
  pool named and onboarding probes from training folds only; S0.9 live onboarding labelled a
  demonstration; S0.11 rewritten (coverage, 1,262-task headline set for cascade claims, "no visible
  tests means no verifier signal"); S0.13 pinned thresholds and S0.14 fallback lead claim added;
  DP1a, DP1b and DP3 given numeric tests; S1.1, S1.11, S2.2, S2.3, S2.7 and S5.2 updated; stale
  "v2", "milestone" and "this semester" wording removed; denominators named for 39.3% and 47%.
- **Citation:** the test-runner placeholder names are at `pipeline/components/test_runner.py:174-177`
  (recounted). The earlier plan-v3 log entry's `177-182` is wrong; S6.2 is correct.

### 2026-11-19 (fourth and fifth reviews: population, observations, nesting)

Two further reviews of plan v3 were checked against code and data before adoption; all fixes
approved by the user, with option (a) for nesting.

- **Checked:**
  - The headline set is easier than the pool (filtered, old binary-only population): best single
    model 62.8% on the 1,262 headline tasks against 29.3% on 2,735 pool tasks; unsolved 13.9%
    against 59.5%. X13 covers the tasks Route Anything could make runnable (2,911 of 2,912
    marked runnable), not a random sample. Family mix shifts toward MBPP (13% to 27% of tasks).
  - 853 of the 1,262 headline tasks (67.6%) and 67.9% of cells have no visible-test signal; on 611
    tasks the visible-test count differs between models.
  - Feasibility under the S0.11 rule (point estimates, 1,262 tasks): no fixed ladder meets both
    thresholds when no-signal cells are all accepted (best gap at ≥ 20% saving −6.3 points) or
    all escalated (about −8 points). The earlier −1.4 to −1.7 figures used the X13 heuristic and
    are withdrawn as a feasibility reference.
  - The observed-score eligibility filter dropped partial-credit tasks: it kept 93 of 800
    CRUXEval, 871 of 1,000 `lcb`, 362 of 869 `lcb_testgen` and 122 of 124 MBPP completion tasks.
    Partial scores are fractions of tests passed (`lcb` 0.2–0.667; CRUXEval and MBPP 0.8).
  - With success = all tests pass over the five execution-scored dimensions (provisional, before
    S1.4): 5,555 pool tasks (best single GPT-5.4, 38.1%); 3,444 filtered (Opus 38.5%, unsolved
    48.4%); 2,747 headline (Opus 57.6%); 1,971 filtered headline (Opus 66.8%, unsolved 10.8%).
  - `router/sim/stream.py:35-36` passes the released score to `router.observe`.
- **Plan changes:**
  - S0: all counts marked provisional; S0.1 success rule for partial credit; S0.3 best single
    model chosen on the scored population inside each training set; S0.5 held-out model excluded
    from all fitting and selection in E1; S0.6 rewritten (20% development split, nested
    cross-fitting with frozen candidates and per-fold automatic selection on the other 80%);
    S0.7 iteration on the development split only; S0.8 clustering by task for repeated live
    outputs; S0.10 restricted-population wording; S0.11 rewritten (selection, difficulty,
    per-cell no-signal share); S0.12 eligibility from documented rules; S0.13 rewritten (PASS /
    FAIL / UNKNOWN observations, ground-truth probe labels, budget-expiry fallback, rescue
    trained on observable histories).
  - Stages reordered: S2 is now evaluation infrastructure and baselines (old S3, plus the
    verifier, the observation types and the nested driver); S3 is analysis and feasibility (old
    S2, run on the development split). DP1 moves to the end of S3.
  - S1.4 recomputes the eligible pool; S1.6 adds the development split; S1.11 tests only
    immutable counts; S5 runs on the development split; S6.3 commits the split manifest before
    inspection; S7.1 runs the nested evaluation; S7.3 requires 2 runs for the noise-corrected
    oracle and clusters by task.
  - C2 reworded as predictive routing with partial verification; R14 updated; R16 added.

### 2026-11-19 (sixth and seventh reviews: feasibility bound, probes, zero-event bound)

Two further reviews were checked against the data; all fixes approved by the user.

- **Recomputed on the provisional 1,971-task headline set** (success = all tests pass, filtered,
  X13 tasks): 1,562 tasks (79.2%) and 79.4% of cells have no visible-test signal; Opus is the best
  single model at 66.8%. Fixed ladders meeting both thresholds: none with no-signal cells all
  accepted (best gap at ≥ 20% saving −13.4 points) or all escalated (about −14). With an oracle
  accept-or-escalate decision on no-signal cells and the verifier elsewhere, 12 ladders meet both
  (best +10.8 points at 24.5% saving). The learned router must recover about 11–12 points; the
  earlier "4–6 points" came from the old 1,262-task set.
- **Plan changes:** DP1b tests the oracle-on-no-signal upper bound and states the gap to close; a
  failed DP1b is a reassessment, not proof that a learned policy cannot work. DP1a runs on the
  full filtered pool with cross-fitted DimensionBest. S0.5 gives every arm with the same k the same
  label source and reports E1 in two pairings (ground truth for all, verifier-only for all). S0.6
  wording fixed: inner folds are formed within the other four outer folds. S0.11, C2 and the
  verifier risk use the 79.2% / 79.4% shares. S0.13 separates calibration probes (charged,
  ground-truth labelled, only on training-pool tasks) from exploration traffic (live requests,
  verifier-only feedback). S3.2 holds the new feasibility table. S7.3 uses the task as the unit for
  the false-pass Clopper-Pearson bound and drops the bootstrap for it. Stale references fixed (S4.1,
  SC4, cut-order, S5.3); R15 and R16 reordered.

### 2026-11-19 (P0.1 and P0.3 done, reviewed and merged)

- **P0.3** merged (`b933171`): `scripts/fetch_data.py` and `external/CHECKSUMS` (187 files, both
  repos pinned to full SHAs). Checked in a fresh directory: plain `shasum -c` passes, the 20 large
  files match the Hub's hashes, an offline re-run fetches nothing, a corrupted file is re-fetched
  alone. The paths to `obs_matrix.json` and `id_decisions.jsonl` were wrong in
  `coderouterbench.py` and three `m1_profile` scripts and were corrected. Tests: 100 passed (the
  CodeRouterBench data test now runs).
- **P0.1** merged (`0a4f40b`): `docs/research/P0.1_novelty_read.md`. C1–C4 each partly covered;
  order unchanged under S0.14; §1 reworded. Route Anything is arXiv 2611.05275. A proposed
  correction of the ORACLE quote from "Fig. 4" to "Fig. 5" was checked against the arXiv v4 PDF
  and rejected: E.8 and B.4 say "Fig. 4", so the quote is verbatim. IRT-Router is still unread.
- **Data facts** (recounted on the pinned data): 1,551 of 23,288 X13 rows are empty responses, 877
  of them with a passing released score; all 611 tasks whose visible-test count "differs between
  models" differ only because of them. The route-anything-data card lists 148 zero-score,
  zero-cost Qwen3-Max rows as missing runs.
- **Plan changes:** §1 contributions and novelty status replaced; S0.5 single pass with no repeated
  problem; S0.11 treats empty and missing cells as unavailable and marks the no-signal shares as
  including them; S1.7 adds the all-zero tasks Route Anything excludes; S1.11 counts empty cells;
  S3.3 reports capture against the perfect-information bound; R2 adds CellBest and the zero-loss
  direct-dispatch route; four risk rows added or replaced; §8 adds Route Anything and moves ORACLE
  to C1; new P0.5 (provisional recount) before P0.4. The power scripts are committed in
  `router/experiments/e0_power/`.
- **Open user decision (after P0.5):** whether S0.3 also requires the router to beat the
  strongest R2 lookup baseline at matched acceptance, not only the best single model.

### 2026-11-19 (P0.5 recount, IRT-Router read, S0.3(c))

- **P0.5** merged (`dcf50ad`): `docs/research/P0.5_recount.md`, `router/experiments/p0_5_recount.py`.
  The inline 2026-11-19 numbers reproduce exactly from the committed script; a reviewer re-run gave
  byte-identical output and 103 tests passed.
- **Unavailable cells barely touch the headline set.** 1,479 of the 1,551 empty X13 rows are
  LCB (excluded); 66 empty cells in 64 tasks remain, and none of the 148 missing Qwen3-Max runs.
  Complete case: 1,907 tasks; 80.0% of tasks and cells have no visible-test signal; the
  no-signal share is real, not an empty-response artefact.
- **S3.2** (complete case, verifier cost, cross-fitted best model, which is Opus in every fold):
  accept 0 ladders / −13.1, escalate 0 / −13.9, oracle bound 12 / +10.9 at 23.2% saving.
- **Route Anything** zero-loss route reproduced on its own population (25.516×, ΔV −0.44,
  0.452 vs 11.655 m$). On our population, direct dispatch only, cross-fitted: zero-loss fails
  S0.3 (gap −1.3 [−3.2, +0.5]); **CellBest passes** (saving 22.9% [20.8, 25.2], gap +3.0
  [+1.5, +4.5]); DimensionBest and cost-aware DimensionBest save only 10.2%. CellBest's
  advantage over DimensionBest comes from the source label, which the router cannot see.
- **IRT-Router** (2506.01048v2) read in full: a new model enters through a text-profile
  embedding with no outcomes; new-model prediction ACC 0.67; no in-stream learning, verifier,
  stopping rule or lookup-table comparison. C1 stays partly covered (UniRoute).
- **User decision (option B):** S0.3 gains condition (c): the router must also beat CellBest with
  predicted cells (lower CI bound of the saving above 0, gap lower bound at least −2 points).
  CellBest with true cells is reported as a reference. DP1b, goal 1, R2, S2.5 and the risk table
  follow; new P0.6 checks the bound against CellBest before `CLAIMS.md`.
- **Other plan changes:** S0.11 and C2 use the complete-case shares (80.0%); S0.3's example uses
  P0.5 pass rates and drops the "wrong population" argument, which no longer holds under
  success = 1.0; S1.7 adds `code_generation|mbpp` (Opus 22.0%).

### 2026-11-19 (P0.6 feasibility against CellBest)

- **P0.6** merged (`36c3371`): `docs/research/P0.6_cellbest_bound.md`,
  `router/experiments/p0_6_cellbest_bound.py`. P0.5's numbers reproduce exactly from its own
  functions; true-cell CellBest recomputed here equals P0.5's to 10⁻⁹. A reviewer re-run gave
  identical numbers (only the runtime field differs) and 109 tests passed.
- **Cell predictor** (prompt text only, inside the cross-fit): 96.2% held-out accuracy. The
  three small bcb cells are not identifiable from the prompt (63 of 67 prompts are identical
  to prompts in other cells) and are all predicted as code_generation|bcb. All cross-cell
  identical-prompt groups share one problem id, so they stay in one fold.
- **Comparator.** CellBest with predicted cells: 70.2%, 4.41 m$ per task; saving 18.2%
  [16.4, 20.2] and gap +2.9 against Opus, below S0.3(a)'s 20% on point estimates on all seeds.
- **DP1b, point estimates:** the oracle bound meets S0.3(c) with fixed ladders (7 of 84, best
  an in-sample pick at 24.7% / −0.6) and with per-cell choice (Rule A 36.6% / +1.6, Rule B
  16.4% / +8.1), on all five seeds.
- **No-oracle per-cell choice** (real verdicts only) meets (c) on point estimates at 1.7% to
  3.3% saving; Rule B with no-signal cells accepted is at the threshold. Rule A with no-signal
  cells escalated also meets (a) and (b) on point estimates (20.9%, +3.1) but fails (a) on its CI
  (lower bound 18.8%). The learned no-signal decision's headroom against CellBest is about 33
  points of saving (3.3% without an oracle against 36.6% in the bound).
- **Power for (c):** gap CI half-width against CellBest 0.3 to 2.3 points, growing with how far
  the policy's routing departs from CellBest.
- **Plan changes (approved by the user):** DP1b states the headroom against CellBest; R2 adds
  per-cell choice without an oracle; S0.3 adds the power note for (c); S0.8 adopts the "at the
  threshold" rule; S1.6 merges byte-identical prompts into one problem (three pairs found);
  S1.7 checks the identical-prompt bcb tasks; S3.2 adds the CellBest results. Next: P0.4.

### 2026-11-20 (P0.4 claim contract, constants, power port; S0 signed off)

- **P0.4** merged (`88a425f`): `docs/CLAIMS.md` drafted from S0.1–S0.14, §1 and §6 at `1921063`,
  with 18 open items; the claim constants in `router/settings.py` (a test pins each value, checks
  it against the `CLAIMS.md` table and against the P0.5/P0.6 scripts); `e0_power.py` ports the four
  power scripts onto the current definitions. A reviewer re-run reproduced every number (only the
  runtime field differs).
- **Power (provisional):** the originals reproduce the plan's 0.85 and 1.38. Ported: no-oracle
  per-cell policies ±1.8–1.9 points against the best single model on the evaluation population
  (power at −2 about 55–60%), ±0.3–0.6 against CellBest. CellBest's saving against the best single
  model ranges from 15.4% to 32.9% across development-split seeds (`code_generation|bcb` flips
  between Opus and qwen3.5-plus). DP1a pool: oracle headroom +12.66 [11.55, 13.91], cross-fitted
  DimensionBest +1.45 [0.68, 2.23] against 3.0.
- **Reviewer finding (O19):** per-cell Rule A with no-signal cells escalated, a buildable policy
  with nothing learned, passes all of S0.3 on the evaluation population (development seed 0). The
  gate stays; C2 and SC2 must beat that policy at matched acceptance.
- **S0 signed off by the user**, accepting the reviewer's recommendation on every open item
  (O1–O19, recorded in `CLAIMS.md`): baselines fitted on the scored population's training folds,
  the router may also use the pool's; gate scored on the complete case; S2.5 predictor frozen
  before evaluation, CellBest's argmax rule kept; 2,000 resamples for gates, 1,000 elsewhere; gate
  seed 0; development-split seed 0; 4 inner folds; exploration exactly 10% until the stopping rule
  or k; group = S1.1 family; AvgPerf = acceptance; verifier rate basis recorded in S1.10; "at the
  threshold" is a label only; price sweep ×0.5 / ×2 per model; DP3 gap = noise-corrected oracle
  headroom; (a) paired; DP1a decided on cross-fitted DimensionBest (provisionally it would fail and
  C4 would lead the accuracy story); C2 must beat per-cell Rule A escalated.
- **Plan changes:** Status moved to S1; P0.4 and S0.1–S0.14 checked; S0 states that `CLAIMS.md`
  wins where more specific; S0.3 power note and paired (a); DP1a, DP3, C2, SC2 updated; §1's
  "score ≥ 0.5" figure marked historical; the P0.6 half-width wording fixed. New constants:
  `DEV_SEED`, `N_INNER_FOLDS`, `N_BOOT_GATE`, `PRICE_SWEEP`.
- **Next:** S1.1.

### 2026-11-20 (S0 implementation check; S1 plan)

- **S0 check (reviewer):** 187 data files verified; 164 tests pass, including the data-gated
  reproductions. P0.5, P0.6 and E0 re-run on the pinned data reproduce their committed JSON
  exactly (only the runtime differs; all three logs complete, no tracebacks). The numbers signed
  off in `CLAIMS.md` and the plan match the JSON: DP1a +1.45 [0.68, 2.23]; gap half-width 1.79–1.87
  on the evaluation population (power at −2 55–59%) and 1.40–1.55 on the complete case (72–77%);
  0.30–0.60 against CellBest.
- Every released `cost_usd` equals tokens × the `models.json` prices (79,844 rows); the 148
  missing-token-record rows are all Qwen3-Max and the S0 scripts treat them as unavailable.
- E0's two DP1a readings give identical DimensionBest numbers because partial credit occurs only
  in `code_completion`, where DimensionBest and the best single model both pick Opus. Not a bug.
- The provisional problem ids miss HumanEval-X (`hex_go_Go/47`) and leave `multiple_*` ids
  unlinked to their source; no S0 number is affected (neither appears in the 1,907-task set or
  the DP1a pool), but S1.6 must link them.
- Quick-check defects for S1 (not used by any S0 result): `scoring.ACCEPT_THRESHOLD = 0.5`
  contradicts S0.1; `coderouterbench.load` scores the 148 rows as 0 at cost 0; `source_of` makes
  the OOD and `multiple_*` ids singletons; `kimi-k2.5` input price $0.01/M looks wrong.
- **Plan changes (user approved):** S1 batch order (A1 ∥ A2, then B, then C; S1.8 side task);
  S1.1 id parsing, OOD families exempt from the 30-task minimum, stop and report instead of
  merging; S1.2 `available` and `unavailable_reason` fields; S1.3 unavailable rows; S1.4 success
  through `SUCCESS_SCORE`, CRUXEval citation; S1.6 HumanEval-X and MultiPL-E links, `_sup` check,
  committed development-split membership file; S1.7 criteria committed before inspection and
  blind labelling; S1.10 verifier rate basis, cost-identity test, kimi price check.
- **Next:** briefs for batches A1 and A2.

### 2026-11-20 (S1 batches A1 and A2 merged; amendment A1)

- **A1 (S1.1–S1.3)** merged (`1ffed9c`): family table (30 ID, 4 OOD), long-format
  `OutcomeMatrix` with availability fields, CodeRouterBench loader with the 148
  missing-token-record cells unavailable. Reviewer check: score and cost matrices identical to
  the pre-A1 loader in both views and both splits; unavailable cells identical to the S0
  scripts'.
- **A2 (S1.9–S1.11)** merged (`e2a572f`): X13 verdict loader (0 mismatches with P0.5), price
  table with sources and the Lambda verifier basis, read-only Phase 5 loader with a response
  cache. Reviewer check: the A2 report script reproduces its committed JSON; the verdict and
  verifier-time figures re-derived independently.
- **User decisions:** `canitedit` waits for S1.4; the singleton gate covers ID families only;
  `old112` keeps its label; the paper view is reproduction-only; empty X13 cells stay unavailable
  (S0.11); the kimi-k2.5 price stays an open note.
- **Amendment A1 (`CLAIMS.md`):** `VERIFIER_RUN_S` 0.074 → 0.200 s and `VERIFIER_RUN_USD`
  $10⁻⁶ → $6.9 × 10⁻⁶. The 0.074 s averaged in the 80% of cells with no visible tests as 0 s;
  0.200 s is the mean over cells where a verifier ran, priced on the O11 Lambda basis. The P0
  records keep $10⁻⁶ (a test pins this) and S3 re-scores on the amended value. A verifier run is
  about 3% of the cheapest model's mean cost per task and 0.1% of Opus's; the P0 numbers were not
  re-run at the new rate, and S3 measures the effect (the ×10 sweep already bracketed it). `e0_power.py` now records the rate it actually used.
- Tests: 247 pass.
- **Next:** batch B briefs (S1.4 ∥ S1.5, then S1.6).

### 2026-11-20 (S1.4 merged)

- **S1.4** merged from `s1-b-scoring` (`4de4264`). Reviewer check: 297 tests pass on the branch
  (298 on `main` after the merge); `s1_4_pool.py` reproduces its JSON; the rule table was
  committed before the pool report and a scramble test shows labels do not depend on scores;
  Table 5 of arXiv 2606.22902v3 read verbatim from the PDF (six Exec dimensions, three Proxy+,
  plus the OOD Sandbox dimension). The eligible pool (20 families, 5,626 tasks) and the
  cross-check findings re-derived independently.
- **User decisions:** `bcb_completion` and `mbpp_completion` stay eligible; DebugBench stays
  ineligible; the S1.7 suspected list gains `hex_cpp`, `hex_go`, `hex_java`, `hex_js`,
  `multiple_rs`, `multiple_ts`, `swebench` and CRUXEval's 0.8 value. `canitedit` is resolved
  (ineligible), so `PENDING_USER_DECISION` is empty.
- **S1.5** is in progress on `s1-b-prompts`: six sources pinned through per-field extracts (193
  verified files); the wrappers are rebuilt and match all 3,297 `tasks.jsonl` prompts; the
  adapters, join report and Outcome are not done.
- Fixed the stale `DEFERRED_COLUMNS` comment in `router/data/schema.py`.
- **Next:** finish S1.5, then S1.6.

### 2026-11-20 (S1.5 merged)

- **S1.5** merged from `s1-b-prompts` (`0474330`), including addendum A. Reviewer check: 326
  tests pass on the branch, 327 on `main`; the join report reproduces its JSON; the new source
  extracts hold only task-text fields; the content field contains no builder instruction text and
  CRUXEval content carries its call; the eligible join (5,555 / 5,626) and the 830 cross-family
  shared-content tasks re-derived independently. `fetch_data.py` run once in the main checkout
  (6 new extracts; 193 verified).
- **User decisions:** the router field is `content_text` (raw task text plus task payload, no
  wrapper), refining the earlier raw-text decision, which dropped CRUXEval's call and
  LiveCodeBench's starter code; no task-kind field; identical content is merged for splits only;
  the gate's 95% is measured on the eligible pool. Written into S1.5, S1.6, the S1 gate, S2.5 and
  S4.4. The "888 `lcb`" figure is corrected (776 `lcb` plus 112 `lcb_testgen`).
- **Next:** S1.6.

### 2026-11-20 (S1.6 merged)

- **S1.6** merged from `s1-b-splits` (`8e0c4cf`). It was done in two checkpoints, with a reviewer
  check between them.
  - **Checkpoint 1, problem ids:** an independent union-find over source keys, content hashes
    and DS-1000 origins (resolved within the library) gives the same partition: 6,997
    problems, 3,487 in the eligible pool.
  - **Checkpoint 2, split and folds:** an independent re-draw from the documented calls
    reproduces the development split and all five seeds' folds. A deliberately injected
    content or source-key leak is caught, once per seed. The development split is identical
    for every seed.
  - **Tests:** 383 pass on the branch, 384 on `main`. `fetch_data.py` was run once in the main
    checkout (one new extract; 194 verified).
- **User decisions:**
  - DS-1000 variants are linked to their original;
  - the fold assignments for seeds 0–4 are committed alongside the development split;
  - no stratification;
  - eligibility stays read from S1.4 and is not frozen in `splits.py`; the pin test guards it.
- **Correction:** the provisional regex already linked `multiple_*` to HumanEval; only
  HumanEval-X was missed. The 2026-11-19 entry above overstated this.
- **Noted, not acted on:**
  - 45 `lcb_testgen` ids have no `lcb` partner (`outside_pool`);
  - 21 `debugbench_sup` numbers also occur in debugbench, but nothing ties them, so they stay
    unlinked;
  - the DS-1000 release has 452 Origin rows, where the paper reports 451.
  - No new S1.7 suspects.
- **Next:** batch C (S1.7 audit and `artefacts.yaml`, then the S1 gate).

### 2026-11-21 (S1.7 checkpoints 1 and 2a reviewed; amendment A2)

- **S1.7 in progress** on `s1-c-audit`. The labels are interim; final labels come after 2c.
  - **Checkpoint 1:** the criteria were committed before any table or response; the coverage
    counts and model mapping were re-derived.
  - **Checkpoint 2a (Python):** the reviewer re-ran 36 sandbox cells, and all matched.
  - **Release-side grading faults** reproduced from the released scores:
    - BigCodeBench: tasks importing matplotlib, sklearn, seaborn, scipy or wordcloud pass
      0.18%, against 41.3% for the rest;
    - LiveCodeBench: LeetCode tasks are never called as functions (9.2% vs 50.9% AtCoder;
      0% for five of the eight models);
    - DS-1000: answers are executed without stripping fences or `</code>` (0.1% pass with
      `</code>`, 34.1% bare).
  - MBPP is cleared. 6,232 stored responses are capped at 2,000 characters.
- **User decisions:**
  - `lcb` excludes its LeetCode subset (422 tasks; the kept part gets its own sample);
  - BigCodeBench is labelled at family level (the library subset is only a sensitivity check);
  - capped responses stay in, and a capped share with a Wilson lower bound ≥ 5% blocks
    "cleared";
  - unexplained cells are investigated in every family;
  - empty `response_matrix.json` cells keep their released scores;
  - glm-5, MiniMax-M2.7 and qwen3.5-plus are mapped by name (unconfirmed);
  - S2.4 stream ids run 2–99.
- **Amendment A2 (`CLAIMS.md`):** CRUXEval success is a released score ≥ 0.8.
  - All 128 sampled 0.8 cells hold the correct output with extra text. CRUXEval's documented
    metric is `assert f(input) == output`.
  - 0.8 is 72% of Sonnet's CRUXEval cells and 12% of gpt-5.4's.
  - Code: `CRUXEVAL_SUCCESS_SCORE`, and `scoring.success(score, family, strict=False)`.
  - The P0 records keep the strict rule (a test pins this).
  - Tests: 390 pass on `main`.
- **Next:** S1.7 checkpoint 2b (C++, Go, Java, JavaScript, Rust, TypeScript), after the agent
  commits the capped-response amendment.

### 2026-11-21 (S1.7 checkpoint 2b reviewed; amendment A3)

- **Checkpoint 2b** (C++, Go, Java, JavaScript, Rust and TypeScript families) on `s1-c-audit`:
  - The amendments, harness, images and reference-solution gate were each committed before any
    2b response was loaded.
  - The reviewer re-ran 12 sandbox cells, and all matched.
  - Interim labels: all six families are confirmed defects, mechanism `wrong_field`.
    - HumanEval-X: the released values are fractions, not pass/fail; for example 0.74 on an
      answer that passes every test.
    - MultiPL-E: every one of 2,328 cells is 0, while our sandbox passes 87–95%.
- **Amendment A3 (`CLAIMS.md`):** A2 is reversed for the headline, so CRUXEval success is again
  1.0.
  - A2 rested on a substring check. The expected value also appears in 17% of 0-scored cells, and
    a fixed rule confirms a stated value in only 39 of 97 cells scored 0.8.
  - CRUXEval ≥ 0.8 is kept as a sensitivity check. Code:
    `scoring.success(score, family, cruxeval_lenient=False)`. 390 tests pass.
- **User decisions:**
  - MultiPL-E is a confirmed defect (`wrong_field`; not "undecided");
  - the whole-text CRUXEval fallback is not adopted (sensitivity only);
  - SWE-bench gets one attempt within the half-day limit, otherwise undecided.
- **Headline so far:** 3,466 of 5,626 eligible tasks are interim defects: BigCodeBench 1,097,
  DS-1000 1,000, LeetCode 422, HumanEval-X C++/Go/Java/JS 656, MultiPL-E 291.
- **Next:** finish 2a (the kept `lcb` sample on stream 140; unexplained cells in every family),
  then 2c (SWE-bench).

### 2026-11-21 (S1.7 checkpoint 2 complete)

- **2a remainder, decision D and 2c** reviewed on `s1-c-audit` (`5e9acb4`; 414 tests).
  - **Kept `lcb` subset** (AtCoder and Codeforces, stream 140): agreement 94/96, but 37/97
    responses are capped, so it is undecided under the capped-response rule.
  - **Decision D:** 32 of the 40 unexplained main-sample cells were reassigned with evidence.
    The reviewer read the two that decided labels (`hex_python` Python/50, CRUXEval
    sample_505) and both were our-side extraction errors.
  - **SWE-bench Lite:** 89/89 agree, but only 4 cells executed (85 responses contain no diff).
    53/97 are capped, so it is undecided.
  - **Mapping test:** no unconfirmed model fails (gaps −0.008 to +0.007).
- **Labels (user, 2026-11-21; final labels are written in checkpoint 3):**
  - **confirmed defect:** BigCodeBench ×5, DS-1000, `lcb` (LeetCode subset excluded by rule),
    HumanEval-X C++/Go/Java/JS, MultiPL-E Rust/TypeScript;
  - **cleared:** `mbpp`, `hex_python`, `cruxeval` (strict 1.0);
  - **undecided:** `humaneval`, `mbpp_completion`, the kept `lcb` subset, `swebench`;
  - `he_completion` was never suspected.
- **Views, provisionally:**
  - artefact-filtered headline: 1,287 eligible tasks, about 62% CRUXEval;
  - sensitivity: 2,160;
  - out: 3,466.

  S3 measures the power of the gates on this smaller headline.
- **Next:** S1.7 checkpoint 3 (labels, merge `main`, `artefacts.yaml`, view builder, then the gain
  shares), then the S1 gate.

### 2026-11-21 (S1.7 merged; S1 gate passed)

- **S1.7** merged from `s1-c-audit` (`b5d5bc0`).
  - **Checkpoint 3:** final labels (`a865dfb`), then the merge of `main`, then
    `artefacts.yaml` and the view builder (`471ddb5`), then the gain shares (`bdf833f`).
    `c72db93` adds only verifier-signal counts; the view task-id hashes are unchanged.
  - **Reviewer recount:** labels 3,466 / 873 / 1,287; filtered view 1,287 tasks, 1,188
    problems and 248 development tasks; sensitivity 2,160; Opus strict passes 901.
  - Tests: 428 pass on `main`, and 206 data files verify.
- **S1 gate passed;** each item is checked under the gate in §4 S1.
- **CLAIMS:** the provisional-numbers table gains the S1.7 frozen values next to the P0.5 ones.
  No rule changed.
- **What S1.7 implies** (the formal verdicts come at the end of S3):
  - DP1a will almost certainly fail: in-sample DimensionBest beats Opus by 1 task in 1,264. C4
    then leads the accuracy story (O18).
  - The oracle's +14.3-point headroom on the headline is 91% CRUXEval (+3.3 under the 0.8
    sensitivity).
  - 90.8% of headline tasks have no visible-test signal.
  - A DP1b failure on CodeRouterBench is therefore likely.
- **User decision:** start S1.8 (the LLMRouterBench loader) now as a hedge, alongside S2.
- **Next:** the S1.8 brief, then S2.
