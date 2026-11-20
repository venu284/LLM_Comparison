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
| Current stage | **Pre-S0 — requirements before S0** (§4) |
| Next action | P0.4: write `docs/CLAIMS.md` (S0.1–S0.14), the constants in `router/settings.py`, and port the power scripts. P0.1–P0.3, P0.5 and P0.6 done |
| Overall progress | Design v3 100% · Quick checks for M1–M3 done (Report 8) · Validated stages: none yet · Write-up: Report 8 |
| Last updated | 2026-11-19 |
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
  model solves.

**Two separate gates.** Finishing the research and reaching the goal are judged separately:
- the **research gate** (S0.4) passes when every decision point and contribution has a verdict
  with quoted evidence. A negative verdict passes it;
- the **performance gate** (S0.3) passes only when goal 1 is met on the cross-fitted evaluation scores (S0.6),
  including S0.3(c): the router must beat the best lookup table, not only the best single model.

### Contributions (ranked) and what the thesis says if each fails

| # | Contribution | Minimal evidence | If it fails |
|---|---|---|---|
| C1 | **Probes-to-decision for a new model, from verified outcomes in one pass.** How many outcomes a new model needs before routing to it matches a lookup table built from the same k outcomes (k-matched DimensionBest, same label source), and whether a pooled hierarchical prior, in-stream outcomes from a partial verifier (PASS/FAIL/UNKNOWN) and a per-group stopping rule shorten that. Scored on a single pass over the stream with no repeated problem (S0.5). Quality against onboarding-set size is already published (UniRoute Fig. 2); the in-stream, stopping-rule and same-label-source comparison is not | E1: four onboarding arms against k-matched DimensionBest on both corpora, with CIs | Report the number of shared probes a lookup table needs (currently about 250 to reach 50.7) as the measured answer to RQ-A |
| C2 | **Give up or escalate, per task, under partial verification.** A calibrated per-task classifier of P(a later rung passes \| earlier rungs failed), with a give-up action for tasks predicted to be unsolvable, under the real visible-test verifier, where about four-fifths of headline tasks (80.0% of the 1,907 complete-case tasks, provisional, P0.5) give no signal (S0.11). The stop-or-escalate rule on post-failure histories (Route Anything Prop. 8) and cell-level truncation under an oracle verifier (Route Anything X10) are prior work that C2 uses | E2: cost at matched acceptance on exec-scored tasks, against cost-aware DimensionBest, the fixed cascade and Route Anything's 21-cell zero-loss direct-dispatch route (R2) | Report how far a realistic predictor falls short of the perfect-truncation bound ($6.0 vs $41.0, Always-Opus $34.0) |
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
| DP1a | End of S3 | **Accuracy:** headroom of the oracle and of cross-fitted DimensionBest over the best single model, on the **full** artefact-filtered eligible pool (not the development split: the oracle needs no fitting and a cross-fitted lookup table is not router development, so no leak; the development split alone, about 690 tasks, would leave the test weak). Passes if the lower end of the 95% cluster-bootstrap CI is ≥ 3.0 AvgPerf points | No: C4 leads the accuracy story |
| DP1b | End of S3 (S3.2) | **Cost (feasibility upper bound):** does the **oracle-on-no-signal bound** meet S0.3 on point estimates on the headline set, with verifier cost included: saving ≥ 20% and acceptance gap ≥ −2 points against the best single model, **and** saving > 0 and gap ≥ −2 points against CellBest with predicted cells (S0.3(c))? In the bound, cascades use the real verifier where visible tests exist and an oracle accept-or-escalate decision on cells with no verifier signal. Fixed ladders cannot pass by construction (S3.2), so they are not the test. Also report the gap the learned router must close (provisional: about 11 points against the best single model, P0.5; against CellBest with predicted cells (P0.6, seed 0): the bound meets (c) on point estimates (per-cell Rule A: 36.6% saving, +1.6 points; Rule B: 16.4%, +8.1). The best policy without an oracle (per-cell Rule A, no-signal cells escalated) reaches 3.3%, +0.3. The learned no-signal decision must recover the difference, about 33 points of saving at similar acceptance) | No: the bound misses the gate, so stop and bring it to the user before S4 (LLMRouterBench becomes a candidate primary corpus for C1 and C2). A failed DP1b means the tested policies have not shown economic feasibility; it does not prove a learned policy cannot succeed, so the pause is a reassessment, not a verdict. This half decides the economic claim |
| DP2 | During S5 | Does the hierarchical model beat k-matched DimensionBest **in routing units** (AvgPerf, cost) at a k in the fixed grid {10, 25, 50, 100}, on the development split? | No: the routing model is the shrunk lookup table, and IRT is reported as an ablation |
| DP3 | S6 pilot | Is the routing gap on the live pool (free models plus the paid top model) more than twice the run-to-run standard deviation, measured on the pilot's ≥ 2 runs per task? | No: live work is reported as a verifier study and a transfer test, not as a routing win |

### Success criteria

| # | Criterion | Measured by |
|---|---|---|
| SC1 | E1 onboarding curves in routing units (AvgPerf, cost) for the onboarding arms against k-matched DimensionBest, with CIs | S5, S7 |
| SC2 | Cascade with stop-or-escalate against cost-aware DimensionBest and the fixed cascade at matched acceptance rate, exec-scored tasks, paired CIs; or an honest negative | S5, S7 |
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
- [ ] **P0.4 Write `docs/CLAIMS.md`** from S0.1–S0.14 below, with the constants in
      `router/settings.py`, and move the power-calculation scripts into
      `router/experiments/e0_power.py` as the evidence for S0.3 and S0.6. The original scripts
      are committed in `router/experiments/e0_power/`; its README lists what the port must change.
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
      +0.3) and is now an R2 baseline. Gap CI half-width against CellBest: 0.3 to 2.3 points
      depending on the policy.

The remaining requirements are placed in the stage that needs them, in priority order:

| Priority | Requirement | Stage |
|---|---|---|
| P1 | Problem ids from source ids, linking `lcb` to `lcb_testgen` through `tasks.jsonl`; split-integrity test | S1.6 |
| P1 | Long-format `OutcomeMatrix` (task, model, run, problem_id, family, score_type, eligible) | S1.2 |
| P1 | Prompt join extended with `tasks.jsonl` (3,297 prompts, including 888 `lcb`) | S1.5 |
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
result can be reinterpreted after it is seen. S0.3, S0.6 and S0.10 were approved by the user on
2026-11-19 on the evidence of the power calculation; the third and fourth reviews of that day
added the population, observation and nesting rules below (progress log).

**All task counts in S0 are provisional.** They come from a prototype that has not passed S1. The
counts that cannot change (release row counts, the X13 coverage of 2,911 tasks) are tested in S1;
every count that depends on eligibility (S0.12) or on the audit (S0.10) is frozen in
`data/artefacts.yaml` only after S1.4 and S1.7 run.

- [ ] **S0.1 Acceptance.** A task is accepted only when the ground truth passes it. **In replay the
      ground truth is the benchmark's released score**: it covers every task and is what the
      baselines and the paper use. For execution-scored families with partial credit (scores
      such as 0.2–0.8 are fractions of tests passed), **success means all tests pass (score =
      1.0)**, the standard pass@1 rule. The X13 hidden-test outcome agrees with the released score
      on 87.7% of cells and has 2,051 harness-mismatch and 2,718 extraction-failure cells, so it is
      reported only as a sensitivity check. In live runs the ground truth is the reserved hidden
      test pool. The router's own verifier decides escalation but never counts as acceptance.
- [ ] **S0.2 Cost.** Cost = model tokens × published price (source URL and access date in
      `router/data/prices.py`) + verifier runs + retries + probe calls, spread over the stream.
      Actual spend is not the metric. Verifier runs are reported as a count and in dollars at a
      stated per-run rate: the X13 measured mean run time (0.074 s per visible-test run) priced at a
      published on-demand cloud vCPU-hour rate, with its source and date in `CLAIMS.md` (about
      $10⁻⁶ per run, about 0.06% of the best single model's per-task cost), swept at ×1, ×10 and
      ×100. The stream length over which probe cost is spread is the number of tasks scored
      (reported with the break-even stream length). A price-sensitivity sweep is run on every
      headline number.
- [ ] **S0.3 Performance gate.** The router passes if, over the nested cross-fitted evaluation
      scores (S0.6):
      - (a) the lower end of the 95% cluster-bootstrap CI of its cost saving against the best
        single model is at least **20%**;
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
      the best model in every training fold. Power: recomputed in S3.7 on the final population;
      the user chose on 2026-11-19 to keep −2 and report the power with the result, even if it is
      about 60–70%. First estimate for (c) (P0.6): gap CI half-width 0.3–1.8 points against
      CellBest; a router that departs from CellBest as much as the bound policies needs a point
      gap of about −0.2 or better.
- [ ] **S0.4 Research gate.** Every decision point and contribution ends in a verdict (holds,
      fails, not reached) with quoted evidence. A negative verdict passes this gate.
- [ ] **S0.5 k and label source.** k is the number of calls made to the new model. Every
      onboarding arm and baseline with the same k, including k-matched DimensionBest, receives
      exactly the same k outcomes **from the same label source**. E1 is reported in two pairings:
      ground-truth labels for all arms, and verifier-only labels (PASS/FAIL/UNKNOWN) for all arms,
      so that a probe-versus-in-stream difference cannot come from label quality alone. In E1 the held-out model's outcomes are absent from
      every fitting and selection step, except the probes and online feedback it is allowed.
      E1 is scored on a **single pass** over the stream with **no repeated problem** (the R13
      problem id, not the task id), so in-stream learning cannot recall its own or a sibling
      task's verdict (ORACLE E.6 and E.8); a test checks this.
- [ ] **S0.6 Development split and nested cross-fitting** (option (a), chosen by the user on
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
- [ ] **S0.7 Iteration rule.** Router changes are made and judged on the development split only.
      The nested evaluation and the live hidden tests are run once, with the frozen candidate
      list and selection rule. Every version tried on the development split is logged and
      reported.
- [ ] **S0.8 Statistics.** Paired bootstrap over tasks and cluster bootstrap over problems, 95%
      CIs; the cluster bootstrap is the one used for the gates. Repeated live outputs of one task
      are resampled together (cluster by task). A seed means the fold assignment plus the stream
      order plus the probe draw; gates use a single pre-registered seed, and the other four seeds
      are reported next to it. Leave-one-model-out results are shown per model, never pooled.
      A point estimate that clears its threshold on the gate seed is reported as **at the
      threshold** when it falls to the threshold or below on another seed, when its CI lower
      bound is at or below the threshold, or when its margin is under 0.1 points (adopted from
      P0.6 on 2026-11-19).
- [ ] **S0.9 Live pool and budget.** The live ladder needs steps in both price and ability, and
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
- [ ] **S0.10 Headline data view.** The headline is the **artefact-filtered** view; the all-tasks
      view is always reported next to it. Rule, fixed before any result: the S1.7 audit decides
      each suspected family on evidence from the raw responses. A family proven defective is
      excluded; a family proven sound is kept; a family the audit cannot decide stays out of the
      headline and appears in the sensitivity view. The headline is therefore a **restricted
      population** (runnable, audited, test-checkable tasks); its results do not establish
      performance across all executable coding tasks.
- [ ] **S0.11 Replay verifier and the headline set for cascade claims.**
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
- [ ] **S0.12 Scope and eligibility.** Router claims cover execution-scored, test-checkable tasks
      only (consistent with R4). **Eligibility is decided by each source benchmark's documented
      scoring rule (S1.4), never by the scores observed** — an unseen model's outcomes must not
      decide which tasks enter its evaluation. Partial-credit execution families are kept, with
      success defined by S0.1; a family whose scoring is not execution-based is excluded as a
      whole. Similarity- and judge-scored tasks are reported separately.
- [ ] **S0.13 Observations, probes and pinned thresholds.**
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
- [ ] **S0.14 Fallback lead claim, fixed in advance.** If P0.1 finds C2 and C3 already covered, C1
      becomes the lead contribution. If C1 is also covered, the lead becomes C4 together with the
      real-verifier economics (cost of a verified cascade under a measured weak verifier).
- **Deliverable:** `docs/CLAIMS.md` with S0.1–S0.14, plus their constants in `router/settings.py`.
- **Gate:** the user signs off on `docs/CLAIMS.md` and it is committed before any S1 code. A later
  change needs a log entry giving the reason, and every result already computed is re-scored.

### S1 — Validate data
- [ ] S1.1 `data/sources.py` [M1.1]: an explicit task-id-to-family table covering every id in the
      probing, ID-test and OOD splits. It replaces `coderouterbench.source_of`, which yields 600
      "families" in the probing split, 572 of them with a single task. Unit test: every task maps to
      one of a fixed list of families, none under 30 tasks (S0.13).
- [ ] S1.2 `data/schema.py` [M1.2]: long-format storage (task, model, run, score, cost, tokens,
      problem_id, family, score_type, eligible), with the wide task × model views derived from it.
      Test that every field survives `subset` and `without`.
- [ ] S1.3 `data/coderouterbench.py` [M1.3]: two views, complete and paper (missing = 0). Tests:
      56,640 probing cells (7,080 × 8), 23,352 ID-test cells (2,919 × 8), 881 missing paper-view
      test cells, no NaN scores.
- [ ] S1.4 `data/scoring.py` [M1.4]: score type per family from the source benchmark's documented
      rule (exec, similarity or judge), not from the observed score distribution, with the citation
      for each rule. Record which families are eligible (S0.12) and which carry partial credit
      (provisional: `lcb` 1,000 tasks, CRUXEval 800, `lcb_testgen` 869, MBPP completion 124 have
      non-binary scores; an observed-score filter kept only 871, 93, 362 and 122 of them).
      Recompute the eligible pool from these rules.
- [ ] S1.5 `data/prompts.py` [M1.5]: add route-anything `tasks.jsonl` (3,297 prompts, including 888
      `lcb`) to the existing join, then adapters for families still missing. Report the join rate
      per family.
- [ ] S1.6 `data/splits.py` (new): `problem_id` taken from the source-benchmark ids in the S1.1
      table (HumanEval, MBPP and BigCodeBench variants linked; `lcb` linked to `lcb_testgen` through
      `tasks.jsonl`), with prompt matching as a cross-check only. Tasks with byte-identical
      prompts are merged into one problem even when their source ids differ (P0.6 found three
      pairs: `ds1000_138`/`180`, `ds1000_137`/`179`, `mbpp_141`/`71`). The development split and the
      five outer folds by problem (S0.6), with recorded seeds, and an automated split-integrity
      test that fails if any problem id appears in two of them (R13). Recount the second review's
      figure of 643 of 1,430 joined test tasks sharing prompt text with the probing split.
- [ ] S1.7 `data/audit.py` and `experiments/e0_audit.py` [M1.6]: a family × model table flagging
      all-zero families, flagship models below the pool median, and the share of oracle and
      DimensionBest gain per family. For each suspected family, inspect the raw responses
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
      differently for the same model (P0.6).
- [ ] S1.8 `data/llmrouterbench.py` [M1.7]: code subsets; drop zero-cost models from economic
      metrics. Validate the loader early, so it is ready if DP1b makes it the primary corpus.
- [ ] S1.9 `data/phase5.py` [M1.8]: read-only loader (R5), including the stored raw responses
      (324 non-empty of 325) read from the database without writing to it.
- [ ] S1.10 `data/prices.py` [M1.9] as the single source of prices, with source URL and access date,
      including reference prices for the live models (S0.2, S0.9).
- [ ] S1.11 `data/verdicts.py` (new): loader for the X13 visible-test and hidden-test verdicts
      (S0.11), joined to CodeRouterBench task ids. Tests cover only counts that cannot change: the
      23,288-row count, 2,911 tasks, and the 2,044/867 split coverage. Headline counts are derived
      after S1.4 and S1.7 and frozen in `artefacts.yaml`. Cells with `w1_n = 0` are marked "no
      signal" (S0.11); `empty` cells are counted separately and marked unavailable (S0.11).
- **Gate:** the S1.3 count tests pass; zero singleton families; the split-integrity test passes;
  eligibility comes from S1.4 rules only; prompt join ≥ 95%, or the gap is documented per family
  and every embedding result is restricted to joined tasks; both matrix views load; the verdict
  loader passes its count tests; `artefacts.yaml` committed with every audit-dependent count; the
  three data views (all eligible tasks, artefact-filtered, headline set) can be built from
  `OutcomeMatrix`.

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
      of S0.6, including automatic per-fold selection from a frozen candidate list.
- [ ] S2.5 Baselines [M2.4]: Always-X, Random, DimensionBest (full, k-matched, cost-aware),
      CellBest with true source cells (reference) and with predicted cells (the S0.3(c)
      comparator), using a cell predictor that infers dimension × source from the prompt (Route
      Anything Table 11) with its accuracy reported, Route Anything's zero-loss direct-dispatch
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
- [ ] S4.4 `core/difficulty.py` [M3.2]: prompt-embedding predictor without the source label;
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
