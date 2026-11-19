# PROBE-CASCADE — Implementation Plan (living document)

> **Read this first in every new session.** It is the single source of truth for what is being
> built, what is done, and what comes next. Update the status block, the checkboxes and the
> progress log at the end of each session (see §9, Session protocol).

> **Plan v2 (2026-11-19).** This version replaces the goal framing and milestones M1–M8 of plan v1
> after an independent review whose numbers were rerun against the data (progress log,
> 2026-11-19). Plan v1 is at commit `b8edb5f` on branch `worktree-progress-report-8`. The M1–M3
> code and numbers from 2026-11-10 were quick checks written for Report 8, not validated
> infrastructure. M1 and M2 are rebuilt here as engineering milestones.

## Status

| Field | Value |
|---|---|
| Current milestone | **M1 — Data layer (rebuild under plan v2)** |
| Next action | M1.1: replace the regex `source_of` with an explicit task-to-family table (zero singleton families). Then M1.6: the audit module and the frozen artefact list |
| Overall progress | Design v2 100% · Quick checks for M1–M3 done (Report 8) · Validated v2 milestones: none yet · Write-up: Report 8 |
| Last updated | 2026-11-19 |
| Final deadline | December 2026 (exact submission and defence dates: **TBD — user to fill in**) |

---

## 1. Goal

Build a router for code-generation tasks that:

1. **reaches the best single model's acceptance rate at clearly lower cost.** The metric is cost
   per accepted task (CPAT), always with an acceptance floor (R3);
2. **brings in a newly added LLM from a small number of measured outcomes, without retraining;**
3. **is usable** as a CLI and static report, with an OpenAI-compatible proxy as an optional extra.
   Web development (the 65 Docker-tested tasks) is the showcase domain.

**Research questions (v2).**

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
  predicted-pass threshold rarely binds. The real cost lever is the 39.3% of test tasks that no
  model solves.

### Contributions (ranked) and what the thesis says if each fails

| # | Contribution | Minimal evidence | If it fails |
|---|---|---|---|
| C1 | **Probes-to-decision.** Onboarding measured in routing quality, not ability error. A pooled hierarchical prior plus in-stream verified outcomes, with a per-group stopping rule, against a lookup table built from the same k | E1: four onboarding arms against k-matched DimensionBest on both corpora, with CIs | Report the number of shared probes a lookup table needs (currently about 250 to reach 50.7) as the measured answer to RQ-A |
| C2 | **Stop or escalate.** Predict tasks no model solves and cut the ladder early | E2: cost at matched acceptance rate on exec-scored tasks, against cost-aware DimensionBest and the fixed cascade | Report how far a realistic predictor falls short of the perfect-truncation bound ($6.0 vs $41.0, Always-Opus $34.0) |
| C3 | **Measured verifier strength.** False-pass rate of a verifier that sees a fraction f of each web-dev task's tests, and its effect on cascade cost | E4: visible vs held-out tests, 3 runs per task | Report that test-suite verifiers are effectively exact on this benchmark; E3's simulated noise becomes a bound |
| C4 | **Headroom audit (guaranteed floor).** How much published routing gain survives artefact removal and run noise | M1.6 audit plus the three data views; noise-corrected oracle on web-dev | If headroom survives, CodeRouterBench is cleared as primary data, which strengthens C1 and C2 |

**Novelty status.** UniRoute (2502.08773) and ICL-Router (2510.09719) publish quality against
onboarding-set size with fixed offline sets and static accuracy; IRT-Router (2506.01048) onboards
from text only; ACRouter (2606.22902) uses a fixed pool. No paper found in the M0.4 check measures
outcomes until the routing decision stabilises, or compares dedicated probes with in-stream
learning under a verifier. **Not yet checked:** the full text of ORACLE (2607.22465) for C3, and of
"The Routing Plateau" (2606.07587) and "Most of the LLM Routing Gap Is Task Type" for C4. Read
them before claiming C3 or C4 (M8.3).

### Decision points

| DP | Where | Test | Consequence |
|---|---|---|---|
| DP1 | End of M1 | Headroom of the oracle and of DimensionBest over the best single model, on the artefact-filtered, exec-scored binary view | Below about 3 AvgPerf points: C4 becomes the lead claim and LLMRouterBench becomes the primary corpus for C1 |
| DP2 | End of M3 | Does the hierarchical model beat k-matched DimensionBest on AvgPerf at any k ≤ 100? | No: the routing model is the shrunk lookup table, and IRT is reported as an ablation |
| DP3 | M6 pilot | Is routing headroom on the free-tier pool larger than run-to-run noise? | No: E4 is reported as a verifier study and a transfer test, not as a routing win |

### Success criteria

| # | Criterion | Measured by |
|---|---|---|
| S1 | E1 onboarding curves in routing units (AvgPerf, cost) for four arms against k-matched DimensionBest, on both corpora, with CIs | M4 |
| S2 | Cascade with truncation against cost-aware DimensionBest and the fixed cascade at matched acceptance rate, exec-scored tasks, paired CIs; or an honest negative | M5 |
| S3 | Verifier noise results: simulated one-sided noise (E3) and the measured false-pass curve on web-dev (E4) | M5, M6 |
| S4 | Headroom audit in three data views | M1, M2 |
| S5 | At least 3 free-tier models × 65 tasks × 3 runs, stored with per-test results; transfer test | M6 |
| S6 | CLI and static-report demo; proxy optional | M7 |
| S7 | Thesis and defence | M8 |

---

## 2. The method, briefly (v2)

1. **Ability model.** P(model m passes task q) = σ(μ_g + δ_m + γ_{m,g} − b_q), where g is the
   task's group (dimension or family), μ_g is the pool's group mean, δ_m a model shift, and
   γ_{m,g} a model-by-group offset shrunk toward 0 with variances estimated from the known
   models. The plain 1PL is the special case with μ_g and γ removed.
2. **New model.** It starts at δ_m = 0, γ = 0, i.e. the pool prior. This is the k = 0 baseline,
   never θ = 0 (R11).
3. **Evidence.** Either k dedicated probes (random or stratified by group and difficulty band) or
   verified outcomes from the live stream, with a per-group stopping rule on the posterior.
4. **Difficulty.** b_q is predicted from the prompt embedding. The source-benchmark label is not
   available at decision time, so it is not a feature; the group may be predicted from the prompt.
5. **Decide.** Cheap-first, using each step's break-even ratio (cost of the cheaper model over the
   next one). Stop the ladder when the predicted P(any model passes) falls below t.
6. **Verify and escalate.** Run-time signals only (tests, sandbox), on exec-scored tasks only.
   Each verified outcome updates the model's posterior.

Deferred until there is evidence they help: Thompson sampling (only if the pass threshold binds)
and the kNN memory residual (only once prompts cover ≥ 95% of tasks). There is no LLM in the
routing decision.

---

## 3. Architecture and repo layout

The new code lives in `router/` at the repo root. It reuses `pipeline/components/` where
possible. Items marked *(quick check)* exist from 2026-11-10 and must be rebuilt or validated in
their milestone before any result from them is cited.

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
    prompts.py           # task -> prompt join for all 11 families  (quick check, 48%)
    scoring.py           # score type per family from the source benchmark's rule  (quick check)
    audit.py             # per family x model anomaly table; writes the frozen artefact list
    artefacts.yaml       # frozen artefact list, committed before any router result
  core/
    irt.py               # 1PL/2PL fit and damped ability solver  (quick check, keep)
    ability.py           # hierarchical ability model with shrinkage
    difficulty.py        # prompt embedding -> b_q, no source label
    onboarding.py        # four onboarding arms
    stopping.py          # per-group posterior stopping rule
    policy.py            # cheap-first by break-even ratio, ladder truncation, budget cap
    anchors.py           # random / stratified / Fisher selection  (quick check)
  baselines/             # always-X, random, DimensionBest (full, k-matched, cost-aware), SourceBest,
                         # fixed cascade, released ACRouter decisions, LinUCB, kNN on prompt embeddings
  sim/
    stream.py            # seeded replay, model arrivals, bandit feedback only  (quick check)
    metrics.py           # AvgPerf, cost, accept rate, CPAT@floor, cost at matched accept, regret incl. probe cost
    stats.py             # paired bootstrap over tasks, cluster bootstrap over problems
    verifier.py          # uniform, one-sided and difficulty-linked false passes  (quick check, extend)
  live/
    registry.py          # free-tier model registry (SQLite, keys from env)
    gateway.py           # thin wrapper over pipeline/components/llm_gateway.py
    sandbox.py           # wraps TestRunner; records per-test results
    testsplit.py         # visible vs held-out test split per web-dev task
  experiments/           # e0_audit.py, e1_onboarding.py, e2_cost.py, e3_verifier.py, e4_webdev.py
  report/                # static HTML report generator
  cli.py                 # router audit | replay | probe | run | report  (proxy optional, M7)
tests/router/            # pytest, mirrors router/
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

## 4. Milestones

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in the log).
Every milestone has an **exit gate**. Do not start the next milestone's core work until the gate
passes, or until a waiver is written in the log with its reason.

### M0 — Setup and novelty check
- [x] Done 2026-10-05: Docker verified, `router/` skeleton, worktree branches merged, novelty
      check (contribution reframed), reference ids checked. See the log.
- [ ] M0.6 Repo-audit backlog: see §4a.

### M1 — Data layer (engineering)
- [ ] M1.1 `data/sources.py`: an explicit task-id-to-family table covering every id in the
      probing, ID-test and OOD splits. It replaces `coderouterbench.source_of`, which yields 600
      "families" in the probing split, 572 of them with a single task (`multiple_*`,
      `lcb_testgen_*`, `bcb_BigCodeBench/*` ids). Unit test: every task maps to one of a fixed
      list of about 20 families, with no singleton families.
- [ ] M1.2 Extend `data/schema.py`: `OutcomeMatrix` gains `family`, `score_type`, `run` and
      `eligible` (binary, exec-scored, verifiable at run time).
- [ ] M1.3 `data/coderouterbench.py`: two views, complete and paper (missing = 0). Tests: 56,640
      probing cells (7,080 × 8), 23,352 ID-test cells (2,919 × 8), 881 missing paper-view test
      cells, no NaN scores.
- [ ] M1.4 `data/scoring.py`: score type per family from the source benchmark's documented rule
      (exec, similarity or judge), not from the observed score distribution. On the test split, 41%
      of cells are continuous and 67% of tasks sit in dimensions that contain continuous scores;
      record which families are cascade-eligible.
- [ ] M1.5 `data/prompts.py`: adapters for the six missing families (CodeXGLUE, Bugs2Fix,
      DebugBench, LiveCodeBench, HumanEval-X, CanItEdit). Report the join rate per family. Tag
      variants that share one prompt (test generation, completion and generation on the same
      HumanEval problem) with a common problem id.
- [ ] M1.6 `data/audit.py` and `experiments/e0_audit.py`: a family × model table flagging
      all-zero families, flagship models below the pool median, and the share of oracle and
      DimensionBest gain per family. Write `data/artefacts.yaml` and commit it **before** any
      router result is computed (R10). Known candidates from 2026-11-19: `mbpp_testgen` (all
      models 0.00, n=31), `he_testgen` (Sonnet and GPT-5.4 0.00, n=39), `lcb_testgen` (Opus 0.47
      vs Qwen3-Max 0.96, n=236), `lcb` (Opus 0.27 vs qwen3.5-plus 0.57, n=279).
- [ ] M1.7 `data/llmrouterbench.py`: code subsets; drop zero-cost models from economic metrics.
- [ ] M1.8 `data/phase5.py`: read-only loader (R5).
- [ ] M1.9 `data/prices.py` as the single source of prices, with source URL and access date.
- **Gate:** the M1.3 count tests pass; zero singleton families; prompt join ≥ 95%, or the gap is
  documented per family and every embedding result is restricted to joined tasks; both matrix
  views load; `artefacts.yaml` committed; the three data views (all tasks, exec-scored binary,
  artefact-filtered) can be built from `OutcomeMatrix`.
- **DP1** (see §1).

### M2 — Replay harness and baselines (engineering)
- [ ] M2.1 `sim/stream.py`: seeded task order, model-arrival events, bandit feedback only (the
      router sees only the outcome of the model it called).
- [ ] M2.2 `sim/metrics.py`: AvgPerf, cost, acceptance rate, CPAT@floor, cost at matched
      acceptance rate, and regret that includes probe cost in dollars and in probe count.
- [ ] M2.3 `sim/stats.py`: paired bootstrap over tasks; cluster bootstrap over problems (variants
      of one problem resampled together); leave-one-model-out results shown as per-model points,
      never pooled into one CI over models.
- [ ] M2.4 Baselines: Always-X, Random, DimensionBest, SourceBest (on the M1.1 families),
      cost-aware DimensionBest (argmax of score − λ·cost), k-matched DimensionBest (table built
      from the same k probes), fixed cascade, released ACRouter decisions, LinUCB with stated
      features, kNN on prompt embeddings (joined tasks only). The metadata "kNN" is dropped: with
      one-hot features it returns random tasks from the same group.
- [ ] M2.5 Tests: oracle 57.00, Always-Opus 43.83 and released ACRouter 50.14 reproduced on the
      paper view; a shuffled-label test in which each learned baseline falls to Random.
- **Gate:** M2.5 passes; every comparison table carries a paired CI; baselines that would need
  the paper's unavailable code are labelled "our definition", with the definition difference
  documented instead of a ±1.0 requirement. Reference points from 2026-11-19: DimensionBest
  vs ACRouter is +1.54 on the complete view and −0.05 [−0.31, 0.23] on the paper view.

### M3 — Outcome model
- [ ] M3.1 `core/ability.py`: the hierarchical model of §2; `core/irt.py` kept as the 1PL case.
- [ ] M3.2 `core/difficulty.py`: prompt-embedding predictor without the source label; optional
      group classifier from the prompt; CV grouped by problem.
- [ ] M3.3 Evaluation under leave-one-model-out (not random-cell holdout): held-out
      log-likelihood; k = 0 scored at the pool prior; rankings reported per single draw.
- [ ] M3.4 Reference curve: k-matched DimensionBest (2026-11-19: 44.4 ± 2.6 at k = 25,
      47.9 ± 2.1 at 50, 49.3 ± 0.9 at 100, 50.7 ± 0.6 at 250, 51.4 ± 0.3 at 1000; full table 51.7).
- **Gate:** the hierarchical model beats per-dimension 1PL on leave-one-model-out
  log-likelihood (recompute the per-dimension 1PL figure under the same split; −0.307 was
  random-cell); single-draw results reported.
- **DP2** (see §1).

### M4 — Onboarding policies (C1)
- [ ] M4.1 `core/onboarding.py`: four arms — zero-probe pool prior, k random probes, k
      stratified probes, in-stream learning with stopping.
- [ ] M4.2 `core/stopping.py`: per-group posterior stopping rule.
- [ ] M4.3 Non-transductive evaluation: test-item difficulty is never fitted from the other
      models' outcomes on those same items (fixes `router/experiments/m3_irt.py:67-74`).
- [ ] M4.4 `experiments/e1_onboarding.py` on CodeRouterBench and LLMRouterBench; 5 seeds; report
      the distribution of probes the stopping rule uses.
- [ ] M4.5 Fisher anchors: run once on LLMRouterBench if its item bank has enough respondents;
      otherwise report them as untestable here.
- **Gate:** E1 curves in routing units against k-matched DimensionBest, with CIs, on both corpora.

### M5 — Cascade with stop or escalate (C2)
- [ ] M5.1 `core/policy.py`: cheap-first by break-even ratio, ladder truncation from predicted
      P(any model passes), budget cap.
- [ ] M5.2 `sim/verifier.py`: uniform, one-sided (false passes only) and difficulty-linked noise.
- [ ] M5.3 `experiments/e2_cost.py`: exec-scored binary tasks only; against cost-aware
      DimensionBest, fixed cascade, always-best and released ACRouter; cost at matched
      acceptance rate.
- [ ] M5.4 `experiments/e3_verifier.py`: ε ∈ {0, 0.1, 0.2, 0.3} for each noise type.
- [ ] M5.5 Optional: Thompson sampling and the kNN memory residual, only under the conditions
      in §2.
- **Gate:** paired CIs on every comparison; an explicit answer to whether truncation recovers
  the cost of unsolvable tasks (reference: full cheapest-first ladder $41.0, Always-Opus $34.0,
  perfect truncation $6.0).

### M6 — Live web-dev (C3 and the transfer test)
- [ ] M6.1 The §4a "before M6" fixes (sandbox limits, `<think>` stripping, true `tests_total`,
      `failure_mode`, per-attempt temp files, TestRunner fixes).
- [ ] M6.2 Store per-test results for every new run, and `live/testsplit.py`: a fixed visible vs
      held-out split per task. Tasks average about 7.5 tests, so record the achievable
      fractions f per task.
- [ ] M6.3 `live/registry.py`, `live/gateway.py`: free-tier pool (Groq, OpenRouter `:free`,
      Gemini free). Record which models are siblings.
- [ ] M6.4 Pilot on about 20 tasks, then **DP3**.
- [ ] M6.5 E4: at least 3 models × 65 tasks × 3 runs, stored in new tables or files (R5).
- [ ] M6.6 Analysis: false-pass rate against f by category; cascade cost and accept rate read
      from that curve; noise-corrected oracle (choose on run 1, score on run 2); transfer test
      of benchmark-derived ability to web-dev ranking.
- **Gate:** the E4 data is stored with per-test results; the false-pass curve and the
  noise-corrected oracle are reported.

### M7 — Demo
- [ ] M7.1 `cli.py`: `router audit | replay | probe | run | report`.
- [ ] M7.2 `report/`: static HTML report (onboarding curves, cost and accept rate, decision log,
      verifier curve).
- [ ] M7.3 Optional, only after the M4–M6 gates pass: FastAPI `/v1/chat/completions` proxy with
      `x-router-model` and `x-router-reason` headers.
- **Gate:** the CLI and report run end to end from a clean start.

### M8 — Thesis, report and defence
- [ ] M8.1 Chapters: problem, Phase 1–6 recap, literature, method, data audit, experiments, demo,
      limitations.
- [ ] M8.2 Report every simple baseline, every negative result and all three data views (R2).
- [ ] M8.3 Novelty re-sweep: full text of ORACLE (2607.22465), "The Routing Plateau"
      (2606.07587) and "Most of the LLM Routing Gap Is Task Type"; fix the seven reference
      titles listed in the 2026-10-05 log.
- [ ] M8.4 Slides and a recorded demo as a backup for the defence.


### 4a. Repo-audit backlog (from the 2026-10-05 pre-implementation audit)

These items were found while auditing the reused components. Each is a **prerequisite of the
milestone named**, so do it at the start of that milestone, not earlier.

**Before M6 (live layer):**
- [ ] Find out why every Phase 5 model output for API-013 and CSS-012 collected 0 tests
      (`extraction_success=True`, `tests_total=0`), when the reference solutions pass 12/12 and 8/8.
      The `defective_task` label for these two is unsupported. Recheck whether `drop_artifactual`
      should still drop them, and whether the RQ1 numbers change.
- [ ] Derive and persist `failure_mode`, `runner_error`, `llm_error` and `extraction_method` when
      writing runs (`results_storage.py:213-221`, `run_eval.py` failure builders). Plan step M6.5
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

**Before M7 (proxy):**
- [ ] Add `LLMGateway.query_messages()` for chat message lists, make the retry backoff
      configurable, and narrow the `"rate"` substring check (`llm_gateway.py:73,123,127`).

**During M1:**
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

Dates are not planned in v2: on 2026-11-17 the user asked to set the plan without a timeline.
Submission and defence dates are still TBD. If time runs short, cut in this order (last item is
cut first): M1 > M2 > C4 audit > M3 and M4 (C1) > M5 (C2) > M6 (C3) > M7 report > M7 proxy. A
decision point may reorder this; record any reordering in the log.

---

## 6. Rules (do not break)

- **R1 — No clairvoyant escalation.** During replay, never escalate on the benchmark's ground
  truth; use the simulated run-time verifier only. In live runs, the verifier's tests and the
  grader's tests must be disjoint.
- **R2 — Always report the simple baselines:** DimensionBest (full and k-matched), cost-aware
  DimensionBest, fixed cascade and always-best. A result without them is not a result.
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

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| CodeRouterBench headroom is mostly harness artefacts | DP1; C4 becomes the lead claim; LLMRouterBench as primary corpus for C1 |
| C1 is incremental over UniRoute / ICL-Router | Measure in routing units against k-matched DimensionBest; in-stream learning with stopping is the differentiator; re-sweep at M8.3 |
| ORACLE (2607.22465) or the task-type papers already cover C3 or C4 | Read their full text before claiming (M8.3); narrow the claim if needed |
| The hierarchical model does not beat the lookup table | DP2; report the lookup table as the routing model and IRT as an ablation |
| The prompt join stays below 95% | Restrict embedding results to joined tasks and state the coverage |
| About 7.5 tests per web-dev task makes the visible-test split coarse | Report the achievable f per task; pool by category |
| The free-tier pool is nested again (Phase 6 found +8.6%) | DP3 pilot before the full E4 run |
| Free-tier rate limits slow E4 | Batch runs overnight, cache every response, drop to 2 runs if needed |
| Docker breaks again | Subprocess sandbox fallback, flagged as less isolated |

---

## 8. Key references

| Paper | Where it's used |
|---|---|
| Agent-as-a-Router / ACRouter (2606.22902) | the base system, data and baselines |
| UniRoute (2502.08773), EmbedLLM (2410.02223), RouteProfile (2605.00180) | new-model representation (M3, E1) |
| IRT-Router (2506.01048), tinyBenchmarks (2402.14992) | IRT and anchors (M3) |
| Dekoninck et al. (2410.10347) | cascade optimality (M5) |
| ORACLE (2607.22465) | verifier-feedback routing; related work for C3 (full text not yet read) |
| PILOT (2508.21141), BaRP (2510.07429), LLM Bandit (2502.02743) | exploration and online learning (M4) |
| LLMRouterBench (2601.07206), kNN baseline (2505.12601), Routing Plateau (2606.07587), 2608.23023 | evaluation honesty (M2, M5) |

The full list is in `Reports/Research_Progress_Report_7.md` §10.

---

## 9. Session protocol (for every agent or session)

1. **At the start:** read §Status, §6 Rules, and the current milestone's section. Check the
   progress log for open issues.
2. **During work:** tick the checkboxes as you go (`[~]` when started, `[x]` when the gate or task is verified).
3. **At the end of the session:**
   - update the Status table: current milestone, next action, progress, date;
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
