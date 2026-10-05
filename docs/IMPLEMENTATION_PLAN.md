# PROBE-CASCADE — Implementation Plan (living document)

> **Read this first in every new session.** It is the single source of truth for what is being
> built, what is done, and what comes next. Update the status block, the checkboxes and the
> progress log at the end of each session (see §9, Session protocol).

## Status

| Field | Value |
|---|---|
| Current milestone | **M1 — Data layer** (M0 gate passed 2026-10-05, with the contribution reframed; see §1 and the log) |
| Next action | M1.1: download CodeRouterBench and clone the ACRouter repo into `external/` |
| Overall progress | Design 100% · Repo prep 100% · Implementation 5% · Experiments 0% · Write-up 0% |
| Last updated | 2026-10-05 |
| Final deadline | December 2026 (exact submission and defence dates: **TBD — user to fill in**) |

---

## 1. Goal

Build a router for code-generation tasks that:

1. **reaches the best single model's pass rate at clearly lower cost.** The metric is cost per
   accepted task (CPAT), always with an acceptance floor;
2. **keeps working when a new LLM is added, without retraining,** by onboarding it with a
   small measured probe of k anchor tasks;
3. **is usable day-to-day** as an OpenAI-compatible proxy with a dashboard, with web
   development as the showcase domain.

**The research question:** how many probe tasks does a new model need before the router
routes to it correctly? We measure that cold-start regret curve against k, and compare it with
using a text description and with no probe at all.

**Contribution (reframed 2026-10-05 after the M0.4 novelty check).** A curve of routing quality
against the size of the onboarding set already exists: UniRoute (2502.08773, Fig. 2) and
ICL-Router (2510.09719, §4.6, 100–1000 exemplars). Our claim is therefore narrower:
- the **small-k regime** (k ∈ 5…50);
- **IRT/Fisher-selected anchors** compared with random anchors and with a UniRoute-style
  cluster representation;
- measured as **stream regret and CPAT@floor inside a run-time-verified cascade**, not as
  static routing accuracy;
- including **live, unseen free-tier endpoints**.

UniRoute and ICL-Router are related work and E1 comparators. They are not the gap.

The background and design rationale are in `Reports/Research_Progress_Report_7.md`. The
literature, method families, and the design-evidence table are in §3–§5 of that report.

### Success criteria for December

| # | Criterion | Measured by |
|---|---|---|
| S1 | The onboarding curve exists: regret recovery as a function of k ∈ {0,5,10,25,50} vs text-only, with CIs | E1 |
| S2 | PROBE-CASCADE matches ACRouter's pass rate at lower cost, or this is honestly shown not to hold | E2 |
| S3 | Results stay stable under 0–30% verifier error, or the degradation is quantified | E3 |
| S4 | Free-tier models the router has never seen are onboarded live and routed on the 65 web-dev tasks | E4 |
| S5 | A working proxy and dashboard demo | M7 |
| S6 | Thesis and defence | M8 |

---

## 2. The method, briefly (PROBE-CASCADE)

1. **Onboard:** a new model runs k IRT-selected anchor tasks, and its ability θ_m is fitted
   with an uncertainty.
2. **Predict:** the task's difficulty b_q is predicted from its embedding. P(pass) = σ(θ_m − b_q),
   plus a memory-kNN residual correction.
3. **Explore:** Thompson-sample P(pass) for each model.
4. **Decide:** pick the cheapest model whose sampled P(pass) ≥ τ, where τ is the user's
   quality/cost dial.
5. **Verify and escalate:** check the answer with run-time signals only (tests, sandbox, AST).
   On failure, escalate to the next model up.
6. **Learn:** each verified outcome updates θ_m and the memory.

There is no LLM in the routing decision.

---

## 3. Architecture and repo layout

The new code lives in `router/` at the repo root (new). It reuses `pipeline/components/` where
possible.

```
router/
  __init__.py
  settings.py            # paths, price tables, τ, k, seeds (not `config.py`: pipeline/ owns bare `config`)
  data/
    schema.py            # OutcomeMatrix: task_id, dimension, model, score, cost_usd, tokens, exec_scored
    coderouterbench.py   # loader for HF Lance1573/CodeRouterBench
    llmrouterbench.py    # loader for HF NPULH/LLMRouterBench (code subsets, priced only)
    phase5.py            # loader for our 325 runs (wraps pipeline/analysis/data.py)
    prices.py            # published $/M-token tables (D7); free tier → modelled price
  core/
    irt.py               # 1PL/2PL fit, ability posterior (Laplace), Fisher-information anchor selection
    difficulty.py        # embedding → b_q regressor (local sentence-transformers / BGE)
    memory.py            # kNN store of verified outcomes, residual correction
    policy.py            # Thompson sampling + cheapest-above-τ + escalation ladder
    verifier.py          # run-time signals: prompt tests, sandbox, AST; noisy-verifier simulator for E3
    onboarding.py        # probe runner: k anchors → θ_m (replay or live)
  baselines/             # always-best, random, fixed cascade, kNN, DimensionBest, LinUCB, ACRouter (wrapped)
  sim/
    stream.py            # task-stream replay engine, leave-one-model-out, model-arrival events
    metrics.py           # cumulative regret, CPAT@floor, cost at matched pass rate, bootstrap CIs
  experiments/           # e1_onboarding.py, e2_cost.py, e3_verifier_noise.py, e4_webdev_live.py
  live/
    registry.py          # BYOK model registry (SQLite)
    gateway.py           # thin wrapper over pipeline/components/llm_gateway.py (LiteLLM)
    sandbox.py           # wraps pipeline/components/test_runner.py (Docker)
  proxy/
    server.py            # FastAPI, OpenAI-compatible /v1/chat/completions
    wrap.py              # `router wrap -- <test cmd>` verification mode
  dashboard/             # static page or Streamlit: models, onboarding curve, decisions, $ saved
  cli.py                 # router add-model | probe | run | serve | wrap | replay
tests/router/            # pytest, mirrors router/ (add "tests/router" to testpaths in pyproject.toml)
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
- `pipeline/components/test_runner.py` (`TestRunner`, the Docker test execution)
- `pipeline/components/code_extractor.py` (`CodeExtractor`)
- `pipeline/components/task_loader.py` (`TaskLoader`, for the 65 tasks in `benchmark/tasks/`)
- `pipeline/analysis/data.py` (`load_experiment_data(csv_path="data/phase5_results.csv",
  prefer_database=False, zero_test_treatment=...)`)
- `pipeline/analysis/failure_taxonomy.py` (derived failure labels; on `main` since 2026-10-05)
- `data/phase5_results.csv`, the frozen 325-run snapshot (SHA-256 in `data/README.md`). Never edit it.

---

## 4. Milestones

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[-]` dropped (say why in the log).
Every milestone has an **exit gate**. Do not start the next milestone's core work until the gate passes.

### M0 — Setup and novelty check (target: Oct 5–11)
- [x] M0.1 Confirm Docker runs. Run `cd benchmark && ./docker-run.sh task API-013` against the reference solution
      to settle the suspected defective tasks (API-013, CSS-012). *Done 2026-10-05:* both reference
      solutions pass, so the tasks are not defective (see the log).
- [x] M0.2 Create the `router/` skeleton and `tests/router/`. Add `sentence-transformers`,
      `fastapi`, `uvicorn` and `huggingface_hub` to requirements. *Done 2026-10-05:* 84 tests pass.
      Only `huggingface_hub` is installed so far; the other three are installed at M3 and M7.
- [x] M0.3 Merge every worktree branch into `main` (done 2026-10-05: data-sufficiency, goal-lock and
      progress-report-7, all `--no-ff`; see the log).
- [x] M0.4 Novelty check (about 1 day). Read FlyRoute, MonoRouter, "Routing with Generated Data:
      Annotation-Free LLM Skill Estimation", and papers that cite ACRouter (2606.22902). Record the
      verdict in the log. *Done 2026-10-05:* the gate tripped partially, and the contribution is reframed in §1.
- [x] M0.5 Check that the unverified ids in Report 7's references ([26], [28]–[31]) resolve.
      *Done 2026-10-05:* all ids resolve, and 7 titles need correcting in the thesis (see the log).
- [ ] M0.6 Repo-audit backlog: see §4a. Nothing there blocks M1–M5.
- **Gate:** Docker works, `pytest` runs, and the novelty verdict is recorded. If a paper already
  does onboarding-k-vs-regret, reframe the contribution before M3.
  **Passed 2026-10-05.** Docker 29.2.0 ran both tasks, 84 tests pass, and the verdict is recorded.
  The contribution was reframed (§1) because UniRoute and ICL-Router already publish k-vs-quality curves.

### M1 — Data layer (target: Oct 8–15)
- [ ] M1.1 Download CodeRouterBench from HF (`Lance1573/CodeRouterBench`) and clone
      `github.com/LanceZPF/agent-as-a-router` (MIT) into `external/`, which is git-ignored.
- [ ] M1.2 Write `router/data/schema.py` (`OutcomeMatrix`) and `coderouterbench.py`. Flag each
      dimension as exec-scored or LLM-judged; the primary evidence uses exec-scored only.
- [ ] M1.3 Write `llmrouterbench.py` for the code subsets (livecodebench, swe-bench/verified,
      humaneval, mbpp). Drop the zero-cost models from all economic metrics.
- [ ] M1.4 Write `phase5.py` over `data/phase5_results.csv`. Never write to the `runs` table (D10);
      evaluation runs now need `--db` to touch the database at all.
- [ ] M1.5 Write `prices.py`. Use the CodeRouterBench Table 6 prices, plus published list prices for the
      free-tier models in the live pool, with the source URL and access date for each.
- [ ] M1.6 Unit tests: row counts match the published stats (CodeRouterBench probe 7,080 /
      ID test 2,919 / OOD 176); no NaN scores; prices present for every model.
- **Gate:** all three corpora load into `OutcomeMatrix`, and the tests pass.

### M2 — Replay harness and baseline reproduction (target: Oct 13–22)
- [ ] M2.1 `sim/stream.py`: deterministic task-stream replay with seeds, a leave-one-model-out
      mode, and model-arrival events (a model joins at task t).
- [ ] M2.2 `sim/metrics.py`: cumulative regret (ACRouter reward r = s − 0.1·USD), CPAT with an
      acceptance floor, cost at matched pass rate, and bootstrap 95% CIs.
- [ ] M2.3 Baselines: always-best, random, fixed cheap→strong cascade, kNN, DimensionBest, LinUCB.
- [ ] M2.4 Reproduce the ACRouter paper's Table 3 (ID split) with their released scripts
      (`scripts/run_id.py`, `run_baselines_ood176.py`) and with our harness.
- **Gate:** our harness reproduces the paper's kNN, DimensionBest and Always-Opus numbers within
  ±1.0 AvgPerf point. If it doesn't, stop and find out why before going further.

### M3 — IRT core and onboarding (target: Oct 20–Nov 2)
- [ ] M3.1 `core/irt.py`: fit 1PL and 2PL on the probing split. Choose by held-out log-likelihood.
- [ ] M3.2 `core/difficulty.py`: embedding → b_q regressor, using local BGE / MiniLM so it costs
      nothing. Report held-out correlation.
- [ ] M3.3 Anchor selection by Fisher information (tinyBenchmarks style), compared against random anchors.
- [ ] M3.4 `core/onboarding.py`: k anchor outcomes → θ_m posterior (mean, sd).
- [ ] M3.5 Sanity check: θ_m from k = 50 anchors ranks the 8 models the same way as their full-data
      pass rate (Spearman ρ ≥ 0.8).
- **Gate:** the M3.5 sanity check passes, and the ability-estimation error falls as k grows.
- Papers: IRT-Router (2506.01048), tinyBenchmarks (2402.14992), UniRoute (2502.08773).

### M4 — Policy: memory, exploration, cascade (target: Oct 27–Nov 9)
- [ ] M4.1 `core/memory.py`: kNN residual correction on top of the IRT prediction (ACRouter idea).
- [ ] M4.2 `core/policy.py`: Thompson sampling, then the cheapest model with P ≥ τ, then an
      escalation ladder with a budget cap.
- [ ] M4.3 `core/verifier.py`: during replay, escalation is driven by a **simulated run-time
      verifier** (ground truth flipped at error rate ε), **never by the raw ground-truth
      score** (see Rule R1).
- [ ] M4.4 Online update of θ_m and memory after each verified outcome.
- [ ] M4.5 Ablations: no memory, no exploration, no escalation, and DimensionBest in place of IRT.
- **Gate:** on the ID split with all 8 models known, PROBE-CASCADE matches or beats kNN on CPAT
  at the best-single acceptance floor. If it doesn't, record it and investigate before running E1/E2.
- Papers: Dekoninck et al. (2410.10347), ACRouter (2606.22902), PILOT (2508.21141).

### M5 — Offline experiments E1–E3 (target: Nov 3–16)
- [ ] M5.1 **E1 onboarding (core):** leave each model out in turn and add it at task t. Conditions:
      k ∈ {0,5,10,25,50}, a text description only (LLM-written card, or RouteProfile-style
      metadata profile, 2605.00180), random anchors vs Fisher anchors, a UniRoute-style cluster
      representation over the same k probes (2502.08773), and an oracle (full data).
      Metric: regret over the next N tasks and the number of tasks to recover. 5 seeds. Run on
      CodeRouterBench and on the LLMRouterBench livecodebench pool.
- [ ] M5.2 **E2 cost:** cost at matched pass rate and CPAT@floor against all baselines and ACRouter.
      5 seeds, with CIs.
- [ ] M5.3 **E3 verifier noise:** ε ∈ {0, 0.1, 0.2, 0.3}.
- [ ] M5.4 Figures: onboarding curve, Pareto (cost vs pass), regret curves, and noise sensitivity.
      Save to `router/experiments/out/` and keep the generating script beside each figure.
- **Gate:** E1 and E2 are complete with CIs. The results feed the thesis, whatever their direction.

### M6 — Live layer and E4 web-dev (target: Nov 10–23)
- [ ] M6.1 `live/registry.py`: BYOK model registry (SQLite, keys from env, never committed).
- [ ] M6.2 `live/gateway.py` over `LLMGateway`. The live pool is **free-tier only**: Groq, OpenRouter
      `:free` models, and Gemini free. These are models the offline router has never seen, which
      makes them a true new-model test.
- [ ] M6.3 Live onboarding. The anchor tasks must be executable locally, so pick anchors from
      CodeRouterBench's exec-scored dimensions whose tests run in our sandbox. Check this first.
- [ ] M6.4 `live/sandbox.py` over `TestRunner` (Docker) for the 65 web-dev tasks.
- [ ] M6.5 **E4:** run the free-tier pool on the 65 tasks, **3 runs each** to fix the old
      `run_number = 1` defect. Record `failure_mode`. Cost uses modelled published prices, since
      the real free-tier cost is $0.
- [ ] M6.6 Respect rate limits (back-off, overnight batch runs). Log every call.
- **Gate:** at least 3 free-tier models are onboarded live and routed on the 65 tasks, with
  results stored.

### M7 — Proxy and dashboard (target: Nov 17–30)
- [ ] M7.1 `proxy/server.py`: FastAPI `/v1/chat/completions`, OpenAI-compatible. Adds an
      `x-router-model` / `x-router-reason` response header.
- [ ] M7.2 Verification modes:
  - (a) the request supplies a test command or tests in its metadata, so verify and escalate;
  - (b) `router wrap -- pytest`, which wraps an agent loop;
  - (c) no tests at all, so prediction-only routing with no escalation, and the dashboard says so.
- [ ] M7.3 `router add-model --name --provider --key-env`, which triggers live onboarding.
- [ ] M7.4 Dashboard: registered models with θ ± sd, the onboarding progress of a new model, a
      decision log, escalations, and $ saved vs always-best (modelled prices).
- [ ] M7.5 Demo script: add an unseen model, watch it get onboarded, then route the web-dev tasks.
- **Gate:** the demo runs end-to-end from a clean start in under 10 minutes.
- **Fallback if behind:** dashboard → a static HTML report; proxy → CLI only. Never cut E1.

### M8 — Thesis, report and defence (target: Dec 1 → deadline)
- [ ] M8.1 Thesis chapters: problem, Phase 1–6 recap, literature, method, experiments, demo,
      limitations.
- [ ] M8.2 Results must report the simple baselines and every negative result (Rule R2).
- [ ] M8.3 Resolve or mark all unverified citations.
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

| Weeks | Milestones |
|---|---|
| Oct 5–11 | M0, start M1 |
| Oct 12–18 | M1, M2 |
| Oct 19–25 | M2 gate, M3 |
| Oct 26–Nov 1 | M3, M4 |
| Nov 2–8 | M4 gate, M5 (E1 first) |
| Nov 9–15 | M5, M6 |
| Nov 16–22 | M6 (E4), M7 |
| Nov 23–29 | M7, buffer |
| Dec 1 → deadline | M8 |

**Priority when short of time:** E1 > E2 > M6/E4 > E3 > M7 dashboard > M7 proxy.

---

## 6. Rules (do not break)

- **R1 — No clairvoyant escalation.** During replay, never escalate on the benchmark's ground
  truth. Use the simulated run-time verifier only.
- **R2 — Always report the simple baselines:** kNN, DimensionBest, fixed cascade and always-best.
  A result without them is not a result.
- **R3 — CPAT always carries an acceptance floor.** Never quote unconstrained CPAT; it picks a
  16.8%-accuracy model.
- **R4 — Primary evidence comes from execution-scored data only.** LLM-judged dimensions are
  reported separately.
- **R5 (D10) — Never write to or repair the Phase 5 `runs` table.** New live runs go to new tables or files.
- **R6 (D5) — No hypothesis may depend on human raters.**
- **R7 — Cost comes from modelled published prices,** with the source and date recorded, plus a
  price-sensitivity sweep (D7).
- **R8 — Multiple seeds and bootstrap CIs on every headline number.**
- **R9 — Keys and secrets live in `.env` only.** Never commit them.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| A 2026 paper already does the onboarding-k study | **Partly happened** (UniRoute, ICL-Router). Reframed in §1 to the small-k, anchor-selection, cascade-regret and CPAT setting. Re-run the citation sweep at M8 |
| The M2 gate fails (can't reproduce ACRouter) | Use their released scripts as the reference; report the discrepancy |
| The IRT fit is poor on 8 models | Add the LLMRouterBench pool (more models); fall back to a 1PL or per-dimension θ |
| Free-tier rate limits slow E4 | Batch runs overnight, cache every response, drop to 2 runs if needed |
| The anchor tasks don't run in our sandbox | Choose anchors only from tasks verified to run locally (M6.3) |
| The proxy has no tests for open-ended prompts | Mode (c) is prediction-only, stated honestly in the demo |
| Docker breaks again | Subprocess sandbox fallback, flagged as less isolated |

---

## 8. Key references

| Paper | Where it's used |
|---|---|
| Agent-as-a-Router / ACRouter (2606.22902) | the base system, data and baselines |
| UniRoute (2502.08773), EmbedLLM (2410.02223), RouteProfile (2605.00180) | new-model representation (M3, E1) |
| IRT-Router (2506.01048), tinyBenchmarks (2402.14992) | IRT and anchors (M3) |
| Dekoninck et al. (2410.10347) | cascade optimality (M4) |
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
- **Decision needed from the user:** confirm the reframe. Submission and defence dates are still TBD.
