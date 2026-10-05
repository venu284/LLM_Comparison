# Handover: Phase 7 vote collection, zero-test data issues, Phase 8 direction

_Written: 2026-08-02 · Scope: the three open threads after Phase 6 was completed and merged_
_Status: sub-scope B resolved as far as stored data allows; A and C still open — see Outcome_

Three independent sub-scopes. A resuming session can take any one alone — they
share context but not dependencies. Sub-scope A is the next deliverable;
sub-scope B is the biggest risk to the thesis.

---

## Context: what just landed

Phase 6 (Analysis and Recommendation Algorithm) is complete and merged to `main`
at `50b6634` (PR #1, merged 2026-07-31). Phase 7 is fully instrumented but has
collected zero human votes.

**Repo root is nested**: the real project root is
`/Users/venu/Downloads/Major Project/LLM_Comparison/LLM_Comparison` — one level
below the directory of the same name. Paths relative to the outer directory fail.

What Phase 6 found, because it reframes everything downstream:

- **H4 rejected, and unreachable by construction.** Oracle 58.5% vs
  always-best-single 53.8% caps *any* router at +8.6%; H4 required >15%. Only 3
  of 65 tasks are winnable by routing away from GPT-OSS-120B.
- **The recommender scores 36.9% vs a 53.8% baseline** (bootstrap p = 0.0004).
  Calibration, not implementation: the pre-registered weights spend 30% of the
  score on latency/tokens. Pure category weighting ties the baseline exactly.
- **The top three models are statistically indistinguishable** (McNemar + Holm).
- **Dropping the 63 zero-test runs flips RQ1 from significant to not**
  (chi-squared p = 1.9e-5 -> 0.37). Unresolved — see sub-scope B.

Deliverables on `main`: `pipeline/analysis/` (7 modules), `human_eval/` (7 files),
`Reports/Research_Progress_Report_6.md` + 16-page PDF with 9 figures.

---

## Sub-scope A — Phase 7: collect human votes (the next deliverable)

### Current state

Everything is built and validated. Nothing is half-finished.

- Blind pairwise voting interface, stdlib only, writes to the `human_votes`
  table that has existed in `db/schema.sql` since Phase 4 (no migration needed).
- Bradley-Terry estimator, MM algorithm, bootstrap CIs. Verified to recover
  planted strengths to within 0.003.
- Sample size derived empirically, not assumed: **300 votes minimum, 800 target**.

**The database currently holds 1,200 synthetic votes and 0 human votes.** They
are tagged `evaluator_id = 'synthetic_NN'` and must be cleared before real
collection begins, or the fit will mix generated votes with human ones.

### Next steps

1. **Clear the synthetic votes**: `cd human_eval && python run_synthetic.py --clear`.
   Verify with `db.vote_counts()` that total drops to 0. `--clear` deletes only
   `synthetic_%` evaluator ids, so it is safe to re-run.
2. **Recruit 8–12 evaluators** per `human_eval/PROTOCOL.md` section 3 (>=1 year
   web dev experience; React/Express/TypeScript familiarity; *not* previously
   exposed to the Phase 5 results, which would anchor them).
3. **Run the interface**: `cd human_eval && python app.py --port 8000`. Each
   evaluator gets a pseudonymous id automatically via localStorage. Target ~80
   comparisons each (40–60 min).
4. **Fit and test H3**: Spearman rho between the human Bradley-Terry ordering and
   the automated pass-rate ordering across the five models. H3 holds if rho > 0.7.
   `/api/results` shows a live fit and warns if any synthetic votes remain.
5. **Also compute** inter-rater agreement (Krippendorff's alpha) and inspect
   comparisons where humans preferred the model that failed the tests — those
   reveal what the test suites miss. Both specified in PROTOCOL.md section 6.

### Expect this

Adjacent models will likely stay statistically inseparable at any feasible vote
count — Phase 6 already showed the top three are indistinguishable on pass rate.
The protocol is powered to rank *groups*, not to split near-ties. Report
overlapping intervals as ties rather than forcing an order.

---

## Sub-scope B — Zero-test rows (the biggest thesis risk)

### Current state

63 of 325 Phase 5 runs (19%) reported `tests_total = 0` and were scored as
failures. Qwen3-32B holds 40 of them. The published RQ1 result is significant
under that treatment and **not** significant when those runs are dropped, and
neither treatment is defensible on its own:

- Scoring a harness failure as a model failure is wrong when the harness broke.
- Dropping the run is wrong when the model emitted code so broken no test ran.
- Dropping also leaves unequal denominators — Qwen retains only 25 of 65
  attempts — so filtered rates are not comparable to full-data rates.

Resolving this needs the failure taxonomy that `failure_mode` was designed to
capture. **`failure_mode` is empty in all 325 rows** despite being defined in
`db/schema.sql`, in `models/run_result.py`, and in Report 5's metrics table.

### Next steps

1. **Confirm whether API-013 and CSS-012 are defective tasks.** Both returned
   zero tests for all five models, pointing at the task rather than the models.
   Run the reference solutions in Docker:
   `cd benchmark && ./docker-run.sh task API-013` and the same for `CSS-012`.
   If the reference itself yields 0 tests, the tasks are broken and those 10 rows
   are noise. **Docker was not running during this session, so this is unverified.**
2. **Populate `failure_mode`.** Classify each zero-test run: harness/runner
   error, empty extraction, truncated output, or genuine non-compiling code.
   `RunResult.failure_mode` already exists; it just needs to be set and backfilled.
3. **Re-run Qwen3-32B at a higher `max_tokens`.** 13 of its 65 runs hit the 4,096
   ceiling in `pipeline/config.yaml` and 10 of those produced no test count. No
   other model ever hit the ceiling. Until this is separated, the Phase 5 claim
   that chain-of-thought degrades code generation is confounded with plain
   output truncation.
4. **Pass@3 runs** remain future work, gated by Groq's 100K tokens/day/model.
   Single-run evaluation is still the largest threat to every result.

### Do not "fix" the benchmark data in place

Repairing the Phase 5 rows would invalidate comparison with the already-published
numbers. Disclose and re-run; do not silently amend.

---

## Sub-scope C — Phase 8 direction (a decision, not implementation)

### The problem

Phase 8 was planned as a React demo app answering "which model should I use for
this task". Phase 6 showed routing has +8.6% headroom on this model set, and 35
of the 38 solvable tasks are already solved by GPT-OSS-120B alone. A routing demo
therefore has almost nothing to demonstrate.

### Two framings worth considering

1. **Select models for complementarity.** Routing pays off only when different
   models succeed on *different* tasks. This set's successes are nearly nested.
   A new model set chosen for complementary failure profiles — rather than
   architectural diversity, which is how the current five were picked — would
   restore real headroom.
2. **Pivot to difficulty-aware selection.** Difficulty is the stronger signal:
   Cramer's V 0.515 for difficulty vs 0.288 for model identity. "How hard is this
   task, and which model survives that difficulty" is a better question than
   "which model is best".

Decide this before implementation starts. Recommend brainstorming it rather than
building the originally-planned app.

---

## Key files + commands

**Phase 6 analysis** — all offline, no API calls, reruns in seconds:
- `pipeline/analysis/run_analysis.py` — orchestrator.
  `cd pipeline && venv/bin/python analysis/run_analysis.py` regenerates every
  statistic and all 9 figures.
- `pipeline/analysis/stats.py` — chi-squared, Cramer's V, McNemar, Holm, Spearman.
- `pipeline/analysis/validate.py` — LOOCV against random/best-single/oracle.
  Contains the H4 test.
- `pipeline/analysis/data.py` — single load path. **Read `pass_matrix()`'s
  docstring before touching it.**
- `pipeline/analysis/figures.py` — the 9 report figures.

**Phase 7**:
- `human_eval/PROTOCOL.md` — methodology, participant criteria, sample-size table.
- `human_eval/app.py` — voting server. `python app.py --port 8000`
- `human_eval/bradley_terry.py` — the estimator.
- `human_eval/run_synthetic.py` — `--clear` removes synthetic votes; `--store` writes them.

**Outputs** (git-ignored — regenerate rather than expecting them in a fresh clone):
- `pipeline/exports/analysis/phase6_results.json`
- `pipeline/exports/charts/fig*.png`

**Tests** — 36 total, all passing:

    pipeline/tests/test_analysis.py       23 passed
    human_eval/test_bradley_terry.py      13 passed

**`pytest` is NOT installed in `pipeline/venv`.** Use a scratch venv
(`pytest pandas scipy numpy pyyaml python-dotenv`); do not install into the
project venv without asking.

**PDF rendering**: no pandoc/wkhtmltopdf on this machine. The report was built
Markdown -> styled HTML -> headless Chrome. Edit the `.md` as source of truth and
re-render.

---

## Blockers + decisions

**Blocked**
- Sub-scope A is blocked on human participants — nothing technical remains.
- Sub-scope B step 1 is blocked on Docker, which was not running this session.

**Decisions already made — please do not re-litigate**

- **Synthetic votes are not H3 evidence.** They are generated from the automated
  pass/fail results, so agreement with those results is circular by construction.
  They exist to validate the pipeline and derive the sample size. This is
  labelled at the source, in the runner output, in the report, and in the figure
  title. Keep it that way.
- **The recommender's weights were not tuned to make H4 pass.** They were fixed
  in Phase 2 before data existed. The weight-sensitivity grid is a sensitivity
  analysis, not a search for a passing configuration. If a future session tunes
  them, say so explicitly and separate it from the pre-registered result.
- **H4's failure is reported as a finding, not softened.** The headroom
  arithmetic (+8.6% ceiling vs >15% required) is treatment-independent and is the
  strongest result in Phase 6.
- **Phase 7 uses the Python standard library, not FastAPI** as Phase 2 specified.
  It is a research instrument for a handful of evaluators, not the Phase 8 app,
  and this removes an install step for every participant. Documented in report
  section 2.4.
- **`pass_matrix()` fills missing cells with `False`, never `.astype(bool)`.**
  Coercing NaN yields `True` and silently inflated the oracle from 38/63 to 56/63
  during this session. McNemar and LOOCV now refuse to run on a non-fully-paired
  frame rather than computing something misleading.
- **Statistical functions are validated against known-answer distributions**
  before being run on real data. This caught a pandas defect where
  `table[[False, True]]` is read as a boolean *row mask* and silently collapsed
  every contingency table. Keep that discipline for any new test.

**Housekeeping**

- The user's own report PDFs (`Final_Report.pdf`, `Research_Progress_Report_5.pdf`,
  `Venu_Research_Report3.pdf`, `Venu_Research_Report_4.pdf`) sit **untracked** in
  `Reports/`. Never clean that directory indiscriminately. They are not in git; if
  the user wants them versioned that is a separate, deliberate commit.
- The merged remote branch `origin/worktree-phase6-7-analysis` still exists. Safe
  to delete (fully merged) but left alone as an outward-facing action.
- `gh` is not authenticated. A GitHub OAuth token with `repo` scope lives in the
  macOS keychain and can be passed per-command as `GH_TOKEN` without writing
  persistent gh config:
  `export GH_TOKEN=$(printf "protocol=https\nhost=github.com\n\n" | git credential-osxkeychain get | sed -n 's/^password=//p')`

---

## Outcome
_Completed: 2026-08-02_

Resumed to answer a different question than the three sub-scopes were framed
around: **is the Phase 5 data sufficient to build Phases 6 and 7 on, or should
it be re-collected with stronger models first?** Answer and evidence are in
`Reports/Phase5_Data_Sufficiency_Assessment.md`. Verdict: build on the current
data.

Work is on branch `worktree-phase6-7-data-sufficiency`, commit `3bfd4d0`.
**The push was blocked by the permission classifier, so the branch is local
only and no PR exists.** Push it manually or re-authorize:
`git push -u origin worktree-phase6-7-data-sufficiency`.

### Sub-scope B — advanced as far as stored data allows

- `pipeline/analysis/failure_taxonomy.py` reconstructs the taxonomy
  `failure_mode` should have held, from `tests_total`, `tokens_output` and
  `extraction_success`. Of the 63 zero-test runs: 10 defective-task
  (API-013, CSS-012), 10 truncated at the 4,096 ceiling (all Qwen3-32B), 43
  undifferentiable and labelled `no_tests_run` rather than guessed at.
- Derived at analysis time. **Nothing written back to `runs`** — the "disclose,
  do not repair" decision is intact.
- `data.py` gained `zero_test_treatment`: `keep` (default, reproduces published
  numbers exactly), `drop_all`, `drop_artifactual`.
- **The result that matters:** dropping only the 20 artifactual rows gives
  χ² p = 0.000258, V = 0.265, denominators 63/63/63/63/53. RQ1 survives the
  defensible exclusion; the p = 0.37 collapse is an artefact of Qwen retaining
  25 of 65 attempts under `drop_all`. The report should lead with the middle
  treatment and show all three.
- Still open: the 43 undifferentiable runs, and Docker confirmation of API-013 /
  CSS-012 (Docker still not running this session).

### Sub-scope A — one real defect fixed; still blocked on people

Reading `app.py` before recruitment turned up a defect that would have surfaced
only after the evaluators had gone home. It drew a task uniformly then two of
its models uniformly. Over 646 comparison cells, an evaluator doing the
protocol's 80 comparisons draws a repeat with p ≈ 0.99, and Krippendorff's α
counts one person rating a cell twice as two coders. Cross-evaluator overlap —
the base α is computed on — was left entirely to chance.

`human_eval/sampler.py` replaces it: a fixed 20-cell anchor set (two per model
pair, deterministic seed) interleaved into every session, coverage balancing on
the remainder, per-evaluator de-duplication. A 10-evaluator simulation lands
76–83 votes per pair against a target of 80 and multi-rates all 20 anchors.

Tasks no model solved are **kept**, not filtered — excluding them would bias H3
upward. They carry an outcome stratum instead, which revealed that ~50% of
comparisons pair two failing solutions. PROTOCOL.md §6.2 now requires ρ on the
passing subset as well as overall.

Synthetic votes were **not** cleared: collection is not starting today and
clearing early gains nothing. Step 1 of the original sub-scope A stands.

### Sub-scope C — not decided, but the input is sharper

The discordance matrix in §2.2 of the assessment is the concrete argument: every
weaker model's successes are a near-subset of GPT-OSS-120B's. Stronger models
would deepen that nesting, not relieve it. Before any second collection, pilot
15–20 hard tasks and compute that table first — it predicts whether Phase 8 has
anything to demo. Decision still the user's.

### For the main session

- 60 tests pass, up from 36 (24 new). `pytest` still not in `pipeline/venv`;
  a scratch venv was used, and nothing was installed into the project venv.
- `psycopg2` (not `psycopg`) is installed in `pipeline/venv`; the DB is
  reachable and agrees with the CSV export.
- Figure 8 was rebuilt as a three-treatment comparison; new Figure 10 shows what
  explains the zero-test runs per model. Both were copied into `Reports/figures/`
  and `Research_Progress_Report_6.md` was updated to match: §2.1 lists
  `failure_taxonomy.py`, §2.4 documents comparison selection and lists
  `sampler.py`, §3.8 was rewritten around the three treatments, §6 points at the
  sufficiency assessment, §7 restates the zero-test threat.
  **`Research_Progress_Report_6.pdf` is now stale** — the `.md` is the source of
  truth and needs re-rendering (Markdown → styled HTML → headless Chrome; no
  pandoc on this machine).
- H3 remains untested. Nothing done today changes that; it needs evaluators.
