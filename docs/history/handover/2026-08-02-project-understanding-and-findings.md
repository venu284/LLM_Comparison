# Handover: project understanding, verified findings, and assessment

_Written: 2026-08-02 · Scope: what this repo is for, everything measured about it so far, and an explicit opinion on where it stands_
_Status: open (reference brief — no task to execute)_

This is a **context brief**, not a task handoff. For the open work items see
`2026-08-02-phase7-and-open-data-issues.md` in this directory.

Everything below is split into **MEASURED** (verified against the repo or the
`runs` table, with the command that produced it) and **OPINION** (my judgement,
arguable). Do not let the two blur — a future session should be able to
re-derive every measured claim and freely disagree with every opinion.

---

## 1. What the project is trying to achieve

Replace *"which LLM is best?"* with *"which LLM is best **for this specific
web-development task**?"* — answered from measured evidence rather than a
general leaderboard.

Three goals, stacked. Most of the project's internal tension comes from these
being treated as one thing.

### Goal 1 — the product
A developer types a web-dev task in plain English. The system classifies the
intent, routes to the model(s) most likely to produce working code for that kind
of task, and returns the best answer — **two models' responses by default**.

| Layer | Status | What it really is |
|---|---|---|
| Free models (GPT-OSS-120B, Llama ×3, Qwen3-32B) | Working | Showcase vehicle. Chosen for free API access, not merit. Disposable. |
| BYOK — user supplies their own API keys | Planned | Convenience: one interface over many providers |
| **The routing algorithm** | Partly built | **The actual contribution** |

The models are interchangeable scaffolding. Swapping all five would not change
the project. That is the correct relationship.

### Goal 2 — the research
Prove that domain-specific, evidence-driven model selection beats naive
selection. Requires, in order: specialization exists (H1/RQ1) → general rankings
don't predict it (H2) → a router can exploit it (H4/RQ3). H3 (automated ranking
correlates with human preference, ρ > 0.7) is the credibility bridge: without
it, the system optimises for passing tests rather than for useful code.

### Goal 3 — the apparatus
The benchmark + pipeline are a deliverable, not setup: a reusable instrument
that takes *any* model set and produces a defensible routing table. This is what
makes BYOK meaningful — a user's own models can be scored the same way.

### The dependency chain
```
benchmark + pipeline  ->  evidence  ->  routing algorithm  ->  the app
  (the apparatus)        (the data)   (the contribution)     (the vehicle)
```
Every defensibility claim traces back to the benchmark. That is why the
data-quality issues mattered so much.

---

## 2. Current state of the repo

**MEASURED.** `main` is at `50b6634` (PR #1, merged 2026-07-31). Contains:
`pipeline/analysis/` (data, stats, recommender, validate, figures,
run_analysis), `human_eval/` (7 files), `Reports/Research_Progress_Report_6.md`
+ 16-page PDF with 9 figures. 36 tests passing.

**MEASURED.** A second, **unmerged** branch `worktree-phase6-7-data-sufficiency`
exists at `72a97fa` in a *locked* worktree under `.claude/worktrees/`, from a
parallel session. 3 commits, +1491/-106 across 14 files. It adds
`pipeline/analysis/failure_taxonomy.py`, `human_eval/sampler.py`,
`Reports/Phase5_Data_Sufficiency_Assessment.md`, a `zero_test_treatment`
parameter on `load_experiment_data`, and a top-2 router analysis.
**None of this is on `main`.** Anything referencing `failure_taxonomy.py` or
`zero_test_treatment` is describing that branch, not the merged codebase.

**Repo root is nested**: the real root is
`/Users/venu/Downloads/Major Project/LLM_Comparison/LLM_Comparison` — one level
below the directory of the same name.

---

## 3. Verified findings

### 3.1 The Phase 5 numbers are sound
**MEASURED.** All published Phase 5 pass rates re-derive exactly from the `runs`
table (GPT-OSS 53.8%, Llama-3.3-70B 44.6%, Llama-4-Scout 43.1%, Llama-3.1-8B
36.9%, Qwen3-32B 12.3%), as do every per-category and per-difficulty figure. The
data collection is trustworthy; the *interpretation* is where the problems were.

### 3.2 H4 is refuted, and was unreachable
**MEASURED** (`analysis/validate.py`, leave-one-task-out over 65 tasks):

| Strategy | Pass rate |
|---|---|
| Oracle (perfect router, upper bound) | 58.5% |
| Best single model (GPT-OSS-120B) | 53.8% |
| Random | 38.2% |
| Recommender, framework weights | 36.9% |

Maximum improvement available to **any** router: **+8.6%**. H4 required >15%.
Only **3 of 65** tasks are winnable by routing away from GPT-OSS-120B, which
alone solves 35 of the 38 tasks any model solves.

### 3.3 The published 55.4% routing figure does not survive validation
**MEASURED.** The "~55.4% vs 53.8%, supports H4" claim in the project overview
is the mean of each category's best model (38.5 + 46.2 + 46.2 + 61.5 + 84.6)/5 =
55.4%. The arithmetic is right, but it selects each winner **using the same 65
tasks it is then scored on** — resubstitution, not validation.

Held out properly, category-aware routing scores **53.8% — identical to always
using GPT-OSS-120B. A +0.0 point gain.**

Note also: the overview's H4 wording drops the threshold, but the pre-registered
Phase 2 H4 said **">15%"**. Expect to be held to the registered version.

### 3.4 Top-2 saturates the ceiling
**MEASURED.** Since the app returns two models:

| Strategy | At least one returned model passes |
|---|---|
| Best single model | 53.8% |
| **Best fixed pair** (GPT-OSS-120B + Llama-4-Scout) | **58.5%** |
| Best fixed trio | 58.5% |
| Oracle, all five | 58.5% |

One fixed pair, chosen once and never varied, reaches the full 5-model ceiling.
A third model adds nothing, and routing adds nothing over that static pair.

### 3.5 The model ranking does not survive testing
**MEASURED** (McNemar, paired, Holm-corrected). Only 5 of 10 pairs are
distinguishable, 4 involving Qwen3-32B. GPT-OSS-120B is **not** distinguishable
from Llama-3.3-70B (p = 0.35) or Llama-4-Scout (p = 0.37). "GPT-OSS-120B had the
highest pass rate" holds as a point estimate, not as a ranking claim.

### 3.6 Difficulty beats model choice as a predictor
**MEASURED.** Cramér's V = 0.515 for difficulty × outcome vs 0.288 for model ×
outcome. How hard the task is predicts success better than which model you use.

### 3.7 Per-category specialization is underpowered
**MEASURED.** Of five per-category chi-squared tests, only Bug Fixing reaches
significance (p = 0.002). Frontend, API, CSS, TypeScript do not, at 13 tasks
each. Every subtable has a minimum expected count below 5.

### 3.8 Data-quality issues
**MEASURED.** 63 of 325 runs (19%) reported `tests_total = 0` and were scored as
failures; Qwen3-32B holds 40. 13 Qwen runs hit the `max_tokens: 4096` ceiling
(no other model ever did), 10 of those are zero-test rows — partly confounding
the "chain-of-thought hurts code generation" claim with plain truncation.
Dropping all 63 flips RQ1 from p = 1.9e-5 to p = 0.37, but leaves ragged
denominators (Qwen retains 25 of 65). `failure_mode` is empty in all 325 rows
despite being in the schema. `run_number` is 1 in all 325 rows despite
`runs_per_task: 3` being pre-registered — no variance estimate exists and pass@3
is uncomputable.

The unmerged branch reports this as largely resolved via a derived taxonomy
(10 defective-task / 10 truncated / 43 undiagnosable; dropping only the 20
artifactual rows gives p = 2.6e-4). **Unverified by me — that work is on the
branch, not `main`.**

### 3.9 The classifier works; the runtime path does not exist
**MEASURED.** Task classifier: 100% on the 65 benchmark prompts, **80%** on ten
freshly written paraphrases. The 100% is inflated by the benchmark's rigid
prompt template; 80% is the deployment-realistic number.

**MEASURED.** `recommender.py:214` — `recommend_for_prompt(prompt,
difficulty="medium")` returns **one** model and **hardcodes difficulty**. The app
needs two models and has no ground-truth difficulty label at runtime. Missing in
code: a difficulty estimator, a top-k path, and any serving layer.

### 3.10 Two bugs found and fixed during Phase 6
**MEASURED.** Both caught by known-answer tests, not by inspection:
- `table[[False, True]]` on boolean column labels is read by pandas as a boolean
  **row mask**, silently collapsing every contingency table to one row.
- `.astype(bool)` on a pivot turns NaN into `True`, inflating the oracle from
  38/63 to 56/63 on filtered data.

Both now have regression tests, plus guards that refuse to compute paired
statistics (McNemar, LOOCV) on a non-fully-paired frame.

---

## 4. Opinion

Clearly marked as judgement. Argue with any of it.

**The project is sound and the app idea is worth building.** The apparatus is
genuinely good: 65 tasks with real test suites, Docker isolation, a clean
pipeline, and a database of honest outcomes. That is the durable asset.

**The H4 failure is a finding, not a failure — and it is the most interesting
result in the project.** Routing pays off only when model competences are
*complementary*. These five are *nested*: three are Llama variants, all five are
free-tier models of similar lineage, and GPT-OSS-120B dominates almost
everything the set can solve. A result that tells other people *when* routing is
worth building is more valuable than a leaked +1.6 points.

**The headroom ceiling is a property of these five models, not of routing.** I
believe — but have **not measured**, and neither has anyone else — that a user
bringing Claude + GPT-4 + Gemini keys would show genuinely different failure
profiles and real headroom. BYOK is therefore not a side feature; it is the
escape hatch that gives the algorithm its honest test.

**Therefore split the claim in two.** They are currently conflated, and that is
the single biggest framing problem:
1. *Does the router pick correctly?* A property of the **algorithm**. Measurable
   now, independent of headroom. The classifier result is real evidence here.
2. *Does picking correctly pay off?* A property of the **model set**. On these
   five: no, and provably cannot. On a user's own set: unknown — and the platform
   is precisely the instrument for finding out.

**Top-2 is the best design decision in the project**, and I suspect it was made
for UX reasons rather than statistical ones. It happens to capture the entire
achievable ceiling. Keep it.

**Biggest risk to the thesis is no longer the data — it is single-run
evaluation.** Every number rests on one sample per task/model. `run_number` = 1
everywhere. If the benchmark is ever re-collected, adding runs 2 and 3 is worth
more than swapping in stronger models.

**Do not swap in stronger models hoping to rescue H4.** Stronger models are
*more* nested, not less — a better model tends to solve a superset of what weaker
ones solve, which shrinks routing headroom further. Complementarity is what you
need, and complementarity is not the same as quality.

**On presentation:** the honest version is stronger than the optimistic one. A
committee that finds an uncross-validated 55.4% will discount everything else.
A committee shown "+8.6% ceiling, here is why, here is when routing does pay"
sees someone who understands their own data.

---

## 5. Corrections needed in project documents

**MEASURED — these are factual mismatches, not preferences.**

1. **The overview's §5 "~55.4% supports H4"** — resubstitution. The validated
   figure is 53.8%, a +0.0 gain. This is the highest-priority fix; it is the
   central claim and it will not survive review.
2. **The overview's §5 Step 1 describes a "lightweight LLM fallback for
   ambiguous prompts."** Not built. Stage 2 is keyword counting across the four
   language categories. Build it or drop it from the doc.
3. **The overview's §6 plans Friedman + Nemenyi.** Implemented instead:
   chi-squared, Cramér's V, McNemar, Holm. McNemar is the more appropriate
   choice (same 65 tasks per model = paired observations), but Friedman is cheap
   to add if the committee expects it.
4. **H4's wording** differs between the overview (no threshold) and the
   pre-registered Phase 2 framework (">15%").

---

## 6. Where the evidence stands against the hypotheses

| | Claim | Status |
|---|---|---|
| **H1 / RQ1** | Specialization exists | Supported overall; per-category underpowered (13 tasks each) |
| **H2** | Elo doesn't predict domain performance | Consistent with the data; **not directly tested** |
| **H3** | Automated ≈ human ranking (ρ > 0.7) | **Untested.** Instrumented and ready; needs voters |
| **H4 / RQ3** | Routing beats single model | **Refuted on this model set**, and unreachable (+8.6% ceiling) |
| **RQ2** | Cost/latency vs quality trade-off | Supported — Qwen3-32B: 9× tokens, 10× latency, worst accuracy |

---

## 7. Key files and commands

**Analysis** (all offline, no API calls, reruns in seconds):
- `pipeline/analysis/run_analysis.py` — orchestrator.
  `cd pipeline && venv/bin/python analysis/run_analysis.py` regenerates every
  statistic and all 9 figures.
- `pipeline/analysis/validate.py` — LOOCV; contains the H4 test and the leakage assertion.
- `pipeline/analysis/recommender.py` — classifier + scoring. Line 214 is the runtime gap.
- `pipeline/analysis/data.py` — single load path. **Read `pass_matrix()`'s docstring before touching it.**

**Phase 7**: `human_eval/PROTOCOL.md` (methodology + sample sizes),
`app.py` (voting server), `bradley_terry.py` (estimator),
`run_synthetic.py --clear` (removes synthetic votes).

**Outputs are git-ignored** — regenerate rather than expecting them in a clone:
`pipeline/exports/analysis/phase6_results.json`, `pipeline/exports/charts/`.

**Environment gotchas:**
- `pytest` is **not** in `pipeline/venv`. Use a scratch venv; do not install into
  the project venv without asking.
- No pandoc/wkhtmltopdf. The report PDF is built Markdown → styled HTML →
  headless Chrome. The `.md` is the source of truth.
- Docker was **not running** during these sessions — anything needing
  `./docker-run.sh` is unverified.
- `gh` is not authenticated; a keychain OAuth token can be passed per-command:
  `export GH_TOKEN=$(printf "protocol=https\nhost=github.com\n\n" | git credential-osxkeychain get | sed -n 's/^password=//p')`
- The user's own PDFs (`Final_Report.pdf`, `Venu_Research_Report3/4`,
  `Research_Progress_Report_5.pdf`) are **untracked** in `Reports/`. Never clean
  that directory indiscriminately.

---

## 8. Decisions already made — do not re-litigate

- **Synthetic Phase 7 votes are not H3 evidence.** Generated from the automated
  pass/fail results, so agreement is circular by construction. They validate the
  pipeline and derived the sample size. Labelled as such at source, in the runner
  output, in the report, and in the figure title.
- **The recommender's weights were not tuned to make H4 pass.** Fixed in Phase 2
  before data existed. The sensitivity grid is a sensitivity analysis, not a
  search for a passing configuration. If a future session tunes them, say so and
  keep it separate from the pre-registered result.
- **H4's failure is reported plainly, not softened.** The headroom arithmetic is
  treatment-independent and is the strongest result in Phase 6.
- **Phase 7 uses the Python standard library, not FastAPI** as Phase 2 specified.
  It is a research instrument for a handful of evaluators, not the Phase 8 app.
- **Nothing is written back to the `runs` table.** Repairing Phase 5 rows would
  break comparison with published numbers. Disclose and re-run; never amend.
- **Statistical functions are validated against known-answer distributions**
  before touching real data. Keep that discipline for anything added.
