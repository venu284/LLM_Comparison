# Phase 6 + Phase 7 Design

**Date:** 2026-07-31
**Project:** Multi-LLM Comparison Platform (CSCI 7200 Master's Project)
**Scope:** Phase 6 (Analysis & Recommendation Algorithm) and Phase 7 (Human Evaluation)

## Context

Phases 1–5 are complete. The first experimental run produced 325 rows (65 tasks × 5
models × 1 run), stored in Neon PostgreSQL and exported to
`pipeline/exports/results.csv`. Published pass rates in `Reports/Final_Report.pdf`
were re-verified against that CSV on 2026-07-31 and match exactly.

Phase 6 requires no new API calls — every statistic and the entire recommendation
algorithm are computable offline from data already collected. Phase 7 requires human
participants, which are not yet recruited.

## Integrity Boundary

This design produces two kinds of numbers, and the report must never blur them:

- **Real results.** All Phase 6 statistics and recommender validation. Derived from
  the 325 collected runs.
- **Synthetic results.** Phase 7 Bradley-Terry output fitted to generated votes.
  These demonstrate that the voting pipeline and the fitting code work. They are
  **not** evidence for H3 and must be labeled synthetic wherever they appear.

H3 ("automated test-driven rankings correlate with human preferences, ρ > 0.7")
remains untested until real votes are collected.

## Phase 6 — Analysis & Recommendation Algorithm

New module: `pipeline/analysis/`.

### `data.py`

Single loading path shared by every downstream consumer. Reads from PostgreSQL when
`DATABASE_URL` resolves, falls back to `exports/results.csv`. Returns a tidy pandas
DataFrame.

Handles the 63 rows where `tests_total == 0` explicitly. These are runs where the
Docker harness returned no test count and the row was scored as a failure. Default
treatment is to keep them as failures (matching how the published numbers were
computed), with a flag to exclude them so the report can state whether any conclusion
depends on that choice.

### `stats.py`

Statistical tests promised in the Final Report's next-steps section:

- Chi-squared test of independence on the model × outcome contingency table, and
  per-category model × outcome tables. Cramér's V for effect size, since chi-squared
  alone says nothing about magnitude.
- Spearman rank correlation between model parameter count and overall pass rate
  (n = 5, so the report must state the correlation is descriptive, not inferential).
- McNemar's test for paired model-vs-model comparison on the same 65 tasks. This is
  the correct test here because the same tasks are given to every model — the samples
  are paired, not independent.
- Bonferroni or Holm correction across the pairwise family.

Expected-count assumptions for chi-squared will be checked and reported; where cells
are too sparse, Fisher's exact test is used instead.

### `recommender.py`

The framework's weighted scoring formula:

```
RecommendationScore(model, task) = w1·CategoryPassRate
                                 + w2·DifficultyPassRate
                                 + w3·(1/NormalizedLatency)
                                 + w4·(1/NormalizedTokens)
```

Defaults w1=0.5, w2=0.2, w3=0.15, w4=0.15, from the Phase 2 framework.

Two stages, matching the design in Report 5:

1. **Task classifier.** Rule-based keyword matching maps a free-text prompt to one of
   the five categories. Evaluated against the 65 known task prompts, which give
   ground-truth labels for free — classifier accuracy is itself a reportable number.
2. **Profile lookup and scoring.** Per-model empirical profiles built from training
   data only, then scored and ranked.

Latency and token terms are min-max normalized across models before inversion, so no
term can dominate through raw scale.

### `validate.py`

Leave-one-task-out cross-validation. For each of the 65 tasks: build profiles from
the other 64, recommend a model for the held-out task, record whether that model
actually passed it. Profiles are never built from the held-out task — that would leak
the answer.

Baselines, as specified in the framework:

- **Random** — uniform model choice, averaged over many seeds for a stable estimate.
- **Always-best-overall** — always GPT-OSS-120B (53.8%).
- **Oracle** — any model that passed, an upper bound.

This is the direct test of H4 (">15% improvement over single-model strategies").
**H4 may well fail.** GPT-OSS-120B leads four of five categories, so the headroom
over always-best-overall is thin by construction. A negative result is reported as a
negative result; the design does not get tuned until it passes.

### `run_analysis.py`

Orchestrator. Writes machine-readable results to `exports/analysis/*.json` and charts
to `exports/charts/phase6_*.png`, so the report quotes generated artifacts rather than
hand-copied numbers.

## Phase 7 — Human Evaluation

New top-level module: `human_eval/`.

### `app.py`

FastAPI service plus a single-page vanilla HTML/JS front end. No build step — this is
a research instrument, not the Phase 8 demo application.

Flow: pick a task, pull two models' `extracted_code` from the `runs` table, display
side by side with model identity hidden and left/right assignment randomized per
comparison, record the vote.

Votes are written to the `human_votes` table already defined in `db/schema.sql`
(`task_id`, `model_a`, `model_b`, `winner`, `evaluator_id`). No schema change needed.

### `bradley_terry.py`

Fits the Bradley-Terry model by maximum likelihood, the same method WebDev Arena uses.
Produces per-model strength parameters, converts them to an Arena-style rating scale,
and derives confidence intervals by bootstrap. Reports convergence diagnostics.

### `synthetic_votes.py`

Generates votes from a latent quality signal derived from each model's real per-task
pass/fail outcome plus tunable noise, so that a "voter" prefers the model that
actually passed, with a realistic error rate.

Purpose is strictly validation: it proves the fitting code recovers a known ranking
and that the end-to-end path (generate → store → fit → rank) works before any human
time is spent. Every output is tagged synthetic at the source.

### Protocol document

Written methodology covering participant criteria and recruitment, blinding, how many
comparisons are needed for stable Bradley-Terry estimates, vote collection procedure,
and the Spearman test of automated rankings against human rankings that decides H3.

## Report

`Reports/Research_Progress_Report_6.md`, mirroring the section structure of
`Final_Report.pdf`: Executive Summary, Work Completed, Results, Findings, Challenges
and Design Decisions, Current Status and Next Steps, Threats to Validity, References.

Markdown; the user converts to docx/PDF for submission.

The Threats to Validity section discloses three data-quality issues found on
2026-07-31 that are absent from the Final Report:

1. 63 of 325 rows (19%) have `tests_total = 0` and were scored as failures.
   Qwen3-32B holds 40 of them. Thirteen Qwen runs hit the `max_tokens: 4096` ceiling
   and ten of those are zero-test rows, which partially confounds the published
   "chain-of-thought hurts code generation" conclusion with simple truncation. No
   other model ever hit the ceiling.
2. API-013 and CSS-012 returned zero tests for all five models, which points at
   defective tasks rather than model failure. Unconfirmed — needs Docker.
3. `failure_mode` is empty in all 325 rows despite being defined in the schema, in
   `RunResult`, and in Report 5's metrics table.

## Testing

- `stats.py` — validated against distributions with known answers (a table with no
  association must yield p ≈ 1; a perfectly separated table must yield a small p).
- `recommender.py` — classifier accuracy measured on the 65 real prompts; scoring
  verified to be scale-invariant to latency and token units.
- `validate.py` — leakage check asserting held-out task data never enters profiles;
  oracle must upper-bound every strategy, random must lower-bound sensibly.
- `bradley_terry.py` — must recover a known planted ranking from synthetic votes.

## Out of Scope

- Pass@3 re-runs. Blocked on Groq's 100K tokens/day/model limit; stays future work.
- The Phase 8 React demo application.
- Fixing the benchmark data issues. They are disclosed, not repaired — repairing them
  would invalidate comparison with already-published Phase 5 numbers.
