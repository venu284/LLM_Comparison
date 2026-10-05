# Is the Phase 5 Data Sufficient for Phases 6 and 7?

**Project:** Multi-LLM Comparison Platform (CSCI 7200)
**Date:** 2026-08-02
**Question asked:** should the benchmark be re-run with stronger models before
Phases 6 and 7 are built, or should those phases be built on the data already
collected and re-run later?

**Short answer: build on the current data. It is sufficient for everything
Phases 6 and 7 actually decide, and a re-run with stronger models would make the
one genuinely weak result *worse*, not better.** The case for re-collection is
real but it rests on a different change — repeated runs and a raised token
ceiling — and it belongs after Phase 7, not before it.

---

## 1. The data, stated plainly

| Property | Value |
|---|---|
| Rows | 325 (65 tasks × 5 models × **1** run) |
| Fully paired | Yes — every model attempted every task |
| Published pass rates | Reproduce exactly from both PostgreSQL and the CSV export |
| Runs with `tests_total = 0` | 63 (19.4%); Qwen3-32B holds 40 |
| Runs that hit the 4,096-token ceiling | 13, all Qwen3-32B |
| Tasks solved by at least one model | 38 of 65 |
| Tasks solved by no model | 27 of 65 |
| `failure_mode` populated | 0 of 325 |
| `run_number` | 1 in all 325 rows, against `runs_per_task: 3` in `config.yaml` |

## 2. Verdict, split by what each phase needs

Lumping "is the data good enough" into one question is what makes it feel
unanswerable. The four things riding on this data have different answers.

### 2.1 Phase 6 statistics — **sufficient**

65 paired tasks per model is an adequate base for the χ² tests, Cramér's V,
McNemar with Holm correction, and the leave-one-task-out validation. The frame is
fully paired, which is the actual precondition for the paired tests. The
published rates reproduce exactly. Nothing in Phase 6's statistical apparatus is
waiting on more data.

### 2.2 H4 and the recommendation algorithm — **the data is not the problem**

H4 (">15% improvement from routing") is not merely rejected; it was unreachable
before the first run. The oracle ceiling is 58.5% against an always-best-single
baseline of 53.8%, which caps *any* router at +8.6%. GPT-OSS-120B alone solves 35
of the 38 tasks that any model solves. Only three tasks — CSS-003, FE-010, TS-005
— are winnable by routing elsewhere.

The reason is visible in the pairwise discordance:

| Pair | A solves, B fails | B solves, A fails |
|---|---:|---:|
| GPT-OSS-120B vs Llama-3.3-70B | 7 | 1 |
| GPT-OSS-120B vs Llama-4-Scout | 10 | 3 |
| GPT-OSS-120B vs Llama-3.1-8B | 13 | 2 |
| GPT-OSS-120B vs Qwen3-32B | 28 | 1 |
| Llama-3.3-70B vs Llama-4-Scout | 5 | 4 |

Every column on the right is near zero. The five models' success sets are almost
perfectly **nested**: each weaker model solves a near-subset of what
GPT-OSS-120B solves. Routing pays only where sets *cross*, and these barely do.

**This is the finding that answers the re-collection question.** Re-running with
"much more performing" models would very likely make it worse. A uniformly
stronger model solves a superset, which deepens the nesting and shrinks the
headroom further. What buys routing headroom is not strength but
**complementarity** — models that fail on different tasks. The routing
literature the algorithm design draws on (RouteLLM [4], RouterDC [5],
GraphRouter [6]) reports gains on model pools deliberately chosen to span
different capability profiles, typically mixing a cheap fast model with an
expensive strong one across task types where the ordering flips. The current five
were selected for architectural diversity, which turned out not to produce
behavioural diversity.

### 2.3 RQ1's robustness — **was insufficient; fixed today without new data**

This was the real problem, and it is the one a reviewer would have found. The
published RQ1 result (models differ significantly on pass rate, χ² p = 1.9e-5)
became p = 0.37 when the 63 zero-test runs were dropped. Two treatments, opposite
conclusions, neither defensible on its own — and no way to choose between them,
because `failure_mode` was never populated.

The taxonomy has now been **reconstructed** from columns that *were* recorded
(`pipeline/analysis/failure_taxonomy.py`). Of the 63 zero-test runs:

| Cause | Runs | Recoverable from stored data? |
|---|---:|---|
| Defective task (API-013, CSS-012 — no model ran a test) | 10 | Yes |
| Truncated at the 4,096-token ceiling | 10 | Yes |
| Cause not recoverable | 43 | No |

That supports a third treatment: drop only the 20 rows whose cause is provably
not the model's competence, and keep the undifferentiable 43 as failures.

| Treatment | n | χ² p | Cramér's V | Denominators |
|---|---:|---:|---:|---|
| Keep all (as published) | 325 | 0.0000195 | 0.288 | 65 / 65 / 65 / 65 / 65 |
| **Drop only artifactual** | **305** | **0.000258** | **0.265** | **63 / 63 / 63 / 63 / 53** |
| Drop all zero-test | 262 | 0.369 | 0.128 | 63 / 60 / 60 / 54 / **25** |

**RQ1 survives the defensible exclusion and collapses only under the
indefensible one.** The collapse in row three is an artefact of Qwen3-32B
retaining 25 of its 65 attempts — dropping four-fifths of one model's hardest
runs inflates its rate from 12.3% to 32.0% and flattens the between-model
difference. Under the middle treatment the denominators stay comparable and the
effect stands.

This does not make the published number airtight, and the report should not
claim it does: 43 runs remain undiagnosed and their true classification could
move the estimate. But the headline result is no longer resting on an arbitrary
choice between two extremes, and that was the open risk.

### 2.4 Phase 7 — **data sufficient; the blocker is people**

`extracted_code` exists for 324 of 325 runs, giving 646 comparison cells across
65 tasks. That is ample stimulus material for the 800-vote target. H3 is blocked
on recruiting 8–12 evaluators and nothing else. No amount of re-collection
changes that.

One property of the data does shape what H3 can mean: 27 of 65 tasks were solved
by no model, so roughly half of all comparisons pit two failing solutions against
each other, where the automated ranking expresses no preference to correlate
with. The protocol now requires ρ to be reported both overall and on the passing
subset (§6.2).

---

## 3. What is genuinely weak, and what would fix it

Two limitations are real, and neither is addressed by swapping in stronger
models.

**Single-run evaluation.** `config.yaml` pre-registered `runs_per_task: 3`;
`run_number` is 1 in all 325 rows. At temperature 0.2 there is still run-to-run
variation in code generation, and with one sample per cell there is no variance
estimate at all — every pass rate is a point with no error bar, and pass@3 cannot
be computed. This is the largest threat to every result in the project and it is
a deviation from the registered protocol. **If anything is re-collected, this is
the change to make first.**

**The token ceiling confounds a headline claim.** Qwen3-32B is the only model
that ever hit 4,096 tokens, because it is the only one emitting chain-of-thought
into the same budget as the code. Ten of its 13 truncated runs produced no test
count. Until it is re-run with a higher ceiling, "chain-of-thought hurts code
generation" cannot be separated from "the model was cut off mid-answer". Note the
direction of the bias: the derived taxonomy already removes those 10 rows from
the middle treatment, and Qwen still finishes last by a wide margin — so the
claim is *probably* directionally right, but it has not been shown.

---

## 4. Recommendation

**Proceed on the current data. Do not re-collect before Phase 7.**

1. **Finish Phase 7 on this data.** Recruitment is the critical path and it is
   the only thing on it. Every day spent re-running the benchmark is a day the
   evaluator pipeline is not moving, and re-collection would not change a single
   input to H3.
2. **Report RQ1 under all three treatments**, with the middle one as the primary
   result and the reason for preferring it stated. This is now generated
   automatically into `phase6_results.json` and Figure 8.
3. **Report H4's rejection as the finding it is.** The +8.6% ceiling is
   treatment-independent and is the strongest, most defensible result in Phase 6.
   It should not be softened, and the weights should not be re-tuned to rescue
   it — they were fixed in Phase 2 before any data existed.
4. **If a second collection happens, change these things and in this order:**
   - `runs_per_task: 3` — actually execute it. Buys variance estimates and
     pass@3. Gated by Groq's 100K tokens/day/model limit, so budget for it.
   - Raise `max_tokens` for reasoning models, or exclude their reasoning tokens
     from the budget. Removes the truncation confound.
   - Populate `failure_mode` at write time. The column exists in the schema, in
     `RunResult`, and in Report 5's metrics table; only the assignment is
     missing. Today's derived taxonomy is a workaround, not a substitute.
   - Fix or drop API-013 and CSS-012 after confirming under Docker.
   - **Select the model set for complementarity, not strength.** Before
     committing a full collection budget, pilot 15–20 hard tasks across candidate
     models and compute the pairwise discordance table in §2.2. If the
     off-diagonal counts are near zero again, routing has nothing to demonstrate
     and Phase 8 should not be a routing demo. Measure this *before* spending the
     budget rather than discovering it afterwards, as happened here.
5. **Reconsider Phase 8's framing before building it.** With +8.6% of headroom, a
   "which model should I use" router is a demo of a null result. Difficulty is
   the stronger signal in this data by a wide margin — Cramér's V 0.515 for
   difficulty against 0.288 for model identity. "How hard is this task, and which
   model survives that difficulty" is the better question this data can answer.

---

## 4a. Addendum: the algorithm as a live top-2 router

Added after the intended deployment was clarified: the algorithm is not only an
offline routing experiment. It is the runtime component of the Phase 8
application — it takes free-text user input, classifies it, and returns **two**
models' responses by default. Three things follow, and none of them are visible
in the numbers reported above.

### 4a.1 The validated number does not describe the deployed pipeline

`validate.py` scores **top-1**, and at line 103 it reads each task's **true
difficulty label** from the benchmark metadata. A user typing a prompt supplies
neither. The live entry point, `Recommender.recommend_for_prompt`, hardcodes
`difficulty="medium"` and returns one model.

Both fixed baselines below are **refit inside each fold** — the best single
model and the best pair are chosen from the other 64 tasks and then scored on
the held-out one — so no strategy in the table enjoys an information advantage
over another. This mirrors what `validate.py:114-115` already does for the
single-model baseline. Measured under actual deployment conditions:

| Strategy | Coverage of 65 tasks | Provenance |
|---|---:|---|
| Oracle (any of the 5) | 58.5% | bound, not a strategy |
| **Best fixed pair — GPT-OSS-120B + Llama-4-Scout** | **58.5%** | LOOCV, refit per fold |
| Best fixed single — GPT-OSS-120B | 53.8% | LOOCV, refit per fold |
| Router top-2, true difficulty | 49.2% | LOOCV |
| Router top-2, `difficulty="medium"` (**as deployed**) | 46.2% | LOOCV |
| Router top-1, true difficulty (the reported 36.9%) | 36.9% | LOOCV |
| Router top-1, `difficulty="medium"` (**as deployed**) | 35.4% | LOOCV |

All 65 folds select the same pair (GPT-OSS-120B + Llama-4-Scout) and the same
single model (GPT-OSS-120B), so the baselines are stable rather than
fold-dependent.

### 4a.2 For a two-response product, routing headroom is zero, not +8.6%

A static pair of GPT-OSS-120B and Llama-4-Scout **reaches the oracle exactly**.
Those two models between them solve every one of the 38 tasks that any of the
five can solve. Once the product returns two answers, there is no correctness
left for a router to win — not 8.6%, nothing. Every fixed pair containing
GPT-OSS-120B scores 55.4–58.5%; every pair without it scores 38.5–50.8%.

The current router returns Llama-3.1-8B + Llama-3.3-70B on 47 of 65 tasks, and
that pair covers 46.2%. **The shipped algorithm would be 12.3 points worse than
hardcoding two models.**

### 4a.3 Why: the router ranks the best model fourth

Score components for a frontend/medium task, weights as pre-registered:

| Model | Total | Category | Difficulty | Latency | Tokens | Actual pass rate |
|---|---:|---:|---:|---:|---:|---:|
| Llama-4-Scout | 0.487 | 0.192 | 0.070 | 0.088 | 0.136 | 43.1% |
| Llama-3.1-8B | 0.471 | 0.115 | 0.060 | 0.150 | 0.146 | 36.9% |
| Llama-3.3-70B | 0.418 | 0.115 | 0.060 | 0.092 | 0.150 | 44.6% |
| **GPT-OSS-120B** | **0.342** | 0.154 | 0.070 | 0.060 | 0.058 | **53.8%** |
| Qwen3-32B | 0.089 | 0.038 | 0.020 | 0.015 | 0.016 | 12.3% |

The most accurate model ranks **fourth of five**. It is slow (1,518 ms against
Llama-3.1-8B's 606 ms) and verbose (640 output tokens against 247), and the
pre-registered weights spend 30% of the score on exactly those two properties.
The efficiency terms are not a tie-breaker here; they are decisive, and they
select against accuracy.

### 4a.4 The code gap, concretely

Three things the application needs that do not exist yet:

- **No top-k path.** `Recommender.recommend_for_prompt` returns
  `Tuple[str, str]` — one model name plus the inferred category. `rank()` already
  produces the full ordering, so exposing top-k is small, but nothing calls it.
- **No difficulty estimator.** Difficulty is an input to the scoring formula
  (w₂ = 0.2) with no way to obtain it at runtime. Validation reads the ground
  truth label; the live path hardcodes `"medium"`. This is the largest missing
  piece, because difficulty is the strongest signal in the data.
- **No serving layer.** The recommender selects a model but nothing calls that
  model. Phase 5's `run_eval.py` invokes models via LiteLLM for the benchmark;
  the application needs an equivalent path for live prompts, returning both
  responses.

### 4a.5 What this means for Phases 6 and 8

The Phase 6 *finding* is unchanged and still correct — it is now simply much
sharper. What it says is that **model selection is the wrong axis for this
product**, and the evidence is stronger for the two-response design than for the
one-response design it was tested under.

That does not leave the application with nothing to do. Three framings the data
does support, in order of how well it supports them:

These are **proposals, not measured results.** Each follows from the data above,
but none has been validated the way H4 was; the first in particular needs its own
LOOCV before it appears in the thesis as anything more than a design direction.

1. **Ship the static pair and route on the second slot only.** GPT-OSS-120B is
   always one of the two answers; the algorithm chooses the *companion* — fast
   and terse when the request looks simple or latency-sensitive, a second strong
   model when it looks hard. Correctness is protected by construction, and the
   algorithm governs the trade-off it can actually measure.
2. **Difficulty-aware escalation.** Difficulty is the dominant signal in the data
   (Cramér's V 0.515 against 0.288 for model identity), and it is the one thing
   the current classifier does not estimate — it is passed in from ground truth
   during validation and hardcoded to `"medium"` at runtime. A difficulty
   estimator on the prompt is the single highest-value addition to the algorithm.
3. **Report the efficiency trade-off explicitly to the user** rather than folding
   it into a hidden weighted score. The 30% efficiency weight is a product
   decision disguised as an algorithmic one; surfacing "this one is 2.5× faster
   and 60% shorter, this one is 17 points more accurate" is both more honest and
   more useful than silently preferring the fast model.

**Do not fix this by re-tuning the weights.** They were pre-registered in Phase 2
before data existed, and the weight-sensitivity grid is a sensitivity analysis,
not a search for a passing configuration. If a runtime profile is introduced for
the application, it must be reported separately from the pre-registered result.

---

## 5. What changed in the repository today

| File | Change |
|---|---|
| `pipeline/analysis/failure_taxonomy.py` | New. Reconstructs the failure taxonomy from `tests_total`, `tokens_output` and `extraction_success`. Derived at analysis time; **nothing is written back to `runs`**, so the published Phase 5 numbers remain comparable. |
| `pipeline/analysis/data.py` | `zero_test_treatment` of `keep` / `drop_all` / `drop_artifactual`. `keep` remains the default and reproduces the published rates exactly. |
| `pipeline/analysis/run_analysis.py` | Emits a `failure_taxonomy` block and all three treatments into `phase6_results.json`; prints the comparison. |
| `pipeline/analysis/figures.py` | Figure 8 rebuilt as a three-treatment comparison with denominators inside the bars. New Figure 10: what explains the zero-test runs, per model. |
| `human_eval/sampler.py` | New. Comparison scheduler — anchor set, coverage balancing, per-evaluator de-duplication, outcome strata. |
| `human_eval/app.py`, `db.py` | Wired to the scheduler; `/api/results` reports coverage. |
| `human_eval/PROTOCOL.md` | §5.1 on comparison selection; §6.2 requires ρ on the passing subset; §6.3 computes α on the anchor set; §7 replaces the task-coverage threat with the both-failed one. |
| `pipeline/tests/test_analysis.py`, `human_eval/test_sampler.py` | 24 new tests. 60 pass, up from 36. |

**Still open and not closable from stored data:** the 43 undifferentiable
zero-test runs, confirmation that API-013 and CSS-012 are defective (needs
Docker), and H3 (needs human evaluators).
