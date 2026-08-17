# Handover: goal-lock session outcome

_Written: 2026-08-16 · Scope: resuming `2026-08-02-phase7-and-open-data-issues.md` sub-scope C (Phase 8 direction), which was left as "not decided"_
_Status: done — direction decided and committed_

Sub-scope C of the 2026-08-02 handover asked for a decision, not implementation:
Phase 8 was planned as a React "which model should I use" demo, and Phase 6 had
shown routing has +8.6% headroom, so the demo had nothing to demonstrate.

**That decision is now made.** The originating handover's two framings —
"select models for complementarity" and "pivot to difficulty-aware selection" —
were both superseded by a third the session found empirically.

## Outcome
_Completed: 2026-08-16_

### What got done

The full direction is in `docs/superpowers/specs/2026-08-16-adaptive-router-design.md`
(published copy: `docs/superpowers/specs/charter.html`). Summary:

**New goal.** A constraint-aware router for code generation that predicts
*whether a cheap model is sufficient*, generalises to unseen models via a bounded
measurement probe, and accepts user-supplied API keys. Metric: cost per accepted
task.

**Why the pivot works.** Nesting kills *selection* routing but *feeds* cascade.
If the weak model's successes are a subset of the strong model's, "try cheap,
escalate on verification failure" reaches the strong model's pass rate for less
money. Computed on the existing 5×65 matrix under Groq list prices:

| Strategy | Pass | $/task | vs baseline |
|---|---|---|---|
| Always GPT-OSS-120B | 53.8% | 0.000422 | — |
| **Llama-3.1-8B → GPT-OSS-120B** | **56.9%** | **0.000377** | **−10.8%** |
| Llama-4-Scout → GPT-OSS-120B | 58.5% | 0.000433 | +2.5% |
| Oracle (cheapest solver) | 58.5% | 0.000089 | −78.8% |

Fixed cascade beats the baseline on **both** axes. This is the day-one result.

**The novelty.** GraphRouter (ICLR 2025) already generalises to unseen LLMs — but
via GPT-4o-*written descriptions* embedded with BERT. That is reputation, and it
breaks on private fine-tunes, post-cutoff models, quantized endpoints, and models
with similar descriptions but different behaviour — exactly the BYOK cases. Our
claim: **measured probe descriptors beat generated descriptions, and the gap
widens where BYOK lives.**

### Findings the design turns on

- **The −78.8% oracle is inflated.** 68.4% of it is clairvoyant abstention on the
  27 tasks (42%) no model solves. Realistic capturable headroom is **≈ −25%**.
  Never quote the oracle undecomposed.
- **Argmax-over-descriptor is flat at ~52% for every probe size k = 3…40**, below
  best-single 53.8%, while descriptor accuracy climbs 63.8% → 74.4%. The
  descriptor is not for picking a winner; it is for thresholding sufficiency.
  H4 reappearing at the descriptor level.
- **Post-failure signal is weak here.** Escalation success by bucket: 27% (no
  tests ran) vs 35% (partial) against 32% blind, n=41. Design pre-execution-first.
  This *inverts* CodeRescue's finding on frontier models — a reportable contrast.
- **Cheap-sufficiency is strongly predictable**: 80% easy / 30% medium / 8% hard;
  69% bugfix / 23% frontend.
- **128 of 325 runs (39%) have partial test credit** (1–99% passing). Every
  published number binarises them to "fail." Unspent graded signal.
- **Four schema columns are empty in all 325 rows**: `eslint_warnings`,
  `compiler_errors`, `failure_mode`, and `ts_any_count` (19/325). Backfilling
  them needs Docker and unlocks a whole feature family — a stronger argument for
  getting Docker up than the API-013 check was.
- `estimated_cost_usd` is `0.000000` in all 325 rows. All prices are modelled.

### Sub-scopes A and B — unchanged

Nothing was done on Phase 7 vote collection or the remaining zero-test rows.
Under the new direction, **H3 is demoted**: no hypothesis may depend on human
evaluators (decision D5), so sub-scope A no longer gates anything. Sub-scope B's
open items (43 undifferentiable rows, Docker confirmation of API-013 / CSS-012)
are unchanged and still open.

### For the main session

- Work is on branch `worktree-phase8-goal-lock`, commits `90f9c3e`, `95095d2`.
- Reproduction scripts committed at `pipeline/analysis/exploratory/` with a
  README documenting the caveats. Verified to re-run and reproduce 56.9% @ −10.8%.
- **Docker is still not running** — fifth consecutive session. Experiments E1–E7
  in the new plan are deliberately scoped to need neither Docker nor new data.
- The two `2026-08-02-*.md` handovers are **untracked** in the main checkout, so
  they could not be edited from this isolated worktree. Their `Status:` lines
  still read as open; sub-scope C is superseded by this file.
- Top validity threat is now **T1: `run_number` = 1 everywhere.** A cascade
  learner *acts on* the failure signal, so single-run noise triggers paid
  escalations. If any collection budget exists, runs 2–3 beat every other use.
