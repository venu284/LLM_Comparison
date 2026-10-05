# Project Direction — LOCKED

_Written: 2026-08-16 · Supersedes the Phase 8 "which model should I use" demo app_
_Status: direction fixed; implementation spec pending approval_

---

## 0. The goal, in one sentence

**Build a constraint-aware LLM router for code generation that learns to predict
whether a cheap model is sufficient, generalises to models it has never seen via a
bounded measurement probe, and lets a user bring their own API keys.**

The unit of success is **cost per accepted task**, not cost per token and not
benchmark pass rate.

---

## 1. Why this problem, and why now

### 1.1 The market signal

- Enterprise LLM API spend passed **$8.4B**, projected **$15B by end of 2026**,
  driven by agentic workloads that consume far more tokens than chatbots.
- Companies are pulling back on uncapped developer AI access. The pattern is
  "give every engineer a coding agent, discover the bill, then restrict."
- The proposed correction across the industry is not "use a worse model" but
  **"let a layer decide which model each request deserves."**
- Enterprises using gateways for cost governance report **40–60%** inference cost
  reductions.

### 1.2 The research signal

The field moved in 2026. Three shifts matter:

| Shift | From | To |
|---|---|---|
| **Granularity** | one-shot prompt routing | routing *inside* execution (step-level, post-failure) |
| **Metric** | cost per token | **cost per accepted task** |
| **Open problem** | which model is best | **transferable policies** — routers are trained per-pool and die when the pool changes |

Cheap inference becomes expensive when failures trigger retries. That is the
insight the whole project now sits on.

### 1.3 Our own evidence for the premise

Phases 5–6 measured, independently, the thing the field asserts:
**prompt-level selection routing saturates.** On a five-model nested pool the
oracle selection ceiling is +8.6% over always-best-single, and the pre-registered
H4 (>15%) is unreachable by construction.

That refutation is now the **motivation**, not damage. It is a stronger opening
than H4 passing would have been.

---

## 2. What is FIXED

These are locked. Re-opening any of them requires a new decision record, not a
conversation.

### D1 — The claim is about *sufficiency*, not *selection*
The router does not answer "which model is best for this task." It answers
**"is the cheap model good enough for this task, and if not, what do I spend
next?"** Selection is refuted on nested pools; sufficiency is not.

### D2 — Never learn model identity
The router learns `f(task_features, model_descriptor) → outcome`, never
`f(task) → model_id`. Identity-based routers cannot accept a model they were not
trained on. This single constraint is what makes BYOK possible.

### D3 — Descriptors are MEASURED, not described
A model's capability vector comes from executing a bounded **probe set** against
its endpoint and observing outcomes. It does **not** come from an LLM writing a
description of the model. This is the novelty (see §4.3).

### D4 — Black-box only
The descriptor must be computable from ordinary chat-completion responses. No
logprobs, no hidden states, no provider-specific fields. An arbitrary
OpenAI-compatible endpoint may expose none of them.

### D5 — Every learning signal is automated
Test outcomes, compile/typecheck/lint results. **No hypothesis depends on
recruiting human evaluators.** Phase 7 stands as proof that human-signal
dependencies do not ship: the apparatus was fully built and collected zero votes
across multiple sessions.

### D6 — Constraints are set by the user, not learned
`w_cost`, `w_latency`, `w_quality` are a weight vector the user supplies. The
router optimises *under* those weights. Personalisation — learning an individual
user's preferences — is **out of scope**: it is a cold-start problem with no data.

> "Self-correcting" means the **route policy** improves from pooled execution
> outcomes. It does **not** mean the system learns each user's taste. Keep these
> apart in every document.

### D7 — Cost is modelled from published per-token prices, with sensitivity analysis
All five current models are free-tier; `estimated_cost_usd` is **0.000000 in all
325 rows**. There is no measured cost signal and there will not be one without
paid collection. Published $/M in-out rates are a defensible modelling assumption
*if* every headline number carries a price-sensitivity sweep.

### D8 — Nesting is a feature now, not a bug
The handover decision "do not swap in stronger models to rescue H4" **stands**.
Selection needs complementarity; cascade and sufficiency-thresholding want
nesting. Changing the pool would weaken, not strengthen, the new design.

### D9 — Two research components, everything else is thin
**C2 (probe profiling) and C4 (sufficiency predictor) carry the contribution.**
Ingress, policy engine, and feedback plumbing are features LiteLLM/Bifrost/Portkey
already ship. Build them demo-grade. Do not spend chapters on them.

### D10 — Disclose, never repair
Nothing is written back to the `runs` table. Phase 5 rows stay as published.
Corrections are disclosed and re-run, never silently amended. (Carried forward
unchanged.)

---

## 3. What we are explicitly NOT doing

| Not doing | Why |
|---|---|
| A "which model is best" recommender UI | Refuted. +8.6% ceiling, and the Phase 6 recommender scored 36.9% vs a 53.8% baseline. |
| Per-user preference learning | No data volume per user. Cold start with nothing to start from. |
| Beating frontier models on quality | Not the claim. The claim is cost per accepted task. |
| Human-preference-gated results | D5. H3 becomes an optional external-validity note. |
| A general-purpose chat router | Verification is unsolved outside executable domains. Scope to code. |
| Re-implementing budget/quota/failover as a contribution | Commodity. Ships in LiteLLM. Demo only. |
| Swapping in stronger models | D8. Deepens nesting, shrinks headroom. |

---

## 4. Research questions and hypotheses

Replaces H1–H4 as the active hypothesis set. H1/RQ1 (specialization exists) and
H4 (selection routing refuted) are retained as **established background**.

### 4.1 RQ-A — Is cheap-model sufficiency predictable?

> **H-A.** Task features plus a measured model descriptor predict whether a cheap
> model will produce a passing solution, materially better than chance.

**Target:** held-out AUC ≥ 0.70.
**Why plausible:** on the existing matrix, cheap-model sufficiency varies from
**80% on easy tasks to 8% on hard tasks**, and from 69% on bug-fixing to 23% on
frontend. That is a strong exploitable gradient.
**Enabler required:** a **difficulty estimator**. At runtime there is no
ground-truth difficulty label — `recommender.py:214` currently hardcodes
`difficulty="medium"`. This is the single highest-value missing component.

### 4.2 RQ-B — Does the policy transfer to unseen models?

> **H-B.** A router trained on model pool A, given only a k-task probe descriptor
> of an unseen model M, retains most of the **CPAT reduction** achieved by an
> otherwise-identical router that had M in its training pool.

**Target:** ≥ 80% of the in-pool CPAT reduction retained.

**Measured on a public corpus (E2), not on the 65-task matrix.** This matters:
our pool is nested, so there is no in-pool *routing gain* here to retain 80% of —
§5/C2 shows argmax-over-descriptor flat at ~52% for every k. H-B is defined
against **cost reduction at held pass rate**, which is the quantity D1 actually
optimises, and it is evaluated where a measurable in-pool gain exists.

**Design:** leave-one-model-out within a public corpus. Isolates the *pool* shift.

### 4.3 RQ-C — Are measured descriptors better than generated ones? ← the novelty

> **H-C.** Descriptors measured by probing outperform descriptors generated by an
> LLM writing about the model, and the gap widens on models the describer cannot
> know about.

**Prior art this contests:** GraphRouter (ICLR 2025) generalises to new LLMs by
using *"a generative LLM such as GPT-4o to generate a descriptive text for each
LLM, outlining key details such as its strengths, token pricing, and context
length"*, then embedding it with BERT.

That is **reputation, not measurement**. It has four failure modes, and all four
are exactly the BYOK case:

1. a user's **private fine-tune** — no describer has heard of it
2. a model **released after** the describer's knowledge cutoff
3. a **self-hosted or quantized** endpoint that behaves unlike its model card
4. **two models with similar descriptions and different real behaviour**

**Design:** same router, two descriptor sources, head-to-head. Stratify results by
whether the model is inside or outside the describer's knowledge. Construct the
outside-knowledge condition deliberately (fine-tunes, quantized variants, models
post-dating the describer).

**This is the contribution nobody owns.**

### 4.4 RQ-D — Does it actually pay?

> **H-D.** A learned sufficiency router beats both always-best-single and fixed
> cascade on cost per accepted task, at equal or better pass rate.

**Endpoints already measured on the existing matrix (Groq list prices):**

| Strategy | Pass | $/task | vs baseline |
|---|---|---|---|
| Phase 6 selection recommender | 36.9% | — | refuted |
| Always GPT-OSS-120B (baseline) | 53.8% | 0.000422 | — |
| **Fixed cascade** Llama-3.1-8B → GPT-OSS-120B | **56.9%** | **0.000377** | **−10.8%, +3.1pp** |
| Fixed cascade Llama-4-Scout → GPT-OSS-120B † | 58.5% | 0.000433 | +2.5% |
| Oracle (cheapest model that solves it) | 58.5% | 0.000089 | −78.8% |

**Price provenance.** Groq list rates were confirmed for **Llama-3.1-8B
($0.05/$0.08), Llama-3.3-70B ($0.59/$0.79), and GPT-OSS-120B ($0.15/$0.60)**.
Rates for **Llama-4-Scout (~$0.11/$0.34) and Qwen3-32B (~$0.29/$0.59) are
unconfirmed** and were taken from recall — rows marked **†** rest on them and must
be re-checked before publication. **The headline result — Llama-3.1-8B →
GPT-OSS-120B at 56.9% / −10.8% — uses only confirmed prices and stands
regardless.** Six-decimal figures reflect arithmetic precision, not measurement
precision; the underlying prices are modelled (D7).

**Honest headroom, decomposed.** The −78.8% oracle figure is inflated: 68.4% of it
comes from the oracle clairvoyantly *declining to attempt* the 27 tasks (42%) no
model solves. A real router does not get that free.

| Task class | n | share of oracle saving | capturable? |
|---|---|---|---|
| Unsolvable — no model solves it | 27 (42%) | 68.4% | only via abstention |
| Cheap-sufficient — 8B alone solves it | 24 (37%) | 20.8% | **yes** |
| Mid-tier needed | 9 (14%) | 10.8% | **yes** |
| Top-tier only | 5 (8%) | 0.0% | no saving exists |

**Realistic target: ≈ −25% cost**, against fixed cascade's already-banked −10.8%.
Claim −25%, not −79%.

### 4.5 RQ-E — Is "don't spend at all" a useful action?

> **H-E.** Predicted-unsolvable flagging reaches useful precision, converting
> wasted spend into an actionable "out of pool" signal.

**Why plausible:** unsolvable rate is **5% on easy, 60% on medium, 56% on hard**.
Difficulty separates it strongly (Cramér's V 0.515 for difficulty × outcome).
**Claim discipline:** abstention reduces spend; it does **not** raise pass rate.
Report as "N tasks flagged out-of-pool," never folded into a pass-rate number.

---

## 5. Architecture

**Design principle (D2):** never learn model identity.

```
        user's existing tooling
                 │  OpenAI-compatible request
                 ▼
    ┌────────────────────────────┐
    │ C1  INGRESS                │  /v1/chat/completions
    │     + constraint profile   │  w_cost, w_latency, w_quality
    │     + user id / budget     │
    └────────────┬───────────────┘
                 ▼
    ┌────────────────────────────┐        ┌─────────────────────────┐
    │ C3  REQUEST UNDERSTANDING  │        │ C2  MODEL REGISTRY      │
    │     category classifier    │        │     user adds API key   │
    │     DIFFICULTY ESTIMATOR ★ │        │     → k-task PROBE RUN  │
    │     complexity features    │        │     → MEASURED          │
    └────────────┬───────────────┘        │       DESCRIPTOR ★      │
                 │                        └───────────┬─────────────┘
                 │      task features                 │ descriptor
                 └──────────────┬─────────────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │ C4  SUFFICIENCY PREDICTOR ★  │
                 │   P(pass | task, descriptor) │
                 │   E[cost], E[latency]        │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │ C5  POLICY ENGINE            │
                 │   utility = f(preds, weights)│
                 │   action ∈ { answer cheap,   │
                 │     cascade, parallel pair,  │
                 │     abstain & flag }         │
                 └──────────────┬───────────────┘
                                ▼
                 ┌──────────────────────────────┐
                 │ C6  VERIFICATION & FEEDBACK  │
                 │   execute → verify → update  │
                 └──────────────┬───────────────┘
                                │ outcome
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
          update descriptor (C2)      update predictor (C4)

          ★ = research contribution; everything else is thin
```

### C1 — Ingress *(thin)*
OpenAI-compatible `/v1/chat/completions` so users repoint existing tooling with an
env var. Carries user id, constraint weights, remaining budget.

### C2 — Model onboarding and profiling ★ *(research core)*
1. User supplies endpoint + key.
2. System runs the **probe set**: k tasks, executed and verified.
3. Emits a **measured descriptor**:
   - pass rate per (category × difficulty) cell, with backoff to marginals
   - mean output tokens → verbosity → cost driver
   - mean and p95 latency
   - truncation rate under a fixed `max_tokens`
   - format compliance / extraction success rate
   - price ($/M in, $/M out) — supplied, not measured
4. Cost is **bounded, disclosed upfront, and charged to the user's own account.**

**Probe-size evidence (already run on the 65-task matrix):**

```
 k probe   descriptor accuracy   argmax-routed pass
      3           63.8%               52.7%
     10           66.8%               52.4%
     20           70.4%               52.7%
     40           74.4%               51.7%
                  ^ still climbing    ^ FLAT, below best-single 53.8%
```

Two findings, both load-bearing:
- Descriptor accuracy is **still improving at k=40** — probe-size economics is a
  real research question, not a footnote. Budget it as an experiment (E4).
- **Argmax-over-descriptor never beats best-single at any k.** The descriptor is
  *not* for picking a winner. It is for **thresholding sufficiency**. This is H4
  reappearing at the descriptor level, and it confirms the D1 reframe.

The descriptor wants 5 categories × 3 difficulties = **15 cells**, so k in the
12–15 range gives roughly one sample per cell — almost certainly too noisy.
Let E4 set k; do not assume it.

### C3 — Request understanding *(enabler)*
- **Category classifier** — already built. 100% on the 65 benchmark prompts, **80%
  on fresh paraphrases**. Use 80% as the deployment-realistic number; the 100% is
  inflated by the benchmark's rigid prompt template.
- **Difficulty estimator ★** — does not exist. Highest-value missing component.
- **Complexity features** — prompt length, # requirements, # files touched,
  # acceptance criteria, framework keywords.

### C4 — Sufficiency predictor ★ *(research core)*
`P(pass | task_features, model_descriptor)`, plus `E[cost]` (verbosity × price)
and `E[latency]`. Trained across many models so it generalises to a descriptor it
has never seen. **Descriptor in, not model id in** — that is the whole trick.

### C5 — Policy engine *(thin)*
Constraint weights → utility. Selects an **action**, not a model:

| Action | When |
|---|---|
| `answer with tier-1` | P(cheap suffices) high |
| `cascade` | P(cheap suffices) moderate; escalate on verification failure |
| `parallel pair` | latency-weighted user, or high-stakes task |
| `abstain & flag` | P(any model solves) low → do not spend |

Budget-aware: remaining quota shapes aggressiveness.

### C6 — Verification and feedback *(enabler)*

**Verification ladder**, in preference order. Every rung is automated (D5):

| Rung | Signal | Quality | Availability |
|---|---|---|---|
| 1 | user's own test suite | best | when the repo has tests |
| 2 | generated tests | noisy | always |
| 3 | compile / typecheck / lint | weak | always |
| 4 | explicit accept/reject | clean | sparse |

**Use the graded signal, not the binary one.** On the existing 325 runs, **128
(39%) sit between 1% and 99% of tests passing**, and every published number
binarises them to "fail." That is a rich continuous quality signal already paid
for and never spent.

**Implicit telemetry** — did the user keep the code, re-prompt within 30s, revert
— is the strongest realistic *product* signal and the field's identified open
problem. It is **future work**, not thesis, because it requires users we do not
have (D5).

### 5.1 The honest hard part

For a general BYOK app on an arbitrary user prompt, **there is no test suite**, so
rungs 2–3 are all you get and they are weak. The self-correcting loop is strongest
where executable verification exists. **State this boundary explicitly and scope
the evaluation inside it.** A committee will respect a stated boundary and will not
respect a claim that it works everywhere.

---

## 6. Data strategy

The 65-task benchmark is too small to train a transferable predictor. It does not
need to be — public corpora with the right shape already exist.

| Corpus | Shape | Role |
|---|---|---|
| **RouterBench** | 11 models × 7 tasks, 405k precomputed inferences, with cost | training + LOMO transfer |
| **LLMRouterBench** (Findings@ACL'26) | 33 models, 21+ datasets, 400k+ instances, 10 routing algorithms | training + baseline comparison |
| **TwinRouterBench** | 970 router-visible prefixes, 520 instances; SWE-bench Verified harness | step-level comparison point |
| **Our web-dev benchmark** | 65 tasks × 5 models, authored test suites, Docker | **held-out cross-domain test** |

**Split pool shift from domain shift.** Never test both at once:

- **E2** shifts the *pool* only (leave-one-model-out inside a public corpus).
- **E5** shifts the *domain* (train on public QA/reasoning → test on our web-dev
  benchmark).

If both shift together and it fails, you cannot say which broke.

---

## 7. Experiments

| # | Experiment | Tests | Needs Docker? | Needs new data? |
|---|---|---|---|---|
| **E1** | Sufficiency predictor, held-out within pool | H-A | no | no |
| **E2** | Leave-one-model-out transfer, public corpus | H-B | no | public download |
| **E3** | **Measured vs. generated descriptors, head-to-head** | **H-C** | no | describer API calls |
| **E4** | Probe-size / cost curve — how big must k be? | C2 design | no | no |
| **E5** | Cross-domain: public-trained router on our web-dev benchmark | H-B, H-D | no | no |
| **E6** | Full economics: cost per accepted task vs baseline & fixed cascade | H-D | no | no |
| **E7** | Abstention precision / recall | H-E | no | no |
| **E8** | Live BYOK validation on new authored tasks | product | **YES** | ~15–20 new tasks |
| **E9** | Backfill `failure_mode`, `eslint_warnings`, `compiler_errors` | feature family | **YES** | re-run |

**E1–E7 need no Docker and no new collection.** The entire thesis is reachable
offline. E8–E9 are strengthening, not gating. Scope accordingly.

---

## 8. Metrics

**Primary — cost per accepted task (CPAT)**
```
CPAT = total spend / tasks passing verification
```
This is the metric the field converged on in 2026. Cheap inference that fails and
retries is not cheap.

**Secondary**
- pass rate (accepted / attempted)
- p50 / p95 end-to-end latency — cascade adds a serial hop; on the existing matrix
  `8B → GPT-OSS` costs **+21% latency** for −10.8% cost. That trade must be shown,
  not hidden.
- escalation rate
- abstention precision / recall
- descriptor accuracy (probe quality)
- **regret vs. oracle** as feedback accumulates — the learning curve

**Reporting discipline**
- Every cost number carries a **price-sensitivity sweep** (D7).
- Abstained tasks are reported as flagged, never folded into pass rate (H-E).
- The **oracle is decomposed**, never quoted as a single headline (§4.4).

---

## 9. Phase plan

| Phase | Deliverable | Gates on |
|---|---|---|
| **P8** | Foundation: public corpus ingest, descriptor schema, probe-set construction, difficulty estimator | — |
| **P9** | C4 sufficiency predictor + E1, E4 | P8 |
| **P10** | E2 leave-one-model-out transfer | P9 |
| **P11** | **E3 measured vs. generated — the novelty experiment** | P9 |
| **P12** | C5 policy engine + E5, E6, E7 (full economics) | P10 |
| **P13** | C1/C2/C6 BYOK gateway, end-to-end demo | P12 |
| **P14** | Governance chapter (budgets, personal-use classification) + write-up | P13 |

P11 is the highest-value phase. If time runs short, P11 > P12 > P13.

---

## 10. Threats to validity

| # | Threat | Severity | Mitigation |
|---|---|---|---|
| **T1** | `run_number` = 1 in all 325 rows. A learner trained on single samples fits noise as signal — and unlike selection routing, it **acts on** the failure signal, so a stochastic failure triggers a paid escalation. | **highest** | Public corpora have repeated sampling. If any collection budget exists, runs 2–3 on a subset buys more than anything else. |
| **T2** | No measured cost signal — all models free-tier, `estimated_cost_usd` = 0 everywhere. | high | D7: modelled prices + mandatory sensitivity sweep. |
| **T3** | 65 tasks → short, noisy learning curves. | high | Public corpora carry the training load; our benchmark is the held-out domain only. |
| **T4** | Docker down across ≥4 sessions. | medium | E1–E7 need no Docker. E8/E9 are strengthening only. |
| **T5** | Post-failure signal is weak on our data. Escalation success by signal bucket: 27% (no tests ran) vs 35% (partial), against 32% blind — barely separated, n=41. | medium | Design the router **pre-execution-first**. Report as a contrast with CodeRescue, which found post-failure signal strong on frontier models. |
| **T6** | 4 schema columns never populated: `eslint_warnings` 0/325, `compiler_errors` 0/325, `failure_mode` 0/325, `ts_any_count` 19/325. | medium | E9. This is the strongest concrete argument for getting Docker up — it unlocks a whole feature family. |
| **T7** | 63 zero-test rows (19%). | resolved-ish | Derived taxonomy: 10 defective-task, 10 truncated, 43 undifferentiable. Middle treatment (drop 20 artifactual) gives χ² p = 2.6e-4; RQ1 survives. Lead with the middle treatment, show all three. |
| **T8** | Classifier is 80% on paraphrases, not 100%. | low | Use 80% in all downstream error budgets. |
| **T9** | Public corpora are QA/reasoning, not code. | medium | That is precisely why E2 and E5 are separate experiments. |
| **T10** | Probe run spends the user's money on their account. | product | Bound it, disclose before running, cap it. Black-box only (D4). |

---

## 11. What carries over from Phases 1–7

**Kept and given a new job:**

| Asset | New role |
|---|---|
| 65-task web-dev benchmark + test suites + Docker | held-out cross-domain test (E5) **and** the probe instrument (C2) |
| `pipeline/` collection + DB + `analysis/` modules | unchanged infrastructure |
| Category classifier | C3 |
| **H4 refutation (+8.6% ceiling)** | the **motivation** for the whole redesign |
| **Fixed cascade result: 56.9% @ −10.8%** | the day-one baseline H-D must beat |
| `failure_taxonomy.py`, `zero_test_treatment` (on the unmerged branch) | data-quality handling; needs merging |
| `human_eval/` + `sampler.py` + Bradley-Terry | optional external-validity note in the discussion chapter |

**Demoted:**
- **H3** (automated ≈ human ranking) — no longer gates anything (D5). Nice to have.
- **Phase 8 as originally planned** (React "which model should I use" demo) —
  superseded. It had nothing to demonstrate at a +8.6% ceiling.

**Retired:**
- The selection recommender as a *product*. Its refutation is a finding; its
  implementation is not the deliverable.
- The uncross-validated **55.4%** routing figure. Held out properly it is 53.8% —
  a +0.0 gain. Must be corrected wherever it appears.

---

## 12. Open items

**Decide early:**
1. **Choose the describer model for E3** and fix its knowledge cutoff — the
   "outside knowledge" stratum, and therefore the entire novelty experiment,
   depends on being able to state it. This gates P11, the highest-value phase.
2. Confirm RouterBench / LLMRouterBench licences permit thesis use, and that the
   per-instance cost fields are present as documented.
3. Confirm Groq list rates for Llama-4-Scout and Qwen3-32B, or drop the †-marked
   rows from any published table (§4.4).
4. Decide whether E8 (live BYOK) is in scope for the thesis or is future work.

**Standing blockers, carried forward:**
- Docker not running. Gates E8, E9, and the API-013 / CSS-012 defective-task check.
- `pytest` is not in `pipeline/venv` — use a scratch venv, do not install into the
  project venv without asking.
- No pandoc/wkhtmltopdf: reports render Markdown → styled HTML → headless Chrome.
  `Research_Progress_Report_6.pdf` is **stale**; the `.md` is source of truth.
- Branch `worktree-phase6-7-data-sufficiency` is unmerged and unpushed.

---

## 13. References

**Prior art this project contests or builds on**
- GraphRouter: A Graph-based Router for LLM Selections (ICLR 2025) —
  https://arxiv.org/abs/2410.03834 · https://github.com/ulab-uiuc/GraphRouter
  *Generalises to new LLMs via LLM-generated descriptions. H-C contests this.*
- LLMRouter (open-source library) — https://github.com/ulab-uiuc/LLMRouter
- Router-R1 (NeurIPS'25) — https://github.com/ulab-uiuc/Router-R1
- RouteLLM — preference-data routing with a cost-quality threshold
- FrugalGPT — cascade-through-cheaper-first

**Benchmarks / corpora**
- RouterBench — https://arxiv.org/abs/2403.12031
- LLMRouterBench (Findings@ACL'26) — https://arxiv.org/html/2601.07206v1 ·
  https://github.com/ynulihao/LLMRouterBench
- TwinRouterBench — https://arxiv.org/abs/2605.18859
- RouterArena (ICLR 2026)

**The 2026 execution-aware turn**
- CodeRescue: Budget-Calibrated Recovery Routing for Coding Agents —
  https://arxiv.org/abs/2607.19338
  *Post-failure recovery routing with conformal risk control; exceeds
  always-escalate solve rate at 35% of its mean recovery cost. Our T5 finds the
  opposite signal strength on a cheap open pool — that contrast is reportable.*
- Agent-as-a-Router: Agentic Model Routing for Coding Tasks —
  https://arxiv.org/abs/2606.22902
- Agentic Routing: The Harness-Native Data Flywheel —
  https://arxiv.org/html/2607.11399
- SeqRoute: Global Budget-Aware Sequential LLM Routing via Offline RL —
  https://arxiv.org/pdf/2605.25424
- State of LLM Routers in 2026 — https://pakodas.substack.com/p/llm-routers

**Market / practitioner context**
- Enterprise AI gateways for LLM cost control (Bifrost, LiteLLM, Portkey,
  Cloudflare) — https://www.getmaxim.ai/articles/5-enterprise-ai-gateways-for-llm-cost-control-in-2026/
- AI cost optimization strategies 2026 — https://www.truefoundry.com/blog/ai-cost-optimization-strategies
- Best LLM routers and model routing platforms 2026 — https://www.braintrust.dev/articles/best-llm-routers-2026
- Groq per-model pricing 2026 — https://www.cloudzero.com/blog/groq-pricing/

**Internal**
- `Reports/Research_Progress_Report_6.md` — Phase 6 analysis, H4 refutation
- `Reports/Phase5_Data_Sufficiency_Assessment.md` — discordance matrix, zero-test
  taxonomy *(unmerged branch)*
- `.claude/handover/2026-08-02-project-understanding-and-findings.md`
- `.claude/handover/2026-08-02-phase7-and-open-data-issues.md`

---

## 14. Provenance of the numbers in this document

Every figure below was computed during the 2026-08-16 session against
`pipeline/exports/results_full.csv` (325 rows, 5 models × 65 tasks), using Groq
published list prices. Scripts are reproducible; prices are a modelling assumption
per D7.

| Figure | Source |
|---|---|
| Cascade 56.9% @ −10.8%; all chain economics | cascade sweep over all 1–5 length permutations |
| Oracle −78.8% and its 4-class decomposition | cheapest-solver-per-task with perfect foresight |
| Cheap-sufficiency by difficulty (80/30/8%) and category | per-cell pass rates |
| Unsolvable by difficulty (5/60/56%) | tasks where no model passes |
| Probe-size curve k = 3…40 | 200-trial resampling, held-out prediction + argmax routing |
| Post-failure signal weakness (27/35 vs 32% blind) | escalation success conditioned on `tests_passed/tests_total` |
| Column fill rates | non-empty non-zero count over 325 rows |
| Partial credit 128/325 (39%) | `0 < tests_passed/tests_total < 1` |

Pre-existing figures (+8.6% ceiling, Cramér's V 0.515, 80% classifier accuracy,
zero-test taxonomy) are carried from Phase 6 and the sufficiency assessment.
