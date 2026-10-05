# Handover: the router goal, for adversarial review

_Written: 2026-08-18 · Scope: the project goal and its supporting claims, packaged so multiple agents can attack it independently_
_Status: open — this is a review brief, not a task brief_

**Read this if you are one of several agents grilling this idea.** It is
deliberately self-contained: you should not need the repo, the earlier phases, or
the prior conversation. Everything you need to attack is below.

**Your job is to find what is wrong with this**, not to improve the prose. See
§9 for the specific attacks that would be most valuable, and §10 for what has
already been settled and should not be re-litigated unless you have new evidence.

---

## 1. The goal

> Build an LLM router that takes a pool of models **the user supplies**, routes
> each request under the user's **stated weights** for cost, latency and quality,
> **learns its routing policy from automatic outcome feedback**, and beats
> **every single model in that pool** on cost per accepted result.

Context: master's major project. Domain is code generation. The user is a
developer who has API keys for several models and wants the cheapest one that
actually solves each task.

### The four components

| | Component | Meaning |
|---|---|---|
| **BYOK** | user supplies the pool | any models, any providers, their own keys |
| **Weights** | user states cost/latency/quality priorities | the router optimises *under* them; it does not infer them |
| **Learning** | policy improves from automatic feedback | tests, compile/typecheck, implicit signals — never surveys |
| **Beat the pool** | cheaper per accepted result than any single member | the deliverable claim |

### Primary metric

**Cost per accepted task (CPAT)** = total spend / tasks passing verification.
Not cost per token. Not raw pass rate. Cheap inference that fails and gets
retried is not cheap.

---

## 2. The claimed contribution

> A **probe-measured model descriptor** that lets a learned routing policy
> transfer to a model it has **never seen** — cold-starting BYOK at bounded,
> disclosed cost instead of making the user pay to train the router from zero.

### The gap this sits in

Two closest systems, two different failures:

| System | Learns from feedback | Handles unseen models | How it knows a model |
|---|---|---|---|
| RouteLLM | yes, preferences | no | trained per-pool |
| **LLM Routing with Dueling Feedback** (arXiv 2510.00841) | yes, pairwise dueling bandits | **no — stated limitation** | pre-computed embeddings |
| **GraphRouter** (ICLR 2025, arXiv 2410.03834) | yes | yes, but | **GPT-4o writes a text description**, embedded with BERT |
| **This project** | yes | yes | **bounded probe run, measured** |

Verbatim from the dueling-feedback paper's limitations:

> *"The framework cannot natively handle entirely new LLMs unseen during
> training. Embeddings are pre-computed for tested models; new models would
> require retraining or embedding adaptation."*

Its named future work includes *"reducing the number of pairwise comparisons
needed during deployment"* — which is the same problem as BYOK cold-start.

GraphRouter's mechanism, verbatim:

> *"utilizes a generative LLM such as GPT-4o to generate a descriptive text for
> each LLM, outlining key details such as its strengths, token pricing, and
> context length"*

That is **reputation, not measurement**. The claimed failure modes — all of them
exactly the BYOK case:

1. a user's **private fine-tune** — no describer has heard of it
2. a model **released after** the describer's knowledge cutoff
3. a **self-hosted or quantized** endpoint that behaves unlike its model card
4. **two models with similar descriptions** but different real behaviour

**Nobody measures the model.** That is the claimed hole.

---

## 3. Hypotheses

| | Claim | Target |
|---|---|---|
| **H-A** | Task features + measured descriptor predict whether a cheap model suffices | held-out AUC ≥ 0.70 |
| **H-B** | Policy transfers to unseen models given only a k-task probe descriptor | ≥80% of in-pool **CPAT reduction** retained |
| **H-C** | **Measured descriptors beat LLM-generated ones, and the gap widens on models the describer cannot know** | **the novelty** |
| **H-D** | Router beats best-single and fixed cascade | on CPAT, at equal or better acceptance |
| **H-E** | Predicted-unsolvable abstention is useful | flags out-of-pool tasks; reduces spend, does *not* raise pass rate |

H-C is the contribution. H-A/B/D/E are support.

---

## 4. Design decisions already fixed

These were argued through and settled. Challenge them only with new evidence.

- **D1 — Sufficiency, not selection.** The router asks "is the cheap model good
  enough?", not "which model is best?" Selection routing is refuted on nested
  pools (see §6).
- **D2 — Never learn model identity.** Learn `f(task_features, model_descriptor)
  → outcome`, never `f(task) → model_id`. This is the *only* reason BYOK is
  possible; an identity-keyed policy cannot accept a model it wasn't trained on.
- **D3 — Descriptors are measured, not described.** From executing a bounded
  probe set. This is the novelty.
- **D4 — Black-box only.** Computable from ordinary chat-completion responses.
  No logprobs, no hidden states. An arbitrary OpenAI-compatible endpoint may
  expose none of them.
- **D5 — Every learning signal is automated.** No hypothesis may depend on
  recruiting human evaluators. *Empirical basis:* a previous phase of this
  project built a complete human-voting apparatus — interface, Bradley-Terry
  estimator, sampling design, protocol — and collected **zero votes** across
  multiple sessions. Human-gated designs did not ship.
- **D6 — Constraints are stated, not learned.** The user sets `w_cost`,
  `w_latency`, `w_quality`. Per-user *preference* learning is out of scope: it
  needs hundreds of judgements from one individual before it beats simply
  asking them. The router learns the **routing policy** globally; the user
  supplies the **weights**.
- **D9 — Two research components; everything else is thin.** Probe profiling and
  the sufficiency predictor carry the contribution. Ingress, policy engine,
  budgets, failover are commodity — LiteLLM/Bifrost/Portkey ship them. Demo-grade
  only.

### The (A)/(B) split — the single most important distinction

```
(A) POLICY LEARNING              (B) PREFERENCE LEARNING
    learns: which model              learns: how much THIS user
    suits which kind of input        weights cost vs speed vs quality
    from:   pooled feedback          from:   that one user's feedback
    needs:  many events              needs:  100s of judgements per person
            (transferable)                   (cold start, no data)
    -> IN SCOPE                      -> OUT OF SCOPE (D6)
```

"The router learns what the user likes" is ambiguous between these. This project
means **(A)**. The dueling-feedback paper independently makes the same cut: its
routing is *global per-query, not per-user personalisation*.

---

## 5. Evidence for feasibility

### Improvement over time is measured in rounds, not users

Simulation, online learner vs. best fixed model, 30 seeds:

```
rounds     learner   best-fixed   delta
   200      70.5%      64.0%      +6.5%
   500      74.3%      63.8%     +10.6%
  1000      76.0%      63.4%     +12.5%
  2000      77.1%      63.4%     +13.7%
  5000      78.1%      63.4%     +14.6%
```

Most gain lands by ~500 rounds. The dueling-feedback paper reports
*"meaningful performance stabilization around 300–500 queries"* with 4–5 models —
two different methods agreeing on the order of magnitude.

**Rounds needed for a stable learning curve**, by feature dimension d and pool
size K:

| d | K=3 | K=5 |
|---|---|---|
| 8 | 555 | 717 |
| 16 | 1,109 | 1,433 |
| 32 | 2,218 | 2,867 |

Implication: **keep d small.** 8 features converges in a third the rounds of 32.
Argues for tabular logistic regression over high-dimensional embeddings.

**Statistical detection** (paired, policy-at-0 vs policy-at-T on the same eval
tasks): +10% needs 186 tasks, +5% needs 373, +3% needs 621, +2% needs 932.
Public corpora supply tens of thousands of instances, so all of this is free.

**Human validation (optional layer only).** To detect Spearman ρ = 0.7 against
null needs ~13 paired comparisons; ρ = 0.5 needs ~29. Recommended design:
**5–8 evaluators × 40 pairwise comparisons ≈ 250 judgements** — heavily
over-powered, also yields inter-rater agreement. This *validates*, it does not
*gate*: with zero evaluators every hypothesis still stands.

### The data exists

`NPULH/LLMRouterBench` (Findings@ACL 2026) ships per-instance:
`origin_query`, `prompt`, `prediction`, `ground_truth`, `score`,
`prompt_tokens`, `completion_tokens`, `cost`. 33 models, 21+ datasets including
HumanEval. That is exactly the schema CPAT needs.

RouterBench (11 models × 7 tasks, 405k inferences) is secondary — its
per-instance token fields were **not confirmed**.

**H-C's hard condition is constructible from public data**: the corpus includes
GPT-5, Claude-4, Gemini-2.5-Pro, DeepSeek-V3.1 and NVIDIA-Nemotron — all
post-dating a GPT-4o-class describer's cutoff. So "models the describer cannot
know" needs no fine-tuning and no extra spend.

---

## 6. Where this came from — the prior negative result

This project previously tested *selection* routing on a 5-model, 65-task
web-development benchmark with authored test suites. Findings that reshaped it:

- **Selection routing is refuted and was unreachable.** Oracle selection ceiling
  is 58.5% vs 53.8% for always-best-single — a **+8.6% maximum** for *any*
  router. The pre-registered hypothesis required >15%. Only 3 of 65 tasks are
  winnable by routing away from the single best model.
- **The pool is nested, not complementary.** Every weaker model's successes are
  a near-subset of the strongest model's. Nesting is fatal for selection.
- **But nesting *feeds* cascade.** Fixed cascade `Llama-3.1-8B → GPT-OSS-120B`
  scores **56.9% at −10.8% cost** — beating always-best-single on *both* axes.
  This is the day-one result the learned router must beat.
- **The raw oracle is misleading.** "Cheapest model that solves it" is −78.8%
  cost, but **68.4% of that saving is clairvoyant abstention** on the 27 of 65
  tasks (42%) no model solves. Realistic capturable headroom is **≈ −25%**.
- **Argmax-over-descriptor never beats best-single**, at any probe size k=3…40
  (flat ~52% vs 53.8%), even while descriptor accuracy climbs 63.8% → 74.4%.
  This is why D1 reframes to *thresholding sufficiency* rather than picking a
  winner.
- **Cheap-sufficiency is strongly predictable**: 80% on easy / 30% medium / 8%
  hard; 69% bugfix / 23% frontend. This is the gradient H-A exploits.
- **Post-failure signal was weak here**: escalation success 27% (no tests ran)
  vs 35% (partial credit) against 32% blind, n=41. So the design is
  **pre-execution-first**. Note this *inverts* CodeRescue (arXiv 2607.19338),
  which found post-failure recovery signal strong on frontier models — the
  contrast is itself reportable.

Caveat on all cost figures above: that benchmark's models were free-tier, so
`estimated_cost_usd` was 0 in all 325 rows and prices are **modelled** from
published list rates. Also `run_number` = 1 everywhere — no variance estimate.

---

## 7. Product form

An **OpenAI-compatible proxy**, so any existing client works by changing one
environment variable (`OPENAI_BASE_URL`, or `ANTHROPIC_BASE_URL` for Claude
Code). Layered as: library core → CLI → server → dashboard. Provider adapters
come from **LiteLLM**; they are not rebuilt.

Two commands are the product surface, and both map onto research claims:

- `smartroute add-model` → runs the probe → writes a measured descriptor (H-C)
- `smartroute feedback` / `smartroute wrap -- pytest` → closes the loop (learning)

Reference point: `llmrouter-lib` (ulab-uiuc) already ships
`train` / `infer` / `chat` / `serve`. It has **no** probe-based onboarding and
**no** online feedback command. Those two absences are the product differentiator.

Every response carries a `_smartroute` block naming the chosen model, the reason,
whether it escalated, the realised cost, and the counterfactual cost. Rationale:
users do not trust an opaque system spending their money.

---

## 8. Build order

| | Milestone | Output | Kills which risk |
|---|---|---|---|
| M0 | Canonical schema + corpus loaders + `OutcomeSource` protocol | oracle ceilings per pool | schema wrong |
| **M1** | **Metric harness + trivial baselines** | **go/no-go** | building a learner nobody needs |
| M2 | Sufficiency predictor, in-pool | H-A | no signal exists |
| M3 | Descriptors + leave-one-model-out | H-B | transfer fails |
| **M4** | **Measured vs generated descriptors** | **H-C** | someone else owns it |
| M5 | Policy engine + full economics | H-D, H-E | doesn't pay |
| M6 | Library → CLI → proxy → dashboard | the product | — |

**M1 is a stated go/no-go**: if fixed rules on 2–3 features capture ≥70% of the
attainable oracle gap, the learned predictor is a footnote and M2–M5 get rescoped
around *why* simple rules suffice. That is a legitimate finding, and declaring the
threshold in advance is what stops sunk-cost momentum.

**If time dies at M4 there is still a thesis.** That is the point of the order.

Key architectural seam:

```python
class OutcomeSource(Protocol):
    def outcome(self, task: Task, model: ModelId) -> Outcome: ...
    # Outcome(passed, score, tokens_in, tokens_out, latency_ms, cost_usd)

ReplaySource   # dict lookup into a fixed matrix — for research
LiveSource     # call endpoint, THEN VERIFY, emit the same Outcome — for product
```

The verifier lives *inside* `LiveSource`. Everything above consumes
`OutcomeSource` and never knows which it has.

---

## 9. Attack this — highest-value challenges

Ranked by how much damage they would do if correct.

1. **Is H-C actually novel?** Find prior work that measures model capability by
   probing and uses it for routing transfer. Adjacent literature to check:
   model/LLM fingerprinting, capability elicitation, task-vector and
   model-selection-for-transfer work in classical ML, meta-learning "dataset2vec"
   style descriptors, and the AutoML cold-start literature. **This is the single
   most valuable check.** If someone owns it, the contribution collapses.
2. **Is the probe descriptor even the right representation?** Per-(category ×
   difficulty) pass rates give 15 cells for 5 categories × 3 difficulties; a
   12–15 task probe gives ~1 sample per cell, which is very noisy. Is there a
   better low-dimensional descriptor? Should probe tasks be *selected* for
   discriminative power rather than sampled?
3. **Does the (A)/(B) split hold under scrutiny?** Is there a defensible middle
   — e.g. hierarchical/partial pooling, where a per-user prior shrinks toward a
   global policy? That would recover some personalisation without the cold-start
   problem. Is it worth the complexity for a master's scope?
4. **Is CPAT the right primary metric?** It ignores partial credit, and it
   assumes verification is binary. 39% of runs in the earlier benchmark had
   *partial* test credit (1–99% passing) and were binarised to "fail". Is a
   graded metric better? Does that break comparability with the literature?
5. **Verification outside a benchmark is the weakest link.** For an arbitrary
   user prompt there is no test suite, so the feedback ladder degrades to
   compile/lint. Is the learning loop honest for general use, or only inside
   executable domains? The current answer is "scope the evaluation to where
   tests exist and say so" — is that sufficient, or fatal to the product claim?
6. **Domain shift.** Public corpora are QA/reasoning-heavy; the target domain is
   code generation. Does a descriptor learned on QA transfer to code at all?
   The plan separates pool shift (E2) from domain shift (E5) precisely so a
   failure is attributable — is that separation adequate?
7. **Probe cost vs value.** The probe spends the user's money on their own key.
   At what k does it stop paying for itself? Is there a cheaper cold-start —
   e.g. hybrid: start from a *generated* descriptor, refine with observed traffic,
   never run an explicit probe? (This would weaken H-C's framing but might be
   the better product.)
8. **Is "beat every model in the pool" achievable or trivially false?** On a
   nested pool the best single model is very strong. The earlier benchmark's
   fixed cascade beat it on both axes — but that is n=1 pool, single-run data.
   Does the claim survive on a pool where the strongest model dominates more?
9. **Scope realism.** M0 alone (canonicalizing public corpora) is estimated at
   30–40% of total effort and is invisible in the write-up. Is M0–M6 achievable
   in a master's timeline, and which milestone should be cut first if not?

---

## 10. Settled — do not re-litigate without new evidence

- **Do not swap in stronger models to rescue selection routing.** Stronger models
  are *more* nested, not less; a better model solves a superset of what weaker
  ones solve, which shrinks selection headroom further. Complementarity ≠ quality.
- **Human-gated hypotheses are banned (D5).** Empirically grounded — see §4.
- **Per-user preference learning is out (D6).** Cold start with no data.
- **Commodity gateway features are not contributions (D9).** Budgets, quotas,
  failover, unified API all ship in LiteLLM/Bifrost/Portkey.
- **Never quote the oracle undecomposed.** 68.4% of the raw −78.8% is clairvoyant
  abstention. Realistic target ≈ −25%.
- **The prior H4 refutation is the motivation, not damage.** The +8.6% ceiling is
  treatment-independent and is the strongest result of the earlier phase.

---

## 11. Known weaknesses — already acknowledged, no need to rediscover

- It is a **mechanism contribution, not a new paradigm** — one component replaced
  in an established architecture, shown to transfer better. Appropriate for a
  master's; would be thin for a PhD.
- **If probe descriptors do not beat generated ones, H-C fails** — though
  "reputation is sufficient, measurement isn't worth the probe cost" is a
  publishable negative that saves other people money.
- **Someone may publish this within 12 months.** The gap is visible to anyone
  reading both papers. Speed matters; M4 is where the novelty lives.
- All prior-phase cost figures rest on **modelled prices** and **single runs**
  (`run_number` = 1). Two of the five list prices used were never confirmed.

---

## 12. References

**Contested prior art**
- GraphRouter: A Graph-based Router for LLM Selections — ICLR 2025 —
  https://arxiv.org/abs/2410.03834 · https://github.com/ulab-uiuc/GraphRouter
- LLM Routing with Dueling Feedback — https://arxiv.org/abs/2510.00841
- LLMRouter library — https://github.com/ulab-uiuc/LLMRouter (`pip install llmrouter-lib`)
- Router-R1 — NeurIPS 2025 — https://github.com/ulab-uiuc/Router-R1
- RouteLLM; FrugalGPT

**Corpora**
- LLMRouterBench — Findings@ACL 2026 — https://arxiv.org/html/2601.07206v1 ·
  https://github.com/ynulihao/LLMRouterBench · HF `NPULH/LLMRouterBench`
- RouterBench — https://arxiv.org/abs/2403.12031
- TwinRouterBench — https://arxiv.org/abs/2605.18859

**Execution-aware routing (2026)**
- CodeRescue: Budget-Calibrated Recovery Routing for Coding Agents —
  https://arxiv.org/abs/2607.19338
- Agent-as-a-Router — https://arxiv.org/abs/2606.22902
- State of LLM Routers in 2026 — https://pakodas.substack.com/p/llm-routers

**Internal**
- `docs/superpowers/specs/2026-08-16-adaptive-router-design.md` — full charter
  (decisions D1–D10, experiments E1–E9, threats T1–T10). **Note: written before
  the preference-learning framing was settled; §1–§4 here supersede its §4.**
- `.claude/handover/2026-08-16-goal-lock-outcome.md` — how the direction was reached
- `pipeline/analysis/exploratory/` — scripts reproducing every §6 number

---

## 13. For the agent writing back

If you are reporting findings on this brief, say explicitly:

1. **Which section you are attacking** (§ number).
2. **Whether the claim survives, is weakened, or is dead** — and on what evidence.
3. **If dead: what replaces it.** A refutation without an alternative direction
   is half a contribution.
4. **Distinguish measured from asserted.** Cite a paper, a dataset field, or a
   computation. "This seems unlikely" is not a finding.
