# Findings: adversarial review of the router goal

_Written: 2026-08-18 · Reviews: `.claude/worktrees/phase8-goal-lock/.claude/handover/2026-08-18-router-goal-for-review.md`_
_Status: done_

> **Note:** this review added a one-line pointer to the brief inside the
> `phase8-goal-lock` worktree. That edit is uncommitted, and the worktree was not created
> by this session — the commit decision is yours.

Format per §13 of the brief: section attacked → survives / weakened / dead → evidence
→ replacement. Every number below is either quoted from a paper's abstract page or
computed in this session from `NPULH/LLMRouterBench` (1.28 GB tarball, 70,678
per-instance code rows). Nothing here is asserted from recall.

**Headline, three things:**

1. The claimed contribution (§2, D3, H-C) is **dead as stated** — the mechanism is
   published prior art in at least four papers, one of which the brief already cites.
2. The primary metric (§1, CPAT) is **degenerate**, and its headline claim is **false as
   written** — shown by direct computation, not argument.
3. **In the project's favour:** the cost prize on the target domain is roughly twice what
   §6 assumes (−33% to −50% CPAT), and it comes from *sufficiency*, not complementarity —
   so the data vindicates D1. §10's nesting claim survives; only the −25% ceiling goes.

---

## Verdict table

| § | Claim | Verdict |
|---|---|---|
| §2 / D3 / H-C | probe-measured descriptor for unseen-model transfer is the novelty | **DEAD** — published |
| §2 | the gap analysis itself | **DEAD** — method built the gap from 2 papers, missed ≥4 |
| §1 | "beats every single model in the pool" on CPAT | **FALSE as written** (computed) |
| §1 | CPAT as primary metric | **DEGENERATE** — selects the weakest model (computed) |
| §5 | simulation + sample-size tables | **UNREPRODUCIBLE** — exist only in the brief |
| §5 | the simulation's premise | **CONTRADICTS §6** |
| §5 | "the data exists" | **WEAKENED** — same zero-cost defect, ragged matrix |
| §9 #2 | 15-cell descriptor from a 12–15 task probe | **DEAD** — noise; off-the-shelf fix exists |
| §9 #6 | domain shift within code | **REAL, but it is a grader effect** — ρ flips +0.36 → −0.33 (computed) |
| §10 | "stronger models are more nested" | **HOLDS** — but the *cost* prize survives it (computed) |
| §6 | realistic headroom ≈ −25% cost | **UNDERSTATED** — −33% to −50% CPAT (computed) |
| §9 #5 | verification outside a benchmark | **SURVIVES**, and is worse than stated |
| §1 vs §9 #4 | weights vs metric | **INTERNALLY INCONSISTENT** |

---

## 1. §2 / D3 / H-C — the contribution is prior art. **DEAD.**

### The kill criterion, declared before searching

H-C dies if one paper does both: **(a)** builds a model descriptor from *measured*
performance on a probe/anchor/validation set, **and (b)** uses it to route to models
**unseen at policy-training time**.

### Four papers meet it or come within one step

**UniRoute — "Universal Model Routing for Efficient LLM Inference"**, Jitkrittum et al.,
[arXiv:2502.08773](https://arxiv.org/abs/2502.08773) (Feb 2025; later version
"Universal LLM Routing with **Correctness-Based Representation**", ICLR Workshop 2025).
Abstract, verbatim:

> "we consider the problem of dynamic routing, where new, previously unobserved LLMs
> are available at test time. We propose UniRoute, a new approach to this problem that
> relies on **representing each LLM as a feature vector, derived based on predictions on
> a set of representative prompts** … Experiments … show the effectiveness of UniRoute
> in routing amongst **more than 30 unseen LLMs**."

That is (a) + (b) verbatim. It is D3 and H-B, published 18 months before this brief.
From the HTML version: the representative set is a labelled validation set "of modest
size (e.g., O(10³))", and the paper explicitly claims it "seamlessly handles any
h ∈ H_all, **including one unobserved during training**".

**EmbedLLM**, [arXiv:2410.02223](https://arxiv.org/abs/2410.02223), **ICLR 2025
spotlight** — compact LLM embeddings learned from observed correctness on QA pairs,
evaluated on model routing and performance forecasting. Measured, not described.

**IRT-Router**, [arXiv:2506.01048](https://arxiv.org/abs/2506.01048), **ACL 2025 Main** —
Item Response Theory ability estimates from observed performance, with an explicit
cold-start evaluation.

**Router-R1**, [arXiv:2506.09033](https://arxiv.org/abs/2506.09033), **NeurIPS 2025** —
**the brief cites this paper in §12 and omits it from the §2 gap table.** Verbatim:

> "Router-R1 also conditions only on simple model descriptors such as pricing, latency,
> and **example performance**, enabling **strong generalization to unseen model selection**."

### The gap analysis method failed, not just its luck

§2 builds the hole from exactly two papers and promotes one paper's stated limitation to
a *field* limitation. The dueling-feedback paper (Oct 2025) saying it "cannot natively
handle entirely new LLMs" is a limitation **of that method**. UniRoute solved it eight
months earlier; EmbedLLM and Router-R1 shipped adjacent solutions at ICLR and NeurIPS.

Worse, the field has already *named the exact tradeoff* the brief claims to have found.
RouteProfile, [arXiv:2605.00180](https://arxiv.org/abs/2605.00180) (Apr–May 2026),
abstract, verbatim:

> "Existing profiles are limited: **LLM-generated descriptions are often coarse, while
> interaction-based embeddings are costly to construct.**"

§2's argument ("nobody measures the model") is not a hole. It is one of two branches the
literature already labels, and RouteProfile takes the *other* branch on purpose.

### What replaces it

The mechanism claim is gone. What I did **not** find (keyword search plus a citation sweep
of both GraphRouter and UniRoute — see §10 for the residual risk) is the **head-to-head at
a stated budget**:

> **H-C′:** measured descriptors vs LLM-generated descriptors, same router, same pool,
> as a function of (i) probe budget *k* and (ii) whether the model post-dates the
> describer's knowledge cutoff. Deliverable: the *break-even k* — the probe spend at
> which measurement overtakes free description, and the pool conditions under which it
> never does.

This reframes the contribution from **mechanism** ("measurement works") to **economics**
("measurement is worth paying for, and here is the price"). Supporting facts: UniRoute
uses O(10³) prompts and analyses no cost; RouteProfile asserts embeddings are "costly"
without quantifying it. Nobody has priced the branch point.

This is a smaller contribution — an empirical comparison, not a new mechanism. §11
already accepts "mechanism contribution, not a new paradigm"; H-C′ is one notch below
that. It is still defensible for a master's, and the negative result ("reputation is
free and good enough above k=N") remains publishable.

**Caveat in the project's favour:** UniRoute is a workshop paper, not main-conference.
That makes it weaker prior art than EmbedLLM (ICLR spotlight) or Router-R1 (NeurIPS) —
but all three must appear in §2's table before any novelty claim is made again.

---

## 2. §1 — CPAT is degenerate and the headline claim is false. **Computed.**

CPAT = total spend / tasks passing verification, with no accuracy constraint. Minimising
it selects **the cheapest model that is not at zero accuracy**, regardless of quality.

Measured on LiveCodeBench, 29 priced models, 1,055 shared instances:

| Ranked by CPAT (as §1 defines it) | accuracy | total cost | CPAT |
|---|---|---|---|
| **Llama-3.1-8B-Instruct** | **16.8%** | $0.016 | **$0.00009** |
| Llama-3.1-8B-UltraMedical | 13.4% | $0.018 | $0.00012 |
| gemma-2-9b-it | 17.3% | $0.025 | $0.00014 |
| … | | | |
| gpt-5 (highest accuracy) | 86.4% | $54.92 | $0.06022 |

The CPAT-optimal single model is a **16.8%-accuracy** model. The *clairvoyant oracle*
router is **+9,905% worse on CPAT** than that model. So:

> **"Beats every single model in that pool on cost per accepted result" (§1) is not
> merely hard — it is unachievable, and the metric that defines it prefers the worst
> model in the pool.**

H-D quietly repairs this ("at equal or better acceptance") but §1, the goal statement
itself, does not. Everything downstream inherits the unconstrained version.

**Replacement:** state the goal as **CPAT subject to an acceptance floor** — "cheaper per
accepted task than the best single model *at no worse acceptance than that model*". Under
that definition the numbers are strong (§4 below). Pick the floor explicitly: matched to
the best single model, or a user-set `min_acceptance`. Never quote unconstrained CPAT.

Second-order: §1 declares three weights (cost, latency, quality) and D6 makes the user set
them, but CPAT contains **no latency term at all**. The deliverable claim is stated in a
metric that ignores one of the three declared axes. Either drop latency from §1's framing
or make the primary metric weight-aware.

---

## 3. §5 — the feasibility evidence is unreproducible, and its premise contradicts §6.

**Unreproducible.** Grepping the whole repo for §5's figures (`70.5`, `74.3`, `78.1`,
`2,218`, `1,109`) returns **only the brief itself**. The four committed exploratory
scripts (`cascade_economics.py`, `oracle_decomposition.py`, `probe_size_curve.py`,
`feature_availability.py`) have a README stating they produced every new number in the
**§6/spec** set — the simulation table and the d/K sample-size table are not among them,
and appear nowhere in `docs/superpowers/specs/`. §6's numbers are backed by committed
code with an explicit caveats list; §5's are typeset identically and backed by nothing.

**Premise contradiction — the more serious half.** §5's simulation reports a learner
reaching 78.1% against a best-fixed model at 63.4%: **+14.6 points**. §6 measures, on the
project's real pool, an *oracle* selection ceiling of 58.5% against 53.8% best-single:
**+8.6 points, treatment-independent**, described in §10 as "the strongest result of the
earlier phase". A simulation whose learner beats best-fixed by 14.6pp is running in a
world with roughly **double the complementarity the project has actually measured**. The
simulator's generative process is assumed, and it assumes away the finding that motivated
the redesign.

**Replacement:** re-run the simulation with the complementarity structure calibrated to a
measured pool (§4 gives four), commit the script alongside the other four, and mark every
number in §5 with its provenance. Until then §5 cannot carry the feasibility argument.

---

## 4. §10 and §6 — nesting **holds**; the cost ceiling does not. This validates D1.

I first read this as a refutation of §10. That was wrong, and the correction matters
because it changes what the data is telling the project.

### Nesting holds on the target domain. §10 stands.

Oracle accuracy exceeding best-single accuracy is **not** evidence of complementarity: a
per-instance max over models absorbs every lucky single-run draw, and this corpus is
single-run, the same `run_number = 1` defect §9 charges against the project's own data.
The discriminating test is the **independence null** — if model successes were independent,
oracle accuracy would be `1 − Π(1−p_m)`. Nesting means the measured oracle sits *below* it:

| pool | best single | measured oracle | independence null | gap vs null |
|---|---|---|---|---|
| SWE-bench, 3 models | 41.6% | 49.0% | 63.5% | **−14.5pp** |
| SWE-bench, 5 models | 41.6% | 54.6% | 80.5% | **−25.9pp** |
| LiveCodeBench, 3 models | 86.4% | 89.8% | 98.6% | **−8.8pp** |
| LiveCodeBench, 5 models | 86.4% | 89.9% | 99.4% | **−9.5pp** |

Every pool falls far below independence. Model successes are strongly positively
correlated at frontier scale too. **§10's nesting claim survives; do not retract it.** The
+3.3pp accuracy gain on LiveCodeBench is thin enough that single-run noise could supply it
unaided.

Independent confirmation that single-shot labels mislead here: **DARS**,
[arXiv:2606.06924](https://arxiv.org/abs/2606.06924) (June 2026) — "single-shot
supervision provides only a noisy observation of a query-model pair's behavior rather than
a reliable capability estimate … single-shot labels can be misleading for model selection".

### But §6's −25% cost ceiling is understated. **Measured.**

The cost prize does not require complementarity at all — it comes from cheap models
*sufficing* on easy instances, which is exactly what D1 locked. Oracle = cheapest model
solving each instance, **paying the best model on instances nobody solves** (clairvoyant
abstention already removed, per §10):

| pool | best single | oracle acc | Δ acc | cost vs best | **CPAT vs best** |
|---|---|---|---|---|---|
| SWE-bench, 3 models | claude-opus-4.1 41.6% | 49.0% | +7.4pp | −21.5% | **−33.4%** |
| SWE-bench, 5 models | claude-opus-4.1 41.6% | 54.6% | +13.0pp | −34.8% | **−50.3%** |
| LiveCodeBench, 3 models | gpt-5 86.4% | 89.8% | +3.3pp | −36.2% | **−38.5%** |
| LiveCodeBench, 5 models | gpt-5 86.4% | 89.9% | +3.4pp | −37.3% | **−39.7%** |

**−33% to −50% CPAT**, against §6's −25% estimate, with unsolvable-by-any at 7.9% on
LiveCodeBench (29 models) versus 42% on the project's own benchmark.

**The headline reading:** nesting holds, so *selection* headroom stays small — §6 and §10
were right. The large prize is **sufficiency-routing cost**, and it is roughly twice what
the project assumed. That is D1's thesis, measured on the target domain at frontier scale.
This data validates the project's central design decision rather than contradicting it.

---

## 5. §9 #6 / #2 — the probe measures the grader as much as the model. **Computed.**

§9 #6 worries that a descriptor learned on QA may not transfer to code. The real problem
is one level down: **within code, a probe can rank models backwards — because it is partly
measuring the grader.**

Spearman rank correlation of per-model pass rate, 20 models present in all three:

| | ρ |
|---|---|
| HumanEval vs MBPP | **+0.835** |
| HumanEval vs LiveCodeBench | **−0.328** |
| MBPP vs LiveCodeBench | **−0.295** |

**Negative** — but the sign is a *grader* effect, not domain divergence, and that
distinction changes the conclusion. Excluding the nine reasoning/CoT-distill models
(OpenThinker3, DeepSeek-R1 variants, MiniCPM4.1, Nemotron-Nano-9B-v2, GLM-Z1, Qwen3-8B,
MiMo-7B, Intern-S1-mini), the correlations **flip positive** on the remaining 11:

| | all 20 models | 11 non-CoT models |
|---|---|---|
| HumanEval vs MBPP | +0.835 | +0.718 |
| HumanEval vs LiveCodeBench | **−0.328** | **+0.355** |
| MBPP vs LiveCodeBench | **−0.295** | **+0.264** |

So the mechanism is: HumanEval/MBPP grading penalises models that emit long chain-of-thought
before the answer, while LiveCodeBench extracts the code block. The probe does not measure
"coding ability"; it measures coding ability *convolved with output-format compliance under
one grader*. Examples from the 20-model table:

| model | HumanEval | MBPP | LiveCodeBench |
|---|---|---|---|
| Fin-R1 | 77.4% | 69.0% | **6.8%** |
| Qwen2.5-Coder-7B-Instruct | 77.4% | 76.0% | **27.9%** |
| MiMo-7B-RL-0530 | 52.4% | 56.8% | **6.4%** |
| DeepSeek-R1-0528-Qwen3-8B | 25.0% | 35.0% | **64.7%** |
| OpenThinker3-7B | **7.3%** | 11.8% | **44.2%** |

**That is precisely the point, and it is worse than plain domain shift.** A measured probe
inherits the artefacts of the probe harness as faithfully as it inherits capability — and a
BYOK user's arbitrary endpoint is exactly the case where format compliance diverges from
the probe harness's assumptions (D4 already anticipates quantized and self-hosted
endpoints). A 12–15 task probe on the wrong grader produces a confidently wrong descriptor,
and the failure is invisible because the descriptor still looks well-formed.

Two consequences:

1. **This is the strongest available argument *for* H-C′.** Model cards quote HumanEval.
   A describer reading a model card will rank Fin-R1 as a strong coder; it scores 6.8% on
   LiveCodeBench. Reputation is measurably wrong here. But the same result bounds the
   claim: measurement only beats reputation when the probe harness matches deployment, and
   the brief has no mechanism for checking that it does. **Probe-harness match should be an
   explicit assumption of H-C′, and ideally a measured covariate.**
2. **Probe-set selection is the experiment, not a hyperparameter.** §9 #2 asks whether
   probe tasks should be *selected* rather than sampled. The answer is yes, and it is
   solved off the shelf: `tinyBenchmarks` ([arXiv:2402.14992](https://arxiv.org/abs/2402.14992))
   estimates MMLU to ~2% average error from **100 IRT-selected anchor items**;
   IRT-Router applies the same machinery to routing directly.

### The 15-cell descriptor is noise, and it breaks §5's own budget

5 categories × 3 difficulties = 15 cells from a 12–15 task probe is **~1 sample per
cell**. A per-cell rate from n=1 is a Bernoulli draw with variance 0.25 — the descriptor
is mostly noise, which is a sufficient explanation for §6's finding that
"argmax-over-descriptor never beats best-single at any k = 3…40".

It also violates the brief's own convergence budget: §5's table says d=8 needs ~555
rounds and d=32 needs ~2,218. The descriptor alone is 15 dimensions **before any task
features**. Either the descriptor shrinks to ~1–2 parameters (an IRT ability estimate) or
§5's round budget triples and the "converges in 500 rounds" claim goes with it.

**Replacement for D3's representation:** a 1–2 parameter IRT ability estimate over
IRT-selected anchor items, not a 15-cell pass-rate table over sampled ones. This fixes
the noise problem, the dimension problem, and §9 #2 in one move — and it makes H-C′'s
"break-even k" question sharper, because IRT gives a principled item-count/error curve.

---

## 6. §5 "the data exists" — **WEAKENED**. Three defects, measured.

I downloaded and audited the corpus (`NPULH/LLMRouterBench`, `bench-release.tar.gz`,
1.28 GB, last modified 2025-12-08, 667 downloads).

**Good news first.** The per-instance schema is exactly as §5 claims —
`index, origin_query, prompt, prompt_tokens, completion_tokens, cost, score, prediction,
ground_truth, raw_output`. Data quality is high: **1 null score in 70,678 code rows**.
Code coverage is *richer* than the brief states — not just HumanEval but `mbpp`,
`livecodebench`, `swe-bench/verified`, plus `arenahard_coding`.

**Defect 1 — the zero-cost problem is not solved, it is inherited.** §6's headline caveat
is that the project's own benchmark had `estimated_cost_usd = 0` in all 325 rows, forcing
modelled prices. In LLMRouterBench: **9 of 22 models on HumanEval, 9 of 20 on MBPP, and 9
of 38 on LiveCodeBench have total cost exactly 0.0.** CPAT is undefined for those models
(cost 0 → CPAT 0 → they win trivially). M0/M1 will need modelled prices *again*, on ~41%
of the small-model pool, reproducing the exact weakness §11 lists as unresolved. Only
`swe-bench/verified` is fully priced (0/15 zero-cost) — which is an argument for making
SWE-bench the primary economic corpus.

**Defect 2 — the model × dataset matrix is ragged and structurally partitioned.** Across
the four code subsets: 39 distinct models, 60.9% density, and **zero models present in all
four**. HumanEval/MBPP are ~8B-class models; SWE-bench is exclusively frontier. There is
no model on both. Leave-one-model-out transfer (H-B, M3) needs a dense-ish matrix within a
comparable pool; the corpus supplies **two disjoint pools**, not one.

**Defect 3 — no HF dataset viewer.** `datasets-server.huggingface.co/splits` returns
*"The split names could not be parsed from the dataset config"*. The repo holds a single
1.28 GB tarball plus a `download.md`; there is no parquet, no loading script, no splits.
Every consumer writes their own loader. This directly supports §9 #9's worry that M0 is
30–40% of effort — the estimate looks **right, possibly low**.

**Also note:** §5 claims the corpus contains "GPT-5, Claude-4, Gemini-2.5-Pro,
DeepSeek-V3.1 and NVIDIA-Nemotron", which is what makes H-C's post-cutoff condition
constructible. True — but they are concentrated in `livecodebench` and `swe-bench`, i.e.
the frontier pool. The H-C′ post-cutoff arm and the cheap-model arm cannot be run on the
same pool. Design around that explicitly.

---

## 7. §9 #4 — binarisation biases *toward* the hypothesis. **Weakened, in the dangerous direction.**

The brief notes 39% of prior-benchmark runs (128/325) had partial test credit and were
binarised to "fail". The framing in §9 #4 is neutral ("is a graded metric better?"). It is
not neutral. Binarising treats a cheap model that passes 80% of tests identically to one
that passes 0%. That **understates cheap-model sufficiency**, which **inflates measured
escalation benefit** — the quantity H-A and H-D exist to demonstrate.

**The corpus cannot support the check.** I audited the `score` field: across `humaneval`,
`mbpp` and `livecodebench` the only distinct values are **{0.0, 1.0}** — strictly binary,
no partial credit anywhere. So LLMRouterBench's code subsets reproduce the exact
binarisation the project already flagged as a threat, and the graded-CPAT robustness check
is **not constructible from this corpus**. Either re-run a subset with a graded harness
(the project's own benchmark already produces partial credit — 128/325 rows), or state
explicitly that binary-only is a limitation inherited from the data and quantify the
direction of the bias.

---

## 8. §9 #5 — verification outside the benchmark. **Survives, and the tension is sharper than stated.**

D5 requires every learning signal to be automated. §7 sells an OpenAI-compatible proxy —
"any existing client works by changing one environment variable". Those two do not
compose: on an arbitrary user prompt through a generic proxy there is no test suite, so
`LiveSource.outcome()` has nothing to verify against and the learning loop is inert
exactly where the product claims generality.

The current answer ("scope the evaluation to where tests exist and say so") is fine for
the *research*. It is not sufficient for the *product* claim as written in §7.

**Replacement:** narrow the product surface to agentic coding with an existing test command
— `smartroute wrap -- pytest` is already the right shape, and it is where CodeRescue and
Claude-Code-style harnesses live. Sell it as a router for *verifiable* workloads, and say
so on the tin. That is a smaller claim that is actually true, and it costs nothing
research-side because §8's `OutcomeSource` seam already assumes it.

---

## 9. §9 #8 — the learned router's real margin is never computed against its own baseline

§8 makes M1 a go/no-go against "≥70% of the attainable **oracle** gap". But §6 names the
day-one baseline as the *fixed cascade*, not the oracle. Recomputing from §6's own
figures: cascade 56.9% accepted at −10.8% cost vs best-single 53.8% ⇒
CPAT ratio = 0.892 × (53.8/56.9) = **0.843, i.e. −15.7% CPAT** on day one, from a rule
with no learning in it. Against §6's realistic −25% ceiling, the learned router's margin
over the fixed cascade is roughly **−11% relative** — computed from modelled prices, two
of five unconfirmed, at `run_number = 1` with no variance estimate.

**A −11% relative margin cannot be resolved by data with no variance estimate and
unverified prices.** M1's gate must therefore be stated against the cascade, not the
oracle, and the D7 price-sensitivity sweep must run *before* M1 can function as a gate
rather than after. As currently specified, M1 will pass or fail on noise.

The §4 numbers offer the way out: on properly-priced frontier pools the oracle margin is
−33% to −50%, comfortably outside price-model noise. Run M1 there.

---

## 10. Residual novelty risk — the citation sweep, and what still needs checking

I pulled the citation lists for GraphRouter (100 citing papers) and UniRoute (86 citing
papers) from Semantic Scholar. The field has moved very fast, and several titles sit
directly on H-C′ or on the fixes recommended below. **I have titles and years only — not
abstracts — for these. Every one must be read before H-C′ is committed to.** In rough
order of threat:

| Paper (2026) | Why it threatens |
|---|---|
| Routing with Generated Data: Annotation-Free LLM Skill Estimation and Expert Selection | skill estimation without labels — the cheap-descriptor branch |
| From Sampled Outcomes to Capability Distributions (DARS, [2606.06924](https://arxiv.org/abs/2606.06924)) | **read** — single-shot capability labels are noisy; supports the n=1 critique |
| MonoRouter: Enforcing Monotonicity in Item Response Theory for LLM Routing | IRT descriptor for routing — i.e. recommendation #3 below |
| Capturing LLM Capabilities via Evidence-Calibrated Query Clustering | measured capability representation |
| FlyRoute: Self-Evolving Agent Profiling via Data Flywheel | profile refined by traffic — i.e. §9 #7's hybrid |
| The Routing Plateau: Understanding and Breaking the Accuracy Limits of LLM Routers | likely generalises §6's +8.6% ceiling |
| Opportunity Is Not Realizability: Selection-Valid Diagnostics for Multi-LLM Routing | likely formalises §6's oracle decomposition |
| RouteGuard: Certifying Routing Gain … When Complementarity Is Not Enough | the nesting result, formalised |
| Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades | the cascade-economics baseline |
| SWE-Router: Routing in Multi-turn Agentic Software Engineering Tasks | the exact target domain |
| LLMRouter / RouteJudge / SEAR (gateways, platforms) | the §7 product form |

Two implications the brief should absorb:

1. **Several of the project's own prior-phase negative results appear to have been
   independently published** (routing plateau, oracle-vs-realizable gap, complementarity
   limits). §10 calls the +8.6% ceiling "the strongest result of the earlier phase" — it
   may now be a *confirmation* rather than a novel finding. Check before writing it up as
   the headline.
2. **The recommended fixes below are off-the-shelf, and that is fine.** MonoRouter and
   IRT-Router mean the descriptor fix is adopt-don't-invent. Use them and spend the novelty
   budget on the break-even economics, which is still the thinnest-covered question.

## 11. Smaller findings

- **§9 #3 (the A/B split) survives.** Hierarchical partial pooling is the defensible
  middle, but D6 is correct for master's scope. Not worth the complexity. The real defect
  in this area is the §1-vs-CPAT weight inconsistency in §2 above.
- **§9 #7's hybrid is closer to the literature than the brief thinks.** Router-R1 already
  conditions on descriptors including example performance and generalises to unseen
  models. A "generated prior, refined by traffic" design is not open ground — but the
  *break-even analysis* between the two remains open, and that is H-C′.
- **§12's internal pointer is stale from the main checkout.** `pipeline/analysis/exploratory/`
  exists only in `.claude/worktrees/phase8-goal-lock/`; the main working tree has no such
  directory. Anyone handed this brief and pointed at the repo will not find the scripts.
- **RouteProfile should be added to §12 and §2.** It is the closest thing to a direct
  competitor for the reframed H-C′ and it post-dates every reference currently in the
  brief.

---

## 12. What to do

1. **Rewrite §2 before anything else.** Add UniRoute, EmbedLLM, IRT-Router, Router-R1 and
   RouteProfile to the gap table. Restate the contribution as H-C′ (break-even probe
   economics + the measured-vs-generated head-to-head), or abandon the novelty claim and
   reposition as a systems/replication contribution.
2. **Fix §1's metric.** CPAT subject to an acceptance floor. Add latency or drop it from
   the goal statement.
3. **Replace the descriptor** with an IRT ability estimate over selected anchor items.
   Cite tinyBenchmarks and IRT-Router; this also answers §9 #2 and repairs the d-vs-rounds
   contradiction.
4. **Move the primary corpus to SWE-bench Verified + LiveCodeBench**, not HumanEval.
   SWE-bench Verified is the only fully-priced code subset (0/15 zero-cost rows), it is a
   frontier pool, and the grader result in §5 makes HumanEval actively misleading as a
   probe set for anything agentic.
5. **Keep §10's nesting claim; retract only §6's −25% cost ceiling.** Nesting holds against
   the independence null on frontier pools. The cost prize is −33% to −50% CPAT and comes
   from sufficiency, not complementarity — which is D1 vindicated, not contradicted.
6. **Re-run or retire §5's simulation.** Calibrate it to a measured pool, commit the
   script, and mark provenance per claim.
7. **Restate M1's gate against the fixed cascade**, and run the D7 price sweep first.
8. **Read the §10 checklist before committing to H-C′.** Eleven 2026 papers sit on or next
   to it, and several of the project's prior-phase negative results may already be
   published — which changes whether they are headline findings or confirmations.

---

## 13. Reproducing the computed numbers

All measurements in §2, §4, §5, §6 come from:

```
https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz
tar xzf bench.tar.gz bench-release/{humaneval,mbpp,livecodebench,swe-bench}
```

Each `<dataset>/<split>/<model>/*.json` holds `performance`, `cost`, `counts`, and a
`records` list with per-instance `index / score / cost / prompt_tokens /
completion_tokens`. Pass rates are `mean(score >= 1.0)`; oracle is the cheapest model
solving each instance, charging the best model on instances no model solves; CPAT is
`total cost / instances accepted`. Models with total cost 0.0 are excluded from every
economic figure.
