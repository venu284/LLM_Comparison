# Phase 7 Human Evaluation Protocol

**Project:** Multi-LLM Comparison Platform (CSCI 7200)
**Purpose:** Test H3 — automated test-driven rankings correlate with human code-quality preference at Spearman ρ > 0.7.
**Status:** Designed and instrumented. No human votes collected as of 2026-07-31.

## 1. What This Tests

Phase 5 ranked five models by automated test pass rate. That ranking measures
functional correctness only. It says nothing about readability, idiomatic style,
or maintainability — the things a reviewer actually weighs when choosing between
two working solutions.

H3 asks whether the cheap automated signal is a usable proxy for the expensive
human one. If ρ > 0.7, automated pass rate can stand in for human judgment in
the recommendation algorithm. If it does not, the recommender needs a human
preference term and the Phase 6 results understate what is missing.

The null result is informative either way and will be reported as found.

## 2. Design

Blind pairwise comparison, the design used by Chatbot Arena and WebDev Arena,
fitted with the Bradley-Terry model.

Pairwise comparison is used rather than absolute 1–5 rating because absolute
scales drift between raters and compress toward the middle. Asking "which of
these two is better" is a question people answer consistently.

**Blinding.** Model identity is never sent to the browser. The server holds the
mapping from an opaque comparison id to the two models and resolves it only when
the vote is posted. Left/right assignment is randomized independently per
comparison, so position bias cannot attach to any model.

**Unit of comparison.** One task, two models' extracted code, side by side. The
task prompt is shown so the reviewer knows what was asked.

## 3. Participants

**Target:** 8–12 evaluators.

**Inclusion criteria:**
- At least one year of professional or academic web development experience
- Working familiarity with React, Express, and TypeScript — the stack the
  benchmark is written in
- Not previously exposed to this project's Phase 5 results, which would anchor
  their judgment toward the automated ranking

**Recruitment:** graduate students and practitioners from the department and
local developer community. Participation is voluntary and uncompensated.

**Instructions given to each evaluator:**

> You will see a programming task and two candidate solutions, A and B. Choose
> the one you would rather inherit and maintain. Weigh correctness first, then
> clarity, structure, and idiomatic style. If you genuinely cannot separate
> them, choose Tie. You do not need to run the code. There is no time limit per
> comparison, but do not agonize — first considered judgment is what we want.

Evaluators are not told which models produced the code, how many models are in
the pool, or what the automated results were.

## 4. Sample Size

Derived empirically rather than assumed. Bradley-Terry was fitted to generated
votes at a range of sample sizes, five seeds each, with 200 bootstrap
resamples per fit:

| Votes | Per model pair | Mean 95% CI width (rating pts) | Top vs bottom separated |
|------:|---------------:|-------------------------------:|------------------------:|
| 100   | 10             | 193                            | 60%                     |
| 200   | 20             | 127                            | 80%                     |
| 300   | 30             | 100                            | 100%                    |
| 500   | 50             | 77                             | 100%                    |
| 800   | 80             | 62                             | 100%                    |
| 1200  | 120            | 49                             | 100%                    |

**Minimum: 300 votes** (30 per model pair). Below this the extremes are not
reliably separated.

**Target: 800 votes** (80 per pair), giving roughly ±30 rating points per model.
At 10 evaluators this is 80 comparisons each, about 40–60 minutes of work.

Adjacent models will likely remain statistically inseparable at any feasible
sample size — the Phase 6 McNemar tests already show GPT-OSS-120B,
Llama-3.3-70B and Llama-4-Scout are indistinguishable on automated pass rate.
The protocol is powered to rank *groups*, not to split near-ties, and the
analysis will report overlapping intervals as ties rather than forcing an order.

## 5. Procedure

1. Confirm eligibility, obtain informed consent, assign an evaluator id.
2. Evaluator opens the voting interface (`python human_eval/app.py`).
3. The interface serves comparisons scheduled by `human_eval/sampler.py` until
   the evaluator stops.
4. Votes are written to the `human_votes` table with the evaluator id.
5. Collection closes at the 800-vote target or after two weeks, whichever first.

### 5.1 How comparisons are chosen

A **comparison cell** is one task and one unordered pair of models. The 65 tasks
give 646 cells — 650 minus the four lost to the single run whose code could not
be extracted. Cells are not drawn uniformly, for three reasons:

- **No evaluator ever sees the same cell twice.** Under uniform sampling an
  evaluator completing 80 comparisons draws a repeat with probability ≈ 0.99
  (birthday bound over 646 cells), and one person rating the same pair twice is
  counted as two coders by Krippendorff's α.
- **A fixed anchor set is rated by everyone.** Twenty cells, two per model pair,
  chosen deterministically from a fixed seed and interleaved through each
  session rather than front-loaded. This is the subset §6.3 computes α on; under
  uniform sampling that subset would exist only by chance, and at the 300-vote
  minimum only about 51 cells would attract a second rating at all.
- **Remaining comparisons go to the least-observed cell.** A Bradley-Terry
  interval is set by the least-observed model pair, so balance buys precision at
  no extra cost. A 10-evaluator, 800-vote simulation lands 76–83 votes on each of
  the 10 pairs against a target of 80.

Tasks that no model solved are **kept** in the pool. Twenty-seven of the 65 are
failed by every model, and excluding them would bias H3 upward by only ever
asking about work the models can do. Each comparison is instead labelled with an
outcome stratum — both passed, one passed, neither passed — and `/api/results`
reports the mix. In the same 800-vote simulation it comes out at roughly 50%
neither-passed, 27% both-passed, 23% split.

**Data recorded per vote:** task, both models, winner (`a` / `b` / `tie`),
evaluator id, timestamp. No personal data beyond the pseudonymous evaluator id.

## 6. Analysis

1. **Fit Bradley-Terry** to all human votes. Report per-model ratings with
   bootstrap 95% intervals (`human_eval/bradley_terry.py`).
2. **Test H3.** Spearman ρ between the human rating ordering and the automated
   pass-rate ordering across the five models. H3 holds if ρ > 0.7.
   With n = 5 the correlation is descriptive and its p-value is weak; the point
   estimate and interval are what get reported, with that limitation stated.
   Report ρ twice: over all comparisons, and restricted to tasks where at least
   one model passed. Half the comparisons are between two solutions that both
   failed their tests, where the automated ranking expresses no preference at
   all, so the two figures answer different questions and both belong in the
   report.
3. **Inter-rater agreement.** Krippendorff's α across evaluators on the anchor
   set (§5.1) — the twenty cells every evaluator rates. Low agreement means
   "human preference" is not a single coherent target and H3 is ill-posed as
   framed — which is itself a reportable finding.
4. **Disagreement analysis.** Inspect comparisons where the human majority
   preferred the model that failed the automated tests. These are the cases that
   reveal what the test suites miss.

## 7. Threats to Validity

- **n = 5 models.** Any correlation across five points is fragile. ρ > 0.7
  requires only that the orderings roughly agree; a single swap moves it a lot.
- **Correctness leakage.** Reviewers weigh correctness first, and broken code is
  often visibly broken. Human and automated rankings may agree because both
  track correctness, not because humans endorse the automated ranking's
  finer distinctions.
- **Self-selected evaluators** from one department are not a random sample of
  web developers.
- **Code shown without execution.** Reviewers judge from reading, so runtime
  behavior that is not visible in the source cannot influence their vote.
- **Comparisons between two failing solutions.** Roughly half of all cells pair
  two solutions that failed their tests, because 27 of the 65 tasks were solved
  by no model. Preference there is a judgment about style between two wrong
  answers, and the automated ranking has no opinion to correlate with. This is
  why §6.2 reports ρ on the passing subset as well; it is a property of the
  benchmark's difficulty, not something the scheduler can remove without biasing
  the result the other way.

## 8. Current Instrumentation Status

Built and validated:

- `app.py` — blind side-by-side voting interface, stdlib only, writes to
  `human_votes`
- `sampler.py` — comparison scheduler: anchor set, coverage balancing,
  per-evaluator de-duplication, outcome strata (§5.1)
- `bradley_terry.py` — MM-algorithm maximum-likelihood fit with bootstrap
  intervals; verified to recover planted strengths to within 0.003
- `synthetic_votes.py` — generated votes for pipeline validation
- `run_synthetic.py` — end-to-end runner

The database currently holds **1,200 synthetic votes and 0 human votes**.
The synthetic votes exist to prove the pipeline works and to derive the sample
size table above. They are tagged `evaluator_id = 'synthetic_NN'` and are
removed with `python run_synthetic.py --clear` before real collection begins.

**Synthetic votes are not evidence for H3.** They were generated from the
automated pass/fail results, so any agreement with those results is circular by
construction. H3 remains untested until human votes exist.
