# S4 router versions log (S0.7)

Every router version tried on the development split, with its development numbers, including
versions that are discarded. Development figures, not results. Rules and definitions:
`docs/research/S4_spec.md`. Source: `router/experiments/s4_v1.py`, outputs
`router/experiments/out/s4_v1*.{csv,json}` (byte-reproducible; two runs compared).

Common to every row: complete case (O2); the scored populations of spec §2 under the recommended
P1 (b): headline 195 tasks / 180 problems / 3 families; all-eligible with an X13 record 459 / 355 /
13; sensitivity with an X13 record 279 / 244 / 6. Folds `stats.dev_folds(tasks, seed)`, fitted on
four folds and scored on the fifth. Gate seed 0; paired cluster-bootstrap 95% CIs over problems
(1,000 resamples). Gap in acceptance points, saving in percent of the reference's cost.

## v1 (2026-11-24): `router_v1`

Spec §3 with the recommendations P2-P6: predicted family refit per fold; beta-binomial shrinkage
with one shared kappa (ANOVA estimator); Rule A on the shrunk values with margin 0.02; the menu is
the eight models plus one two-rung cascade with no-signal cells escalated.

Fitted kappa per fold (seed 0): headline 0.82-1.02, all-eligible 1.22-1.34, sensitivity 1.00-1.15
(never infinite). The cascade chosen in every fold of every population at seed 0:
`Qwen3-Max > claude-opus-4-6` on the headline and sensitivity; on all-eligible
`kimi-k2.5 > Opus`, `gpt-5.4 > Opus` or `Qwen3-Max > Opus` by fold. Each reached the training best
single model's acceptance.

Gate seed 0:

| Population | Acceptance | Cost (m$/task) | vs best single: gap / saving | vs CellBest-pred: gap / saving | vs Rule A escalated: gap / saving | vs fixed cascade (esc.): gap / saving |
|---|---|---|---|---|---|---|
| headline | 84.62 [78.57, 89.64] | 5.28 | −0.51 [−1.58, 0.00] / 11.11 [5.70, 16.98] | 0.00 [0.00, 0.00] / −1.99 [−7.13, 1.98] | +0.51 [0.00, 1.55] / −2.60 [−4.86, −0.83] | −0.51 / 4.08 [2.79, 6.23] |
| all-eligible | 72.77 [67.90, 77.24] | 3.62 | +1.96 [−1.84, 5.95] / 29.15 [23.82, 34.41] | +0.22 [0.00, 0.68] / 2.97 [−0.96, 6.41] | +0.44 [−0.68, 1.72] / −9.01 [−15.79, −3.95] | +2.40 / 23.80 [18.07, 29.25] |
| sensitivity | 86.38 [82.27, 90.22] | 4.14 | +0.72 [−1.06, 2.54] / 23.77 [17.51, 30.00] | +0.36 [0.00, 1.10] / 3.38 [−4.60, 9.86] | +1.08 [0.00, 2.42] / −8.83 [−17.21, −2.85] | +0.72 / 8.46 [5.52, 11.80] |

- Verifier runs: 22 (headline), 81, 80. CPAT at the floor (best single's acceptance): not
  defined on the headline (84.62 < 85.13); 4.97 and 4.79 m$ per accepted task elsewhere.
- Regret against the cost-aware oracle: 12.31, 17.21 and 10.04 points.
- Seeds 1-4 (point estimates; CIs in `s4_v1_seeds.csv`): headline saving vs best single 9.93 to
  12.32 at a gap of −0.51 every seed; vs Rule A escalated saving −5.25 to −2.14. All-eligible
  saving vs best single 26.01 to 29.91; vs Rule A escalated −14.56 to −9.57. Sensitivity 20.89
  to 24.04; vs Rule A escalated −8.62 to −6.39.
- CRUXEval ≥ 0.8 (seed 0): the decisions are fitted under the main rule, so cost is unchanged;
  acceptance 98.97 (headline), 78.87, 96.42.
- P1 (a) variant (seed 0, no-record tasks included, labelled): all-eligible 966 tasks,
  acceptance 34.47, saving vs best single 50.74; sensitivity 294 tasks, 81.97, 12.36.
- Shuffled-label check (headline, S2.6 protocol, 20 shuffles): shuffled mean 46.56
  [43.40, 49.73], Random's expected 50.38 outside it. Diagnostic: shuffle seeds 20-119 give 51.33
  [49.75, 52.92] (inside). The test is marked `xfail(strict)` and left for the reviewer.
- Status: built; pending the reviewer's rulings on P1-P6.

## v1 accept (2026-11-24): `router_v1_accept`

v1 with no-signal cells accepted inside the cascade (spec §3.4). Logged as its own version.

| Population | Acceptance | Cost (m$/task) | vs best single: gap / saving | vs CellBest-pred: gap / saving | vs Rule A escalated: gap / saving |
|---|---|---|---|---|---|
| headline | 84.62 | 5.18 | −0.51 [−1.58, 0.00] / 12.84 [7.36, 18.63] | 0.00 / 0.00 | +0.51 / −0.60 [−3.53, 2.44] |
| all-eligible | 72.55 | 3.73 | +1.74 [−1.99, 5.71] / 26.98 [22.12, 31.80] | 0.00 / 0.00 | +0.22 / −12.35 [−18.14, −7.27] |
| sensitivity | 86.02 | 4.28 | +0.36 [−1.44, 2.40] / 21.11 [15.56, 26.58] | 0.00 / 0.00 | +0.72 / −12.64 [−17.22, −8.55] |

- At seed 0 it matches CellBest-pred on all three populations: the gap is 0 and the saving rounds
  to 0.00. On all-eligible the accept-mode cascade (`Opus > glm-5`) is chosen for a group in some
  folds (Opus first, verified: 3 verifier runs in all). Seeds 1-4:
  the same, except sensitivity seed 2 (gap +2.51, saving −5.63) and seed 4 (gap 0, saving −0.13).
- Verifier runs: 0, 3, 0.
- Status: built; reported beside v1.

## Diagnostic (given more information, not a version): `v1_true_groups`

v1 reading the true family from the task handle. Seed 0: headline 84.62, cost 5.32, saving vs best
single 10.46; all-eligible 73.20, 3.58, 29.84; sensitivity 86.38, 4.11, 24.17.
