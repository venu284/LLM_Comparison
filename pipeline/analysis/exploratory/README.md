# Exploratory analysis — 2026-08-16 goal-lock session

Four throwaway-grade scripts that produced every new number in
`docs/superpowers/specs/2026-08-16-adaptive-router-design.md`. Kept for
provenance, not as production modules — they read the CSV export directly rather
than going through `analysis/data.py`, and they do not honour
`zero_test_treatment`.

Run any of them with the project venv:

    cd pipeline && venv/bin/python analysis/exploratory/<script>.py

| Script | Produces |
|---|---|
| `cascade_economics.py` | Cascade chain economics over all 1–5 length permutations under Groq list prices. Source of **56.9% @ −10.8%** and the oracle **−78.8%**. |
| `oracle_decomposition.py` | Splits the oracle saving into 4 task classes. Source of the **68.4% unsolvable / 20.8% cheap-sufficient** split, and the cheap-sufficiency and unsolvable rates by category and difficulty. |
| `probe_size_curve.py` | Probe-size sweep k = 3…40, 200 resampling trials. Source of the **descriptor accuracy climbing / routed pass flat** finding. |
| `feature_availability.py` | Column fill rates and post-failure signal strength. Source of the **4 empty schema columns**, **128/325 partial credit**, and the **T5 weak-post-failure-signal** result. |

## Caveats — read before quoting these numbers

1. **Prices are modelled, not measured.** `estimated_cost_usd` is `0.000000` in
   all 325 rows (free-tier). `cascade_economics.py` and `oracle_decomposition.py`
   hardcode Groq published $/M rates. Llama-4-Scout and Qwen3-32B rates were not
   directly confirmed in the source search and are approximate. **Every headline
   cost number needs a price-sensitivity sweep before publication** (decision D7).
2. **`zero_test_treatment` is not applied.** These use `pass_fail` as stored,
   i.e. the `keep` treatment that reproduces the published Phase 5 numbers.
   Re-running under `drop_artifactual` is an open robustness check.
3. **Single run per cell.** `run_number` = 1 everywhere, so none of these carry a
   variance estimate. This is threat T1 in the spec and it bites cascade harder
   than it bit selection routing.
4. `probe_size_curve.py` uses a deliberately simple descriptor (per-cell pass rate
   with marginal backoff). It is a floor on what a learned descriptor achieves,
   not a ceiling.
