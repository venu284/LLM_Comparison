# E0 power scripts (originals)

Ported in P0.4 to `router/experiments/e0_power.py` (results: `docs/research/P0.4_power.md`; the
originals' output on the pinned data: `router/experiments/out/e0_power_originals.txt`).

These are the scripts behind the power and margin numbers in S0.3 and S0.6 of
`docs/IMPLEMENTATION_PLAN.md`, committed as they were run on 2026-11-19 so the numbers can be
traced. P0.4 ports them into `router/experiments/e0_power.py`; until then they are the record.

Run each from the repo root with the CodeRouterBench mirror as the only argument:

```
pipeline/venv/bin/python router/experiments/e0_power/power.py external/CodeRouterBench
```

| Script | What it computes |
|---|---|
| `power.py` | CI half-width and P(gate passes) over 40 random 60/20/20 splits, for DimensionBest and two fixed cascades with a noisy one-sided verifier; cluster bootstrap over problems |
| `power2.py` | Per-model acceptance and cost, oracle cheapest-pass saving, and every cost-ordered 2- and 3-rung ladder at verifier false-pass 0 and 0.1, against qwen3.5-plus |
| `power3.py` | Three two-rung ladders over 30 random 60/20/20 problem-level splits, on all exec-binary tasks and with the suspected families removed; P(gate passes) by margin and saving |
| `power4.py` | The same two populations under 5-fold cross-fitting by problem (the design S0.6 adopted); CI half-width and the margin for 80% power |

Caveats for the port:
- They use the old eligibility rule (tasks whose observed scores are all 0 or 1), which R16
  replaced with documented rules and the success = 1.0 rule. Their numbers are provisional.
- They read `cost_usd` with missing values filled as 0. That counts the 148 missing Qwen3-Max runs
  as free failures (P0.1 flag 8); the port must treat them as missing.
- They simulate the verifier with a random false-pass rate. The port must use the real X13 verdicts
  (R1).
- `power4.py` line 2 is a leftover no-op (`exec("")`).
- The feasibility numbers in S3.2 (the 1,971-task headline set and the oracle-on-no-signal bound)
  were computed inline and are not in these scripts; the provisional recount rebuilds them.
