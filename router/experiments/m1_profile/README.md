# M1 pre-implementation profile of CodeRouterBench

These scripts back the descriptive numbers in `Reports/Research_Progress_Report_8.md` §3.
They were run on 2026-11-10 before the `router/data/` loaders existed, so each one reads the
raw files directly. M1.2 replaces them with `OutcomeMatrix` loaders and tests.

## Data snapshot

Hugging Face dataset `Lance1573/CodeRouterBench`, `lastModified` 2026-06-23T17:53:28Z.
Files used:

- `id_probing_results_long.csv`, `id_test_results_long.csv`, `ood176_results_long.csv`
- `id_tasks.jsonl`
- `raw_matrices/phase1_acrouter_v2/obs_matrix.json` (the uncleaned matrix the paper's tables
  were computed on)
- `outputs/current/id_decisions.jsonl` (the released ACRouter ID decisions)

Download them into one directory (git-ignored, for example `external/CodeRouterBench/`) with:

```
hf download Lance1573/CodeRouterBench --repo-type dataset --local-dir external/CodeRouterBench
```

## Scripts

| Script | What it prints |
|---|---|
| `profile.py` | Score types per dimension, DimensionBest (probing to test), nesting, clairvoyant cheapest-oracle cost, task sources, OOD176 resolve rates |
| `oracle.py` | Max-score and reward oracles at several cost weights; Always-Opus metrics |
| `check.py` | Scores the released ACRouter decisions; oracle on the uncleaned matrix; dimension-best and source-best rules |
| `ragged.py` | Missing task-model cells in the uncleaned matrix on the ID test split |
| `ragged_zero.py` | Oracle, Always-Opus, released ACRouter decisions and DimensionBest on the uncleaned matrix, with missing cells skipped and scored 0 |

Each script takes the snapshot directory as its only argument. The outputs captured on
2026-11-10 are in `out/`.

## Caveat

Every oracle number here uses ground-truth scores, so it is a clairvoyant upper bound (Rule R1),
not something a router can reach.
