# Frozen data

## `phase5_results.csv`

A snapshot of the 325 Phase 5 evaluation runs (5 models × 65 tasks, one run each), exported from
the Neon `runs` table. It is the same file as `pipeline/exports/results.csv`, which is git-ignored.
It is committed here so that every analysis can be reproduced from a clean clone, without
database access.

| Field | Value |
|---|---|
| Rows | 325 (plus a header row) |
| SHA-256 | `abaceb9a17c65c42333ef98d6762b8685db0ccb2a4d0dec52e3563cdd39f04c5` |
| Frozen | 2026-10-05 |

Rules:
- **Never edit this file.** The published Phase 5–7 numbers depend on it (plan rule R5, decision D10).
- Derived labels, such as the failure taxonomy, are computed at analysis time and never written back.
- New runs go to new files or tables, never into this snapshot.

To load it:

```python
from analysis.data import load_experiment_data
data = load_experiment_data(csv_path="data/phase5_results.csv", prefer_database=False)
```

Known properties are documented in `Reports/Research_Progress_Report_6.md` §3.8 and
`Reports/Phase5_Data_Sufficiency_Assessment.md`:
- `run_number = 1` in every row;
- `estimated_cost_usd = 0` in every row (free tier);
- `failure_mode` is empty in every row;
- 63 rows have `tests_total = 0`.
