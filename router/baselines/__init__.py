"""S2.5 baselines on the replay simulator (`router/sim/`). See `common.py` for the fit-and-route
contract, `run.py` for the development-split runner and the full list (`default_baselines`).

- `static.py`: Always-X, always-best (best single model on the training folds), Random;
- `lookup.py`: DimensionBest (full, cost-aware, k-matched), SourceBest, CellBest (predicted cells =
  the S0.3(c) comparator; true cells = reference), P0.6's per-cell Rules A and B (no-signal cells
  accepted or escalated; Rule A escalated is C2's bar, O19), Route Anything's zero-loss route;
- `cells.py`: the S2.5 cell predictor;
- `cascade.py`: the fixed cascade;
- `acrouter.py`: the released ACRouter decisions;
- `online.py`: LinUCB and kNN on the task text; `embeddings.py`: the pinned embedding model.

The 2026-11-10 quick-check baselines are kept, unchanged, in
`router/experiments/m2_quickcheck_baselines/` for the m2 record only.
"""
