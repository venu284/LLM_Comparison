"""Score baselines on the ragged paper matrix with missing cells counted as 0.

Tests whether the paper's Table 3 numbers come from the uncleaned obs_matrix.json
with a missing task-model cell scored as a failure rather than skipped.

Usage: python ragged_zero.py <CodeRouterBench snapshot dir>
"""

import json
import sys

import numpy as np
import pandas as pd

D = sys.argv[1]
obs = json.load(open(f"{D}/obs_matrix.json"))
test = pd.read_csv(f"{D}/id_test_results_long.csv")
probe = pd.read_csv(f"{D}/id_probing_results_long.csv")
models = sorted(test.model.unique())


def matrix(ids):
    perf = pd.DataFrame(np.nan, index=ids, columns=models)
    cost = pd.DataFrame(np.nan, index=ids, columns=models)
    for i in ids:
        for m, v in obs.get(i, {}).items():
            perf.loc[i, m] = v.get("perf")
            cost.loc[i, m] = v.get("cost")
    return perf, cost


tid = sorted(test.task_id.unique())
pid = sorted(probe.task_id.unique())
tp, tc = matrix(tid)
pp, _ = matrix(pid)

for label, fill in [("missing skipped", None), ("missing = 0", 0.0)]:
    p = tp if fill is None else tp.fillna(fill)
    c = tc.fillna(0.0)
    print(f"== {label}")
    print(f"  oracle {p.max(axis=1).mean() * 100:.2f}")
    print(f"  Always-Opus {p['claude-opus-4-6'].mean() * 100:.2f}"
          f"  Perf/$ {p['claude-opus-4-6'].mean() * 100 / c['claude-opus-4-6'].sum():.2f}")

    dec = pd.read_json(f"{D}/id_decisions.jsonl", lines=True).set_index("task_id").chosen_model
    perf = np.nanmean([p.loc[i, dec[i]] for i in tid])
    cost = np.sum([c.loc[i, dec[i]] for i in tid])
    print(f"  ACRouter release decisions {perf * 100:.2f}  $Total {cost:.2f}  Perf/$ {perf * 100 / cost:.2f}")

    ppf = pp if fill is None else pp.fillna(fill)
    dim_p = probe.drop_duplicates("task_id").set_index("task_id").dimension.reindex(pid)
    dim_t = test.drop_duplicates("task_id").set_index("task_id").dimension.reindex(tid)
    best = ppf.groupby(dim_p).mean().idxmax(axis=1)
    perf = np.nanmean([p.loc[i, best[dim_t[i]]] for i in tid])
    cost = np.sum([c.loc[i, best[dim_t[i]]] for i in tid])
    print(f"  DimensionBest (probing->test) {perf * 100:.2f}  $Total {cost:.2f}  Perf/$ {perf * 100 / cost:.2f}")
    print(f"  DimensionBest choices {best.to_dict()}")
