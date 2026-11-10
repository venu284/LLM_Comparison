import json, re, sys
import numpy as np
import pandas as pd

D = sys.argv[1]
d = pd.concat([pd.read_csv(f"{D}/id_probing_results_long.csv"), pd.read_csv(f"{D}/id_test_results_long.csv")])
g = d.groupby("dimension").score
print(pd.DataFrame({"binary_frac": g.apply(lambda s: s.isin([0, 1]).mean()).round(2),
                    "nuniq": g.nunique(), "mean": g.mean().round(3)}))
t = d[d.split == "id_test"]
p = t.pivot_table(index="task_id", columns="model", values="score")
c = t.pivot_table(index="task_id", columns="model", values="cost_usd")
dim = t.drop_duplicates("task_id").set_index("task_id").dimension.reindex(p.index)
pr = d[d.split == "probing"].pivot_table(index="task_id", columns="model", values="score")
dp = d[d.split == "probing"].drop_duplicates("task_id").set_index("task_id").dimension.reindex(pr.index)
best = pr.groupby(dp).mean().idxmax(axis=1)
print("DimBest choice", best.to_dict())
sel = [p.loc[i, best[dim[i]]] for i in p.index]
print("DimBest probing->test", round(np.mean(sel) * 100, 2),
      "cost/task", round(np.mean([c.loc[i, best[dim[i]]] for i in p.index]), 5))
b = p >= 0.5
for m in p.columns:
    print(m, "solves", int(b[m].sum()), "also-by-opus", round((b[m] & b["claude-opus-4-6"]).sum() / b[m].sum(), 2))
print("best-single solves", int(b.sum().max()), "any solves", int(b.any(axis=1).sum()), "of", len(p))
order = c.mean().sort_values().index
cost = []
for i in p.index:
    mx = p.loc[i].max()
    ok = [m for m in order if mx > 0 and p.loc[i, m] >= mx]
    cost.append(c.loc[i, ok[0]] if ok else c.loc[i, order[0]])
print("cheapest-max-score oracle cost/task", round(np.mean(cost), 5), "vs opus", round(c["claude-opus-4-6"].mean(), 5))
ids = [json.loads(l) for l in open(f"{D}/id_tasks.jsonl")]
print("id task keys", list(ids[0].keys()))
src = pd.Series([re.sub(r"_\d+$", "", re.sub(r"^dim\d+_", "", x["task_id"])) for x in ids]).value_counts()
print(src.head(40).to_dict())
o = pd.read_csv(f"{D}/ood176_results_long.csv")
print(o.columns.tolist())
print(o.groupby("model").resolved.mean().round(3).to_dict())
print(o.drop_duplicates("task_id").dimension.value_counts().to_dict())
