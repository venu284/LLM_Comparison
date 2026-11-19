import json, re, sys
import numpy as np
import pandas as pd

D = sys.argv[1]
t = pd.read_csv(f"{D}/id_test_results_long.csv")
t["cost_usd"] = t.cost_usd.fillna(0)
p = t.pivot_table(index="task_id", columns="model", values="score")
c = t.pivot_table(index="task_id", columns="model", values="cost_usd")

# 1. score the released ACRouter ID decisions against the released matrix
dec = pd.read_json(f"{D}/id_decisions.jsonl", lines=True).set_index("task_id")
print("decisions", len(dec), "voters", dec.voter.value_counts().to_dict(), "modes", dec.matched_mode.value_counts().to_dict())
common = [i for i in dec.index if i in p.index]
perf = np.mean([p.loc[i, dec.loc[i, "chosen_model"]] for i in common])
cost = np.sum([c.loc[i, dec.loc[i, "chosen_model"]] for i in common])
print(f"released ACRouter decisions on released matrix: n={len(common)} AvgPerf {perf*100:.2f} $Total {cost:.2f} Perf/$ {perf*100/cost:.2f}")
print("chosen model share", dec.chosen_model.value_counts(normalize=True).round(3).to_dict())

# 2. oracle / DimensionBest on the uncleaned obs_matrix
obs = json.load(open(f"{D}/obs_matrix.json"))
rows = [(tid, m, v.get("perf"), v.get("cost")) for tid, mm in obs.items() for m, v in mm.items()]
o = pd.DataFrame(rows, columns=["task_id", "model", "perf", "cost"])
print("obs_matrix tasks", o.task_id.nunique(), "models", o.model.nunique(), "NaN perf", int(o.perf.isna().sum()))
op = o.pivot_table(index="task_id", columns="model", values="perf")
oc = o.pivot_table(index="task_id", columns="model", values="cost")
ot = op.reindex([i for i in p.index if i in op.index])
print("test tasks found in obs_matrix", len(ot))
diff = (ot - p.reindex(ot.index)).abs()
print("cells differing from released matrix", int((diff > 1e-9).sum().sum()), "per model", (diff > 1e-9).sum().to_dict())
print(f"obs_matrix test: oracle {ot.max(axis=1).mean()*100:.2f}  Always-Opus {ot['claude-opus-4-6'].mean()*100:.2f}")

# 3. source-level and dimension-level best rules, fit on probing, scored on test
pr = pd.read_csv(f"{D}/id_probing_results_long.csv")
pp = pr.pivot_table(index="task_id", columns="model", values="score")
def src(x):
    return re.sub(r"_\d+$", "", re.sub(r"/\d+$", "", re.sub(r"^dim\d+_", "", x)))
dim_t = t.drop_duplicates("task_id").set_index("task_id").dimension
dim_p = pr.drop_duplicates("task_id").set_index("task_id").dimension
for name, kp, kt in [("dimension", dim_p.reindex(pp.index), dim_t.reindex(p.index)),
                     ("source", pd.Series([src(i) for i in pp.index], index=pp.index),
                      pd.Series([src(i) for i in p.index], index=p.index))]:
    best = pp.groupby(kp).mean().idxmax(axis=1)
    fallback = pp.mean().idxmax()
    ch = [best.get(kt[i], fallback) for i in p.index]
    perf = np.mean([p.loc[i, m] for i, m in zip(p.index, ch)])
    cost = np.sum([c.loc[i, m] for i, m in zip(p.index, ch)])
    print(f"{name}-best rule: AvgPerf {perf*100:.2f} $Total {cost:.2f} Perf/$ {perf*100/cost:.2f}")
