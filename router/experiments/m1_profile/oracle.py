import sys
import numpy as np
import pandas as pd

D = sys.argv[1]
t = pd.read_csv(f"{D}/id_test_results_long.csv")
t["cost_usd"] = t.cost_usd.fillna(0)
p = t.pivot_table(index="task_id", columns="model", values="score")
c = t.pivot_table(index="task_id", columns="model", values="cost_usd")
for lam in [0.0, 0.1, 1.0, 10.0]:
    r = p - lam * c
    pick = r.idxmax(axis=1)
    perf = np.mean([p.loc[i, m] for i, m in pick.items()])
    cost = np.sum([c.loc[i, m] for i, m in pick.items()])
    print(f"reward oracle lam={lam}: AvgPerf {perf*100:.2f}  $Total {cost:.2f}  Perf/$ {perf*100/cost:.2f}")
print("Always-Opus $Total", round(c["claude-opus-4-6"].sum(), 2), "Perf/$", round(p["claude-opus-4-6"].mean() * 100 / c["claude-opus-4-6"].sum(), 2))
