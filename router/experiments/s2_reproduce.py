"""S2.6 reproductions: the ACRouter paper's published numbers on the paper view (REPRODUCTION ONLY).

The paper (arXiv 2606.22902 v3, Table 3 §5.1 and Table 12 App. D.6) scores its routers on the
2,919 ID-test tasks with AvgPerf = the mean released score of the chosen model, in percent (O10),
on the ragged `obs_matrix.json` with missing cells scored 0 (the paper view; App. D.2's
"missing-as-0"). This script uses that metric (`router.sim.paper`) and the paper's own split
(probing -> ID-test, R13) and nothing here is claim-facing.

Reproduced:
- the oracle (highest released score per task; ties to the cheapest such model);
- Always-Opus;
- the released ACRouter decisions (`outputs/current/id_decisions.jsonl`).

ACRouter target: the plan's 50.14 is the release's own metrics file
(`outputs/current/summary.json`: AvgPerf 50.14, CumReg 202.0, $Total 22.31, Perf/$ 2.25), which the
released decisions reproduce. The paper's text gives 49.98 (Table 3; also Table 12, §5.2 and
App. D.5) with CumReg 205.5 and $13.21. Both CumReg and cost differ, so the paper's row is a
different ACRouter run from the released decisions; Table 12's cost also counts router-side
tokens. 49.98 cannot be rebuilt from the release; the reproduction targets 50.14.

Reference figure (plan §4 S2): DimensionBest (best mean released score per dimension on the
probing split, the m2 definition) minus released ACRouter, paired over tasks, on the complete and
the paper view. CIs: paired bootstrap over tasks (`N_BOOT`, seed 0), and the cluster bootstrap over
problems next to it.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s2_reproduce
Writes router/experiments/out/s2_reproduce.json and s2_reproduce.csv.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from router.data import coderouterbench as crb
from router.data import splits as S
from router.settings import N_BOOT, REPO_ROOT
from router.sim import stats
from router.sim.paper import from_choice, paper_avgperf, paper_summary

OUT = REPO_ROOT / "router" / "experiments" / "out"
OPUS = "claude-opus-4-6"
SUMMARY_JSON = crb.CRB_DIR / "outputs" / "current" / "summary.json"
# Published values (2606.22902 v3): Table 3 (AvgPerf, CumReg, Perf/$) and Table 12 ($Total).
PAPER = {
    "oracle": {"PaperAvgPerf": 57.00, "CumReg": 0.0, "Perf/$": 8.20, "$Total": 6.95},
    "always_opus": {"PaperAvgPerf": 43.83, "CumReg": 387.1, "Perf/$": 1.29, "$Total": 34.02},
    "acrouter_released": {"PaperAvgPerf": 49.98, "CumReg": 205.5, "Perf/$": 3.79, "$Total": 13.21},
}
TARGET = {"oracle": 57.00, "always_opus": 43.83, "acrouter_released": 50.14}
TOLERANCE = 0.005           # a reproduction matches when |ours - target| < 0.005 (2 decimals)
REFERENCE = {"complete": {"point": 1.54}, "paper": {"point": -0.05, "lo": -0.31, "hi": 0.23}}


def dimension_best(train, test) -> pd.Series:
    """m2's DimensionBest: best mean released score per dimension on the training split."""
    table = train.score.groupby(train.dimension).mean().idxmax(axis=1)
    fallback = train.score.mean().idxmax()
    return pd.Series([table.get(d, fallback) for d in test.dimension], index=test.tasks)


def oracle(test) -> pd.Series:
    """Highest released score per task; ties to the cheapest of those models (Table 12 calls the
    paper's oracle cost-aware). AvgPerf does not depend on the tie-break; $Total does."""
    s, c = test.score.to_numpy(), test.cost.to_numpy()
    cc = np.where(s >= s.max(axis=1, keepdims=True) - 1e-12, c, np.inf)
    return pd.Series(np.asarray(test.models)[cc.argmin(axis=1)], index=test.tasks)


def _interval(iv) -> dict:
    return {"point": iv.point, "lo": iv.lo, "hi": iv.hi, "unit": iv.unit, "n_units": iv.n_units,
            "n_boot": iv.n_boot, "seed": iv.seed}


def reproduce() -> dict:
    problems = S.load_problems().set_index("task_id").problem_id
    released = json.loads(SUMMARY_JSON.read_text())["id"]
    dec = crb.released_acrouter_decisions()
    out = {"release_summary_id": released, "paper": PAPER, "target": TARGET, "views": {}}
    for view in ("paper", "complete"):
        train, test = crb.load("probing", view), crb.load("id_test", view)
        if not dec.index.isin(test.tasks).all() or len(dec) != len(test):
            raise ValueError("released decisions do not cover the ID-test tasks one to one")
        choices = {"oracle": oracle(test), "always_opus": pd.Series(OPUS, index=test.tasks),
                   "acrouter_released": dec, "dimension_best": dimension_best(train, test)}
        outs = {k: from_choice(test.score, test.cost, c) for k, c in choices.items()}
        frame = pd.DataFrame({k: o.score for k, o in outs.items()})
        rows = {}
        for k, o in outs.items():
            s = paper_summary(o, test.score, test.cost)
            iv = stats.bootstrap(frame, stats.rate(k), n_boot=N_BOOT)
            rows[k] = {**s, "ci_task": _interval(iv)}
            if view == "paper" and k in TARGET:
                rows[k]["target"] = TARGET[k]
                rows[k]["matches_target"] = bool(abs(s["PaperAvgPerf"] - TARGET[k]) < TOLERANCE)
                rows[k]["minus_paper_table3"] = s["PaperAvgPerf"] - PAPER[k]["PaperAvgPerf"]
        diff = stats.mean_diff("dimension_best", "acrouter_released")
        clusters = problems.reindex(test.tasks)
        ref = {"task": _interval(stats.bootstrap(frame, diff, n_boot=N_BOOT)),
               "problem": _interval(stats.bootstrap(frame, diff, clusters=clusters, n_boot=N_BOOT)),
               "reference": REFERENCE[view]}
        ref["point_matches_reference_2dp"] = bool(
            round(ref["task"]["point"], 2) == REFERENCE[view]["point"])
        out["views"][view] = {"n_test": len(test), "n_train": len(train), "routers": rows,
                              "dimension_best_minus_acrouter": ref,
                              "clusters_missing": int(clusters.isna().sum())}
    return out


def table(res: dict) -> pd.DataFrame:
    rows = []
    for view, v in res["views"].items():
        for k, r in v["routers"].items():
            rows.append({"view": view, "router": k, "paper_table3": PAPER.get(k, {}).get("PaperAvgPerf"),
                         "target": r.get("target"), "PaperAvgPerf": r["PaperAvgPerf"],
                         "ci_lo": r["ci_task"]["lo"], "ci_hi": r["ci_task"]["hi"],
                         "$Total": r["$Total"], "Perf/$": r["Perf/$"], "CumReg": r["CumReg"],
                         "matches_target": r.get("matches_target")})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res = reproduce()
    (OUT / "s2_reproduce.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    t = table(res)
    t.to_csv(OUT / "s2_reproduce.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    for view, v in res["views"].items():
        d = v["dimension_best_minus_acrouter"]
        print(view, "DimensionBest - ACRouter:", {k: (round(d[k]["point"], 3), round(d[k]["lo"], 3),
                                                      round(d[k]["hi"], 3)) for k in ("task", "problem")},
              "reference", d["reference"])
