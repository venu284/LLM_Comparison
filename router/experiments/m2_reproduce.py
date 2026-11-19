"""M1 data checks (A1), baseline reproduction (A2) and fixed-cascade cost (A3).

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.m2_reproduce
Writes CSVs to router/experiments/out/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.baselines import cascade, static
from router.baselines.online import KNN, LinUCB, metadata_features
from router.data import coderouterbench as crb
from router.data.prompts import join
from router.data.scoring import score_types
from router.settings import REPO_ROOT
from router.sim.metrics import bootstrap_ci, from_choice, summary
from router.sim.stream import replay
from router.sim.verifier import Verifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
OPUS = "claude-opus-4-6"
PAPER = {"Oracle": 57.00, "ACRouter (released decisions)": 49.98, "DimensionBest": 47.50,
         "kNN retrieval": 47.18, "LinUCB": 46.84, "Random (10 seeds)": 38.75, "Always-Opus": 43.83}
SEEDS = range(5)


def a1() -> pd.DataFrame:
    rows = []
    for split, n in (("probing", 7080), ("id_test", 2919)):
        long = crb.load_long(split)
        rows.append({"check": f"{split} rows", "expected": n * 8, "observed": len(long)})
        rows.append({"check": f"{split} NaN scores", "expected": 0, "observed": int(long.score.isna().sum())})
        rows.append({"check": f"{split} zero-token cost rows", "expected": {"probing": 106, "id_test": 42}[split],
                     "observed": int((long.cost_source == "missing_token_record_zero_total").sum())})
    rows.append({"check": "id_test missing cells in paper matrix", "expected": 881,
                 "observed": crb.missing_cells("id_test")})
    tasks = crb.load("probing").tasks.append(crb.load("id_test").tasks)
    prompts = join(tasks)
    src = pd.Series({t: crb.source_of(t) for t in tasks})
    joined = prompts.notna().groupby(src).mean()
    rows.append({"check": "prompt join rate, all ID tasks", "expected": ">=0.95",
                 "observed": round(float(prompts.notna().mean()), 3)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a1_integrity.csv", index=False)
    joined.sort_values().to_csv(OUT / "a1_join_rate_by_source.csv", header=["join_rate"])
    m = crb.load("probing")
    score_types(m.score, m.dimension).to_csv(OUT / "a1_score_types.csv")
    return df


def a2() -> pd.DataFrame:
    rows = []
    for view in ("paper", "complete"):
        train, test = crb.load("probing", view), crb.load("id_test", view)
        res = {}
        res["Oracle"] = summary(from_choice(test.score, test.cost, static.oracle(test)), test.score, test.cost)
        dec = crb.released_acrouter_decisions()
        res["ACRouter (released decisions)"] = summary(from_choice(test.score, test.cost, dec), test.score, test.cost)
        res["DimensionBest"] = summary(from_choice(test.score, test.cost, static.group_best(train, test, "dimension")), test.score, test.cost)
        res["SourceBest"] = summary(from_choice(test.score, test.cost, static.group_best(train, test, "source")), test.score, test.cost)
        ftr, vocab = metadata_features(train)
        fte, _ = metadata_features(test, vocab)
        knn = [summary(from_choice(test.score, test.cost, KNN(train, ftr, fte, k=10, seed=s).choose_all(test.tasks)), test.score, test.cost) for s in SEEDS]
        res["kNN retrieval"] = pd.DataFrame(knn).mean().to_dict()
        lin = []
        for s in SEEDS:
            r = LinUCB(test.models, fte)
            r.warm_start(train, ftr)
            out, _ = replay(r, test, seed=s)
            lin.append(summary(out, test.score, test.cost))
        res["LinUCB"] = pd.DataFrame(lin).mean().to_dict()
        rnd = [summary(from_choice(test.score, test.cost, static.random_choice(test, s)), test.score, test.cost) for s in range(42, 52)]
        res["Random (10 seeds)"] = pd.DataFrame(rnd).mean().to_dict()
        res["Always-Opus"] = summary(from_choice(test.score, test.cost, static.always(test, OPUS)), test.score, test.cost)
        for k, v in res.items():
            rows.append({"view": view, "router": k, "paper": PAPER.get(k, np.nan), **v})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a2_reproduction.csv", index=False)
    return df


def a3() -> pd.DataFrame:
    train, test = crb.load("probing"), crb.load("id_test")
    floor = summary(from_choice(test.score, test.cost, static.always(test, OPUS)), test.score, test.cost)["AcceptRate"]
    rows = []

    def add(name, outs):
        sm = pd.DataFrame([summary(o, test.score, test.cost, floor) for o in outs])
        lo, hi = bootstrap_ci(outs[0].score * 100)
        rows.append({"policy": name, "AvgPerf": sm.AvgPerf.mean(), "AvgPerf_lo": lo, "AvgPerf_hi": hi,
                     "AcceptRate": sm.AcceptRate.mean(), "$Total": sm["$Total"].mean(),
                     "CPAT": sm.CPAT.mean(), "MeetsFloor": bool(sm.AcceptRate.mean() >= floor)})

    add("Always-Opus", [from_choice(test.score, test.cost, static.always(test, OPUS))])
    add("Always-GPT-5.4", [from_choice(test.score, test.cost, static.always(test, "gpt-5.4"))])
    for ladder in (["Qwen3-Max", OPUS], ["Qwen3-Max", "gpt-5.4"], ["Qwen3-Max", "gpt-5.4", OPUS]):
        for eps in (0.0, 0.1, 0.2):
            outs = [cascade.run(test, ladder, Verifier(eps, seed=s)) for s in SEEDS]
            add(f"Cascade {' > '.join(ladder)}, eps={eps}", outs)
    add("DimensionBest", [from_choice(test.score, test.cost, static.group_best(train, test, "dimension"))])
    df = pd.DataFrame(rows)
    opus_cost = df.loc[df.policy == "Always-Opus", "$Total"].iloc[0]
    df["cost_vs_opus_%"] = (df["$Total"] / opus_cost - 1) * 100
    df.attrs["floor"] = floor
    df.to_csv(OUT / "a3_cascade_cost.csv", index=False)
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)
    print(a1().to_string(index=False))
    print(a2().round(2).to_string(index=False))
    d = a3()
    print("floor (Always-Opus accept rate %):", round(d.attrs["floor"], 2))
    print(d.round(3).to_string(index=False))
