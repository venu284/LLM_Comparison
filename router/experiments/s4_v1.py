"""S4.1: router v1 on the development split, beside the R2 baselines (docs/research/S4_spec.md §3).

Populations (spec §2, P1 (b)), complete case (O2), development split only:
- headline: all 195 tasks (every one has an X13 record);
- all_eligible and sensitivity: their tasks with an X13 record (cascade claims, S0.11);
- variant (P1 (a), labelled, gate seed only): all_eligible and sensitivity with every
  complete-case task, no-record tasks included.

Each policy is fitted on four problem-grouped folds (`stats.dev_folds`) and scores the fifth
(`baselines.run.run_dev`), seeds 0-4; seed 0 is the gate seed and seeds 1-4 are reported beside it,
never averaged (R12). The predictor of the predicted family is refit on each training fold of the
population scored (spec P2 (a)); every predicted-cell comparator uses the same pool and folds.

Per policy (gate seed): acceptance (S0.1) with its cluster-bootstrap CI, cost per task (S0.2,
verifier runs included), CPAT at the best single model's acceptance as the floor (R3), regret
against the cost-aware oracle, and paired cluster-bootstrap CIs (gap in points, saving in percent)
against the best single model, CellBest-pred (S0.3(c)), Rule A escalated with predicted cells (O19)
and the fixed cascade escalated. Seeds 1-4 get the same paired comparisons. Also: v1's fit records
per fold, the CRUXEval >= 0.8 sensitivity (gate seed), and the shuffled-label check for v1 on the
headline (S2.6 protocol, `N_SHUFFLES` shuffles).

These are development figures for comparing router versions, not results.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s4_v1
Writes router/experiments/out/s4_v1_seed0.csv, s4_v1_seeds.csv, s4_v1_variants.csv and s4_v1.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from router.baselines.cascade import FixedCascade
from router.baselines.common import ACCEPT, ESCALATE, Context, success
from router.baselines.lookup import CellBest, CellRule, DimensionBest, ZeroLoss
from router.baselines.run import run_dev, shuffled_mean_interval
from router.baselines.static import BestSingle
from router.core.policy import RouterV1
from router.data import splits as S
from router.data.audit import build_views, load_release
from router.data.prompts import join
from router.experiments.s2_baselines import N_SHUFFLES, compare, row
from router.settings import N_BOOT, REPO_ROOT
from router.sim import metrics, stats
from router.sim.verifier import ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
POPULATIONS = ("headline", "all_eligible", "sensitivity")
REFS = {"best_single": "best_single", "cellbest_pred": "cellbest_pred",
        "ruleA_escalate_pred": "ruleA_escalate_pred", "fixed_cascade": "fixed_cascade_escalate"}
V1_NAMES = ("router_v1", "router_v1_accept", "v1_true_groups")


def policies():
    """Router v1 (and its logged variants), the diagnostic, then the R2 baselines."""
    return [RouterV1(), RouterV1(mode=ACCEPT), RouterV1(groups="true"),
            BestSingle(), CellBest("pred"), CellBest("true"),
            *[CellRule(r, mode, "pred") for mode in (ACCEPT, ESCALATE) for r in ("A", "B")],
            FixedCascade(ACCEPT), FixedCascade(ESCALATE),
            DimensionBest(), DimensionBest(cost_aware=True), ZeroLoss("pred"), ZeroLoss("true")]


# ------------------------------------------------------------------------------- populations
_IN: dict = {}


def inputs():
    if not _IN:
        views = build_views(load_release())
        ts = S.task_splits().set_index("task_id")
        dev = set(ts.index[ts.split == S.DEV])
        verifier = ReplayVerifier.from_x13()
        mats, counts = {}, {}
        for name in POPULATIONS:
            v = views[name]
            d = v.subset([t for t in v.tasks if t in dev])
            cc = d.subset(list(d.tasks[d.available.astype(bool).all(axis=1).to_numpy()]))
            rec = cc.subset([t for t in cc.tasks if verifier.has_record(t)])
            mats[name] = rec
            if len(rec) != len(cc):
                mats[name + "__all_complete_case"] = cc
            counts[name] = {"dev_tasks": len(d), "complete_case": len(cc), "with_x13_record": len(rec),
                            "scored": len(rec), "problems": int(ts.problem_id.reindex(rec.tasks).nunique()),
                            "families": int(rec.family.nunique()),
                            "dropped_unavailable_by_family":
                                {str(k): int(x) for k, x in
                                 d.family[~d.available.astype(bool).all(axis=1).to_numpy()]
                                 .value_counts().sort_index().items()}}
        tasks = sorted({t for m in mats.values() for t in m.tasks})
        content = join(tasks)
        for name, m in mats.items():
            key = name.split("__")[0]
            counts.setdefault(key, {})
            counts[key][("no_text_scored" if "__" not in name else "no_text_all_complete_case")] = \
                int(content.reindex(m.tasks).isna().sum())
        _IN.update(mats=mats, counts=counts, problems=ts.problem_id, verifier=verifier,
                   content=content)
    return _IN


def run_population(m, seed: int, *, lenient: bool = False, only=None) -> dict[str, pd.DataFrame]:
    d = inputs()
    ctx = Context(d["verifier"], content=d["content"], problem_of=d["problems"], seed=seed)
    folds = stats.dev_folds(m.tasks, seed)
    return {p.name: run_dev(p, m, ctx, folds, seed=seed, cruxeval_lenient=lenient)
            for p in policies() if only is None or p.name in only}


def v1_fits(o: pd.DataFrame) -> list:
    return [{"fold": f["fold"], "n_train": f["n_train"], "n_scored": f["n_scored"],
             "table": f["table"], "fallback": f["fallback"], "chosen": f["chosen"]}
            for f in o.attrs["fits"]]


def seed_rows(pop: str, seed: int, outs: dict, problems) -> list:
    rows = []
    for name, o in outs.items():
        r = {"population": pop, "seed": seed, "gate_seed": seed == stats.GATE_SEED, "policy": name,
             "given_more_information": ",".join(o.attrs.get("more_information", ())),
             "acceptance": metrics.acceptance(o), "cost_musd_per_task": 1e3 * float(o.cost_usd.mean())}
        for tag, ref in REFS.items():
            if ref == name:
                continue
            c = compare(o, outs[ref], problems)
            r[f"gap_vs_{tag}"], r[f"gap_vs_{tag}_lo"], r[f"gap_vs_{tag}_hi"] = c["gap_points"]
            r[f"saving_vs_{tag}"], r[f"saving_vs_{tag}_lo"], r[f"saving_vs_{tag}_hi"] = c["saving_pct"]
        rows.append(r)
    return rows


def gate_rows(pop: str, outs: dict, m, problems, label: str = "") -> list:
    floor = metrics.acceptance(outs["best_single"])
    refs = {tag: outs[ref] for tag, ref in REFS.items()}
    out = []
    for name, o in outs.items():
        r = row(name, o, refs, problems, floor, m)
        r = {"population": pop, "variant": label or "main", **r, "floor": floor}
        out.append(r)
    return out


def shuffled_check(m, problems) -> dict:
    """S2.6 protocol for router v1 on one population, gate seed."""
    d = inputs()
    ctx = Context(d["verifier"], content=d["content"], problem_of=problems, seed=stats.GATE_SEED)
    folds = stats.dev_folds(m.tasks, stats.GATE_SEED)
    b = RouterV1()
    ok = success(m)
    rand_exp = 100 * float((ok.sum(axis=1) / m.available.sum(axis=1)).mean())
    accs = [metrics.acceptance(run_dev(b, m, ctx, folds, seed=stats.GATE_SEED, shuffle_seed=k))
            for k in range(N_SHUFFLES)]
    mean, lo, hi = shuffled_mean_interval(accs)
    return {"policy": b.name, "n_shuffles": N_SHUFFLES, "shuffle_seeds": [0, N_SHUFFLES - 1],
            "shuffled_mean": mean, "shuffled_mean_lo": lo, "shuffled_mean_hi": hi,
            "shuffled_min": float(np.min(accs)), "shuffled_max": float(np.max(accs)),
            "random_expected": rand_exp, "random_expected_inside": bool(lo <= rand_exp <= hi)}


def run():
    d = inputs()
    problems = d["problems"]
    res = {"populations": d["counts"], "n_boot": N_BOOT, "seeds": list(stats.SEEDS),
           "gate_seed": stats.GATE_SEED, "references": REFS, "fits": {}, "oracle": {},
           "spec": "docs/research/S4_spec.md (recommendations P1-P8, pending rulings)"}
    seed0, seeds, variants = [], [], []
    for pop in POPULATIONS:
        m = d["mats"][pop]
        res["fits"][pop] = {}
        for s in stats.SEEDS:
            outs = run_population(m, s)
            seeds += seed_rows(pop, s, outs, problems)
            res["fits"][pop][s] = {n: v1_fits(outs[n]) for n in V1_NAMES}
            if s == stats.GATE_SEED:
                seed0 += gate_rows(pop, outs, m, problems)
        orc = metrics.oracle_outcomes(m, m.tasks)
        res["oracle"][pop] = {"definition": metrics.OUR_DEFINITION,
                              "acceptance": 100 * float(orc["accepted"].mean()),
                              "cost_musd_per_task": 1e3 * float(orc.cost_usd.mean())}
        # sensitivity: CRUXEval >= 0.8 counts as success (gate seed)
        variants += gate_rows(pop, run_population(m, stats.GATE_SEED, lenient=True), m, problems,
                              "cruxeval_lenient")
        # P1 (a): every complete-case task, no-record tasks included (gate seed)
        if pop + "__all_complete_case" in d["mats"]:
            mc = d["mats"][pop + "__all_complete_case"]
            variants += gate_rows(pop, run_population(mc, stats.GATE_SEED), mc, problems,
                                  "all_complete_case_incl_no_record")
    res["shuffled_labels_headline"] = shuffled_check(d["mats"]["headline"], problems)
    return res, pd.DataFrame(seed0), pd.DataFrame(seeds), pd.DataFrame(variants)


def _round(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(10)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, seed0, seeds, variants = run()
    _round(seed0).to_csv(OUT / "s4_v1_seed0.csv", index=False, lineterminator="\n")
    _round(seeds).to_csv(OUT / "s4_v1_seeds.csv", index=False, lineterminator="\n")
    _round(variants).to_csv(OUT / "s4_v1_variants.csv", index=False, lineterminator="\n")
    (OUT / "s4_v1.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    cols = ["population", "baseline", "acceptance", "acceptance_lo", "acceptance_hi",
            "cost_musd_per_task", "gap_vs_best_single", "saving_vs_best_single",
            "gap_vs_cellbest_pred", "saving_vs_cellbest_pred", "gap_vs_ruleA_escalate_pred",
            "saving_vs_ruleA_escalate_pred", "gap_vs_fixed_cascade", "saving_vs_fixed_cascade"]
    print(seed0[cols].round(2).to_string(index=False))
    print(json.dumps(res["shuffled_labels_headline"]))
