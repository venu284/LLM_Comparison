"""S3.1 and DP1a: accuracy headroom of cross-fitted DimensionBest and of the oracle over the
cross-fitted best single model (S3 spec D1, D2, D3; CLAIMS §3 DP1a, O1, O18).

- `filtered = headline` (the verdict): the whole view through `stats.fit_free_headroom`, the
  closed evaluation-population path (D3 (a)); six groups (the development split and S1.6's five
  outer folds, D1 (i)); every task scored by tables fitted on the other five groups.
- `all_eligible` and `sensitivity`: their development splits only (D3 (α)), with `stats.dev_folds`
  and `baselines.run.run_dev` (BestSingle and DimensionBest fitted on the other four folds).

Headroom = acceptance(route) - acceptance(best single), paired per task, in points, with
`stats.gate_interval` (cluster bootstrap over problems, 2,000 resamples, seed 0). Seeds 0 to 4;
seed 0 decides. DP1a holds when seed 0's CI lower bound of DimensionBest's headroom on the
`filtered` complete case is >= 3.0 (`DP1A_MIN_LOWER`). The oracle (any available model succeeds)
is the ceiling and decides nothing (O18).

Variants (information only): CRUXEval >= 0.8 at scoring (tables chosen under S0.1); empty
`response_matrix.json` cells unavailable; all tasks (unavailable dispatch gives up); and the mean
released score of the chosen model in place of acceptance (O10; tables still chosen by acceptance,
the oracle takes the maximum available score). The paper metric module is not used.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s3_dp1a
Writes router/experiments/out/s3_1_dp1a.csv, s3_1_dp1a_seeds.csv and s3_1_dp1a.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from router.baselines.common import Context, success
from router.baselines.lookup import GroupBest
from router.baselines.run import run_dev
from router.baselines.static import BestSingle
from router.data import splits as S
from router.data.audit import build_views, load_release, response_key, response_stats
from router.data.schema import EMPTY_RESPONSE
from router.settings import DP1A_MIN_LOWER, N_BOOT_GATE, REPO_ROOT
from router.sim import stats
from router.sim.verifier import ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
VARIANTS = ("main", "cruxeval_lenient", "rm_empty_unavailable", "all_tasks")
METRICS = ("acceptance", "mean_score")
ROUTES = ("dimension_best", "oracle")
FLAGS = {"main": {}, "cruxeval_lenient": {"cruxeval_lenient": True},
         "rm_empty_unavailable": {"empty_unavailable": True}, "all_tasks": {"all_tasks": True}}
DEV_VIEWS = ("all_eligible", "sensitivity")


def population(var: str) -> str:
    return "all tasks (available cells)" if var == "all_tasks" else "complete case"


# ------------------------------------------------------------------------------------ filtered
def filtered_frame(seed: int, var: str) -> tuple[pd.DataFrame, dict, list]:
    r = stats.fit_free_headroom(seed, **FLAGS[var])
    t = r.tasks
    f = pd.DataFrame({"best_acc": t.best_single_accepted.astype(float),
                      "dimension_best_acc": t.dimension_best_accepted.astype(float),
                      "oracle_acc": t.oracle_accepted.astype(float),
                      "best_score": t.best_single_score, "dimension_best_score": t.dimension_best_score,
                      "oracle_score": t.oracle_score, "problem_id": t.problem_id}, index=t.index)
    fits = [{"group": x["group"], "n_train": x["n_train"], "n_scored": x["n_scored"],
             "best_single": x["best_single"], "dimension_best": x["dimension_best"]} for x in r.fits]
    return f, r.record, fits


# ------------------------------------------------------------------------------ dev-split views
_DEV = {}


def _dev_inputs():
    if not _DEV:
        m = load_release()
        views = build_views(m)
        ts = S.task_splits().set_index("task_id")
        st = response_stats()
        keys = {response_key(x): x for x in m.models}
        st = st[st.is_empty & st.key.isin(set(keys))]
        _DEV.update(views=views, dev=set(ts.index[ts.split == S.DEV]), problems=ts.problem_id,
                    rm_empty=[(t, keys[k]) for t, k in zip(st.task_id, st.key)],
                    verifier=ReplayVerifier.from_x13())
    return _DEV


def dev_frame(view: str, seed: int, var: str) -> tuple[pd.DataFrame, dict, list]:
    d = _dev_inputs()
    v = d["views"][view]
    m = v.subset([t for t in v.tasks if t in d["dev"]])
    if var == "rm_empty_unavailable":
        m = m.mark_unavailable(d["rm_empty"], EMPTY_RESPONSE)
    if var != "all_tasks":
        m = m.subset(list(m.tasks[m.available.astype(bool).all(axis=1).to_numpy()]))
    lenient = var == "cruxeval_lenient"
    folds = stats.dev_folds(m.tasks, seed)
    ctx = Context(d["verifier"], problem_of=d["problems"], seed=seed)
    outs, fits = {}, []
    for key, route in (("best", BestSingle()), ("dimension_best", GroupBest("dimension"))):
        o = run_dev(route, m, ctx, folds, seed=seed, cruxeval_lenient=lenient)
        chosen = o.final_model.where(o.action == "accept")
        outs[f"{key}_acc"] = o["accepted"].astype(float)
        outs[f"{key}_score"] = pd.Series([float(m.score.at[i, c]) if isinstance(c, str) else 0.0
                                          for i, c in zip(o.index, chosen)], index=o.index)
        fits.append({"route": key, "fits": [{k: x[k] for k in ("fold", "table", "chosen")}
                                            for x in o.attrs["fits"]]})
    ok = success(m, lenient)
    av = m.available.astype(bool)
    f = pd.DataFrame(outs).loc[m.tasks]
    f["oracle_acc"] = ok.any(axis=1).astype(float)
    f["oracle_score"] = m.score.where(av).max(axis=1).fillna(0.0)
    f["problem_id"] = d["problems"].reindex(m.tasks)
    rec = {"view": view, "split": "development", "folds": "stats.dev_folds", "seed": seed,
           "n_tasks": len(m), "n_evaluation_tasks_read": 0}
    return f, rec, fits


# --------------------------------------------------------------------------------- statistics
def headroom_rows(f: pd.DataFrame, view: str, var: str, seed: int) -> list[dict]:
    rows = []
    cl = f.problem_id
    for metric in METRICS:
        suffix = "acc" if metric == "acceptance" else "score"
        for route in ROUTES:
            a, b = f"{route}_{suffix}", f"best_{suffix}"
            iv = stats.gate_interval(f, stats.mean_diff(a, b), cl)
            rows.append({"view": view, "variant": var, "population": population(var),
                         "metric": metric, "route": route, "seed": seed,
                         "gate_seed": seed == stats.GATE_SEED,
                         "headroom_points": iv.point, "lo": iv.lo, "hi": iv.hi,
                         "route_points": 100 * float(f[a].mean()),
                         "best_single_points": 100 * float(f[b].mean()),
                         "n_tasks": len(f), "n_problems": iv.n_units})
    return rows


def run():
    rows, records, fits = [], {}, {}
    for seed in stats.SEEDS:
        for var in VARIANTS:
            f, rec, ft = filtered_frame(seed, var)
            rows += headroom_rows(f, "filtered = headline", var, seed)
            records[f"filtered = headline|{var}|{seed}"] = rec
            fits[f"filtered = headline|{var}|{seed}"] = ft
            for view in DEV_VIEWS:
                f, rec, ft = dev_frame(view, seed, var)
                rows += headroom_rows(f, f"{view} (development split)", var, seed)
                records[f"{view}|{var}|{seed}"] = rec
                fits[f"{view}|{var}|{seed}"] = ft
    seeds = pd.DataFrame(rows)

    # DP1a verdict: filtered, main, acceptance, DimensionBest, seed 0's CI lower bound
    d = seeds[(seeds.view == "filtered = headline") & (seeds.variant == "main")
              & (seeds.metric == "acceptance") & (seeds.route == "dimension_best")].set_index("seed")
    lower = d.lo.to_dict()
    g = d.loc[stats.GATE_SEED]
    verdict = {
        "statistic": "CI lower bound of cross-fitted DimensionBest's acceptance headroom over the "
                     "cross-fitted best single model, filtered = headline complete case, gate seed 0",
        "threshold": DP1A_MIN_LOWER, "point": g.headroom_points, "lo": g.lo, "hi": g.hi,
        "lower_bound": g.lo, "margin": g.lo - DP1A_MIN_LOWER, "holds": bool(g.lo >= DP1A_MIN_LOWER),
        "label": stats.threshold_label(lower, g.lo, DP1A_MIN_LOWER),
        "label_definition": "stats.threshold_label on the CI lower bounds of seeds 0 to 4, with "
                            "seed 0's lower bound as the gate value (spec §7, 2026-11-23)",
        "lower_bound_by_seed": lower, "point_by_seed": d.headroom_points.to_dict(),
        "n_tasks": int(g.n_tasks), "n_problems": int(g.n_problems)}
    o = seeds[(seeds.view == "filtered = headline") & (seeds.variant == "main")
              & (seeds.metric == "acceptance") & (seeds.route == "oracle")].set_index("seed")
    ceiling = {"point": o.loc[0].headroom_points, "lo": o.loc[0].lo, "hi": o.loc[0].hi,
               "note": "ceiling (O18); decides nothing"}
    res = {"spec": "docs/research/S3_analysis_spec.md D1, D2, D3", "dp1a": verdict,
           "oracle_ceiling": ceiling, "records": records, "fits": fits,
           "seed_spread_note": "D1 (i): the development group is the same on every seed, so seeds 1 "
                               "to 4 vary only the evaluation folds; the seed spread is narrower "
                               "than a full fold redraw would give",
           "cruxeval_lenient_note": "tables chosen under S0.1, scored with CRUXEval >= 0.8 (spec §7)",
           "n_boot": N_BOOT_GATE}
    return res, seeds


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (frozenset, set)):
        return sorted(_clean(v) for v in x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, seeds = run()
    keys = ["view", "variant", "metric", "route", "seed"]
    seeds = seeds.sort_values(keys, kind="stable")
    seeds.to_csv(OUT / "s3_1_dp1a_seeds.csv", index=False, lineterminator="\n")
    seeds[seeds.gate_seed].to_csv(OUT / "s3_1_dp1a.csv", index=False, lineterminator="\n")
    (OUT / "s3_1_dp1a.json").write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    print(json.dumps(_clean({"dp1a": res["dp1a"], "oracle_ceiling": res["oracle_ceiling"]}), indent=1))
