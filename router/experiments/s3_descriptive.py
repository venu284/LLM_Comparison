"""S3 checkpoint 1: S3.6 (verifier and population profile) and the descriptive part of S3.3
(unsolved share, rescue rates per pair of rungs, historical cost figures), as fixed in
`docs/research/S3_analysis_spec.md` §5.

Descriptive only: nothing is fitted, no DP number is computed. Every view is read in full, the
evaluation population included, as S1.7 did; no number here feeds a decision (plan S3 header).

Views: `all_eligible`, `filtered = headline` (one column; the script checks the two are the same
tasks) and `sensitivity`. Variants:
- `main`: success = score 1.0 (S0.1, A3), complete case (O2);
- `cruxeval_lenient`: CRUXEval >= 0.8 counts as success (sensitivity);
- `rm_empty_unavailable`: empty `response_matrix.json` cells also unavailable (S1.7), complete case;
- `all_tasks`: every task, unavailable cells skipped (O2 sensitivity).

CIs: Clopper-Pearson per family x model; otherwise the cluster bootstrap over problems
(`stats.bootstrap`, N_BOOT, seed 0). Qwen3-Max's released cost is a per-dimension constant (S1.7
data note), so its per-task cost carries no information within a dimension.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s3_descriptive
Writes router/experiments/out/s3_6_*.csv, s3_3_*.csv and s3_descriptive.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from router.baselines.common import success
from router.data import coderouterbench as crb
from router.data import splits as S
from router.data import verdicts as V
from router.data.audit import (build_views, load_artefacts, load_release, response_key,
                               response_stats)
from router.data.prices import price_order
from router.data.schema import EMPTY_RESPONSE
from router.settings import N_BOOT, REPO_ROOT, VERIFIER_COST_SWEEP, VERIFIER_RUN_USD, PRICE_SWEEP
from router.sim import profile as P
from router.sim import stats

OUT = REPO_ROOT / "router" / "experiments" / "out"
VIEWS = ("all_eligible", "filtered = headline", "sensitivity")
VARIANTS = ("main", "cruxeval_lenient", "rm_empty_unavailable", "all_tasks")
OPUS = "claude-opus-4-6"
QWEN_NOTE = "Qwen3-Max's released cost is a per-dimension constant (S1.7 data note)"


# ------------------------------------------------------------------------------------- inputs
def load():
    m = load_release()
    views = build_views(m)
    if list(views["filtered"].tasks) != list(views["headline"].tasks):
        raise AssertionError("filtered and headline are no longer the same tasks")
    vs = {"all_eligible": views["all_eligible"], "filtered = headline": views["filtered"],
          "sensitivity": views["sensitivity"]}
    v = V.load_verdicts()
    obs = v.pivot(index="task_id", columns="model", values="observation")
    st = response_stats()
    keys = {response_key(x): x for x in m.models}
    st = st[st.is_empty & st.key.isin(set(keys))]
    rm_empty = [(t, keys[k]) for t, k in zip(st.task_id, st.key)]
    problems = S.load_problems().set_index("task_id")["problem_id"]
    split = S.task_splits().set_index("task_id")["split"]
    return m, vs, obs, rm_empty, problems, split


def variant(vm, name: str, rm_empty):
    """(matrix, ok, tasks) for one variant: `tasks` is the scored task index."""
    if name == "rm_empty_unavailable":
        vm = vm.mark_unavailable(rm_empty, EMPTY_RESPONSE)
    ok = success(vm, cruxeval_lenient=(name == "cruxeval_lenient"))
    av = vm.available.astype(bool)
    tasks = vm.tasks if name == "all_tasks" else vm.tasks[av.all(axis=1).to_numpy()]
    return vm, ok, tasks


def verdict_frame(vm, obs: pd.DataFrame) -> pd.DataFrame:
    """X13 observation per cell of `vm`; None where there is no record or the cell is unavailable."""
    v = obs.reindex(index=vm.tasks, columns=vm.models).astype(object)
    v = v.where(v.notna(), None)
    return v.mask(~vm.available.astype(bool), None)


def ci(frame: pd.DataFrame, stat, clusters: pd.Series) -> tuple[float, float, float]:
    if len(frame) == 0:
        return float("nan"), float("nan"), float("nan")
    iv = stats.bootstrap(frame, stat, clusters=clusters.reindex(frame.index), n_boot=N_BOOT)
    return iv.point, iv.lo, iv.hi


# --------------------------------------------------------------------------------------- S3.6
def verifier_tables(vs, obs, rm_empty, problems):
    fm_rows, pooled_rows = [], []
    for view, vm0 in vs.items():
        for var in ("main", "cruxeval_lenient", "rm_empty_unavailable"):
            vm, ok, _ = variant(vm0, var, rm_empty)
            cells = P.verifier_cells(ok, verdict_frame(vm, obs), vm.available)
            cells["family"] = vm.family.reindex(cells.task_id).to_numpy()
            for (fam, mod), g in cells.groupby(["family", "model"], sort=True):
                c = P.confusion(g)
                fp = P.clopper_pearson(c["pass_failure"], c["failures"])
                fr = P.clopper_pearson(c["fail_success"], c["successes"])
                fm_rows.append({"view": view, "variant": var, "family": fam, "model": mod, **c,
                                "false_pass_lo": fp[0], "false_pass_hi": fp[1],
                                "false_reject_lo": fr[0], "false_reject_hi": fr[1]})
            groups = [("view", "all", cells)]
            groups += [("family", f, g) for f, g in cells.groupby("family", sort=True)]
            groups += [("model", x, g) for x, g in cells.groupby("model", sort=True)]
            for level, key, g in groups:
                c = P.confusion(g)
                f = P.confusion_frame(g)
                cl = problems.reindex(f.task_id).set_axis(f.index)
                fp = ci(f, P.ratio("fp_num", "fp_den"), cl) if c["failures"] else (np.nan,) * 3
                fr = ci(f, P.ratio("fr_num", "fr_den"), cl) if c["successes"] else (np.nan,) * 3
                pooled_rows.append({"view": view, "variant": var, "level": level, "key": key, **c,
                                    "false_pass_pct_lo": fp[1], "false_pass_pct_hi": fp[2],
                                    "false_reject_pct_lo": fr[1], "false_reject_pct_hi": fr[2]})
    return label_empty_rates(pd.DataFrame(fm_rows)), label_empty_rates(pd.DataFrame(pooled_rows))


def label_empty_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Spec §5.1: a zero denominator is written "no signal" (no signal cells) or "none" (signal
    cells, but no failures / no successes), never 0."""
    df = df.copy()
    for col, den in (("false_pass_rate", "failures"), ("false_reject_rate", "successes")):
        lab = np.where(df.signal_cells == 0, P.NO_SIGNAL, np.where(df[den] == 0, P.NONE, ""))
        df[col] = [v if not s else s for v, s in zip(df[col].astype(object), lab)]
    return df


def no_signal_table(vs, obs):
    rows = []
    for view, vm in vs.items():
        ok = success(vm)
        cells = P.verifier_cells(ok, verdict_frame(vm, obs), vm.available)
        cc = set(vm.tasks[vm.available.astype(bool).all(axis=1).to_numpy()])
        rec = set(obs.index)
        fams = [("all", vm.tasks)] + [(f, vm.tasks[(vm.family == f).to_numpy()])
                                      for f in sorted(vm.family.unique())]
        for fam, tasks in fams:
            tasks = [t for t in tasks if t in rec]
            c = cells[cells.task_id.isin(set(tasks))]
            a = P.no_signal_share(c, tasks)
            b = P.no_signal_share(c[c.task_id.isin(cc)], [t for t in tasks if t in cc])
            rows.append({"view": view, "family": fam, "x13_tasks": a["tasks"],
                         "cells": a["cells"], "no_signal_cells": a["no_signal_cells"],
                         "no_signal_cell_share": a["no_signal_cell_share"],
                         "no_signal_tasks": a["no_signal_tasks"],
                         "no_signal_task_share": a["no_signal_task_share"],
                         "complete_case_tasks": b["tasks"],
                         "complete_case_no_signal_tasks": b["no_signal_tasks"],
                         "complete_case_no_signal_task_share": b["no_signal_task_share"]})
    return pd.DataFrame(rows)


def population_tables(vs, obs, rm_empty, problems, split):
    pop, mix, acc = [], [], []
    for view, vm0 in vs.items():
        fam = vm0.family
        cc0 = vm0.available.astype(bool).all(axis=1)
        dim = vm0.dimension
        for f in sorted(fam.unique()):
            ids = vm0.tasks[(fam == f).to_numpy()]
            mix.append({"view": view, "family": f, "dimension": str(dim[ids[0]]),
                        "tasks": len(ids), "share": len(ids) / len(vm0),
                        "complete_case_tasks": int(cc0.loc[ids].sum()),
                        "development_tasks": int((split.reindex(ids) == S.DEV).sum()),
                        "evaluation_tasks": int((split.reindex(ids) == S.EVAL).sum())})
        for var in VARIANTS:
            vm, ok, tasks = variant(vm0, var, rm_empty)
            o = ok.loc[tasks]
            cl = problems.reindex(tasks)
            rates = {}
            for mod in vm.models:
                av = vm.available.loc[tasks, mod].astype(bool)
                f = pd.DataFrame({"a": o.loc[av.to_numpy(), mod].astype(float)})
                p, lo, hi = ci(f, stats.rate("a"), cl)
                rates[mod] = p
                acc.append({"view": view, "variant": var, "model": mod, "tasks": len(f),
                            "successes": int(f.a.sum()), "acceptance": p,
                            "acceptance_lo": lo, "acceptance_hi": hi})
            best = sorted(vm.models, key=lambda x: (-rates[x], x))[0]
            uns = P.unsolved(o, vm.available.loc[tasks])
            f = pd.DataFrame({"u": uns.astype(float)})
            up, ulo, uhi = ci(f, stats.rate("u"), cl)
            pop.append({"view": view, "variant": var, "tasks_scored": len(tasks),
                        "tasks": len(vm), "problems": int(problems.reindex(vm.tasks).nunique()),
                        "development_tasks": int((split.reindex(vm.tasks) == S.DEV).sum()),
                        "evaluation_tasks": int((split.reindex(vm.tasks) == S.EVAL).sum()),
                        "x13_tasks": int(vm.tasks.isin(obs.index).sum()),
                        "families": int(fam.nunique()),
                        "best_single_model": best, "best_single_acceptance": rates[best],
                        "solved_by_some_model_pct": 100 - up,
                        "unsolved_pct": up, "unsolved_pct_lo": ulo, "unsolved_pct_hi": uhi})
    return pd.DataFrame(pop), pd.DataFrame(mix), pd.DataFrame(acc)


# --------------------------------------------------------------------------------------- S3.3
def unsolved_table(vs, rm_empty, problems):
    rows = []
    for view, vm0 in vs.items():
        for var in VARIANTS:
            vm, ok, tasks = variant(vm0, var, rm_empty)
            uns = P.unsolved(ok.loc[tasks], vm.available.loc[tasks])
            fam = vm.family.loc[tasks]
            for f in ["all"] + sorted(fam.unique()):
                u = uns if f == "all" else uns[(fam == f).to_numpy()]
                p, lo, hi = ci(pd.DataFrame({"u": u.astype(float)}), stats.rate("u"),
                               problems.reindex(u.index))
                rows.append({"view": view, "variant": var, "family": f, "tasks": len(u),
                             "unsolved": int(u.sum()), "unsolved_pct": p,
                             "unsolved_pct_lo": lo, "unsolved_pct_hi": hi})
    return pd.DataFrame(rows)


def rescue_table(vs, obs, rm_empty, problems):
    rows = []
    for view, vm0 in vs.items():
        for var in ("main", "cruxeval_lenient", "rm_empty_unavailable"):
            vm, ok, tasks = variant(vm0, var, rm_empty)
            o = ok.loc[tasks]
            vf = verdict_frame(vm, obs).loc[tasks]
            corder = P.mean_cost_order(vm.cost.loc[tasks], vm.available.loc[tasks])
            r = P.rescue_pairs(o, vf, price_order(vm.models), corder)
            r.insert(0, "variant", var)
            r.insert(0, "view", view)
            if var == "main":
                cl = problems.reindex(tasks)
                lims = []
                for lo, hi in zip(r.lower, r.higher):
                    f = P.rescue_frame(o, vf, lo, hi)
                    t = ci(f, P.ratio("t_num", "t_den"), cl) if f.t_den.sum() else (np.nan,) * 3
                    b = ci(f, P.ratio("o_num", "o_den"), cl) if f.o_den.sum() else (np.nan,) * 3
                    lims.append((t[1], t[2], b[1], b[2]))
                r[["true_rescue_pct_lo", "true_rescue_pct_hi", "observable_rescue_pct_lo",
                   "observable_rescue_pct_hi"]] = lims
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def _ladder_row(label, view, var, ok, cost, avail, order, verifier_usd=0.0, price=None):
    lc = P.ladder_costs(ok, cost, avail, order, verifier_usd)
    av = avail.astype(bool)
    acc = (ok & av).mean()
    best = sorted(ok.columns, key=lambda x: (-acc[x], x))[0]
    s = P.ladder_summary(lc, cost[OPUS].where(av[OPUS], 0.0))
    return {"definition": label, "view": view, "variant": var, "order": ">".join(order),
            "verifier_usd_per_rung": verifier_usd, "price_scaled": price or "",
            "best_single_model": best,
            "always_best_usd": float(cost[best].where(av[best], 0.0).sum()), **s}


def old_historical_row() -> tuple[dict, pd.Index]:
    """The 2026-11-10 definition: ID-test split, complete view, all dimensions, score >= 0.5,
    missing token records counted as $0 and score 0, no unavailable cells. Returns the row and
    its task ids."""
    L = crb.long_table("id_test", "complete")
    sc = L.pivot(index="task_id", columns="model", values="score")
    co = L.pivot(index="task_id", columns="model", values="cost_usd").reindex(sc.index)[sc.columns].fillna(0.0)
    every = pd.DataFrame(True, index=sc.index, columns=sc.columns)
    return (_ladder_row("old: id_test, all dimensions, score >= 0.5", "id_test (2026-11-10)", "old",
                        sc >= 0.5, co, every, P.mean_cost_order(co, every)), sc.index)


def historical_table(vs, m, rm_empty):
    old, old_ids = old_historical_row()
    rows = [old]
    # (2) the same tasks under S0.1 with unavailable rungs skipped
    ids = [t for t in m.tasks if t in set(old_ids)]
    mi = m.subset(ids)
    rows.append(_ladder_row("bridge: id_test, all dimensions, success 1.0, unavailable skipped",
                            "id_test (2026-11-10)", "all_tasks", success(mi), mi.cost,
                            mi.available, P.mean_cost_order(mi.cost, mi.available)))
    # (3) the new definition per view, with its sensitivities
    sweep = []
    for view, vm0 in vs.items():
        for var in VARIANTS:
            vm, ok, tasks = variant(vm0, var, rm_empty)
            o, c, a = ok.loc[tasks], vm.cost.loc[tasks], vm.available.loc[tasks]
            corder = P.mean_cost_order(c, a)
            rows.append(_ladder_row("new: both ID splits, success 1.0", view, var, o, c, a, corder))
            if var != "main":
                continue
            rows.append(_ladder_row("new, price-order ladder", view, var, o, c, a,
                                    price_order(vm.models)))
            for k in VERIFIER_COST_SWEEP:
                rows.append(_ladder_row(f"new, A1 verifier charged x{k} per rung reached", view,
                                        var, o, c, a, corder, verifier_usd=k * VERIFIER_RUN_USD))
            for mod in vm.models:
                for fct in PRICE_SWEEP:
                    cs = c.copy()
                    cs[mod] = cs[mod] * fct
                    # the ladder order is kept: a rescored sweep (spec D11, descriptive figures)
                    sweep.append(_ladder_row("new, price sweep (rescored)", view, var, o, cs, a,
                                             corder, price=f"{mod} x{fct:g}"))
    return pd.DataFrame(rows), pd.DataFrame(sweep)


# ------------------------------------------------------------------------------------- bridges
def plan_verifier_bridge() -> dict:
    """The plan's 2026-11-19 figures: every X13 cell with a binary released score, any family,
    empty cells included; false-pass = P(accept | truth 0), false-reject = P(reject | truth 1)."""
    raw = V.load_raw()
    truth = pd.to_numeric(raw.truth, errors="coerce")
    n = raw.w1_n.astype(int)
    acc = raw.w1_accept.eq("True")
    out = {}
    for name, sel in (("visible_tests", n > 0), ("no_visible_tests_heuristic", n == 0)):
        b = truth.isin([0.0, 1.0]) & sel
        t1 = truth[b] == 1.0
        a = acc[b]
        out[name] = {"cells": int(b.sum()),
                     "false_pass_pct": 100 * float(a[~t1].mean()),
                     "false_reject_pct": 100 * float((~a[t1]).mean())}
    return out


def checks(pop, ns, pooled, art) -> dict:
    """Recounts against artefacts.yaml (S1.7) and the plan."""
    f = pop[(pop.view == "filtered = headline")].set_index("variant")
    h = art["views"]["headline"]
    rm = pop[pop.variant == "rm_empty_unavailable"].set_index("view").tasks_scored
    rm_checks = {f"{v}_complete_case_if_empty_responses_unavailable":
                 [int(rm[v]), art["views"][k]["complete_case_tasks_if_empty_responses_unavailable"]]
                 for v, k in (("all_eligible", "all_eligible"), ("filtered = headline", "headline"),
                              ("sensitivity", "sensitivity"))}
    nsf = ns[(ns.view == "filtered = headline") & (ns.family == "all")].iloc[0]
    out = {
        "filtered_complete_case_tasks": [int(f.loc["main", "tasks_scored"]), h["complete_case_tasks"]],
        "filtered_unsolved_strict": [round(float(f.loc["main", "unsolved_pct"]) * f.loc["main", "tasks_scored"] / 100),
                                     h["strict"]["complete_case_tasks_solved_by_no_model"]],
        "filtered_unsolved_lenient": [round(float(f.loc["cruxeval_lenient", "unsolved_pct"]) * f.loc["cruxeval_lenient", "tasks_scored"] / 100),
                                      h["cruxeval_lenient_sensitivity"]["complete_case_tasks_solved_by_no_model"]],
        "filtered_complete_case_no_signal_tasks": [int(nsf.complete_case_no_signal_tasks),
                                                   h["verifier_signal"]["complete_case_tasks_no_signal_any_model"]],
        "filtered_signal_cells": [int(nsf.cells - nsf.no_signal_cells),
                                  h["verifier_signal"]["cells_with_signal"]],
        **rm_checks,
    }
    return {k: {"here": v[0], "artefacts_yaml": v[1], "equal": v[0] == v[1]} for k, v in out.items()}


# ---------------------------------------------------------------------------------------- main
def _csv(df: pd.DataFrame, name: str, keys) -> None:
    df.sort_values(list(keys), kind="stable").to_csv(OUT / name, index=False, lineterminator="\n")


def run():
    m, vs, obs, rm_empty, problems, split = load()
    fm, pooled = verifier_tables(vs, obs, rm_empty, problems)
    ns = no_signal_table(vs, obs)
    pop, mix, acc = population_tables(vs, obs, rm_empty, problems, split)
    uns = unsolved_table(vs, rm_empty, problems)
    resc = rescue_table(vs, obs, rm_empty, problems)
    hist, sweep = historical_table(vs, m, rm_empty)
    art = load_artefacts()
    rng = (sweep.groupby(["view", "variant"])[["ladder_usd", "always_usd", "perfect_stop_usd"]]
           .agg(["min", "max"]))
    res = {
        "spec": "docs/research/S3_analysis_spec.md §5",
        "views": {v: {"tasks": len(x), "task_ids_sha256": art["views"]["headline" if v.startswith("filtered") else v]["task_ids_sha256"]}
                  for v, x in vs.items()},
        "filtered_equals_headline": True,
        "plan_verifier_bridge": plan_verifier_bridge(),
        "historical_old_definition": {k: hist.iloc[0][k] for k in
                                      ("tasks", "ladder_usd", "always_usd", "perfect_stop_usd", "unsolved_share")},
        "historical_price_sweep_range": {f"{v}|{var}": {f"{a}_{b}": float(rng.loc[(v, var), (a, b)])
                                                        for a, b in rng.columns}
                                         for v, var in rng.index},
        "checks_against_artefacts": checks(pop, ns, pooled, art),
        "rm_empty_cells": len(rm_empty),
        "notes": [QWEN_NOTE,
                  "cells: available (not an empty X13 response, not a missing token record) with "
                  "an X13 record; artefacts.yaml's signal counts keep missing-token cells",
                  "rates with a zero denominator are empty in the CSVs: 'no signal' when there are "
                  "no signal cells, 'none' when there are cells but no failures or no successes"],
        "n_boot": N_BOOT,
    }
    return res, {"s3_6_verifier_family_model.csv": (fm, ["view", "variant", "family", "model"]),
                 "s3_6_verifier_pooled.csv": (pooled, ["view", "variant", "level", "key"]),
                 "s3_6_no_signal.csv": (ns, ["view", "family"]),
                 "s3_6_population.csv": (pop, ["view", "variant"]),
                 "s3_6_family_mix.csv": (mix, ["view", "family"]),
                 "s3_6_model_acceptance.csv": (acc, ["view", "variant", "model"]),
                 "s3_3_unsolved.csv": (uns, ["view", "variant", "family"]),
                 "s3_3_rescue_pairs.csv": (resc, ["view", "variant", "lower", "higher"]),
                 "s3_3_historical.csv": (hist, []),
                 "s3_3_historical_price_sweep.csv": (sweep, ["view", "variant", "price_scaled"])}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in sorted(x.items())}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, tables = run()
    for name, (df, keys) in tables.items():
        if keys:
            _csv(df, name, keys)
        else:
            df.to_csv(OUT / name, index=False, lineterminator="\n")
    (OUT / "s3_descriptive.json").write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    print(json.dumps(_clean({k: res[k] for k in ("plan_verifier_bridge", "historical_old_definition",
                                                 "checks_against_artefacts")}), indent=1))
