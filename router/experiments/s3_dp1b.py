"""S3.2 and DP1b: the oracle-on-no-signal feasibility bound on the development split (S3 spec D4,
D5, D6, D11; plan §1; CLAIMS S0.3). The only module allowed to import `baselines.oracle_bound`.

Population (D6, user): the headline development split, complete case (195 tasks, 180 problems,
3 cells), `stats.dev_folds` per seed; every policy is fitted on four folds and scores the fifth
(`baselines.run.run_dev`); the cell predictor is refit per training fold. Other views: the
development split's X13 tasks of `all_eligible` and `sensitivity`, complete case, main variant.

Decision (D4 (a)): per-cell Rule A in bound mode with predicted cells (`bound_ruleA_pred`), gate
seed 0, point estimates, x1 verifier cost, base prices. DP1b holds when, at the same time:
(a) saving >= 20% and gap >= -2 points against the best single model; (c) saving > 0 and gap >= -2
points against CellBest with predicted cells. CIs (`gate_interval`), seeds 1 to 4, sweeps and every
other row are information only.

Reported beside it: Rule B in bound mode; Rules A and B with no-signal cells accepted and
escalated; the fixed cascades; CellBest with true cells; the 84 fixed ladders in accept, escalate
and bound mode (in-sample maxima, labelled so); the gap the learned router must close (D5 (a)); the
verifier-cost and price sweeps, refitting every policy at each point (D11 (a)).

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s3_dp1b
Writes router/experiments/out/s3_2_dp1b.csv, s3_2_dp1b_seeds.csv, s3_2_ladders.csv,
s3_2_sweeps.csv and s3_2_dp1b.json.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd

from router.baselines.cascade import FixedCascade
from router.baselines.common import (ACCEPT, ESCALATE, Context, ladders, policy_outcomes, success)
from router.baselines.lookup import CellBest, CellRule
from router.baselines.oracle_bound import (VERIFIER_MODE, BoundRule, OracleOnNoSignal,
                                           oracle_ladders)
from router.baselines.run import run_dev
from router.baselines.static import BestSingle
from router.data import splits as S
from router.data.audit import build_views, load_release, response_key, response_stats
from router.data.prompts import join
from router.data.schema import EMPTY_RESPONSE, OutcomeMatrix
from router.settings import PRICE_SWEEP, REPO_ROOT, VERIFIER_COST_SWEEP, VERIFIER_RUN_USD
from router.sim import metrics, stats
from router.sim.verifier import ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
HEADLINE = "filtered = headline"
VARIANTS = ("main", "cruxeval_lenient", "rm_empty_unavailable", "all_tasks")
OTHER_VIEWS = ("all_eligible", "sensitivity")
BEST, CBP, BOUND, RULEA_ESC = "best_single", "cellbest_pred", "bound_ruleA_pred", "ruleA_escalate_pred"
REFS = (BEST, CBP)
SAVING_MIN_BEST, GAP_MIN = 20.0, -2.0
QWEN_NOTE = "Qwen3-Max's released cost is a per-dimension constant (S1.7 data note)"


def policies():
    """(policy, uses the oracle verifier). kNN is not run in S3 (its embedding pin has not moved
    into fetch_data.py; brief, reviewer item 3)."""
    return [(BestSingle(), False), (CellBest("pred"), False), (CellBest("true"), False),
            *[(CellRule(r, mode, "pred"), False) for mode in (ACCEPT, ESCALATE) for r in ("A", "B")],
            (FixedCascade(ACCEPT), False), (FixedCascade(ESCALATE), False),
            (BoundRule("A", "pred"), True), (BoundRule("B", "pred"), True),
            (BoundRule("A", "true"), True)]


def population_label(var: str) -> str:
    return "all tasks (available cells)" if var == "all_tasks" else "complete case"


# ------------------------------------------------------------------------------- populations
_IN = {}


def inputs():
    if not _IN:
        m = load_release()
        views = build_views(m)
        ts = S.task_splits().set_index("task_id")
        st = response_stats()
        keys = {response_key(x): x for x in m.models}
        st = st[st.is_empty & st.key.isin(set(keys))]
        v = ReplayVerifier.from_x13()
        dev = set(ts.index[ts.split == S.DEV])
        _IN.update(views=views, dev=dev, problems=ts.problem_id, verifier=v,
                   rm_empty=[(t, keys[k]) for t, k in zip(st.task_id, st.key)])
        tasks = sorted({t for name in ("headline", *OTHER_VIEWS) for t in views[name].tasks if t in dev})
        _IN["content"] = join(tasks)
    return _IN


def matrix(view: str, var: str) -> OutcomeMatrix:
    d = inputs()
    v = d["views"]["headline" if view == HEADLINE else view]
    m = v.subset([t for t in v.tasks if t in d["dev"] and d["verifier"].has_record(t)])
    if var == "rm_empty_unavailable":
        m = m.mark_unavailable(d["rm_empty"], EMPTY_RESPONSE)
    if var != "all_tasks":
        m = m.subset(list(m.tasks[m.available.astype(bool).all(axis=1).to_numpy()]))
    return m


def scaled(m: OutcomeMatrix, model: str, factor: float) -> OutcomeMatrix:
    cost = m.cost.copy()
    cost[model] = cost[model] * factor
    return OutcomeMatrix(m.score.copy(), cost, m.dimension.copy(), m.family.copy(), m.name,
                         available=m.available.copy(), unavailable_reason=m.unavailable_reason.copy())


# --------------------------------------------------------------------------------------- runs
def run_all(m: OutcomeMatrix, seed: int, *, lenient: bool = False, verifier_scale: float = 1.0,
            only=None) -> dict[str, pd.DataFrame]:
    d = inputs()
    base = d["verifier"]
    ctx = Context(base, content=d["content"], problem_of=d["problems"], seed=seed,
                  verifier_usd=verifier_scale * VERIFIER_RUN_USD)
    octx = replace(ctx, verifier=OracleOnNoSignal.from_matrix(base, m, cruxeval_lenient=lenient))
    folds = stats.dev_folds(m.tasks, seed)
    out = {}
    for pol, oracle in policies():
        if only is not None and pol.name not in only:
            continue
        o = run_dev(pol, m, octx if oracle else ctx, folds, seed=seed, cruxeval_lenient=lenient)
        if oracle and o.attrs.get("verifier_mode") != VERIFIER_MODE:
            raise AssertionError("bound rows must carry the oracle verifier mode")
        if verifier_scale != 1.0:      # scored cost at the swept verifier cost (A1 x k per run)
            o = o.copy()
            o["verifier_usd"] = o.n_verifier_runs * VERIFIER_RUN_USD * verifier_scale
            o["cost_usd"] = o.model_cost_usd + o.verifier_usd + o.probe_usd
        out[pol.name] = o
    return out


def compare(o, ref, problems) -> dict:
    f = metrics.paired_frame(o, ref, ("x", "r"))
    cl = problems.reindex(f.index)
    gap = stats.gate_interval(f, stats.mean_diff("accepted_x", "accepted_r"), cl)
    sav = stats.gate_interval(f, stats.saving_pct("cost_x", "cost_r"), cl)
    return {"gap": gap.point, "gap_lo": gap.lo, "gap_hi": gap.hi,
            "saving": sav.point, "saving_lo": sav.lo, "saving_hi": sav.hi, "n_problems": gap.n_units}


def point_compare(o, ref) -> dict:
    o, ref = o.loc[ref.index], ref
    return {"gap": 100 * float(o["accepted"].mean() - ref["accepted"].mean()),
            "saving": 100 * (1 - float(o.cost_usd.sum()) / float(ref.cost_usd.sum()))}


def rows_for(outs: dict, view: str, var: str, seed: int, problems) -> list[dict]:
    rows = []
    for name, o in outs.items():
        r = {"view": view, "variant": var, "population": population_label(var), "seed": seed,
             "gate_seed": seed == stats.GATE_SEED, "policy": name,
             "verifier_mode": o.attrs.get("verifier_mode"),
             "buildable": not name.startswith("bound"),
             "given_more_information": ",".join(o.attrs.get("more_information", ())),
             "acceptance": metrics.acceptance(o), "cost_usd": float(o.cost_usd.sum()),
             "cost_musd_per_task": 1e3 * float(o.cost_usd.mean()),
             "n_verifier_runs": int(o.n_verifier_runs.sum()), "n_tasks": len(o)}
        for tag in REFS:
            if name == tag:
                continue
            c = compare(o, outs[tag], problems)
            for k, v in c.items():
                if k == "n_problems":
                    continue
                stat, _, end = k.partition("_")          # gap, gap_lo -> gap_vs_<tag>, gap_vs_<tag>_lo
                r[f"{stat}_vs_{tag}" + (f"_{end}" if end else "")] = v
            r["n_problems"] = c["n_problems"]
        rows.append(r)
    return rows


# --------------------------------------------------------------------------------- decision
def decision(seeds: pd.DataFrame) -> dict:
    d = seeds[(seeds.view == HEADLINE) & (seeds.variant == "main") & (seeds.policy == BOUND)]
    d = d.set_index("seed")
    g = d.loc[stats.GATE_SEED]
    crit = {
        "a_saving_vs_best_single": ("saving_vs_best_single", SAVING_MIN_BEST, False),
        "a_gap_vs_best_single": ("gap_vs_best_single", GAP_MIN, False),
        "c_saving_vs_cellbest_pred": ("saving_vs_cellbest_pred", 0.0, True),
        "c_gap_vs_cellbest_pred": ("gap_vs_cellbest_pred", GAP_MIN, False)}
    out = {}
    for key, (col, thr, strict) in crit.items():
        p = float(g[col])
        ok = p > thr if strict else p >= thr
        out[key] = {"point": p, "threshold": thr, "strict": strict, "meets": bool(ok),
                    "margin": p - thr, "ci": [float(g[col + "_lo"]), float(g[col + "_hi"])],
                    "label": stats.threshold_label(d[col].to_dict(), float(g[col + "_lo"]), thr,
                                                   strict=strict),
                    "by_seed": d[col].to_dict()}
    return {"statistic": "per-cell Rule A in bound mode, predicted cells (spec D4 (a)); gate seed 0, "
                         "point estimates, x1 verifier cost, base prices; headline development "
                         "split, complete case",
            "policy": BOUND, "n_tasks": int(g.n_tasks), "n_problems": int(g.n_problems),
            "criteria": out, "holds": bool(all(v["meets"] for v in out.values())),
            "note": "CIs, seeds 1 to 4 and the at-threshold labels are information only (S0.3)"}


def router_gap(outs: dict, problems) -> dict:
    """D5 (a): (delta saving, delta gap) of the bound over Rule A escalated, per reference, and the
    paired CI of the bound against Rule A escalated directly."""
    b, r = outs[BOUND], outs[RULEA_ESC]
    res = {}
    for tag in REFS:
        cb, cr = point_compare(b, outs[tag]), point_compare(r, outs[tag])
        res[f"vs_{tag}"] = {"delta_saving_points": cb["saving"] - cr["saving"],
                            "delta_gap_points": cb["gap"] - cr["gap"],
                            "bound": cb, "rule_a_escalated": cr}
    res["bound_vs_rule_a_escalated"] = compare(b, r, problems)
    return res


# ------------------------------------------------------------------------------ fixed ladders
def ladder_table(m: OutcomeMatrix, outs0: dict, lenient: bool) -> tuple[pd.DataFrame, dict]:
    """The 84 fixed ladders on the whole population in accept, escalate and bound mode (in-sample:
    ladders ordered by the population's mean cost and evaluated on the same tasks), against the
    seed-0 best single model and CellBest-pred."""
    base = inputs()["verifier"]
    bound_v = OracleOnNoSignal.from_matrix(base, m, cruxeval_lenient=lenient)
    rows = []
    sets = {ACCEPT: (ladders(m, ACCEPT), base), ESCALATE: (ladders(m, ESCALATE), base),
            "bound": (oracle_ladders(m), bound_v)}
    for mode, (pols, ver) in sets.items():
        A, C = policy_outcomes(m, ver, pols, cruxeval_lenient=lenient)
        for j, p in enumerate(pols):
            r = {"mode": mode, "ladder": ">".join(p.rungs), "acceptance": 100 * A[:, j].mean(),
                 "cost_usd": float(C[:, j].sum())}
            for tag in REFS:
                ref = outs0[tag].loc[m.tasks]
                r[f"gap_vs_{tag}"] = 100 * (A[:, j].mean() - ref["accepted"].mean())
                r[f"saving_vs_{tag}"] = 100 * (1 - C[:, j].sum() / ref.cost_usd.sum())
            r["meets_a"] = r["saving_vs_best_single"] >= SAVING_MIN_BEST and r["gap_vs_best_single"] >= GAP_MIN
            r["meets_c"] = r["saving_vs_cellbest_pred"] > 0 and r["gap_vs_cellbest_pred"] >= GAP_MIN
            rows.append(r)
    t = pd.DataFrame(rows)
    summ = {}
    for mode, g in t.groupby("mode"):
        a20 = g[g.saving_vs_best_single >= SAVING_MIN_BEST]
        c0 = g[g.saving_vs_cellbest_pred > 0]
        summ[mode] = {"n_ladders": len(g), "n_meeting_a": int(g.meets_a.sum()),
                      "n_meeting_c": int(g.meets_c.sum()), "n_meeting_both": int((g.meets_a & g.meets_c).sum()),
                      "best_gap_vs_best_single_among_saving_ge_20": float(a20.gap_vs_best_single.max()) if len(a20) else None,
                      "best_gap_vs_cellbest_pred_among_saving_gt_0": float(c0.gap_vs_cellbest_pred.max()) if len(c0) else None,
                      "label": "in-sample maximum"}
    return t, summ


# ------------------------------------------------------------------------------------- sweeps
SWEEP_POLICIES = {BEST, CBP, RULEA_ESC, BOUND}


def sweep_rows(m: OutcomeMatrix) -> pd.DataFrame:
    rows = []
    points = [("verifier", f"x{k}", None, k) for k in VERIFIER_COST_SWEEP]
    points += [("price", f"{mod} x{f:g}", (mod, f), 1.0) for mod in m.models for f in PRICE_SWEEP]
    for kind, label, sc, vk in points:
        mm = m if sc is None else scaled(m, *sc)
        outs = run_all(mm, stats.GATE_SEED, verifier_scale=vk, only=SWEEP_POLICIES)
        for name in (BOUND, RULEA_ESC, CBP):
            r = {"sweep": kind, "point": label, "policy": name, "refit": True,
                 "acceptance": metrics.acceptance(outs[name]),
                 "cost_usd": float(outs[name].cost_usd.sum())}
            for tag in REFS:
                if tag == name:
                    continue
                c = point_compare(outs[name], outs[tag])
                r[f"gap_vs_{tag}"], r[f"saving_vs_{tag}"] = c["gap"], c["saving"]
            if name == BOUND:
                r["meets_a"] = r["saving_vs_best_single"] >= SAVING_MIN_BEST and r["gap_vs_best_single"] >= GAP_MIN
                r["meets_c"] = r["saving_vs_cellbest_pred"] > 0 and r["gap_vs_cellbest_pred"] >= GAP_MIN
            rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- main
def run():
    d = inputs()
    problems = d["problems"]
    seed_rows, outs0, pops = [], {}, {}
    for var in VARIANTS:
        m = matrix(HEADLINE, var)
        pops[f"{HEADLINE}|{var}"] = {"tasks": len(m), "problems": int(problems.reindex(m.tasks).nunique()),
                                     "cells": int((m.dimension.astype(str) + "|" + m.family.astype(str)).nunique())}
        for seed in stats.SEEDS:
            outs = run_all(m, seed, lenient=var == "cruxeval_lenient")
            if seed == stats.GATE_SEED:
                outs0[var] = (m, outs)
            seed_rows += rows_for(outs, HEADLINE, var, seed, problems)
    for view in OTHER_VIEWS:
        m = matrix(view, "main")
        pops[f"{view}|main"] = {"tasks": len(m), "problems": int(problems.reindex(m.tasks).nunique()),
                                "cells": int((m.dimension.astype(str) + "|" + m.family.astype(str)).nunique())}
        for seed in stats.SEEDS:
            outs = run_all(m, seed)
            if seed == stats.GATE_SEED:
                outs0[view] = (m, outs)
            seed_rows += rows_for(outs, f"{view} (development X13)", "main", seed, problems)
    seeds = pd.DataFrame(seed_rows)

    m0, o0 = outs0["main"]
    ladders_t, ladders_s = ladder_table(m0, o0, False)
    sweeps = sweep_rows(m0)
    sw = sweeps[sweeps.policy == BOUND]
    res = {"spec": "docs/research/S3_analysis_spec.md D4, D5, D6, D11",
           "populations": pops, "dp1b": decision(seeds),
           "gap_learned_router_must_close": router_gap(o0, problems),
           "gap_by_view": {k: router_gap(v[1], problems) for k, v in outs0.items() if k != "main"},
           "fixed_ladders": ladders_s,
           "sweeps_bound_range": {c: [float(sw[c].min()), float(sw[c].max())] for c in
                                  ("saving_vs_best_single", "gap_vs_best_single",
                                   "saving_vs_cellbest_pred", "gap_vs_cellbest_pred")},
           "sweeps_bound_meeting_a_and_c": int((sw.meets_a & sw.meets_c).sum()),
           "sweeps_points": len(sw),
           "fits_seed0": {name: [{k: f[k] for k in ("fold", "table", "fallback", "chosen")}
                                 for f in o.attrs["fits"]] for name, o in o0.items()},
           "notes": [QWEN_NOTE, "bound rows read the truth on no-signal cells and are not buildable "
                                "(verifier_mode oracle_on_no_signal)",
                     "kNN is not run in S3: its embedding pin is not in fetch_data.py",
                     "CRUXEval >= 0.8: rules chosen under S0.1, scored lenient; the bound's truth "
                     "is lenient (spec §7)"]}
    return res, seeds, ladders_t, sweeps


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, seeds, lad, sweeps = run()
    seeds = seeds.sort_values(["view", "variant", "policy", "seed"], kind="stable")
    seeds.to_csv(OUT / "s3_2_dp1b_seeds.csv", index=False, lineterminator="\n")
    seeds[seeds.gate_seed].to_csv(OUT / "s3_2_dp1b.csv", index=False, lineterminator="\n")
    lad.sort_values(["mode", "ladder"], kind="stable").to_csv(OUT / "s3_2_ladders.csv", index=False,
                                                             lineterminator="\n")
    sweeps.to_csv(OUT / "s3_2_sweeps.csv", index=False, lineterminator="\n")
    (OUT / "s3_2_dp1b.json").write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    print(json.dumps(_clean({"dp1b": res["dp1b"], "gap": res["gap_learned_router_must_close"],
                             "ladders": res["fixed_ladders"], "populations": res["populations"]}),
                     indent=1))
