"""P0.5 provisional recount (docs/IMPLEMENTATION_PLAN.md, Pre-S0).

Every number in docs/research/P0.5_recount.md comes from this script, run from the repo root on
the pinned data (scripts/fetch_data.py --verify-only must print `fetched: 0, verified: 187`):

    pipeline/venv/bin/python router/experiments/p0_5_recount.py

Modes (default `all`):
  repro    re-implements the 2026-11-19 inline computation unchanged and asserts its numbers;
  recount  treats empty X13 responses and the 148 missing Qwen3-Max runs as unavailable (S0.11),
           recomputes the no-signal shares and the S3.2 feasibility table (complete case, with the
           old treatment next to it), adds verifier cost, and repeats the table with the best
           single model chosen inside training folds;
  ra       reproduces Route Anything's CodeRouterBench zero-loss route (arXiv 2611.05275, Table 10)
           and then scores its lookup baselines on our population against S0.3.

Results are written to router/experiments/out/p0_5_recount.json. All numbers are provisional:
S0.1-S0.14 are not signed off and S1-S3 redo them on the final population.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.settings import EXTERNAL_DIR  # noqa: E402

X13_VERDICTS = EXTERNAL_DIR / "route-anything-data/experiments_x/X13_memory/cache/verdicts.jsonl"
CRB_DIR = EXTERNAL_DIR / "CodeRouterBench"
CRB_FILES = ("id_probing_results_long.csv", "id_test_results_long.csv")
OUT_JSON = REPO_ROOT / "router/experiments/out/p0_5_recount.json"

OPUS = "claude-opus-4-6"
EXEC_DIMS = ["code_generation", "data_science", "algorithm", "code_completion", "test_generation"]
SUSPECTED = re.compile(r"^dim\d+_(lcb_|mbpp_testgen|he_testgen)")
MISSING_RUN = "missing_token_record_zero_total"
# S0.2: X13 mean visible-test run time (0.074 s) at an on-demand vCPU-hour rate, about 1e-6 USD.
VERIFIER_RUN_USD = 1e-6
GATE_SAVING, GATE_GAP = 20.0, -2.0          # S0.3 thresholds (percent, points)
N_FOLDS, N_SEEDS, GATE_SEED = 5, 5, 0       # S0.6 cross-fitting; S0.8 gate on one pre-registered seed


# ----------------------------------------------------------------------------- data

def problem_id(t: str) -> str:
    """Problem id that keeps variants of one problem together.

    Copied from router/experiments/e0_power/power.py (importing it would run that script).
    Provisional: S1.6 replaces it with ids from source ids and tasks.jsonl.
    """
    s = re.sub(r"^dim\d+_", "", t)
    m = re.search(r"HumanEval[_/]?(\d+)", s)
    if m:
        return f"he{m.group(1)}"
    m = re.search(r"mbpp(?:_completion|_testgen)?_(\d+)", s)
    if m:
        return f"mbpp{m.group(1)}"
    m = re.search(r"BigCodeBench/(\d+)", s)
    if m:
        return f"bcb{m.group(1)}"
    m = re.search(r"lcb(?:_testgen)?_(.+)$", s)
    if m:
        return f"lcb{m.group(1)}"
    return s


def load():
    """Load X13 verdicts and the CodeRouterBench long CSVs as task x model frames."""
    v = pd.read_json(X13_VERDICTS, lines=True, dtype=str)
    L = pd.concat([pd.read_csv(CRB_DIR / f) for f in CRB_FILES], ignore_index=True)
    assert len(L) == 79_992 and len(v) == 23_288
    L["cost_usd"] = L.cost_usd.fillna(0)
    d = {"v": v, "L": L}
    S = L.pivot_table(index="task_id", columns="model", values="score")
    d["S"] = S
    d["C"] = L.pivot_table(index="task_id", columns="model", values="cost_usd").reindex(S.index)[S.columns]
    meta = L.drop_duplicates("task_id").set_index("task_id").reindex(S.index)
    d["dim"], d["split"] = meta.dimension, meta.split
    d["missing_run"] = (L.assign(x=L.cost_source == MISSING_RUN)
                        .pivot_table(index="task_id", columns="model", values="x", aggfunc="first")
                        .reindex(S.index)[S.columns].astype(bool))
    v["acc"] = v.w1_accept == "True"
    v["n"] = v.w1_n.astype(int)
    v["is_empty"] = v["empty"] == "True"
    piv = lambda col: v.pivot_table(index="task_id", columns="model", values=col, aggfunc="first")
    d["A"], d["N"], d["E"] = piv("acc"), piv("n"), piv("is_empty")
    d["family"] = v.drop_duplicates("task_id").set_index("task_id").source
    return d


def headline_mask(d):
    """The provisional 1,971-task headline set (S0.11): exec dimensions, suspected families removed,
    tasks with X13 verdicts."""
    S = d["S"]
    ex = d["dim"].isin(EXEC_DIMS)
    susp = pd.Series([bool(SUSPECTED.search(t)) for t in S.index], index=S.index)
    return ex & ~susp & S.index.isin(set(d["v"].task_id))


class Pop:
    """One population: binary success B (score = 1.0, S0.1), cost c, X13 verdicts A and test counts N."""

    def __init__(self, d, mask):
        self.ids = d["S"].index[mask]
        self.B = (d["S"].loc[self.ids] == 1.0).astype(float)
        self.c = d["C"].loc[self.ids]
        self.A = d["A"].reindex(self.ids)[self.B.columns]
        self.N = d["N"].reindex(self.ids)[self.B.columns]
        self.E = d["E"].reindex(self.ids)[self.B.columns].fillna(False).astype(bool)
        self.M = d["missing_run"].loc[self.ids]
        self.unavail = self.E | self.M
        self.family = d["family"].reindex(self.ids)
        self.dim = d["dim"].loc[self.ids]
        self.problem = pd.Series([problem_id(t) for t in self.ids], index=self.ids)
        self.models = list(self.B.columns)

    def __len__(self):
        return len(self.ids)

    def subset(self, keep):
        p = object.__new__(Pop)
        for k, val in self.__dict__.items():
            p.__dict__[k] = val[keep] if isinstance(val, (pd.Series, pd.DataFrame)) else val
        p.ids = self.ids[np.asarray(keep)]
        return p


# ----------------------------------------------------------------------------- ladders (S3.2)

def run_ladder(p: Pop, lad, mode, ver_cost=0.0):
    """Replay a fixed ladder. A rung is accepted when its visible-test verdict accepts (cells with
    visible tests); on cells with no visible tests the decision is `accept`, `escalate`, or the
    released score (`bound`, the oracle-on-no-signal bound and the only place it is read). The last
    rung is always accepted. Returns per-task acceptance and cost arrays. With ver_cost > 0 one
    verifier run is charged on every reached rung whose cell has visible tests."""
    n = len(p)
    acc, cost, done = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for i, m in enumerate(lad):
        sig = p.N[m].values > 0
        cost += np.where(done, 0, p.c[m].values + ver_cost * sig)
        ver = p.A[m].fillna(False).values.astype(bool)
        if mode == "bound":
            ok = np.where(sig, ver, p.B[m].values == 1)
        elif mode == "accept":
            ok = np.where(sig, ver, True)
        else:
            ok = np.where(sig, ver, False)
        take = ~done & (ok | (i == len(lad) - 1))
        acc = np.where(take, p.B[m].values, acc)
        done |= take
    return acc, cost


def ladders(p: Pop):
    """Cost-ordered 2- and 3-rung permutations (cost order taken on the whole population)."""
    return [lad for k in (2, 3) for lad in itertools.permutations(p.models, k)
            if p.c[list(lad)].mean().is_monotonic_increasing]


def feasibility(p: Pop, base_acc, base_cost, ver_cost=0.0):
    """S3.2 table against per-task baseline arrays: ladders meeting both thresholds (point
    estimates) and the best gap among ladders saving >= 20%."""
    out = {}
    lads = ladders(p)
    for mode in ("accept", "escalate", "bound"):
        rows = []
        for lad in lads:
            a, c = run_ladder(p, lad, mode, ver_cost)
            rows.append((">".join(lad), 100 * (a.mean() - base_acc.mean()),
                         100 * (1 - c.sum() / base_cost.sum())))
        R = pd.DataFrame(rows, columns=["ladder", "gap", "saving"])
        top = R[R.saving >= GATE_SAVING].sort_values("gap", ascending=False).head(3)
        out[mode] = {"n_ladders": len(R),
                     "n_meet_both": int(((R.gap >= GATE_GAP) & (R.saving >= GATE_SAVING)).sum()),
                     "top3_at_20pct_saving": top.round(4).to_dict("records")}
    return out


def whole_set_baseline(p: Pop):
    best = p.B.mean().idxmax()
    return best, p.B[best].values, p.c[best].values


def no_signal(p: Pop):
    return {"n_tasks": len(p),
            "tasks_no_signal_any_model": int((p.N == 0).all(axis=1).sum()),
            "share_tasks_no_signal": float((p.N == 0).all(axis=1).mean()),
            "share_cells_no_signal": float((p.N == 0).values.mean())}


# ----------------------------------------------------------------------------- mode: repro

def mode_repro(d):
    """Step 2: the 2026-11-19 computation, unchanged (unavailable cells kept as recorded, no
    verifier cost, best single model on the whole set, point estimates)."""
    p = Pop(d, headline_mask(d))
    best, ba, bc = whole_set_baseline(p)
    ns = no_signal(p)
    out = {"n": len(p), "best_single": best, "best_single_acceptance": float(ba.mean()),
           "share_unsolved_by_any_model": float((p.B.max(axis=1) == 0).mean()), **ns,
           "feasibility": feasibility(p, ba, bc)}
    f = out["feasibility"]
    checks = [
        (out["n"] == 1971, "n = 1,971"),
        (ns["tasks_no_signal_any_model"] == 1562, "1,562 tasks with no signal"),
        (round(100 * ns["share_tasks_no_signal"], 1) == 79.2, "79.2% of tasks"),
        (round(100 * ns["share_cells_no_signal"], 1) == 79.4, "79.4% of cells"),
        (best == OPUS and round(100 * ba.mean(), 1) == 66.8, "Opus best at 66.8%"),
        (round(100 * out["share_unsolved_by_any_model"], 1) == 10.8, "10.8% unsolved"),
        (f["accept"]["n_meet_both"] == 0, "accept: 0 ladders"),
        (round(f["accept"]["top3_at_20pct_saving"][0]["gap"], 1) == -13.4, "accept: -13.4"),
        (f["escalate"]["n_meet_both"] == 0, "escalate: 0 ladders"),
        (round(f["escalate"]["top3_at_20pct_saving"][0]["gap"]) == -14, "escalate: about -14"),
        (f["bound"]["n_meet_both"] == 12, "bound: 12 ladders"),
        (round(f["bound"]["top3_at_20pct_saving"][0]["gap"], 1) == 10.8, "bound: +10.8"),
        (round(f["bound"]["top3_at_20pct_saving"][0]["saving"], 1) == 24.5, "bound: 24.5% saving"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    out["reproduced"] = not failed
    if failed:
        raise SystemExit(f"Reproduction failed, stop and report: {failed}\n{json.dumps(out, indent=1)}")
    return out


# ----------------------------------------------------------------------------- mode: recount

def folds(p: Pop, seed: int) -> np.ndarray:
    """5-fold assignment by problem (S0.6); returns the fold of each task."""
    probs = np.array(sorted(p.problem.unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(rng.permutation(probs), np.arange(len(probs)) % N_FOLDS))
    return p.problem.map(fold_of).values


def crossfit_baseline(p: Pop, fold):
    """Best single model chosen on the training folds, scored on the held-out fold."""
    ba, bc, picks = np.zeros(len(p)), np.zeros(len(p)), []
    for f in range(N_FOLDS):
        te = fold == f
        best = p.B[~te].mean().idxmax()
        picks.append(best)
        ba[te], bc[te] = p.B[best].values[te], p.c[best].values[te]
    return ba, bc, picks


def summarise_seeds(per_seed):
    """Gate on the pre-registered seed (S0.8, R12); other seeds as mean and range."""
    def stats(get):
        xs = [get(r) for r in per_seed]
        return {"seed0": xs[GATE_SEED], "mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))}
    out = {}
    for mode in ("accept", "escalate", "bound"):
        out[mode] = {
            "n_meet_both": stats(lambda r: r["feasibility"][mode]["n_meet_both"]),
            "best_gap_at_20pct": stats(lambda r: r["feasibility"][mode]["top3_at_20pct_saving"][0]["gap"]),
            "seed0_top3": per_seed[GATE_SEED]["feasibility"][mode]["top3_at_20pct_saving"]}
    out["best_single_picks"] = {s: r["picks"] for s, r in enumerate(per_seed)}
    out["baseline_acceptance"] = stats(lambda r: r["base_acc"])
    return out


def family_mix(p: Pop):
    return p.family.value_counts().sort_index().astype(int).to_dict()


def treatment(p: Pop):
    """No-signal shares and the S3.2 table for one population, old simplifications and with
    verifier cost, then with the best single model chosen inside training folds."""
    best, ba, bc = whole_set_baseline(p)
    out = {"n": len(p), "n_problems": int(p.problem.nunique()), "best_single": best,
           "best_single_acceptance": float(ba.mean()),
           "pass_rate": {m: float(p.B[m].mean()) for m in p.models},
           "share_unsolved_by_any_model": float((p.B.max(axis=1) == 0).mean()),
           "family_mix": family_mix(p), "no_signal": no_signal(p),
           "feasibility_no_verifier_cost": feasibility(p, ba, bc),
           "feasibility": feasibility(p, ba, bc, VERIFIER_RUN_USD)}
    per_seed = []
    for s in range(N_SEEDS):
        fa, fc, picks = crossfit_baseline(p, folds(p, s))
        per_seed.append({"picks": picks, "base_acc": float(fa.mean()),
                         "feasibility": feasibility(p, fa, fc, VERIFIER_RUN_USD)})
    out["crossfit"] = summarise_seeds(per_seed)
    return out


def unavailable_counts(d, p: Pop):
    v, M = d["v"], d["missing_run"]
    n_empty, n_missing = int(v.is_empty.sum()), int(M.values.sum())
    assert n_empty == 1551 and n_missing == 148, (n_empty, n_missing)
    in_head = {
        "empty_cells": int(p.E.values.sum()), "missing_run_cells": int(p.M.values.sum()),
        "unavailable_cells": int(p.unavail.values.sum()),
        "both_empty_and_missing": int((p.E & p.M).values.sum()),
        "tasks_with_any_unavailable": int(p.unavail.any(axis=1).sum()),
        "by_model": {m: {"empty": int(p.E[m].sum()), "missing_run": int(p.M[m].sum())} for m in p.models},
        "by_family": {f: {"tasks": int((p.family == f).sum()),
                          "unavailable_cells": int(p.unavail[p.family == f].values.sum()),
                          "tasks_with_any_unavailable": int(p.unavail[p.family == f].any(axis=1).sum())}
                      for f in sorted(p.family.unique())},
        "empty_with_passing_released_score": int((p.E & (p.B == 1)).values.sum()),
    }
    split = d["split"]
    return {"overall": {"empty_x13_rows": n_empty,
                        "empty_by_model": v[v.is_empty].model.value_counts().astype(int).to_dict(),
                        "empty_with_released_truth_ge_1": int((v.is_empty & (v.truth.astype(float) >= 1)).sum()),
                        "empty_by_x13_source": v[v.is_empty].source.value_counts().astype(int).to_dict(),
                        "empty_by_dimension": v[v.is_empty].task_id.map(d["dim"]).value_counts().astype(int).to_dict(),
                        "empty_in_headline_tasks": int(v[v.is_empty].task_id.isin(set(p.ids)).sum()),
                        "missing_qwen3max_runs": n_missing,
                        "missing_by_model": M.sum().astype(int)[lambda s: s > 0].to_dict(),
                        "missing_by_split": M.any(axis=1).groupby(split).sum().astype(int).to_dict(),
                        "missing_by_dimension": M.any(axis=1).groupby(d["dim"]).sum().astype(int)[lambda s: s > 0].to_dict(),
                        "missing_rows_in_x13_tasks": int(M.loc[M.index.isin(set(v.task_id))].values.sum())},
            "headline_1971": in_head}


def mode_recount(d):
    full = Pop(d, headline_mask(d))
    keep = ~full.unavail.any(axis=1).values
    cc = full.subset(keep)
    assert not cc.unavail.values.any()
    E, N = full.E.values, full.N.values
    # Cell-level share on all 1,971 tasks with only the empty cells removed from the denominator.
    ne = ~E
    tasks_ns_avail = [(N[i][ne[i]] == 0).all() for i in range(len(full))]
    return {
        "unavailable": unavailable_counts(d, full),
        "empty_removed_share": {
            "share_cells_no_signal": float(((N == 0) & ne).sum() / ne.sum()),
            "cells_in_denominator": int(ne.sum()),
            "share_tasks_no_signal_among_nonempty_cells": float(np.mean(tasks_ns_avail))},
        "old_full_1971": treatment(full),
        "complete_case": treatment(cc),
    }


# ----------------------------------------------------------------------------- mode: ra
# Port of Route Anything's price-path code, InfRec-Cardinal/route-anything at e137f3f,
# experiments/quality_floor/crb_core.py (primitives, argmax_policy, pol_vd, ratio_opt, price_path,
# fitted_floor, realized) and crb.py (lp_floor). Changes: the menu, the reference action and the
# set of rows are arguments instead of globals, and cells with no fitting rows get weight 0 and are
# sent to the reference.

RA_PRIMARIES = ["Qwen3-Max", "kimi-k2.5", "qwen3.5-plus", "glm-5", "gpt-5.4", "claude-sonnet-4-6", "MiniMax-M2.7"]
RA_THETAS = [1.0, 0.5]
RA_BOOT_SEED = 20260925          # crb_core.SEED
RA_MIX_TOL = 1e-12


def ra_cells(ids: pd.Index, dim: pd.Series) -> pd.Series:
    """21 dimension x source cells on the full release: source = second `_` token of the task id."""
    return dim.loc[ids] + "|" + pd.Series(ids, index=ids).str.split("_").str[1]


def ra_menu(Y, Cm, models, oracle_cascades: bool):
    """Direct dispatch to each model; with oracle_cascades, also the 14 attempt-then-Opus actions
    that escalate when the primary's released score is below theta (Route Anything's X9 menu; an
    oracle verifier, used here only to reproduce the paper)."""
    col = {m: j for j, m in enumerate(models)}
    names = [f"direct:{m}" for m in models]
    V, D = [Y[:, j] for j in range(len(models))], [Cm[:, j] for j in range(len(models))]
    if oracle_cascades:
        yB, cB = Y[:, col[OPUS]], Cm[:, col[OPUS]]
        for m in RA_PRIMARIES:
            for th in RA_THETAS:
                H = (Y[:, col[m]] >= th - 1e-12).astype(float)
                names.append(f"attempt:{m}@{th:g}")
                V.append(H * Y[:, col[m]] + (1 - H) * yB)
                D.append(Cm[:, col[m]] + (1 - H) * cB)
    return names, np.column_stack(V), np.column_stack(D)


def ra_primitives(rows, sidx, X, Vm, Dm):
    s = sidx[rows]
    nx = np.bincount(s, minlength=X).astype(float)
    U = Vm.shape[1]
    v = np.column_stack([np.bincount(s, weights=Vm[rows, u], minlength=X) for u in range(U)]) / np.maximum(nx, 1)[:, None]
    c = np.column_stack([np.bincount(s, weights=Dm[rows, u], minlength=X) for u in range(U)]) / np.maximum(nx, 1)[:, None]
    return nx / nx.sum(), v, c, nx > 0


def ra_pol_vd(g, pi, v, c):
    ix = np.arange(len(g))
    return float(pi @ v[ix, g]), float(pi @ c[ix, g])


def ra_ratio_opt(pi, v, c, a_ref):
    vb, cb = ra_pol_vd(np.full(len(pi), a_ref), pi, v, c)
    lam = vb / cb
    for _ in range(200):
        g = np.argmax(v - lam * c, axis=1)
        ev, ed = ra_pol_vd(g, pi, v, c)
        new = ev / ed
        if new <= lam * (1 + 1e-13):
            return lam
        lam = new
    raise RuntimeError("Dinkelbach did not converge")


def ra_price_path(pi, v, c, a_ref):
    """Maximizers of v - lambda c on the breakpoint intervals of [0, eta*], in increasing lambda."""
    eta = ra_ratio_opt(pi, v, c, a_ref)
    U = v.shape[1]
    bps = []
    for x in range(len(pi)):
        for u in range(U):
            for w in range(u + 1, U):
                dc = c[x, u] - c[x, w]
                if abs(dc) > 1e-15:
                    lam = (v[x, u] - v[x, w]) / dc
                    if 0 < lam < eta:
                        bps.append(lam)
    grid = np.concatenate([[0.0], np.unique(bps), [eta]])
    path, lam_lo = [], []
    for lo, hi in zip(grid[:-1], grid[1:]):
        g = np.argmax(v - 0.5 * (lo + hi) * c, axis=1)
        if not path or not np.array_equal(g, path[-1]):
            path.append(g)
            lam_lo.append(lo)
    return eta, path, np.array(lam_lo)


def ra_lp_floor(pi, v, c, eps, a_ref):
    """Fitted floor optimum E[V] >= vbar - eps (crb.py lp_floor): the last path router meeting the
    floor, or the mixture of it and the next one whose fitted value equals vbar - eps."""
    eta, path, lam_lo = ra_price_path(pi, v, c, a_ref)
    vb, _ = ra_pol_vd(np.full(len(pi), a_ref), pi, v, c)
    k = 0
    for j, g in enumerate(path):
        if ra_pol_vd(g, pi, v, c)[0] >= vb - eps - 1e-12:
            k = j + 1
    if k == len(path):
        return path[-1], len(path)
    va, _ = ra_pol_vd(path[k - 1], pi, v, c)
    vn, _ = ra_pol_vd(path[k], pi, v, c)
    w = float((vb - eps - vn) / (va - vn))
    return (path[k - 1], path[k], w), len(path)


def ra_realized(g, rows, sidx, Vm, Dm):
    """Per-task value and cost; a mixture is scored at its expected per-task value and cost."""
    if isinstance(g, tuple):
        ga, gb, w = g
        va, da = ra_realized(ga, rows, sidx, Vm, Dm)
        vb, db = ra_realized(gb, rows, sidx, Vm, Dm)
        return w * va + (1 - w) * vb, w * da + (1 - w) * db
    a = g[sidx[rows]]
    return Vm[rows, a], Dm[rows, a]


def ra_route_actions(g, names, cells):
    """Readable per-cell actions of a (possibly mixed) router."""
    if isinstance(g, tuple):
        ga, gb, w = g
        return {x: (names[ga[i]] if ga[i] == gb[i] else f"{names[ga[i]]} {w:.3f} / {names[gb[i]]} {1 - w:.3f}")
                for i, x in enumerate(cells)}
    return {x: names[g[i]] for i, x in enumerate(cells)}


def ra_reproduce(d):
    """Route Anything's headline on its own population: the 21-cell zero-loss route fitted on the
    7,080 probing tasks, scored on the 2,919 ID-test tasks, mean released score, against
    Always-Opus (Table 5, Table 10)."""
    S, C = d["S"], d["C"]
    models = list(S.columns)
    names, Vm, Dm = ra_menu(S.values, C.values, models, oracle_cascades=True)
    assert len(names) == 22
    cells = ra_cells(S.index, d["dim"])
    cell_names = sorted(cells.unique())
    assert len(cell_names) == 21
    sidx = cells.map({x: i for i, x in enumerate(cell_names)}).values
    PR = np.where(d["split"].values == "probing")[0]
    TE = np.where(d["split"].values == "id_test")[0]
    assert len(PR) == 7080 and len(TE) == 2919
    a_ref = names.index(f"direct:{OPUS}")
    pi, v, c, _ = ra_primitives(PR, sidx, 21, Vm, Dm)
    g, path_len = ra_lp_floor(pi, v, c, 0.0, a_ref)
    yB, cB = Vm[TE, a_ref], Dm[TE, a_ref]
    rng = np.random.default_rng(RA_BOOT_SEED)
    W = np.vstack([np.ones(len(TE)), rng.multinomial(len(TE), np.full(len(TE), 1 / len(TE)), size=1000)])

    def ev(val, cost):
        Wv, Wd, n = W @ val, W @ cost, W.sum(1)
        r = (Wv / Wd) / ((W @ yB) / (W @ cB))
        dv = (Wv - W @ yB) / n
        ci = lambda a: [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
        return {"efficiency_vs_opus": ci(r), "dV_points": [100 * x for x in ci(dv)],
                "cost_m_usd_per_task": float(cost.mean() * 1e3)}

    zl_v, zl_d = ra_realized(g, TE, sidx, Vm, Dm)
    acts = ra_route_actions(g, names, cell_names)
    # CellBest: each cell to its best probing model by direct dispatch (lowest index on ties).
    cb = np.argmax(v[:, : len(models)], axis=1)
    cb_v, cb_d = ra_realized(cb, TE, sidx, Vm, Dm)
    out = {"always_opus_cost_m_usd_per_task": float(cB.mean() * 1e3),
           "always_opus_mean_score": float(yB.mean()),
           "price_path_length": path_len,
           "zero_loss": ev(zl_v, zl_d), "zero_loss_actions": acts,
           "zero_loss_calls_opus": any(f"{OPUS}" in a for a in acts.values()),
           "zero_loss_all_direct": all(not a.startswith("attempt") and " attempt" not in a for a in acts.values()),
           "cellbest": ev(cb_v, cb_d),
           "cellbest_minus_zero_loss_dV_points": float(100 * (cb_v.mean() - zl_v.mean())),
           "paper": {"efficiency_vs_opus": 25.516, "dV_points": [-0.44, -1.75, 0.97],
                     "cost_m_usd_per_task": 0.452, "always_opus_cost_m_usd_per_task": 11.655,
                     "cellbest_minus_zero_loss": 8.7}}
    z = out["zero_loss"]
    ok = (abs(z["efficiency_vs_opus"][0] - 25.516) < 0.01 and abs(z["dV_points"][0] + 0.44) < 0.01
          and abs(z["cost_m_usd_per_task"] - 0.452) < 0.001 and abs(out["always_opus_cost_m_usd_per_task"] - 11.655) < 0.001)
    out["reproduced"] = bool(ok)
    return out


def cluster_boot(problem: np.ndarray, cols: dict, n_boot: int, seed: int):
    """Resample problems with replacement; returns a function mapping per-task arrays to
    resampled sums (row 0 is the point estimate)."""
    codes, uniq = pd.factorize(problem)
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones(len(uniq)), rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=n_boot)])
    return {k: W @ np.bincount(codes, weights=a, minlength=len(uniq)) for k, a in cols.items()}


def ra_routes_fold(p: Pop, tr, te, sidx, X, Vm, Dm, names):
    """Fit the lookup routes on the training rows, return per-task acceptance and cost on te."""
    M = len(p.models)
    pi, v, c, seen = ra_primitives(tr, sidx, X, Vm, Dm)
    a_ref = int(np.argmax(Vm[tr].mean(0)))                 # best single model on the training folds
    send_ref = lambda g: np.where(seen, g, a_ref)
    # Zero-loss direct-dispatch route (eps = 0 relative to the training-fold best single model).
    g, _ = ra_lp_floor(pi, v, c, 0.0, a_ref)
    g = (send_ref(g[0]), send_ref(g[1]), g[2]) if isinstance(g, tuple) else send_ref(g)
    routes = {"zero_loss": g}
    # CellBest: best training acceptance per cell; ties to the cheaper model.
    routes["cellbest"] = send_ref(np.array([np.lexsort((c[x], -v[x]))[0] for x in range(X)]))
    # DimensionBest and cost-aware DimensionBest on the 5 exec dimensions.
    dims = p.dim.values
    dim_names = sorted(set(dims))
    dacc = {k: Vm[tr][dims[tr] == k].mean(0) for k in dim_names if (dims[tr] == k).any()}
    dcost = {k: Dm[tr][dims[tr] == k].mean(0) for k in dacc}
    db = {k: int(np.lexsort((dcost[k], -dacc[k]))[0]) for k in dacc}
    cadb = {}
    for k in dacc:
        ok = np.where(dacc[k] >= dacc[k].max() - 0.02)[0]       # within 2 points of the dimension's best
        cadb[k] = int(ok[np.argmin(dcost[k][ok])])                 # cheapest of those
    out = {}
    for name, g in routes.items():
        out[name] = ra_realized(g, te, sidx, Vm, Dm)
    for name, table in (("dimensionbest", db), ("cost_aware_dimensionbest", cadb)):
        a = np.array([table.get(k, a_ref) for k in dims[te]])
        out[name] = (Vm[te, a], Dm[te, a])
    out["best_single"] = (Vm[te, a_ref], Dm[te, a_ref])
    picks = {"best_single": names[a_ref],
             "held_out_tasks_in_cells_without_training_tasks": int((~seen[sidx[te]]).sum()),
             "zero_loss": ra_route_actions(routes["zero_loss"], names, [f"cell{x}" for x in range(X)]),
             "cellbest": ra_route_actions(routes["cellbest"], names, [f"cell{x}" for x in range(X)]),
             "cost_aware_dimensionbest": {k: names[j] for k, j in cadb.items()},
             "dimensionbest": {k: names[j] for k, j in db.items()}}
    return out, picks


def ra_on_population(p: Pop, d, n_boot: int):
    """Route Anything's lookup routes on our population against S0.3: success = score 1.0,
    direct-dispatch menu only (no released-score escalation, R1), fitted on training folds."""
    Vm, Dm = p.B.values, p.c.values
    names = [f"direct:{m}" for m in p.models]
    cells = ra_cells(p.ids, d["dim"])
    cell_names = sorted(cells.unique())
    sidx = cells.map({x: i for i, x in enumerate(cell_names)}).values
    X = len(cell_names)
    route_names = ["zero_loss", "cellbest", "dimensionbest", "cost_aware_dimensionbest"]
    per_seed = []
    for s in range(N_SEEDS):
        fold = folds(p, s)
        acc = {r: np.zeros(len(p)) for r in route_names + ["best_single"]}
        cost = {r: np.zeros(len(p)) for r in acc}
        fold_picks = []
        for f in range(N_FOLDS):
            te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
            res, picks = ra_routes_fold(p, tr, te, sidx, X, Vm, Dm, names)
            fold_picks.append(picks)
            for r, (a, c) in res.items():
                acc[r][te], cost[r][te] = a, c
        cols = {f"a_{r}": acc[r] for r in acc} | {f"c_{r}": cost[r] for r in cost} | {"n": np.ones(len(p))}
        Bt = cluster_boot(p.problem.values, cols, n_boot, seed=1000 + s)
        ci = lambda a: [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]
        rows = {}
        for r in route_names + ["best_single"]:
            gap = 100 * (Bt[f"a_{r}"] - Bt["a_best_single"]) / Bt["n"]
            saving = 100 * (1 - Bt[f"c_{r}"] / Bt["c_best_single"])
            rows[r] = {"acceptance_pct": 100 * float(acc[r].mean()),
                       "cost_m_usd_per_task": 1e3 * float(cost[r].mean()),
                       "saving_pct": ci(saving), "gap_points": ci(gap),
                       "passes_s0_3": bool(np.percentile(saving[1:], 2.5) >= GATE_SAVING
                                           and np.percentile(gap[1:], 2.5) >= GATE_GAP)}
        per_seed.append({"routes": rows, "fold_picks": fold_picks,
                         "unseen_cell_tasks": sum(fp["held_out_tasks_in_cells_without_training_tasks"] for fp in fold_picks)})
    summary = {}
    for r in route_names:
        xs = [ps["routes"][r] for ps in per_seed]
        summary[r] = {"seed0": xs[GATE_SEED],
                      "other_seeds": {k: {"mean": float(np.mean([x[k] if np.isscalar(x[k]) else x[k][0] for x in xs])),
                                          "min": float(np.min([x[k] if np.isscalar(x[k]) else x[k][0] for x in xs])),
                                          "max": float(np.max([x[k] if np.isscalar(x[k]) else x[k][0] for x in xs]))}
                                      for k in ("acceptance_pct", "cost_m_usd_per_task", "saving_pct", "gap_points")},
                      "passes_s0_3_by_seed": [x["passes_s0_3"] for x in xs]}
    summary["best_single"] = {"seed0": per_seed[GATE_SEED]["routes"]["best_single"]}
    cell_stats = {x: {"tasks": int((cells == x).sum()),
                      "acceptance_pct": {m: round(100 * float(p.B[m][(cells == x).values].mean()), 1) for m in p.models}}
                  for x in cell_names}
    return {"n": len(p), "n_problems": int(p.problem.nunique()), "n_cells": X, "cells": cell_names,
            "cell_stats_whole_set": cell_stats,
            "held_out_tasks_in_cells_without_training_tasks_by_seed": [ps["unseen_cell_tasks"] for ps in per_seed],
            "routes": summary, "seed0_fold_picks": per_seed[GATE_SEED]["fold_picks"]}


def mode_ra(d, n_boot):
    rep = ra_reproduce(d)
    out = {"reproduction": rep}
    if not rep["reproduced"]:
        print("Route Anything reproduction failed; not running on our population")
        return out
    full = Pop(d, headline_mask(d))
    cc = full.subset(~full.unavail.any(axis=1).values)
    out["complete_case"] = ra_on_population(cc, d, n_boot)
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["repro", "recount", "ra", "all"], default="all")
    ap.add_argument("--boot", type=int, default=1000, help="cluster-bootstrap resamples")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    d = load()
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    if a.mode in ("repro", "all"):
        res["repro"] = mode_repro(d)
        print("repro: all 2026-11-19 numbers reproduced")
    if a.mode in ("recount", "all"):
        res["recount"] = mode_recount(d)
        print("recount: done")
    if a.mode in ("ra", "all"):
        res["ra"] = mode_ra(d, a.boot)
        print(f"ra: reproduction {'matched' if res['ra']['reproduction']['reproduced'] else 'FAILED'}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {a.out.relative_to(REPO_ROOT) if a.out.is_relative_to(REPO_ROOT) else a.out}")


if __name__ == "__main__":
    main()
