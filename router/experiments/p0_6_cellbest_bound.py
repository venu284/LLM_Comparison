"""P0.6 feasibility against CellBest (docs/IMPLEMENTATION_PLAN.md, Pre-S0).

Every number in docs/research/P0.6_cellbest_bound.md comes from this script, run from the repo
root on the pinned data (scripts/fetch_data.py --verify-only must print `fetched: 0, verified: 187`):

    pipeline/venv/bin/python router/experiments/p0_6_cellbest_bound.py

Steps:
  1. reproduce P0.5 on the complete-case headline set (1,907 tasks) by calling its functions, and
     stop if any number differs;
  2. predict each task's dimension x source cell from the prompt text only, inside the 5-fold
     cross-fit by problem;
  3. CellBest with predicted cells (the S0.3(c) comparator) and with true cells (reference);
  4. two policy families in three modes (no-signal cells accepted, escalated, or decided by the
     released score = the oracle-on-no-signal bound): (i) the 84 fixed P0.5 ladders, (ii) per-cell
     choice fitted on the training folds (Rule A: match CellBest's acceptance at least cost;
     Rule B: at most CellBest's cost, highest acceptance);
  5. point estimates (the DP1b criterion) and paired 95% cluster-bootstrap CIs (informational)
     against the cross-fitted best single model and against CellBest.

Results go to router/experiments/out/p0_6_cellbest_bound.json. All numbers are provisional.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.experiments import p0_5_recount as rc  # noqa: E402
from router.settings import EXTERNAL_DIR  # noqa: E402

TASKS_JSONL = EXTERNAL_DIR / "route-anything-data/experiments_x/common/tasks.jsonl"
OUT_JSON = REPO_ROOT / "router/experiments/out/p0_6_cellbest_bound.json"
MODES = ("accept", "escalate", "bound")
C_SAVING, C_GAP = 0.0, -2.0           # S0.3(c): saving against CellBest > 0, gap >= -2 points
CLF_SEED = 0                          # random_state of the cell classifier
TOL = 1e-12


# ----------------------------------------------------------------------------- population

def complete_case(d) -> rc.Pop:
    """The complete-case headline set of P0.5: headline tasks with no unavailable cell."""
    full = rc.Pop(d, rc.headline_mask(d))
    return full.subset(~full.unavail.any(axis=1).values)


# ----------------------------------------------------------------------------- step 1: reproduction

def reproduce(d, cc: rc.Pop, n_boot: int):
    """Call P0.5's functions on the complete case and assert its published numbers."""
    ra = rc.ra_on_population(cc, d, n_boot)
    cb = ra["routes"]["cellbest"]["seed0"]
    cf = rc.treatment(cc)["crossfit"]
    r1 = lambda xs: [round(x, 1) for x in xs]
    checks = [
        (len(cc) == 1907, "n = 1,907"),
        (r1(cb["saving_pct"]) == [22.9, 20.8, 25.2], "CellBest saving 22.9 [20.8, 25.2]"),
        (r1(cb["gap_points"]) == [3.0, 1.5, 4.5], "CellBest gap +3.0 [+1.5, +4.5]"),
        (cf["accept"]["n_meet_both"]["seed0"] == 0, "accept: 0 ladders"),
        (round(cf["accept"]["best_gap_at_20pct"]["seed0"], 1) == -13.1, "accept: -13.1"),
        (cf["escalate"]["n_meet_both"]["seed0"] == 0, "escalate: 0 ladders"),
        (round(cf["escalate"]["best_gap_at_20pct"]["seed0"], 1) == -13.9, "escalate: -13.9"),
        (cf["bound"]["n_meet_both"]["seed0"] == 12, "bound: 12 ladders"),
        (round(cf["bound"]["best_gap_at_20pct"]["seed0"], 1) == 10.9, "bound: +10.9"),
        (round(cf["bound"]["seed0_top3"][0]["saving"], 1) == 23.2, "bound: 23.2% saving"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    if failed:
        raise SystemExit(f"P0.5 reproduction failed, stop and report: {failed}")
    return {"reproduced": True, "checks": [msg for _, msg in checks],
            "cellbest_seed0": cb, "cellbest_saving_ci_lower_by_seed": ra["routes"]["cellbest"]["saving_ci_lower_by_seed"],
            "cellbest_gap_ci_lower_by_seed": ra["routes"]["cellbest"]["gap_ci_lower_by_seed"],
            "s3_2_seed0": {m: {"n_meet_both": cf[m]["n_meet_both"]["seed0"],
                               "best_gap_at_20pct": cf[m]["best_gap_at_20pct"]["seed0"],
                               "top": cf[m]["seed0_top3"][0]} for m in MODES}}


# ----------------------------------------------------------------------------- step 2: policy matrices

def policy_matrices(cc: rc.Pop, mode: str):
    """Per-task acceptance and cost of every candidate policy in one mode: direct dispatch to each
    model (no verifier charge) followed by the P0.5 fixed ladders (verifier cost at S0.2's rate)."""
    names = [f"direct:{m}" for m in cc.models]
    A, C = [cc.B[m].values for m in cc.models], [cc.c[m].values for m in cc.models]
    for lad in rc.ladders(cc):
        a, c = rc.run_ladder(cc, lad, mode, rc.VERIFIER_RUN_USD)
        names.append(">".join(lad))
        A.append(a)
        C.append(c)
    return names, np.column_stack(A), np.column_stack(C)


# ----------------------------------------------------------------------------- step 3: cell predictor

def load_prompts(ids: pd.Index) -> pd.Series:
    t = pd.read_json(TASKS_JSONL, lines=True, dtype={"task_id": str})
    t = t.drop_duplicates("task_id").set_index("task_id")
    return t.prompt.reindex(ids), len(t)


def make_classifier():
    """TF-IDF on word 1-2 grams and character 2-5 grams, then logistic regression. Settings are
    fixed in advance, not tuned."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                        max_features=200_000))
    return make_pipeline(feats, LogisticRegression(C=10.0, max_iter=5000, random_state=CLF_SEED))


def predict_cells(prompts: np.ndarray, cells: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Held-out cell predictions: vectorizer and classifier fitted on the training folds only."""
    pred = np.empty(len(cells), dtype=object)
    for f in range(rc.N_FOLDS):
        te, tr = fold == f, fold != f
        clf = make_classifier().fit(prompts[tr], cells[tr])
        pred[te] = clf.predict(prompts[te])
    return pred


def predictor_report(true: np.ndarray, pred: np.ndarray, cell_names):
    conf = pd.crosstab(pd.Series(true, name="true"), pd.Series(pred, name="pred")).reindex(
        index=cell_names, columns=cell_names, fill_value=0)
    return {"accuracy": float((true == pred).mean()),
            "n_wrong": int((true != pred).sum()),
            "per_cell": {x: {"tasks": int((true == x).sum()),
                             "accuracy": float((pred[true == x] == x).mean())} for x in cell_names},
            "confusion_true_rows_pred_cols": {x: {y: int(conf.loc[x, y]) for y in cell_names if conf.loc[x, y]}
                                              for x in cell_names}}


# ----------------------------------------------------------------------------- per-cell rules

def cell_means(rows: np.ndarray, sidx: np.ndarray, X: int, M: np.ndarray):
    """Mean of each column of M over the given rows, per cell; and which cells have rows."""
    s = sidx[rows]
    n = np.bincount(s, minlength=X).astype(float)
    S = np.zeros((X, M.shape[1]))
    np.add.at(S, s, M[rows])
    return S / np.maximum(n, 1)[:, None], n > 0


def cellbest_table(v: np.ndarray, c: np.ndarray, n_direct: int) -> np.ndarray:
    """Per cell, the direct-dispatch column with the highest training acceptance, ties to the
    cheaper one (P0.5's CellBest)."""
    return np.array([np.lexsort((c[x, :n_direct], -v[x, :n_direct]))[0] for x in range(len(v))])


def rule_tables(v: np.ndarray, c: np.ndarray, cb: np.ndarray):
    """Per-cell choice among all candidate columns, relative to CellBest's column cb[x].

    Rule A: the cheapest candidate whose training acceptance is at least CellBest's; ties to higher
    acceptance, then lower column index.
    Rule B: the highest-acceptance candidate whose training cost is at most CellBest's; ties to
    lower cost, then lower column index.
    CellBest's own column always qualifies, so both rules always have a choice."""
    X, K = v.shape
    idx = np.arange(K)
    ra, rb = np.empty(X, int), np.empty(X, int)
    for x in range(X):
        k = cb[x]
        cand = idx[v[x] >= v[x, k] - TOL]
        ra[x] = cand[np.lexsort((cand, -v[x, cand], c[x, cand]))[0]]
        cand = idx[c[x] <= c[x, k] * (1 + TOL)]
        rb[x] = cand[np.lexsort((cand, c[x, cand], -v[x, cand]))[0]]
        assert v[x, ra[x]] >= v[x, k] - TOL and c[x, rb[x]] <= c[x, k] * (1 + TOL)
    return ra, rb


# ----------------------------------------------------------------------------- scoring

def ci(a):
    return [float(a[0]), float(np.percentile(a[1:], 2.5)), float(np.percentile(a[1:], 97.5))]


def point_compare(a, c, ra, rcost):
    """Point saving (%) and acceptance gap (points) of per-task arrays (a, c) against (ra, rcost)."""
    return 100 * (1 - c.sum() / rcost.sum()), 100 * (a.mean() - ra.mean())


def boot_metrics(Bt, key, refs):
    """CIs of one policy from cluster-bootstrap sums, against each reference key."""
    out = {"acceptance_pct": ci(100 * Bt[f"a_{key}"] / Bt["n"]),
           "cost_m_usd_per_task": ci(1e3 * Bt[f"c_{key}"] / Bt["n"])}
    for name, ref in refs.items():
        sav = 100 * (1 - Bt[f"c_{key}"] / Bt[f"c_{ref}"])
        gap = 100 * (Bt[f"a_{key}"] - Bt[f"a_{ref}"]) / Bt["n"]
        out[f"vs_{name}"] = {"saving_pct": ci(sav), "gap_points": ci(gap)}
    vb, vc = out.get("vs_best_single"), out.get("vs_cellbest_pred")
    if vb and vc:
        out["dp1b_point"] = {
            "meets_ab": bool(vb["saving_pct"][0] >= rc.GATE_SAVING and vb["gap_points"][0] >= rc.GATE_GAP),
            "meets_c": bool(vc["saving_pct"][0] > C_SAVING and vc["gap_points"][0] >= C_GAP)}
        out["dp1b_point"]["meets_both"] = out["dp1b_point"]["meets_ab"] and out["dp1b_point"]["meets_c"]
        out["ci_informational"] = {
            "ab_lower_bounds_pass": bool(vb["saving_pct"][1] >= rc.GATE_SAVING and vb["gap_points"][1] >= rc.GATE_GAP),
            "c_lower_bounds_pass": bool(vc["saving_pct"][1] > C_SAVING and vc["gap_points"][1] >= C_GAP),
            "gap_vs_cellbest_pred_half_width": float((vc["gap_points"][2] - vc["gap_points"][1]) / 2),
            "saving_vs_cellbest_pred_half_width": float((vc["saving_pct"][2] - vc["saving_pct"][1]) / 2)}
    return out


# ----------------------------------------------------------------------------- one seed

def run_seed(cc: rc.Pop, s: int, prompts: np.ndarray, cells: np.ndarray, cell_names, mats, n_boot: int):
    n, M = len(cc), len(cc.models)
    X = len(cell_names)
    cix = {x: i for i, x in enumerate(cell_names)}
    fold = rc.folds(cc, s)
    pred = predict_cells(prompts, cells, fold)
    tidx = np.array([cix[x] for x in cells])
    pidx = np.array([cix[x] for x in pred])
    ba, bc, picks = rc.crossfit_baseline(cc, fold)

    pol = {"best_single": (ba, bc)}
    arr = lambda: (np.zeros(n), np.zeros(n))
    keys = ["cellbest_pred", "cellbest_true"] + [f"{r}_{m}_{src}" for m in MODES for r in ("ruleA", "ruleB")
                                                 for src in ("pred", "true")]
    for k in keys:
        pol[k] = arr()
    D0 = mats[MODES[0]]
    names = D0[0]
    picks_out = []
    for f in range(rc.N_FOLDS):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        a_ref = cc.models.index(picks[f])
        v_d, seen = cell_means(tr, tidx, X, D0[1][:, :M])
        c_d, _ = cell_means(tr, tidx, X, D0[2][:, :M])
        cb = np.where(seen, cellbest_table(v_d, c_d, M), a_ref)
        fp = {"best_single": picks[f], "cellbest": {cell_names[x]: names[cb[x]] for x in range(X)},
              "held_out_predicted_into_unseen_cell": int((~seen[pidx[te]]).sum())}
        for src, sx in (("pred", pidx), ("true", tidx)):
            j = cb[sx[te]]
            pol[f"cellbest_{src}"][0][te] = cc.B.values[te, j]
            pol[f"cellbest_{src}"][1][te] = cc.c.values[te, j]
        for m in MODES:
            _, A, C = mats[m]
            v, _ = cell_means(tr, tidx, X, A)
            c, _ = cell_means(tr, tidx, X, C)
            ra, rb = rule_tables(v, c, cb)
            ra, rb = np.where(seen, ra, a_ref), np.where(seen, rb, a_ref)
            fp[f"ruleA_{m}"] = {cell_names[x]: names[ra[x]] for x in range(X)}
            fp[f"ruleB_{m}"] = {cell_names[x]: names[rb[x]] for x in range(X)}
            for r, tab in (("ruleA", ra), ("ruleB", rb)):
                for src, sx in (("pred", pidx), ("true", tidx)):
                    j = tab[sx[te]]
                    pol[f"{r}_{m}_{src}"][0][te] = A[te, j]
                    pol[f"{r}_{m}_{src}"][1][te] = C[te, j]
        picks_out.append(fp)

    # Fixed ladders: no fitting; point estimates for all of them, CIs for the best against (c).
    lad = {}
    cba, cbc = pol["cellbest_pred"]
    for m in MODES:
        nm, A, C = mats[m]
        rows = []
        for j in range(M, len(nm)):
            sb, gb = point_compare(A[:, j], C[:, j], ba, bc)
            sc, gc = point_compare(A[:, j], C[:, j], cba, cbc)
            rows.append({"ladder": nm[j], "col": j, "saving_vs_best": sb, "gap_vs_best": gb,
                         "saving_vs_cellbest_pred": sc, "gap_vs_cellbest_pred": gc})
        R = pd.DataFrame(rows)
        ab = (R.saving_vs_best >= rc.GATE_SAVING) & (R.gap_vs_best >= rc.GATE_GAP)
        cc_ok = (R.saving_vs_cellbest_pred > C_SAVING) & (R.gap_vs_cellbest_pred >= C_GAP)
        ok_gap = R[R.gap_vs_cellbest_pred >= C_GAP].sort_values("saving_vs_cellbest_pred", ascending=False)
        if len(ok_gap):
            best = ok_gap.iloc[0]
            how = "largest saving against CellBest among ladders with point gap >= -2"
        else:
            best = R.sort_values("gap_vs_cellbest_pred", ascending=False).iloc[0]
            how = "no ladder has point gap >= -2 against CellBest; the ladder with the largest gap"
        pol[f"ladder_{m}"] = (A[:, int(best.col)], C[:, int(best.col)])
        lad[m] = {"n_ladders": len(R), "n_meet_ab_point": int(ab.sum()), "n_meet_c_point": int(cc_ok.sum()),
                  "n_meet_both_point": int((ab & cc_ok).sum()), "best_ladder": best.ladder, "best_ladder_selection": how,
                  "top5_vs_c": ok_gap.head(5).drop(columns="col").round(4).to_dict("records")}

    cols = {f"a_{k}": a for k, (a, _) in pol.items()} | {f"c_{k}": c for k, (_, c) in pol.items()} | {"n": np.ones(n)}
    Bt = rc.cluster_boot(cc.problem.values, cols, n_boot, seed=1000 + s)
    refs = {"best_single": "best_single", "cellbest_pred": "cellbest_pred", "cellbest_true": "cellbest_true"}
    metrics = {k: boot_metrics(Bt, k, {r: v for r, v in refs.items() if v != k}) for k in pol}
    for m in MODES:
        lad[m]["best_ladder_metrics"] = metrics.pop(f"ladder_{m}")
    return {"predictor": predictor_report(cells, pred, cell_names),
            "misrouted_tasks_cellbest": int((pol["cellbest_pred"][1] != pol["cellbest_true"][1]).sum()),
            "metrics": metrics, "fixed_ladders": lad, "fold_picks": picks_out}


# ----------------------------------------------------------------------------- summary

def summarise(per_seed):
    """Seed 0 in full (the gate seed, S0.8); seeds 0-4 as mean, min and max of point estimates."""
    def rng(xs):
        return {"mean": float(np.mean(xs)), "min": float(np.min(xs)), "max": float(np.max(xs))}

    def pol_range(key):
        ms = [ps["metrics"][key] for ps in per_seed]
        out = {}
        for vs in ("vs_best_single", "vs_cellbest_pred"):
            if vs in ms[0]:
                for q in ("saving_pct", "gap_points"):
                    out[f"{vs}.{q}.point"] = rng([m[vs][q][0] for m in ms])
                    out[f"{vs}.{q}.ci_lower"] = rng([m[vs][q][1] for m in ms])
        if "dp1b_point" in ms[0]:
            out["dp1b_point_meets_c_by_seed"] = [m["dp1b_point"]["meets_c"] for m in ms]
            out["dp1b_point_meets_both_by_seed"] = [m["dp1b_point"]["meets_both"] for m in ms]
        return out

    keys = list(per_seed[0]["metrics"])
    lad = {m: {"n_meet_ab_point": [ps["fixed_ladders"][m]["n_meet_ab_point"] for ps in per_seed],
               "n_meet_c_point": [ps["fixed_ladders"][m]["n_meet_c_point"] for ps in per_seed],
               "n_meet_both_point": [ps["fixed_ladders"][m]["n_meet_both_point"] for ps in per_seed],
               "best_ladder": [ps["fixed_ladders"][m]["best_ladder"] for ps in per_seed],
               "best_saving_vs_cellbest_pred_point": rng([ps["fixed_ladders"][m]["best_ladder_metrics"]["vs_cellbest_pred"]["saving_pct"][0] for ps in per_seed]),
               "best_gap_vs_cellbest_pred_point": rng([ps["fixed_ladders"][m]["best_ladder_metrics"]["vs_cellbest_pred"]["gap_points"][0] for ps in per_seed])}
           for m in MODES}
    return {"predictor_accuracy_by_seed": [ps["predictor"]["accuracy"] for ps in per_seed],
            "misrouted_tasks_cellbest_by_seed": [ps["misrouted_tasks_cellbest"] for ps in per_seed],
            "policies_by_seed": {k: pol_range(k) for k in keys},
            "fixed_ladders_by_seed": lad}


def check_cellbest_true(rep, per_seed):
    """True-cell CellBest here must equal P0.5's CellBest exactly (same table, same draws)."""
    m0 = per_seed[rc.GATE_SEED]["metrics"]["cellbest_true"]["vs_best_single"]
    p5 = rep["cellbest_seed0"]
    lo_s = [ps["metrics"]["cellbest_true"]["vs_best_single"]["saving_pct"][1] for ps in per_seed]
    lo_g = [ps["metrics"]["cellbest_true"]["vs_best_single"]["gap_points"][1] for ps in per_seed]
    ok = (np.allclose(m0["saving_pct"], p5["saving_pct"], atol=1e-9) and np.allclose(m0["gap_points"], p5["gap_points"], atol=1e-9)
          and np.allclose(lo_s, rep["cellbest_saving_ci_lower_by_seed"][: len(per_seed)], atol=1e-9)
          and np.allclose(lo_g, rep["cellbest_gap_ci_lower_by_seed"][: len(per_seed)], atol=1e-9))
    if not ok:
        raise SystemExit("True-cell CellBest does not match P0.5's CellBest; stop and report")
    return True


def check_ladder_counts(rep, per_seed):
    """The (a)+(b) point counts of the fixed ladders must equal P0.5's S3.2 table on seed 0."""
    s0 = per_seed[rc.GATE_SEED]["fixed_ladders"]
    bad = [m for m in MODES if s0[m]["n_meet_ab_point"] != rep["s3_2_seed0"][m]["n_meet_both"]]
    if bad:
        raise SystemExit(f"Fixed-ladder (a)+(b) counts differ from P0.5 in modes {bad}; stop and report")
    return True


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000, help="cluster-bootstrap resamples")
    ap.add_argument("--seeds", type=int, default=rc.N_SEEDS)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    t0 = time.time()
    d = rc.load()
    cc = complete_case(d)
    rep = reproduce(d, cc, a.boot)
    print("step 1: P0.5 numbers reproduced")
    prompts, n_jsonl = load_prompts(cc.ids)
    join = int(prompts.notna().sum())
    if join != len(cc):
        raise SystemExit(f"Prompt join {join}/{len(cc)}; stop and report")
    cells = rc.ra_cells(cc.ids, d["dim"]).values
    cell_names = sorted(set(cells))
    mats = {m: policy_matrices(cc, m) for m in MODES}
    per_seed = []
    for s in range(a.seeds):
        per_seed.append(run_seed(cc, s, prompts.values, cells, cell_names, mats, a.boot))
        print(f"seed {s}: predictor accuracy {per_seed[-1]['predictor']['accuracy']:.4f}")
    res = {
        "population": {"n": len(cc), "n_problems": int(cc.problem.nunique()), "cells": cell_names,
                       "cell_tasks": {x: int((cells == x).sum()) for x in cell_names},
                       "n_candidates": len(mats[MODES[0]][0]), "n_ladders": len(mats[MODES[0]][0]) - len(cc.models)},
        "reproduction": rep,
        "prompts": {"tasks_jsonl_rows": n_jsonl, "joined": join, "of": len(cc)},
        "cellbest_true_matches_p0_5": check_cellbest_true(rep, per_seed),
        "ladder_ab_counts_match_p0_5": check_ladder_counts(rep, per_seed),
        "seed0": per_seed[rc.GATE_SEED],
        "by_seed": summarise(per_seed),
        "runtime_s": round(time.time() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {a.out.relative_to(REPO_ROOT) if a.out.is_relative_to(REPO_ROOT) else a.out} in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
