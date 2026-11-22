"""S2.5 baselines on the development split (S0.6, S0.7, R14), with paired CIs, the cell predictor
report and the shuffled-label check (S2.6).

Population: the development split of the headline set (S0.10, S0.11), complete case (O2): every
policy is scored on the same tasks, and the tasks dropped for an unavailable cell are counted.
Each baseline is fitted on four problem-grouped development folds (`stats.dev_folds`) and scores
the fifth once (`baselines.run.run_dev`); every task is scored once per seed. No evaluation-fold
task is read. These are development numbers for comparing baselines and checking the
infrastructure, not gate numbers.

Per baseline, on the gate seed (0): acceptance (S0.1), cost per task (S0.2, verifier runs
included), CPAT at the best single model's acceptance as the floor (R3), regret against the
cost-aware oracle (our definition), and paired CIs against the best single model and against
CellBest with predicted cells (the S0.3(c) comparator): acceptance gap in points and cost saving in
percent, cluster bootstrap over problems (`N_BOOT`, seed 0). The other four seeds are reported as
point estimates next to it (R12).

Also: the released ACRouter decisions on the development tasks they cover; the cell predictor's
held-out accuracy on `content_text` and, as a sensitivity, on the wrapped prompt, for both
training populations (headline development split, the user's decision; all eligible development
tasks, sensitivity), and the frozen predictor's training record (O3); CellBest-pred and Rule A
escalated with the sensitivity predictor; and the shuffled-label check for every learned baseline.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s2_baselines
Writes router/experiments/out/s2_baselines*.csv and s2_baselines.json.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd

from router.baselines import embeddings
from router.baselines.acrouter import ReleasedACRouter, covered
from router.baselines.cells import CellPredictor, out_of_fold_accuracy
from router.baselines.common import Context, success, true_cells
from router.baselines.lookup import CellBest, CellRule
from router.baselines.run import default_baselines, run_dev
from router.data import splits as S
from router.data.audit import build_views, load_release
from router.data.prompts import join, task_texts
from router.settings import N_BOOT, REPO_ROOT
from router.sim import metrics, stats
from router.sim.stream import ReplayView, replay
from router.sim.verifier import ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
SEEDS = stats.SEEDS
N_SHUFFLES = 5
REF_BEST, REF_C = "best_single", "cellbest_pred"


# ------------------------------------------------------------------------------- population
def population():
    m = load_release()
    views = build_views(m)
    ts = S.task_splits().set_index("task_id")
    dev = set(ts.index[ts.split == S.DEV])
    head = views["headline"]
    hd = head.subset([t for t in head.tasks if t in dev])
    cc = hd.available.all(axis=1)
    pop = hd.subset(list(hd.tasks[cc.to_numpy()]))
    dropped = hd.family[~cc.to_numpy()].value_counts().to_dict()
    elig = views["all_eligible"]
    ed = elig.subset([t for t in elig.tasks if t in dev])
    return pop, hd, ed, ts.problem_id, {"headline_dev_tasks": len(hd), "complete_case": len(pop),
                                        "dropped_unavailable_by_family": dropped,
                                        "problems": int(ts.problem_id.reindex(pop.tasks).nunique())}


# --------------------------------------------------------------------------------- statistics
def compare(o: pd.DataFrame, ref: pd.DataFrame, problems: pd.Series) -> dict:
    """Paired comparison of policy `o` against `ref` on the same tasks: gap (points) and saving
    (%), cluster bootstrap over problems."""
    f = metrics.paired_frame(o, ref, ("x", "r"))
    cl = problems.reindex(f.index)
    gap = stats.bootstrap(f, stats.mean_diff("accepted_x", "accepted_r"), clusters=cl, n_boot=N_BOOT)
    sav = stats.bootstrap(f, stats.saving_pct("cost_x", "cost_r"), clusters=cl, n_boot=N_BOOT)
    return {"gap_points": [gap.point, gap.lo, gap.hi], "saving_pct": [sav.point, sav.lo, sav.hi]}


def row(name: str, o: pd.DataFrame, refs: dict, problems: pd.Series, floor: float, m) -> dict:
    f = o.assign(a=o["accepted"].astype(float))
    acc = stats.bootstrap(f, stats.rate("a"), clusters=problems.reindex(f.index), n_boot=N_BOOT)
    cpat = metrics.cpat_at_floor(o, floor)
    reg = metrics.regret(o, m)
    r = {"baseline": name, "given_more_information": ",".join(o.attrs.get("more_information", ())),
         "definition": o.attrs.get("definition") or "",
         "acceptance": acc.point, "acceptance_lo": acc.lo, "acceptance_hi": acc.hi,
         "cost_musd_per_task": 1e3 * float(o.cost_usd.mean()),
         "n_calls_per_task": float(o.n_calls.mean()), "n_verifier_runs": int(o.n_verifier_runs.sum()),
         "verifier_usd": float(o.verifier_usd.sum()), "n_give_up": int((o.action == "give_up").sum()),
         "cpat_at_floor_musd": 1e3 * cpat["cpat"], "meets_floor": cpat["meets_floor"],
         "regret_points": reg["acceptance_regret_points"], "regret_usd": reg["cost_regret_usd"]}
    for tag, ref in refs.items():
        if ref is None or ref.attrs.get("name") == name:
            continue
        c = compare(o, ref, problems)
        r[f"gap_vs_{tag}"], r[f"gap_vs_{tag}_lo"], r[f"gap_vs_{tag}_hi"] = c["gap_points"]
        r[f"saving_vs_{tag}"], r[f"saving_vs_{tag}_lo"], r[f"saving_vs_{tag}_hi"] = c["saving_pct"]
    return r


def fits_summary(o: pd.DataFrame) -> list:
    return [{k: v for k, v in f.items() if k in ("fold", "table", "fallback", "chosen")}
            for f in o.attrs["fits"]]


# ------------------------------------------------------------------------------ cell predictor
def predictor_report(hd, ed, problems, content_all) -> dict:
    tt = task_texts()
    cells_h = true_cells(hd)
    folds = stats.dev_folds(hd.tasks, 0)
    wrapped = pd.Series({t: (tt[t].wrapped_prompt if t in tt else None) for t in ed.tasks})
    cells_e = true_cells(ed)
    out = {"training_population": "headline development split (user decision 2026-11-21)",
           "n_headline_dev": len(hd), "headline_cells": sorted(cells_h.unique()),
           "n_all_eligible_dev": len(ed), "all_eligible_dev_cells": int(cells_e.nunique()),
           "all_eligible_dev_with_text": int(content_all.reindex(ed.tasks).notna().sum()),
           "folds": "stats.dev_folds(seed 0) over the headline development split"}
    acc = {}
    for field, texts in (("content_text", content_all), ("wrapped_prompt", wrapped)):
        th = texts.reindex(hd.tasks)
        if th.isna().any():
            acc[field] = {"error": f"{int(th.isna().sum())} headline tasks have no {field}"}
            continue
        acc[field] = {
            "headline_dev": out_of_fold_accuracy(th, cells_h, folds, problem_of=problems),
            "all_eligible_dev_sensitivity": out_of_fold_accuracy(
                th, cells_h, folds, pool_texts=texts.reindex(ed.tasks).dropna(),
                pool_cells=cells_e[texts.reindex(ed.tasks).notna()], problem_of=problems)}
    out["held_out_accuracy"] = acc
    # The frozen predictor (O3): trained once on the whole headline development split.
    th = content_all.reindex(hd.tasks)
    frozen = CellPredictor(th.tolist(), cells_h.tolist())
    rec = "\n".join(f"{t}\t{c}\t{hashlib.sha256(x.encode()).hexdigest()}"
                    for t, c, x in sorted(zip(hd.tasks, cells_h, th)))
    out["frozen"] = {"n_train": frozen.n_train, "cells": frozen.cells, "field": "content_text",
                     "training_record_sha256": hashlib.sha256(rec.encode()).hexdigest(),
                     "settings": "P0.6 make_classifier: TF-IDF word 1-2 + char_wb 2-5, "
                                 "LogisticRegression C=10, random_state 0",
                     "training_accuracy": float((frozen.predict(th.tolist()) == cells_h.to_numpy()).mean())}
    return out


# ------------------------------------------------------------------------------------- main
def run() -> dict:
    pop, hd, ed, problems, comp = population()
    content_all = join(sorted(set(ed.tasks) | set(hd.tasks)))
    verifier = ReplayVerifier.from_x13()
    embeddings.verify()
    ctx = Context(verifier, content=content_all, problem_of=problems, embed=embeddings.embed)
    baselines = default_baselines(pop.models)
    res = {"population": comp, "seeds": {}, "n_boot": N_BOOT,
           "embedding": {"repo": embeddings.MINILM_REPO, "revision": embeddings.MINILM_REVISION}}
    per_seed_rows, outcomes0 = [], {}
    for s in SEEDS:
        folds = stats.dev_folds(pop.tasks, s)
        outs = {b.name: run_dev(b, pop, ctx, folds, seed=s) for b in baselines}
        if s == stats.GATE_SEED:
            outcomes0 = outs
        for name, o in outs.items():
            ref_b, ref_c = outs[REF_BEST], outs[REF_C]
            per_seed_rows.append({
                "seed": s, "gate_seed": s == stats.GATE_SEED, "baseline": name,
                "acceptance": metrics.acceptance(o), "cost_musd_per_task": 1e3 * o.cost_usd.mean(),
                "gap_vs_best_single": metrics.acceptance(o) - metrics.acceptance(ref_b),
                "saving_vs_best_single": 100 * (1 - o.cost_usd.sum() / ref_b.cost_usd.sum()),
                "gap_vs_cellbest_pred": metrics.acceptance(o) - metrics.acceptance(ref_c),
                "saving_vs_cellbest_pred": 100 * (1 - o.cost_usd.sum() / ref_c.cost_usd.sum())})
        res["seeds"][s] = {name: fits_summary(o) for name, o in outs.items()}

    # gate-seed table with paired CIs
    floor = metrics.acceptance(outcomes0[REF_BEST])
    refs = {"best_single": outcomes0[REF_BEST], "cellbest_pred": outcomes0[REF_C]}
    seed0 = [row(n, o, refs, problems, floor, pop) for n, o in outcomes0.items()]
    orc = metrics.oracle_outcomes(pop, pop.tasks)
    res["oracle"] = {"definition": metrics.OUR_DEFINITION, "acceptance": 100 * float(orc["accepted"].mean()),
                     "cost_musd_per_task": 1e3 * float(orc.cost_usd.mean())}
    res["floor"] = {"acceptance": floor, "source": "best single model, gate seed"}

    # sensitivity: the predictor trained on all eligible development tasks
    pool = true_cells(ed)[content_all.reindex(ed.tasks).notna().to_numpy()]
    sctx = replace(ctx, cell_pool=pool)
    folds0 = stats.dev_folds(pop.tasks, stats.GATE_SEED)
    sens = []
    for b in (CellBest("pred"), CellRule("A", "escalate", "pred")):
        o = run_dev(b, pop, sctx, folds0, seed=stats.GATE_SEED)
        r = row(b.name + "__pool_all_eligible_dev", o, refs, problems, floor, pop)
        sens.append(r)

    # released ACRouter decisions on the development tasks they cover
    cov = covered(pop.tasks)
    ac = {"n_covered": len(cov), "n_population": len(pop),
          "note": "ID-test tasks only: the release has no decisions for probing tasks"}
    if len(cov) >= 5:
        sub = pop.subset(list(cov))
        rr = replay(ReleasedACRouter().fit(sub, ctx),
                    ReplayView.from_matrix(sub, content_all.reindex(sub.tasks)), verifier,
                    seed=stats.GATE_SEED)                      # not fitted: no folds needed
        o_ac = metrics.task_outcomes(rr, sub)
        o_bs = outcomes0[REF_BEST].loc[cov]
        o_cb = outcomes0[REF_C].loc[cov]
        ac.update(acceptance=metrics.acceptance(o_ac), cost_musd_per_task=1e3 * o_ac.cost_usd.mean(),
                  best_single_on_covered=metrics.acceptance(o_bs),
                  vs_best_single=compare(o_ac, o_bs, problems),
                  vs_cellbest_pred=compare(o_ac, o_cb, problems))
    res["acrouter_released_dev"] = ac

    # shuffled labels: every learned baseline should fall to Random
    ok = success(pop)
    rand_exp = 100 * float((ok.sum(axis=1) / pop.available.sum(axis=1)).mean())
    rnd = outcomes0["random"]
    shuf = []
    for b in baselines:
        if not b.learned:
            continue
        accs, first = [], None
        for k in range(N_SHUFFLES):
            o = run_dev(b, pop, ctx, folds0, seed=stats.GATE_SEED, shuffle_seed=k)
            accs.append(metrics.acceptance(o))
            first = o if first is None else first
        c = compare(first, rnd, problems)
        u = compare(outcomes0[b.name], rnd, problems)["gap_points"]
        shuf.append({"baseline": b.name, "unshuffled": metrics.acceptance(outcomes0[b.name]),
                     "shuffled_mean": float(np.mean(accs)), "shuffled_min": float(np.min(accs)),
                     "shuffled_max": float(np.max(accs)), "random_expected": rand_exp,
                     "random_run": metrics.acceptance(rnd),
                     "shuffle0_gap_vs_random": c["gap_points"][0],
                     "shuffle0_gap_vs_random_lo": c["gap_points"][1],
                     "shuffle0_gap_vs_random_hi": c["gap_points"][2],
                     "unshuffled_gap_vs_random": u[0], "unshuffled_gap_vs_random_lo": u[1],
                     "unshuffled_gap_vs_random_hi": u[2]})
    res["predictor"] = predictor_report(hd, ed, problems, content_all)
    return res, pd.DataFrame(seed0), pd.DataFrame(per_seed_rows), pd.DataFrame(sens), pd.DataFrame(shuf)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, seed0, seeds, sens, shuf = run()
    seed0.to_csv(OUT / "s2_baselines_seed0.csv", index=False, lineterminator="\n")
    seeds.to_csv(OUT / "s2_baselines_seeds.csv", index=False, lineterminator="\n")
    sens.to_csv(OUT / "s2_baselines_sensitivity.csv", index=False, lineterminator="\n")
    shuf.to_csv(OUT / "s2_shuffled_labels.csv", index=False, lineterminator="\n")
    (OUT / "s2_baselines.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["baseline", "given_more_information", "acceptance", "acceptance_lo", "acceptance_hi",
            "cost_musd_per_task", "gap_vs_best_single", "gap_vs_best_single_lo",
            "gap_vs_best_single_hi", "saving_vs_best_single", "saving_vs_best_single_lo",
            "saving_vs_best_single_hi", "gap_vs_cellbest_pred", "saving_vs_cellbest_pred"]
    print(json.dumps(res["population"]))
    print(seed0[cols].round(2).to_string(index=False))
    print(sens[cols].round(2).to_string(index=False))
    print(shuf.round(2).to_string(index=False))
    print(json.dumps(res["acrouter_released_dev"], default=float, indent=1))
    print(json.dumps({k: v for k, v in res["predictor"].items() if k != "held_out_accuracy"}, indent=1))
    for f, a in res["predictor"]["held_out_accuracy"].items():
        print(f, {k: (v.get("accuracy"), v.get("n_wrong")) for k, v in a.items()})
