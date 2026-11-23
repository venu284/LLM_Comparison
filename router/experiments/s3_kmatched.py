"""S3.4: k-matched DimensionBest curves for onboarding a new model (S3 spec D9; CLAIMS S0.5, R8,
R11). Development split only.

Populations: the headline development split and the `all_eligible` development split, complete
case, `stats.dev_folds`, seeds 0 to 4. Leave-one-model-out: each of the 8 models in turn is the new
model; the other 7 keep their full training tables (`lookup.KMatchedDimensionBest`). Results are per
model, never pooled (R8).

k grid: `DP2_K_GRID` = {10, 25, 50, 100} plus 250. Label sources: the ground truth and the verifier
(PASS = 1, FAIL = 0, UNKNOWN dropped; k still counts every probe call, and the effective number of
labels, PASS plus FAIL, is reported per fold). Probe draws use stream 3 (`probe_draw`) under the
seed; the probe cost is spread over the scored tasks (S0.2).

Comparators, paired on the same folds: DimensionBest on all 8 models with full tables (k = inf) and
the table without the new model (k = 0, the pool prior, R11).

Unreachable k: `KMatchedDimensionBest.fit` caps k at the number of training tasks where the new
model is available, silently. Here a (population, model, k) whose smallest training-fold
availability is below k is reported as "unreachable" with that availability, and nothing is run.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s3_kmatched
Writes router/experiments/out/s3_4_kmatched.csv, s3_4_kmatched_seeds.csv and s3_4_kmatched.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from router.baselines.common import Context
from router.baselines.lookup import DimensionBest, KMatchedDimensionBest
from router.baselines.run import run_dev
from router.data import splits as S
from router.data.audit import build_views, load_release
from router.data.schema import OutcomeMatrix
from router.settings import DP2_K_GRID, N_BOOT, REPO_ROOT
from router.sim import metrics, stats
from router.sim.verifier import FAIL, PASS, ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
K_GRID = tuple(DP2_K_GRID) + (250,)
LABELS = ("ground_truth", "verifier")
POPS = {"filtered = headline": "headline", "all_eligible": "all_eligible"}


class Unreachable(ValueError):
    """k exceeds the new model's availability in some training fold."""


def min_training_availability(m: OutcomeMatrix, folds: pd.Series, model: str) -> int:
    av = m.available[model].astype(bool)
    return int(min(av.loc[folds.index[folds != f]].sum() for f in sorted(folds.unique())))


def require_reachable(m: OutcomeMatrix, folds: pd.Series, model: str, k: int) -> None:
    n = min_training_availability(m, folds, model)
    if k > n:
        raise Unreachable(f"k = {k} > {n} training tasks with {model} available")


def probe_labels(train: OutcomeMatrix, model: str, k: int, seed: int, verifier) -> dict:
    """The probes `KMatchedDimensionBest.fit` draws (same stream, same order) and how many carry
    a verifier label (PASS or FAIL)."""
    g = stats.rng(seed, "probe_draw")
    avail = train.tasks[train.available[model].astype(bool).to_numpy()]
    kk = min(k, len(avail))
    probes = pd.Index(sorted(avail))[np.sort(g.choice(len(avail), size=kk, replace=False))]
    n_lab = sum(verifier.check(t, model).verdict in (PASS, FAIL) for t in probes)
    return {"n_probes": int(kk), "n_verifier_labels": int(n_lab),
            "probe_usd": float(train.cost.loc[probes, model].sum())}


def populations():
    m = load_release()
    views = build_views(m)
    ts = S.task_splits().set_index("task_id")
    dev = set(ts.index[ts.split == S.DEV])
    out = {}
    for name, key in POPS.items():
        x = views[key]
        x = x.subset([t for t in x.tasks if t in dev])
        out[name] = x.subset(list(x.tasks[x.available.astype(bool).all(axis=1).to_numpy()]))
    return out, ts.problem_id


def compare(o, ref, problems, with_ci: bool) -> dict:
    f = metrics.paired_frame(o, ref, ("x", "r"))
    gap = stats.mean_diff("accepted_x", "accepted_r")
    sav = stats.saving_pct("cost_x", "cost_r")
    one = np.ones(len(f))
    out = {"gap": gap(f, one), "saving": sav(f, one)}
    if with_ci:
        cl = problems.reindex(f.index)
        g = stats.bootstrap(f, gap, clusters=cl, n_boot=N_BOOT)
        s = stats.bootstrap(f, sav, clusters=cl, n_boot=N_BOOT)
        out.update(gap_lo=g.lo, gap_hi=g.hi, saving_lo=s.lo, saving_hi=s.hi)
    return out


def run():
    pops, problems = populations()
    v = ReplayVerifier.from_x13()
    rows, unreachable = [], []
    for pname, m in pops.items():
        ctx = Context(v, problem_of=problems)
        for seed in stats.SEEDS:
            folds = stats.dev_folds(m.tasks, seed)
            full = run_dev(DimensionBest(), m, ctx, folds, seed=seed)
            ci = seed == stats.GATE_SEED
            for model in m.models:
                k0 = run_dev(KMatchedDimensionBest(model, 0), m, ctx, folds, seed=seed)
                base = {"population": pname, "population_kind": "development split, complete case",
                        "seed": seed, "gate_seed": ci, "new_model": model,
                        "min_training_availability": min_training_availability(m, folds, model)}
                for label, o in (("k=inf (full DimensionBest)", full), ("k=0 (pool prior)", k0)):
                    rows.append({**base, "k": label, "labels": "", "status": "comparator",
                                 "acceptance": metrics.acceptance(o),
                                 "cost_musd_per_task": 1e3 * float(o.cost_usd.mean())})
                for k in K_GRID:
                    for lab in LABELS:
                        r = {**base, "k": str(k), "labels": lab}
                        try:
                            require_reachable(m, folds, model, k)
                        except Unreachable:
                            rows.append({**r, "status": "unreachable"})
                            unreachable.append({k_: r[k_] for k_ in ("population", "seed", "new_model", "k",
                                                                     "labels", "min_training_availability")})
                            continue
                        o = run_dev(KMatchedDimensionBest(model, k, lab), m, ctx, folds, seed=seed)
                        per_fold = []
                        for f in sorted(folds.unique()):
                            tr = m.subset(list(folds.index[folds != f]))
                            per_fold.append(probe_labels(tr, model, k, seed, v))
                        n_lab = [p["n_verifier_labels"] if lab == "verifier" else p["n_probes"]
                                 for p in per_fold]
                        r.update(status="run", acceptance=metrics.acceptance(o),
                                 cost_musd_per_task=1e3 * float(o.cost_usd.mean()),
                                 probe_usd=float(o.probe_usd.sum()),
                                 effective_labels_min=int(min(n_lab)),
                                 effective_labels_mean=float(np.mean(n_lab)))
                        for tag, ref in (("full", full), ("k0", k0)):
                            c = compare(o, ref, problems, ci)
                            r.update({f"{a}_vs_{tag}": b for a, b in c.items()})
                        rows.append(r)
    t = pd.DataFrame(rows)
    res = {"spec": "docs/research/S3_analysis_spec.md D9", "k_grid": list(K_GRID),
           "populations": {k: len(m) for k, m in pops.items()}, "n_boot": N_BOOT,
           "unreachable": unreachable,
           "note": "per model, never pooled (R8); CIs on the gate seed only; probe cost spread over "
                   "the scored tasks (S0.2)"}
    return res, t


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, t = run()
    t.to_csv(OUT / "s3_4_kmatched_seeds.csv", index=False, lineterminator="\n")
    t[t.gate_seed].to_csv(OUT / "s3_4_kmatched.csv", index=False, lineterminator="\n")
    (OUT / "s3_4_kmatched.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=int) + "\n")
    g = t[t.gate_seed & (t.status != "unreachable")]
    cols = ["population", "new_model", "k", "labels", "acceptance", "cost_musd_per_task",
            "gap_vs_full", "gap_vs_k0", "effective_labels_min"]
    print(g[[c for c in cols if c in g]].round(2).to_string(index=False))
