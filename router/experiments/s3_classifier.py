"""S3.3 fitted part: how much a give-up classifier captures against its perfect-information bound
(S3 spec D8; compare Route Anything C.5). Development split only.

Populations: the headline development split, complete case; the `all_eligible` development
split's X13 tasks, complete case, as a second population. `stats.dev_folds`, seeds 0 to 4.

Ladder: in each training fold, S2's `FixedCascade("escalate")` (the cheapest P0.5 ladder whose
training acceptance is at least the training best single model's).

Post-failure classifier (primary). A decision point is a rung j of the ladder, not the last
available one, whose verdict is FAIL or UNKNOWN (so an escalate ladder would go on). Target
y = 1 when any later available rung of the ladder succeeds (S0.1). Features, all observable when
the decision is made (S0.13): the predicted cell (S2.5 predictor, fitted on the training fold), the
rung position, the verdict history of the rungs tried, and the task's `content_text` (TF-IDF as in
`cells.make_classifier`); logistic regression (C = 10, as there). No released score, family or
dimension. Policy: give up (accept nothing, pay nothing more) when the predicted P(y = 1) < t. t is
the largest value whose acceptance loss against the full ladder is at most 2 points on out-of-fold
predictions inside the training folds (each training fold predicted by a classifier fitted on the
other training folds). Perfect-information bound: give up exactly when y = 0; same acceptance as
the full ladder, largest saving.

Pre-attempt classifier (secondary, D8 ruling). One decision before rung 1: y = 1 when any available
rung of the ladder succeeds; features: the predicted cell and `content_text`. Same t rule and bound.

Capture = held-out saving of the classifier policy / held-out saving of the perfect-information
policy, on totals over the held-out tasks (pooled over folds); the acceptance loss is beside it.
Also the held-out curve over t and the held-out AUC, both with cluster-bootstrap CIs over problems
(`N_BOOT`, seed 0) on the gate seed. Route Anything C.5 measures pre-attempt signals within a cell
(-0.013 to +0.078); its definition differs, so the comparison is qualitative.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s3_classifier
Writes router/experiments/out/s3_3_classifier.csv, s3_3_classifier_curve.csv and
s3_3_classifier.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from router.baselines.cascade import FixedCascade
from router.baselines.cells import CLF_SEED, fit_predictor
from router.baselines.common import ESCALATE, Context, Policy, success, true_cells, verdicts
from router.data import splits as S
from router.data.audit import build_views, load_release
from router.data.prompts import join
from router.data.schema import OutcomeMatrix
from router.settings import N_BOOT, REPO_ROOT, VERIFIER_RUN_USD
from router.sim import stats
from router.sim.verifier import PASS, ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
MAX_LOSS = 2.0                 # points of acceptance, on out-of-fold training predictions
KINDS = ("post_failure", "pre_attempt")
CURVE_T = tuple(np.round(np.linspace(0.0, 1.0, 21), 2))
POPS = ("filtered = headline", "all_eligible")


# ------------------------------------------------------------------------------------- traces
@dataclass
class Trace:
    """The full ladder's walk on one task, and its decision points."""
    rung_models: list            # available rungs, in order
    step_cost: list              # cost of each rung reached (model + verifier run)
    verdict: list                # verdict of each rung reached
    ok: list                     # S0.1 success of each rung reached
    stop_at: int                 # index of the rung the full ladder accepts (-1: no rung)
    rung_ok: list                # S0.1 success of every available rung, reached or not


def traces(m: OutcomeMatrix, verifier, pol: Policy, verifier_usd: float = VERIFIER_RUN_USD) -> dict:
    """Walk the escalate ladder `pol` on every task of `m`, as `common.policy_outcomes` does."""
    ok = success(m)
    v, ran = verdicts(m, verifier)
    av = m.available.astype(bool)
    out = {}
    for t in m.tasks:
        rungs = [x for x in pol.rungs if av.at[t, x]]
        tr = Trace([], [], [], [], -1, [bool(ok.at[t, x]) for x in rungs])
        for i, x in enumerate(rungs):
            tr.rung_models.append(x)
            tr.step_cost.append(float(m.cost.at[t, x]) + verifier_usd * bool(ran.at[t, x]))
            tr.verdict.append(v.at[t, x])
            tr.ok.append(bool(ok.at[t, x]))
            if v.at[t, x] == PASS or i == len(rungs) - 1:
                tr.stop_at = i
                break
        out[t] = tr
    return out


def full_outcome(tr: Trace) -> tuple[bool, float]:
    return (tr.ok[tr.stop_at] if tr.stop_at >= 0 else False), float(sum(tr.step_cost))


def decision_rows(trs: dict, kind: str) -> pd.DataFrame:
    """One row per decision point: task, position j (-1 = before rung 1), history, y."""
    rows = []
    for t, tr in trs.items():
        if kind == "pre_attempt":
            if tr.rung_models:
                rows.append({"task_id": t, "j": -1, "history": "", "y": int(any(tr.rung_ok))})
            continue
        for j in range(tr.stop_at if tr.stop_at >= 0 else 0):
            hist = " ".join(f"r{i}:{tr.verdict[i]}" for i in range(j + 1))
            rows.append({"task_id": t, "j": j, "history": hist, "y": int(any(tr.rung_ok[j + 1:]))})
    return pd.DataFrame(rows, columns=["task_id", "j", "history", "y"])


def simulate(trs: dict, stop: dict) -> tuple[dict, dict]:
    """Per task (accepted, cost) when the policy gives up at the first decision point with
    stop[(task, j)] True."""
    acc, cost = {}, {}
    for t, tr in trs.items():
        if not tr.rung_models:
            acc[t], cost[t] = False, 0.0
            continue
        if stop.get((t, -1), False):
            acc[t], cost[t] = False, 0.0
            continue
        a, c = full_outcome(tr)
        for j in range(tr.stop_at if tr.stop_at >= 0 else 0):
            if stop.get((t, j), False):
                a, c = False, float(sum(tr.step_cost[:j + 1]))
                break
        acc[t], cost[t] = a, c
    return acc, cost


# --------------------------------------------------------------------------------- classifier
class GiveUpClassifier:
    """P(y = 1) from the predicted cell, the position, the verdict history and the task text."""

    def __init__(self, rows: pd.DataFrame, texts: pd.Series, cells: pd.Series):
        from scipy.sparse import hstack
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_union
        self._hstack = hstack
        self.const = None
        y = rows.y.to_numpy()
        if len(rows) == 0 or len(set(y)) < 2:
            self.const = float(y.mean()) if len(rows) else 0.5
            return
        self.text = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                            max_features=200_000))
        self.meta = DictVectorizer()
        X = self._hstack([self.text.fit_transform(texts.reindex(rows.task_id).tolist()),
                          self.meta.fit_transform(self._meta(rows, cells))]).tocsr()
        self.clf = LogisticRegression(C=10.0, max_iter=5000, random_state=CLF_SEED).fit(X, y)

    @staticmethod
    def _meta(rows, cells):
        out = []
        for r in rows.itertuples():
            d = {f"cell={cells[r.task_id]}": 1.0, f"pos={r.j}": 1.0}
            for tok in r.history.split():
                d[tok] = 1.0
            out.append(d)
        return out

    def predict(self, rows: pd.DataFrame, texts: pd.Series, cells: pd.Series) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0)
        if self.const is not None:
            return np.full(len(rows), self.const)
        X = self._hstack([self.text.transform(texts.reindex(rows.task_id).tolist()),
                          self.meta.transform(self._meta(rows, cells))]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


def choose_t(trs: dict, rows: pd.DataFrame, p: np.ndarray, max_loss: float = MAX_LOSS) -> float:
    """The largest t whose acceptance loss against the full ladder, on these tasks, is at most
    `max_loss` points (give up when p < t). Candidates: 0 and every predicted value, plus 1."""
    keys = list(zip(rows.task_id, rows.j))
    full = np.mean([full_outcome(tr)[0] for tr in trs.values()]) * 100
    best = 0.0
    for t in sorted({0.0, *map(float, p), 1.0 + 1e-9}):
        acc, _ = simulate(trs, {k: bool(q < t) for k, q in zip(keys, p)})
        if full - 100 * np.mean(list(acc.values())) <= max_loss + 1e-12:
            best = max(best, t)
    return best


# ---------------------------------------------------------------------------------- one run
def run_population(m: OutcomeMatrix, ctx: Context, seed: int, kind: str) -> dict:
    folds = stats.dev_folds(m.tasks, seed)
    held_rows, chosen = [], []
    all_trs = {}
    for f in sorted(folds.unique()):
        te, tr_ids = folds.index[folds == f], folds.index[folds != f]
        train, test = m.subset(list(tr_ids)), m.subset(list(te))
        fctx = ctx.for_fold(set(ctx.problem_of.reindex(te)))
        lad = FixedCascade(ESCALATE).fit(train, fctx)
        pol = Policy(tuple(lad.table["all"].split("@")[0].split(">")), True, ESCALATE)
        pred = fit_predictor(true_cells(train), fctx)
        cells = pd.Series({t: pred.predict_one(ctx.content.get(t)) for t in m.tasks})
        tr_tr, te_tr = traces(train, ctx.verifier, pol), traces(test, ctx.verifier, pol)
        rows_tr, rows_te = decision_rows(tr_tr, kind), decision_rows(te_tr, kind)
        # t from out-of-fold predictions inside the training folds
        inner = folds.reindex(tr_ids)
        p_oof = pd.Series(np.nan, index=rows_tr.index)
        for g in sorted(inner.unique()):
            fit_ids = set(inner.index[inner != g])
            fit_rows = rows_tr[rows_tr.task_id.isin(fit_ids)]
            pr = rows_tr[~rows_tr.task_id.isin(fit_ids)]
            clf = GiveUpClassifier(fit_rows, ctx.content, cells)
            p_oof.loc[pr.index] = clf.predict(pr, ctx.content, cells)
        t = choose_t(tr_tr, rows_tr, p_oof.to_numpy())
        clf = GiveUpClassifier(rows_tr, ctx.content, cells)
        p = clf.predict(rows_te, ctx.content, cells)
        held_rows.append(rows_te.assign(p=p, fold=int(f), t=t))
        chosen.append({"fold": int(f), "ladder": pol.name, "t": t, "n_train_rows": len(rows_tr),
                       "n_train_y1": int(rows_tr.y.sum())})
        all_trs.update(te_tr)
    rows = pd.concat(held_rows, ignore_index=True)
    return {"rows": rows, "traces": all_trs, "fits": chosen}


def summarise(res: dict, problems: pd.Series, with_ci: bool) -> tuple[dict, pd.DataFrame]:
    rows, trs = res["rows"], res["traces"]
    keys = list(zip(rows.task_id, rows.j))
    tasks = pd.Index(sorted(trs))
    full = pd.DataFrame([full_outcome(trs[t]) for t in tasks], index=tasks, columns=["acc", "cost"])
    cls_acc, cls_cost = simulate(trs, {k: bool(q < t) for k, q, t in zip(keys, rows.p, rows.t)})
    per_acc, per_cost = simulate(trs, {k: bool(y == 0) for k, y in zip(keys, rows.y)})
    f = pd.DataFrame({"full_acc": full.acc.astype(float), "full_cost": full.cost,
                      "cls_acc": pd.Series(cls_acc).reindex(tasks).astype(float),
                      "cls_cost": pd.Series(cls_cost).reindex(tasks),
                      "per_acc": pd.Series(per_acc).reindex(tasks).astype(float),
                      "per_cost": pd.Series(per_cost).reindex(tasks)}, index=tasks)
    cl = problems.reindex(tasks)

    def capture(fr, w):
        s_cls = np.sum(w * (fr.full_cost - fr.cls_cost))
        s_per = np.sum(w * (fr.full_cost - fr.per_cost))
        return 100.0 * float(s_cls / s_per) if s_per > 0 else float("nan")

    def saving(col):
        return lambda fr, w: 100.0 * float(np.sum(w * (fr.full_cost - fr[col])) / np.sum(w * fr.full_cost))

    out = {"n_tasks": len(tasks), "n_decision_points": len(rows), "n_y1": int(rows.y.sum()),
           "full_acceptance": 100 * f.full_acc.mean(), "classifier_acceptance": 100 * f.cls_acc.mean(),
           "perfect_acceptance": 100 * f.per_acc.mean(),
           "acceptance_loss_points": 100 * (f.full_acc.mean() - f.cls_acc.mean()),
           "full_cost_usd": float(f.full_cost.sum()),
           "classifier_saving_pct": saving("cls_cost")(f, np.ones(len(f))),
           "perfect_saving_pct": saving("per_cost")(f, np.ones(len(f))),
           "capture_pct": capture(f, np.ones(len(f))), "n_gave_up": int(((f.cls_cost < f.full_cost) & (f.cls_acc == 0)).sum())}
    auc = _auc(rows.y.to_numpy(), rows.p.to_numpy(), np.ones(len(rows)))
    out["auc"] = auc
    if with_ci:
        iv = stats.bootstrap(f, capture, clusters=cl, n_boot=N_BOOT)
        out["capture_pct_ci"] = [iv.lo, iv.hi]
        lo = stats.bootstrap(f, stats.mean_diff("full_acc", "cls_acc"), clusters=cl, n_boot=N_BOOT)
        out["acceptance_loss_ci"] = [lo.lo, lo.hi]
        if len(set(rows.y)) == 2:
            r = rows.set_index(pd.RangeIndex(len(rows)))
            a = stats.bootstrap(r, lambda fr, w: _auc(fr.y.to_numpy(), fr.p.to_numpy(), w),
                                clusters=problems.reindex(r.task_id).set_axis(r.index), n_boot=N_BOOT)
            out["auc_ci"] = [a.lo, a.hi]
    curve = []
    for t in CURVE_T:
        a, c = simulate(trs, {k: bool(q < t) for k, q in zip(keys, rows.p)})
        ca, cc = pd.Series(a).reindex(tasks).astype(float), pd.Series(c).reindex(tasks)
        curve.append({"t": float(t), "acceptance_loss_points": 100 * (f.full_acc.mean() - ca.mean()),
                      "saving_pct": 100 * (1 - cc.sum() / f.full_cost.sum())})
    return out, pd.DataFrame(curve)


def _auc(y, p, w) -> float:
    from sklearn.metrics import roc_auc_score
    keep = w > 0
    if len(set(y[keep])) < 2:
        return float("nan")
    return float(roc_auc_score(y[keep], p[keep], sample_weight=w[keep]))


# --------------------------------------------------------------------------------------- main
def populations():
    m = load_release()
    views = build_views(m)
    ts = S.task_splits().set_index("task_id")
    dev = set(ts.index[ts.split == S.DEV])
    v = ReplayVerifier.from_x13()
    out = {}
    for name, key in (("filtered = headline", "headline"), ("all_eligible", "all_eligible")):
        x = views[key]
        x = x.subset([t for t in x.tasks if t in dev and v.has_record(t)])
        out[name] = x.subset(list(x.tasks[x.available.astype(bool).all(axis=1).to_numpy()]))
    return out, v, ts.problem_id


def run():
    pops, v, problems = populations()
    content = join(sorted({t for m in pops.values() for t in m.tasks}))
    rows, curves, res = [], [], {"spec": "docs/research/S3_analysis_spec.md D8", "fits": {},
                                  "populations": {k: len(m) for k, m in pops.items()},
                                  "max_loss_points": MAX_LOSS, "n_boot": N_BOOT}
    for pname, m in pops.items():
        ctx = Context(v, content=content, problem_of=problems)
        for kind in KINDS:
            for seed in stats.SEEDS:
                r = run_population(m, ctx, seed, kind)
                s, curve = summarise(r, problems, with_ci=seed == stats.GATE_SEED)
                rows.append({"population": pname, "population_kind": "development split, complete case",
                             "classifier": kind, "seed": seed, "gate_seed": seed == stats.GATE_SEED,
                             **{k: v_ for k, v_ in s.items() if not isinstance(v_, list)},
                             **{f"{k}_lo": v_[0] for k, v_ in s.items() if isinstance(v_, list)},
                             **{f"{k}_hi": v_[1] for k, v_ in s.items() if isinstance(v_, list)}})
                curves.append(curve.assign(population=pname, classifier=kind, seed=seed))
                res["fits"][f"{pname}|{kind}|{seed}"] = r["fits"]
    res["route_anything_c5"] = ("pre-attempt signals within a cell, -0.013 to +0.078; definition "
                                "differs, qualitative comparison only")
    return res, pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, table, curve = run()
    table.to_csv(OUT / "s3_3_classifier.csv", index=False, lineterminator="\n")
    curve[["population", "classifier", "seed", "t", "acceptance_loss_points", "saving_pct"]].to_csv(
        OUT / "s3_3_classifier_curve.csv", index=False, lineterminator="\n")
    (OUT / "s3_3_classifier.json").write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    cols = ["population", "classifier", "seed", "n_decision_points", "n_y1", "acceptance_loss_points",
            "classifier_saving_pct", "perfect_saving_pct", "capture_pct", "auc"]
    print(table[cols].round(3).to_string(index=False))
