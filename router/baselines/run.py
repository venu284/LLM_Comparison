"""Run a baseline out of sample on the development split, and the shuffled-label check (S2.5, S2.6).

`run_dev` fits the baseline on the other development folds (`stats.dev_folds`, problem-grouped),
replays it on the held-out fold, and joins the decisions to the outcomes with `sim.metrics`
(acceptance under S0.1, cost under S0.2). Every task is scored once, by a fit that never saw its
problem. The evaluation population is never touched (`dev_folds` refuses it).

`shuffle_models` is the shuffled-label check: inside each training task, the eight models' cell
records (score, cost, availability and verifier verdict) are permuted together, so the models become
exchangeable in the training data and a learned baseline has nothing to learn; it should fall to
Random. Permuting across tasks instead would keep each model's overall rate, and learners would
fall to always-best, not to Random.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from router.baselines.acrouter import ReleasedACRouter
from router.baselines.cascade import FixedCascade
from router.baselines.common import ACCEPT, ESCALATE, MODES, Context
from router.baselines.lookup import CellBest, CellRule, DimensionBest, SourceBest, ZeroLoss
from router.baselines.online import KNN, LinUCB
from router.baselines.static import Always, BestSingle, RandomChoice
from router.data.schema import OutcomeMatrix
from router.sim import metrics
from router.sim.stats import rng
from router.sim.stream import ReplayView, replay


def default_baselines(models) -> list:
    """Every S2.5 baseline run on the development split (R2, O19), in report order."""
    out = [BestSingle(), RandomChoice(), *[Always(m) for m in models],
           DimensionBest(), DimensionBest(cost_aware=True), SourceBest(),
           CellBest("pred"), CellBest("true")]
    out += [CellRule(r, mode, cells) for cells in ("pred", "true") for mode in MODES
            for r in ("A", "B")]
    out += [ZeroLoss("true"), ZeroLoss("pred"), FixedCascade(ACCEPT), FixedCascade(ESCALATE),
            LinUCB(), KNN()]
    return out


# ----------------------------------------------------------------------------- label shuffle
class PermutedVerifier:
    """The replay verifier seen through a per-task model permutation: `perm[t][m]` is the model
    whose record now sits in column m of task t."""

    def __init__(self, base, perm: dict[str, dict[str, str]]):
        self.base, self.perm = base, perm
        self.mode, self.sensitivity = base.mode, base.sensitivity

    def check(self, task_id, model):
        return self.base.check(task_id, self.perm.get(task_id, {}).get(model, model))

    def has_record(self, task_id):
        return self.base.has_record(task_id)

    def empty_cells(self):
        inv = {t: {v: k for k, v in p.items()} for t, p in self.perm.items()}
        return {(t, inv.get(t, {}).get(m, m)) for t, m in self.base.empty_cells()}


def shuffle_models(train: OutcomeMatrix, verifier, seed: int) -> tuple[OutcomeMatrix, PermutedVerifier]:
    """Permute the models' cell records inside each training task (stream `label_shuffle`)."""
    g = rng(seed, "label_shuffle")
    models = list(train.models)
    order = sorted(train.tasks)
    P = np.stack([g.permutation(len(models)) for _ in order])      # row i: source column per column
    P = P[pd.Index(order).get_indexer(train.tasks)]                # back to the matrix's task order
    perm = {t: dict(zip(models, (models[j] for j in P[i]))) for i, t in enumerate(train.tasks)}
    rows = np.arange(len(train))[:, None]

    def permuted(frame):
        return pd.DataFrame(frame.to_numpy()[rows, P], index=frame.index, columns=frame.columns)

    score, cost = permuted(train.score), permuted(train.cost)
    av, reason = permuted(train.available).astype(bool), permuted(train.unavailable_reason)
    m = OutcomeMatrix(score, cost, train.dimension.copy(), train.family.copy(),
                      train.name + "/shuffled", available=av.astype(bool),
                      unavailable_reason=reason)
    return m, PermutedVerifier(verifier, perm)


# --------------------------------------------------------------------------------- dev runner
def run_dev(baseline, m: OutcomeMatrix, ctx: Context, folds: pd.Series, *, seed: int,
            shuffle_seed: int | None = None, cruxeval_lenient: bool = False) -> pd.DataFrame:
    """Out-of-sample outcomes of `baseline` on the tasks of `folds` (one row per task). The frame's
    attrs carry the baseline's labels and each fold's fitted table."""
    if ctx.problem_of is None:
        raise ValueError("run_dev needs Context.problem_of")
    tasks = pd.Index(folds.index)
    if not tasks.isin(m.tasks).all():
        raise KeyError("fold tasks missing from the matrix")
    parts, fits = [], []
    for f in sorted(folds.unique()):
        te, tr = tasks[folds.to_numpy() == f], tasks[folds.to_numpy() != f]
        train, test = m.subset(list(tr)), m.subset(list(te))
        fctx = replace(ctx.for_fold(set(ctx.problem_of.reindex(te))), seed=seed)
        if shuffle_seed is not None:
            train, pv = shuffle_models(train, ctx.verifier, shuffle_seed * 1000 + int(f))
            fctx = replace(fctx, verifier=pv)
        router = baseline.fit(train, fctx)
        content = None if ctx.content is None else ctx.content.reindex(test.tasks)
        res = replay(router, ReplayView.from_matrix(test, content), ctx.verifier, seed=seed)
        o = metrics.task_outcomes(res, test, cruxeval_lenient=cruxeval_lenient,
                                  probe_usd=float(getattr(router, "probe_usd", 0.0)),
                                  n_probes=int(getattr(router, "n_probes", 0)))
        o["fold"] = int(f)
        parts.append(o)
        fits.append({"fold": int(f), "n_train": len(train), "n_scored": len(test),
                     "table": getattr(router, "table", None),
                     "fallback": getattr(router, "fallback", None),
                     "chosen": getattr(router, "chosen", None),
                     "given_more_information": list(res.given_more_information)})
    out = pd.concat(parts).loc[tasks]
    out.attrs.update(name=baseline.name, learned=bool(baseline.learned),
                     more_information=tuple(baseline.more_information),
                     definition=baseline.definition, seed=seed, shuffle_seed=shuffle_seed,
                     fits=fits)
    return out


def shuffled_mean_interval(accs, level: float = 0.95) -> tuple[float, float, float]:
    """Mean acceptance over the shuffles and its t interval (n - 1 degrees of freedom). The
    shuffled-label check passes when Random's expected acceptance lies inside it."""
    from scipy import stats as st
    a = np.asarray(accs, dtype=float)
    m = float(a.mean())
    if len(a) < 2:
        return m, m, m
    h = float(st.t.ppf(0.5 + level / 2, len(a) - 1) * a.std(ddof=1) / np.sqrt(len(a)))
    return m, m - h, m + h


__all__ = ["default_baselines", "run_dev", "shuffle_models", "shuffled_mean_interval",
           "PermutedVerifier", "ReleasedACRouter"]
