"""Lookup-table baselines (S2.5; CLAIMS §6, R2, O19).

Each one fits a table on the training folds (O1) and routes a scored task by looking up its group
or cell. Groups the router may not use are labelled **given more information**:
- `DimensionBest` (full, cost-aware, k-matched) and `SourceBest` read the dimension or the S1.1
  family from the task handle;
- `CellBest`, the per-cell `Rule` A and B, and the zero-loss route read the cell either from the
  task text through the S2.5 cell predictor (`cells="pred"`, router-visible; CellBest-pred is the
  S0.3(c) comparator) or from the true labels (`cells="true"`, given more information; CellBest-true
  is a reference, not a gate).

A group or cell with no training task goes to the training best single model (as P0.5 and P0.6).

Per-cell Rules A and B are re-implemented from `p0_6_cellbest_bound.rule_tables`'s stated
definition, over P0.6's candidates (direct dispatch to each model, then the fixed ladders) in one
mode (no-signal cells accepted or escalated):
- Rule A: the cheapest candidate whose training acceptance is at least CellBest's; ties to higher
  acceptance, then the lower column. **Rule A escalated is the bar C2 must beat (O19).**
- Rule B: the highest-acceptance candidate whose training cost is at most CellBest's; ties to lower
  cost, then the lower column.

The zero-loss route is Route Anything's direct-dispatch route at eps = 0 against the training best
single model (2611.05275 C.2, Table 10), re-implemented from P0.5's port of `crb.py lp_floor`
(`p0_5_recount.ra_*`): the fitted per-cell price path, then the last path router whose fitted
acceptance is at least the reference's, or a mixture of it and the next one. A mixture is played
per task with a draw from stream `route_mixture`. Values are acceptance under S0.1, not the
released score; the menu is direct dispatch only (no oracle cascades, R1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from router.baselines.cells import fit_predictor
from router.baselines.common import (OUR_DEFINITION, TOL, Context, PlanRouter, candidates, direct,
                                     policy_outcomes, success, true_cells)
from router.baselines.static import best_single_model
from router.data.schema import OutcomeMatrix
from router.sim.stats import rng
from router.sim.verifier import FAIL, PASS

TRUE_CELL_INFO = ("family", "dimension")


# --------------------------------------------------------------------------- table primitives
def group_means(groups: np.ndarray, M: np.ndarray, names) -> tuple[np.ndarray, np.ndarray]:
    """Mean of each column of M per group (rows in `names` order), and which groups have rows."""
    idx = {g: i for i, g in enumerate(names)}
    s = np.array([idx[g] for g in groups])
    n = np.bincount(s, minlength=len(names)).astype(float)
    S = np.zeros((len(names), M.shape[1]))
    np.add.at(S, s, M)
    return S / np.maximum(n, 1)[:, None], n > 0


def cellbest_column(v: np.ndarray, c: np.ndarray, n_direct: int) -> int:
    """P0.6's CellBest in one cell: the direct-dispatch column with the highest training
    acceptance, ties to the cheaper one (then the lower column)."""
    return int(np.lexsort((c[:n_direct], -v[:n_direct]))[0])


def rule_a_column(v: np.ndarray, c: np.ndarray, k: int) -> int:
    idx = np.arange(len(v))
    cand = idx[v >= v[k] - TOL]
    return int(cand[np.lexsort((cand, -v[cand], c[cand]))[0]])


def rule_b_column(v: np.ndarray, c: np.ndarray, k: int) -> int:
    idx = np.arange(len(v))
    cand = idx[c <= c[k] * (1 + TOL)]
    return int(cand[np.lexsort((cand, c[cand], -v[cand]))[0]])


# ------------------------------------------------------------------------------- cell sources
def _cell_reader(cells: str, train: OutcomeMatrix, ctx: Context):
    """(function handle -> cell, more_information) for `cells` in {"pred", "true"}."""
    if cells == "true":
        return (lambda h: f"{h.info['dimension']}|{h.info['family']}"), TRUE_CELL_INFO
    if cells == "pred":
        pred = fit_predictor(true_cells(train), ctx)
        return (lambda h: pred.predict_one(h.content_text)), ()
    raise ValueError(cells)


def _info_reader(key: str):
    return lambda h: h.info[key]


# -------------------------------------------------------------------------------- group lookup
@dataclass
class GroupBest:
    """Best direct-dispatch model per group: DimensionBest (`key="dimension"`), SourceBest
    (`key="family"`), or by cell (`key="cell"`, CellBest). `cost_aware=True`: the cheapest model
    within `margin` (acceptance fraction) of the group's best (P0.5's cost-aware DimensionBest)."""
    key: str
    cells: str = "pred"           # for key="cell" only
    cost_aware: bool = False
    margin: float = 0.02
    learned: bool = True

    @property
    def name(self) -> str:
        base = {"dimension": "dimension_best", "family": "source_best", "cell": "cellbest"}[self.key]
        if self.key == "cell":
            base += f"_{self.cells}"
        return base + ("_cost_aware" if self.cost_aware else "")

    @property
    def more_information(self) -> tuple:
        if self.key == "cell":
            return TRUE_CELL_INFO if self.cells == "true" else ()
        return (self.key,)

    @property
    def definition(self):
        if self.cost_aware:
            return (f"{OUR_DEFINITION}: P0.5's cost-aware DimensionBest, the cheapest model within "
                    f"{100 * self.margin:g} points of the group's best training acceptance")
        return None

    def fit(self, train: OutcomeMatrix, ctx: Context):
        ok = success(train).to_numpy(float)
        cost = train.cost.to_numpy(float)
        if self.key == "cell":
            groups = true_cells(train)
            read, info = _cell_reader(self.cells, train, ctx)
        else:
            groups = getattr(train, self.key).astype(str)
            read, info = _info_reader(self.key), (self.key,)
        names = sorted(groups.unique())
        v, seen = group_means(groups.to_numpy(), ok, names)
        c, _ = group_means(groups.to_numpy(), cost, names)
        table = {}
        for i, g in enumerate(names):
            if self.cost_aware:
                good = np.flatnonzero(v[i] >= v[i].max() - self.margin - TOL)
                j = int(good[np.lexsort((good, c[i, good]))[0]])
            else:
                j = cellbest_column(v[i], c[i], len(train.models))
            table[g] = direct(train.models[j])
        fallback = direct(best_single_model(train))
        router = PlanRouter(lambda h: table.get(read(h), fallback), info)
        router.table = {g: p.name for g, p in table.items()}
        router.fallback = fallback.name
        return router


def DimensionBest(cost_aware: bool = False) -> GroupBest:
    return GroupBest("dimension", cost_aware=cost_aware)


def SourceBest() -> GroupBest:
    return GroupBest("family")


def CellBest(cells: str = "pred") -> GroupBest:
    return GroupBest("cell", cells=cells)


# ----------------------------------------------------------------------------- per-cell rules
@dataclass
class CellRule:
    """P0.6's per-cell Rule A or B in one mode (no-signal cells `accept`ed or `escalate`d)."""
    rule: str                      # "A" or "B"
    mode: str                      # "accept" or "escalate"
    cells: str = "pred"
    learned: bool = True
    definition: str | None = None

    @property
    def name(self) -> str:
        return f"rule{self.rule}_{self.mode}_{self.cells}"

    @property
    def more_information(self) -> tuple:
        return TRUE_CELL_INFO if self.cells == "true" else ()

    def fit(self, train: OutcomeMatrix, ctx: Context):
        if self.rule not in ("A", "B"):
            raise ValueError(self.rule)
        pols = candidates(train, self.mode)
        A, C = policy_outcomes(train, ctx.verifier, pols)
        groups = true_cells(train)
        names = sorted(groups.unique())
        v, seen = group_means(groups.to_numpy(), A, names)
        c, _ = group_means(groups.to_numpy(), C, names)
        M = len(train.models)
        pick = rule_a_column if self.rule == "A" else rule_b_column
        table = {}
        for i, g in enumerate(names):
            k = cellbest_column(v[i], c[i], M)
            table[g] = pols[pick(v[i], c[i], k)]
        read, info = _cell_reader(self.cells, train, ctx)
        fallback = direct(best_single_model(train))
        router = PlanRouter(lambda h: table.get(read(h), fallback), info)
        router.table = {g: p.name for g, p in table.items()}
        router.fallback = fallback.name
        return router


# --------------------------------------------------------------------------- zero-loss route
def _pol_vd(g, pi, v, c):
    ix = np.arange(len(g))
    return float(pi @ v[ix, g]), float(pi @ c[ix, g])


def ratio_opt(pi, v, c, a_ref):
    """Dinkelbach: the best fitted acceptance-per-dollar ratio (P0.5 `ra_ratio_opt`)."""
    vb, cb = _pol_vd(np.full(len(pi), a_ref), pi, v, c)
    lam = vb / cb
    for _ in range(200):
        g = np.argmax(v - lam * c, axis=1)
        ev, ed = _pol_vd(g, pi, v, c)
        new = ev / ed
        if new <= lam * (1 + 1e-13):
            return lam
        lam = new
    raise RuntimeError("Dinkelbach did not converge")


def price_path(pi, v, c, a_ref):
    """Maximizers of v - lambda c on the breakpoint intervals of [0, eta*] (P0.5 `ra_price_path`)."""
    eta = ratio_opt(pi, v, c, a_ref)
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


def lp_floor(pi, v, c, eps, a_ref):
    """The fitted floor optimum E[V] >= vbar - eps (P0.5 `ra_lp_floor`): a router array, or
    (router_a, router_b, weight of a) for a mixture."""
    eta, path, lam_lo = price_path(pi, v, c, a_ref)
    vb, _ = _pol_vd(np.full(len(pi), a_ref), pi, v, c)
    k = 0
    for j, g in enumerate(path):
        if _pol_vd(g, pi, v, c)[0] >= vb - eps - 1e-12:
            k = j + 1
    if k == len(path):
        return path[-1]
    va, _ = _pol_vd(path[k - 1], pi, v, c)
    vn, _ = _pol_vd(path[k], pi, v, c)
    w = float((vb - eps - vn) / (va - vn))
    return (path[k - 1], path[k], w)


@dataclass
class ZeroLoss:
    """Route Anything's zero-loss direct-dispatch route (2611.05275 C.2, Table 10)."""
    cells: str = "true"
    eps: float = 0.0
    learned: bool = True

    @property
    def name(self) -> str:
        return f"zero_loss_{self.cells}"

    @property
    def more_information(self) -> tuple:
        return TRUE_CELL_INFO if self.cells == "true" else ()

    @property
    def definition(self):
        d = ("acceptance under S0.1 in place of the released score; cells are dimension x S1.1 "
             "family")
        if self.cells == "pred":
            d = f"{OUR_DEFINITION}: predicted cells; " + d
        return d

    def fit(self, train: OutcomeMatrix, ctx: Context):
        ok = success(train).to_numpy(float)
        cost = train.cost.to_numpy(float)
        groups = true_cells(train)
        names = sorted(groups.unique())
        v, _ = group_means(groups.to_numpy(), ok, names)
        c, _ = group_means(groups.to_numpy(), cost, names)
        n = groups.value_counts().reindex(names).to_numpy(float)
        pi = n / n.sum()
        a_ref = train.models.index(best_single_model(train))
        g = lp_floor(pi, v, c, self.eps, a_ref)
        if isinstance(g, tuple):
            ga, gb, w = g
        else:
            ga, gb, w = g, g, 1.0
        ta = {x: direct(train.models[ga[i]]) for i, x in enumerate(names)}
        tb = {x: direct(train.models[gb[i]]) for i, x in enumerate(names)}
        fallback = direct(train.models[a_ref])
        read, info = _cell_reader(self.cells, train, ctx)
        draw = rng(ctx.seed, "route_mixture")

        def plan(h):
            x = read(h)
            if x not in ta:
                return fallback
            return ta[x] if draw.random() < w else tb[x]

        router = PlanRouter(plan, info)
        router.table = {x: (ta[x].name if ta[x] == tb[x] else
                            f"{ta[x].name} {w:.3f} / {tb[x].name} {1 - w:.3f}") for x in names}
        router.fallback = fallback.name
        return router


# ----------------------------------------------------------------------- k-matched DimensionBest
@dataclass
class KMatchedDimensionBest:
    """DimensionBest for onboarding a new model from k outcomes (S0.5): every other model's table
    entries use all training tasks; the new model's per-dimension acceptance uses only k probe
    tasks drawn from the training tasks (stream `probe_draw`), labelled by the ground truth or by
    the verifier (PASS 1, FAIL 0, UNKNOWN dropped). A dimension with no probe label for the new
    model leaves it out there. The router's `n_probes` and `probe_usd` carry the probe cost."""
    new_model: str
    k: int
    labels: str = "ground_truth"      # or "verifier"
    learned: bool = True
    more_information: tuple = ("dimension",)
    definition: str | None = None

    @property
    def name(self) -> str:
        return f"dimension_best_k{self.k}_{self.labels}"

    def fit(self, train: OutcomeMatrix, ctx: Context):
        if self.new_model not in train.models:
            raise ValueError(self.new_model)
        if self.labels not in ("ground_truth", "verifier"):
            raise ValueError(self.labels)
        ok = success(train).astype(float)
        dims = train.dimension.astype(str)
        g = rng(ctx.seed, "probe_draw")
        avail = train.tasks[train.available[self.new_model].astype(bool).to_numpy()]
        k = min(self.k, len(avail))
        probes = pd.Index(sorted(avail))[np.sort(g.choice(len(avail), size=k, replace=False))]
        if self.labels == "ground_truth":
            lab = ok.loc[probes, self.new_model]
        else:
            vals = {}
            for t in probes:
                verdict = ctx.verifier.check(t, self.new_model).verdict
                if verdict in (PASS, FAIL):
                    vals[t] = float(verdict == PASS)
            lab = pd.Series(vals, dtype=float)
        others = [m for m in train.models if m != self.new_model]
        table = {}
        for d in sorted(dims.unique()):
            rows = dims.index[dims == d]
            acc = {m: ok.loc[rows, m].mean() for m in others}
            cst = {m: train.cost.loc[rows, m].mean() for m in others}
            pl = lab[dims.reindex(lab.index) == d]
            if len(pl):
                acc[self.new_model] = pl.mean()
                cst[self.new_model] = train.cost.loc[pl.index, self.new_model].mean()
            ms = list(acc)
            j = int(np.lexsort((np.arange(len(ms)), [cst[m] for m in ms], [-acc[m] for m in ms]))[0])
            table[d] = direct(ms[j])
        # the fallback must not see the new model's outcomes beyond its k probes
        oacc = ok[others].mean().to_numpy()
        ocst = train.cost[others].mean().to_numpy()
        fallback = direct(others[int(np.lexsort((np.arange(len(others)), ocst, -oacc))[0])])
        router = PlanRouter(lambda h: table.get(h.info["dimension"], fallback), ("dimension",))
        router.table = {d: p.name for d, p in table.items()}
        router.n_probes = int(k)
        router.probe_usd = float(train.cost.loc[probes, self.new_model].sum())
        return router
