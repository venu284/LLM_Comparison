"""Router v1 (plan S4.1; `docs/research/S4_spec.md` §3): a shrunk per-group lookup table plus a
fixed two-rung cascade.

Fitting (`RouterV1.fit(train, ctx)`) reads the training folds only (O1, S0.6):
1. The fixed cascade (§3.4): among P0.5's two-rung ladders (`common.ladders`, non-decreasing
   training mean cost) in mode `mode`, the cheapest whose training acceptance is at least the
   training best single model's; ties to the higher acceptance, then the lower index; if none
   reaches it, the highest-acceptance two-rung ladder (ties to the cheaper).
2. The menu (§3.2, P6 (a)): direct dispatch to each model, then that cascade. `menu="rule_a"`
   replaces it by P0.6's full candidate list (8 direct + every 2- and 3-rung ladder), used only for
   the reduction test against per-cell Rule A.
3. Per group (the training tasks' true S1.1 family, which equals the cell in every development
   population; P3 (a)): training acceptance and mean cost of each candidate
   (`common.policy_outcomes`, which matches replay), the acceptance shrunk toward the pool by
   `core.lookup` with one shared kappa (P4 (a)), and Rule A on the shrunk values with margin
   `margin` (P5 (a)).
4. The pool row: the pool-wide acceptance and mean cost, for a task whose predicted group has no
   training row or that has no text (§3.1, R11).

At decision time the router reads the scored task's group from its `content_text` through the S2.5
predictor refit on the training folds (`groups="pred"`, P2 (a)). `groups="true"` reads the true
family from the handle: a diagnostic, labelled given more information, never a router version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from router.baselines.cells import fit_predictor
from router.baselines.common import (ACCEPT, ESCALATE, MODES, OUR_DEFINITION, Context, PlanRouter,
                                     Policy, best_column, candidates, direct, ladders,
                                     policy_outcomes, success, true_cells)
from router.baselines.lookup import TRUE_CELL_INFO
from router.baselines.static import best_single_model
from router.core import lookup as L
from router.data.schema import OutcomeMatrix

DEFAULT_MARGIN = 0.02      # S4 spec §3.2, fixed before any run (P0.5's cost-aware DimensionBest)


def two_rung_cascade(train: OutcomeMatrix, ctx: Context, mode: str) -> tuple[Policy, bool]:
    """The fixed cascade of S4 spec §3.4, and whether it reached the best single model."""
    pols = [p for p in ladders(train, mode) if len(p.rungs) == 2]
    A, C = policy_outcomes(train, ctx.verifier, pols, verifier_usd=ctx.verifier_usd)
    v, c = A.mean(0), C.mean(0)
    ref = float(success(train)[best_single_model(train)].mean())
    idx = np.arange(len(pols))
    ok = idx[v >= ref - L.TOL]
    if len(ok):
        return pols[int(ok[np.lexsort((ok, -v[ok], c[ok]))[0]])], True
    return pols[best_column(v, c)], False


@dataclass
class RouterV1:
    mode: str = ESCALATE
    margin: float = DEFAULT_MARGIN
    shrinkage: str = "anova"       # "anova" (P4 (a)) or "none" (kappa = 0)
    groups: str = "pred"           # "pred" (router) or "true" (diagnostic, given more information)
    menu: str = "v1"               # "v1" (models + one two-rung cascade) or "rule_a" (P0.6's list)
    learned: bool = True

    def __post_init__(self):
        if self.mode not in MODES or self.shrinkage not in ("anova", "none") \
                or self.groups not in ("pred", "true") or self.menu not in ("v1", "rule_a"):
            raise ValueError(self)

    @property
    def name(self) -> str:
        base = "router_v1" if self.groups == "pred" else "v1_true_groups"
        if self.mode == ACCEPT:
            base += "_accept"
        if self.shrinkage != "anova":
            base += "_noshrink"
        if self.margin != DEFAULT_MARGIN:
            base += f"_m{self.margin:g}"
        if self.menu != "v1":
            base += f"_menu_{self.menu}"
        return base

    @property
    def more_information(self) -> tuple:
        return TRUE_CELL_INFO if self.groups == "true" else ()

    @property
    def definition(self) -> str:
        return (f"{OUR_DEFINITION}: S4 spec §3; groups = {self.groups} S1.1 family, shrinkage = "
                f"{self.shrinkage}, margin = {self.margin:g}, no-signal cells {self.mode}d")

    def fit(self, train: OutcomeMatrix, ctx: Context):
        cascade, reached = two_rung_cascade(train, ctx, self.mode)
        if self.menu == "v1":
            pols = [direct(x) for x in train.models] + [cascade]
        else:
            pols = candidates(train, self.mode)
        n_direct = len(train.models)
        A, C = policy_outcomes(train, ctx.verifier, pols, verifier_usd=ctx.verifier_usd)
        groups = true_cells(train).to_numpy()
        names = sorted(set(groups))
        S, n = L.group_counts(groups, A, names)
        Csum, _ = L.group_counts(groups, C, names)
        kappa = 0.0 if self.shrinkage == "none" else L.kappa_from_rho(L.anova_rho(S, n))
        a_hat, p = L.shrink(S, n, kappa)
        cost = Csum / np.maximum(n, 1)[:, None]
        pool_cost = C.mean(0)
        table = {g: pols[L.choose(a_hat[i], cost[i], n_direct, self.margin)]
                 for i, g in enumerate(names)}
        pool = pols[L.choose(p, pool_cost, n_direct, self.margin)]
        if self.groups == "true":
            read = lambda h: f"{h.info['dimension']}|{h.info['family']}"   # noqa: E731
        else:
            pred = fit_predictor(true_cells(train), ctx)
            read = lambda h: pred.predict_one(h.content_text)               # noqa: E731
        router = PlanRouter(lambda h: table.get(read(h), pool), self.more_information)
        router.table = {g: pol.name for g, pol in table.items()}
        router.fallback = pool.name
        router.chosen = {"kappa": None if math.isinf(kappa) else round(float(kappa), 6),
                         "kappa_inf": math.isinf(kappa), "cascade": cascade.name,
                         "cascade_reached_best_single": reached,
                         "n_train": int(n.sum()),
                         "group_n": {g: int(x) for g, x in zip(names, n)}}
        return router


__all__ = ["RouterV1", "two_rung_cascade", "DEFAULT_MARGIN"]
