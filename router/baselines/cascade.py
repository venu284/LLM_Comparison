"""The fixed cascade (S2.5, R2 item 7; our definition).

One cheap-to-strong ladder for every task, chosen on the training folds among P0.5's fixed ladders
(2- and 3-rung, non-decreasing training mean cost) in one mode (no-signal cells accepted or
escalated): the cheapest ladder whose training acceptance is at least the training best single
model's; ties to higher acceptance, then the lower index. If no ladder reaches it, the ladder with
the highest training acceptance (ties to the cheaper). The rule mirrors P0.6's Rule A with one
global cell and the best single model as the reference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from router.baselines.common import (OUR_DEFINITION, TOL, Context, PlanRouter, best_column,
                                     ladders, policy_outcomes, success)
from router.baselines.static import best_single_model
from router.data.schema import OutcomeMatrix


@dataclass
class FixedCascade:
    mode: str
    learned: bool = True
    more_information: tuple = ()
    definition: str = (f"{OUR_DEFINITION}: the cheapest P0.5 ladder whose training acceptance is "
                       "at least the training best single model's")

    @property
    def name(self) -> str:
        return f"fixed_cascade_{self.mode}"

    def fit(self, train: OutcomeMatrix, ctx: Context):
        pols = ladders(train, self.mode)
        A, C = policy_outcomes(train, ctx.verifier, pols)
        v, c = A.mean(0), C.mean(0)
        ref = float(success(train)[best_single_model(train)].mean())
        idx = np.arange(len(pols))
        ok = idx[v >= ref - TOL]
        j = int(ok[np.lexsort((ok, -v[ok], c[ok]))[0]]) if len(ok) else best_column(v, c)
        chosen = pols[j]
        router = PlanRouter(lambda h: chosen)
        router.table = {"all": chosen.name}
        router.reached_reference = bool(len(ok))
        return router
