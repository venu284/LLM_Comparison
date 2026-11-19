"""Fixed cheap-to-strong cascade, escalating only on the simulated verifier (Rule R1)."""

from __future__ import annotations

import pandas as pd

from router.data.schema import OutcomeMatrix
from router.sim.metrics import Outcome
from router.sim.verifier import Verifier


def run(m: OutcomeMatrix, ladder: list[str], verifier: Verifier) -> Outcome:
    scores, costs = [], []
    s = m.score[ladder].to_numpy()
    c = m.cost[ladder].to_numpy()
    for i in range(len(m)):
        spent = 0.0
        for j in range(len(ladder)):
            spent += c[i, j]
            if j == len(ladder) - 1 or verifier.passes(s[i, j]):
                scores.append(s[i, j])
                break
        costs.append(spent)
    return Outcome(pd.Series(scores, m.tasks), pd.Series(costs, m.tasks))
