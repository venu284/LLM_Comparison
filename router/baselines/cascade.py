"""Fixed cheap-to-strong cascade, escalating only on the simulated verifier (Rule R1)."""

from __future__ import annotations

import pandas as pd

from router.data.schema import OutcomeMatrix
from router.experiments.m2_quickcheck_sim import Outcome  # quick-check; checkpoint B rebuilds on router.sim
from router.experiments.m2_quickcheck_sim import Verifier  # quick-check; checkpoint B rebuilds on router.sim


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
