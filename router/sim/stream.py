"""Seeded task-stream replay for online routers.

A router sees tasks one at a time, picks a model from those available, and observes the
outcome of that one model only. Models can arrive mid-stream (leave-one-model-out onboarding).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import pandas as pd

from router.data.schema import OutcomeMatrix
from router.sim.metrics import Outcome


class Router(Protocol):
    def choose(self, task: str, available: list[str]) -> str: ...
    def observe(self, task: str, model: str, score: float, cost: float) -> None: ...


def replay(router: Router, m: OutcomeMatrix, seed: int,
           arrivals: dict[str, int] | None = None) -> tuple[Outcome, pd.Series]:
    """Run the stream in a seeded random order. Returns the outcome and the choices."""
    order = np.random.default_rng(seed).permutation(len(m))
    tasks = m.tasks[order]
    arrivals = arrivals or {}
    scores, costs, choices = {}, {}, {}
    for t_idx, task in enumerate(tasks):
        available = [x for x in m.models if arrivals.get(x, 0) <= t_idx]
        model = router.choose(task, available)
        s = float(m.score.at[task, model])
        c = float(m.cost.at[task, model])
        router.observe(task, model, s, c)
        scores[task], costs[task], choices[task] = s, c, model
    idx = m.tasks
    return (Outcome(pd.Series(scores).reindex(idx), pd.Series(costs).reindex(idx)),
            pd.Series(choices).reindex(idx))
