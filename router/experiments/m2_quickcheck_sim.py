"""FROZEN RECORD: the 2026-11-10 quick-check simulator (router/sim/ before S2).

Moved here verbatim at S2 checkpoint A so `m2_reproduce.py` and the quick-check baselines keep
their 2026-11-10 numbers, while `router/sim/` is rebuilt to S0.1, S0.11 and S0.13. Do not use it
in new code: it accepts at the legacy threshold (score >= 0.5, contradicts S0.1), passes the
released score to the router (contradicts S0.13) and draws from bare `default_rng(seed)`.
Sections, in order: metrics.py, verifier.py, stream.py (only the import lines are merged).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from router.data.schema import OutcomeMatrix
from router.data.scoring import LEGACY_ACCEPT_THRESHOLD

# --------------------------------------------------------------------------- metrics.py
LAMBDA = 0.1


@dataclass
class Outcome:
    """Per-task result of a routing policy: the final score kept and the cost spent."""
    score: pd.Series
    cost: pd.Series


def from_choice(score: pd.DataFrame, cost: pd.DataFrame, choice: pd.Series) -> Outcome:
    idx = score.index
    choice = choice.reindex(idx)
    rows = np.arange(len(idx))
    cols = score.columns.get_indexer(choice.values)
    if (cols < 0).any():
        raise ValueError("choice names a model not in the matrix")
    return Outcome(pd.Series(score.values[rows, cols], idx), pd.Series(cost.values[rows, cols], idx))


def summary(out: Outcome, score: pd.DataFrame, cost: pd.DataFrame, floor: float | None = None) -> dict:
    perf = out.score.mean() * 100
    total = out.cost.sum()
    oracle_reward = (score - LAMBDA * cost).max(axis=1)
    reward = out.score - LAMBDA * out.cost
    accepted = (out.score >= LEGACY_ACCEPT_THRESHOLD).sum()
    res = {
        "AvgPerf": perf,
        "$Total": total,
        "Perf/$": perf / total if total > 0 else np.inf,
        "CumReg": float((oracle_reward - reward).sum()),
        "AcceptRate": accepted / len(out.score) * 100,
        "CPAT": total / accepted if accepted else np.inf,
    }
    if floor is not None:
        res["MeetsFloor"] = res["AcceptRate"] >= floor
    return res


def bootstrap_ci(values: pd.Series, n: int = 2000, seed: int = 0, stat=np.mean) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    draws = [stat(v[rng.integers(0, len(v), len(v))]) for _ in range(n)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# --------------------------------------------------------------------------- verifier.py
class Verifier:
    def __init__(self, eps: float = 0.0, seed: int = 0, threshold: float = LEGACY_ACCEPT_THRESHOLD):
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must be in [0, 1]")
        self.eps = eps
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

    def passes(self, true_score: float) -> bool:
        truth = true_score >= self.threshold
        flip = self.rng.random() < self.eps
        return truth != flip


# --------------------------------------------------------------------------- stream.py
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
