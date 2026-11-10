"""Routing metrics on an OutcomeMatrix.

Definitions follow the ACRouter paper where it gives one:
- AvgPerf: mean score of the chosen model, in percent;
- $Total: summed USD cost of everything that was run;
- Perf/$: AvgPerf divided by $Total (matches the published Always-Opus 1.29);
- CumReg: summed per-task gap to the reward oracle, reward = score - 0.1 * USD.
CPAT (cost per accepted task) is only reported with an acceptance floor (Rule R3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from router.data.scoring import ACCEPT_THRESHOLD

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
    accepted = (out.score >= ACCEPT_THRESHOLD).sum()
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
