"""Toy data for the sim tests and examples: a small OutcomeMatrix with partial credit, an
unavailable cell, and a matching X13-style verdict table. No external data is read."""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.data.schema import EMPTY_RESPONSE, OutcomeMatrix
from router.sim.verifier import FAIL, PASS, UNKNOWN

MODELS = ("cheap", "mid", "strong")
COSTS = {"cheap": 0.01, "mid": 0.05, "strong": 0.40}


def matrix(n: int = 24, seed: int = 7, name: str = "toy/complete") -> OutcomeMatrix:
    """`n` tasks, half cruxeval and half mbpp; scores in {0, 0.8, 1}; task t1 / mid unavailable."""
    g = np.random.default_rng(seed)
    tasks = pd.Index([f"t{i}" for i in range(n)])
    p = {"cheap": 0.35, "mid": 0.55, "strong": 0.8}
    score = pd.DataFrame({m: np.where(g.random(n) < p[m], 1.0, np.where(g.random(n) < 0.3, 0.8, 0.0))
                          for m in MODELS}, index=tasks)
    cost = pd.DataFrame({m: COSTS[m] * (1 + g.random(n)) for m in MODELS}, index=tasks)
    family = pd.Series(["cruxeval" if i % 2 == 0 else "mbpp" for i in range(n)], index=tasks)
    dimension = pd.Series(["dim1" if i % 2 == 0 else "dim2" for i in range(n)], index=tasks)
    available = pd.DataFrame(True, index=tasks, columns=list(MODELS))
    reason = pd.DataFrame(None, index=tasks, columns=list(MODELS), dtype=object)
    available.loc["t1", "mid"] = False
    reason.loc["t1", "mid"] = EMPTY_RESPONSE
    return OutcomeMatrix(score, cost, dimension, family, name, available=available,
                         unavailable_reason=reason)


def verdict_table(m: OutcomeMatrix, seed: int = 11, no_record: tuple[str, ...] = ("t3",)) -> pd.DataFrame:
    """X13-style rows for every cell of `m` except the tasks in `no_record`. Every third task has
    visible tests (signal); the rest are no-signal. Verdicts are drawn independently of the scores,
    so they stay fixed when the scores change. Unavailable cells are empty."""
    g = np.random.default_rng(seed)
    rows = []
    for i, t in enumerate(m.tasks):
        if t in no_record:
            continue
        for mdl in m.models:
            empty = not bool(m.available.at[t, mdl])
            signal = (i % 3 == 0) and not empty
            obs = None if empty else ((PASS if g.random() < 0.5 else FAIL) if signal else UNKNOWN)
            rows.append({"task_id": t, "model": mdl, "observation": obs, "signal": signal,
                         "unavailable": empty})
    return pd.DataFrame(rows)


def heuristic_table(v: pd.DataFrame, seed: int = 13) -> pd.DataFrame:
    g = np.random.default_rng(seed)
    return pd.DataFrame({"task_id": v.task_id, "model": v.model,
                         "heuristic_accept": g.random(len(v)) < 0.6})


def with_scores(m: OutcomeMatrix, score: pd.DataFrame) -> OutcomeMatrix:
    return OutcomeMatrix(score, m.cost.copy(), m.dimension.copy(), m.family.copy(), m.name,
                         available=m.available.copy(),
                         unavailable_reason=m.unavailable_reason.copy())
