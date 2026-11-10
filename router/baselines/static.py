"""Static baselines: each returns a model choice per task (a pd.Series indexed by task)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.data.schema import OutcomeMatrix


def always(m: OutcomeMatrix, model: str) -> pd.Series:
    return pd.Series(model, index=m.tasks)


def random_choice(m: OutcomeMatrix, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.choice(m.models, size=len(m)), index=m.tasks)


def best_single(train: OutcomeMatrix) -> str:
    return train.score.mean().idxmax()


def group_best(train: OutcomeMatrix, test: OutcomeMatrix, key: str = "dimension") -> pd.Series:
    """Best model per group on the training split (DimensionBest when key='dimension')."""
    g_train = getattr(train, key)
    g_test = getattr(test, key)
    table = train.score.groupby(g_train).mean().idxmax(axis=1)
    fallback = best_single(train)
    return pd.Series([table.get(g, fallback) for g in g_test], index=test.tasks)


def oracle(m: OutcomeMatrix, lam: float = 0.0) -> pd.Series:
    return (m.score - lam * m.cost).idxmax(axis=1)
