"""Online baselines over task metadata features (dimension and source one-hot).

The ID tasks of CodeRouterBench ship without prompts, so these baselines use the metadata the
release provides. The paper's own notes describe its online kNN as using "task metadata/hash
features". Reward is score - 0.1 * USD, as in the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.data.schema import OutcomeMatrix
from router.sim.metrics import LAMBDA


def metadata_features(m: OutcomeMatrix, vocab: list[str] | None = None) -> tuple[dict[str, np.ndarray], list[str]]:
    keys = ("dim=" + m.dimension.astype(str)), ("src=" + m.source.astype(str))
    if vocab is None:
        vocab = sorted(set(keys[0]) | set(keys[1]))
    pos = {k: i for i, k in enumerate(vocab)}
    feats = {}
    for t, a, b in zip(m.tasks, keys[0], keys[1]):
        x = np.zeros(len(vocab) + 1)
        x[-1] = 1.0
        for k in (a, b):
            if k in pos:
                x[pos[k]] = 1.0
        feats[t] = x
    return feats, vocab


class LinUCB:
    """Disjoint LinUCB, one ridge model per arm, warm-started from probing rewards."""

    def __init__(self, models: list[str], features: dict[str, np.ndarray], alpha: float = 0.5):
        d = len(next(iter(features.values())))
        self.A = {mdl: np.eye(d) for mdl in models}
        self.b = {mdl: np.zeros(d) for mdl in models}
        self.x = features
        self.alpha = alpha

    def warm_start(self, train: OutcomeMatrix, features: dict[str, np.ndarray]) -> None:
        for mdl in self.A:
            if mdl not in train.models:
                continue
            X = np.stack([features[t] for t in train.tasks])
            r = (train.score[mdl] - LAMBDA * train.cost[mdl]).to_numpy()
            self.A[mdl] += X.T @ X
            self.b[mdl] += X.T @ r

    def choose(self, task: str, available: list[str]) -> str:
        x = self.x[task]
        best, best_v = None, -np.inf
        for mdl in available:
            Ainv = np.linalg.inv(self.A[mdl])
            v = x @ Ainv @ self.b[mdl] + self.alpha * np.sqrt(x @ Ainv @ x)
            if v > best_v:
                best, best_v = mdl, v
        return best

    def observe(self, task: str, model: str, score: float, cost: float) -> None:
        x = self.x[task]
        self.A[model] += np.outer(x, x)
        self.b[model] += (score - LAMBDA * cost) * x


class KNN:
    """Retrieve the k most similar probing tasks; pick the model with the best mean reward."""

    def __init__(self, train: OutcomeMatrix, train_feats: dict[str, np.ndarray],
                 test_feats: dict[str, np.ndarray], k: int = 10, seed: int = 0):
        self.X = np.stack([train_feats[t] for t in train.tasks])
        self.R = (train.score - LAMBDA * train.cost).to_numpy()
        self.models = train.models
        self.test = test_feats
        self.k = k
        self.rng = np.random.default_rng(seed)

    def choose_all(self, tasks) -> pd.Series:
        out = {}
        Xn = self.X / np.linalg.norm(self.X, axis=1, keepdims=True)
        for t in tasks:
            q = self.test[t] / np.linalg.norm(self.test[t])
            sim = Xn @ q
            # random tie-break among equally similar neighbours
            jitter = self.rng.random(len(sim)) * 1e-9
            nn = np.argsort(-(sim + jitter))[: self.k]
            out[t] = self.models[int(self.R[nn].mean(axis=0).argmax())]
        return pd.Series(out)
