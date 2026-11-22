"""LinUCB and kNN baselines on the task text (S2.5; our definitions).

Both read only router-visible inputs at routing time: the scored task's `content_text` from its
handle and, for LinUCB, the verdicts of its own calls. Neither sees a scored task's score or
realised cost (S0.13). The reward is the paper's, acceptance minus `LAMBDA` x USD (2606.22902:
score - 0.1 x USD), with acceptance under S0.1.

- **LinUCB** (stated features): disjoint LinUCB, one ridge model per model, on `TextFeatures`:
  TF-IDF of word 1-2 grams fitted on the training texts, projected onto the top `n_components`
  right singular vectors (a deterministic SVD), plus a bias term. Warm-started on the training
  folds' outcomes (acceptance and realised cost, O1). Online it makes verified calls (verifier cost
  is charged) and updates on PASS (1) or FAIL (0) minus `LAMBDA` x the model's training mean cost;
  an UNKNOWN gives no update.
- **kNN**: the `k` training tasks nearest in prompt-embedding space (`baselines/embeddings.py`,
  cosine similarity), then the model with the best mean training reward over them (ties to the
  cheaper training mean cost). Direct dispatch. Joined tasks only: a task with no text goes to the
  training best single model, and the count is kept.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from router.baselines.common import OUR_DEFINITION, Context, success
from router.baselines.static import best_single_model
from router.data.schema import OutcomeMatrix
from router.sim.stream import GIVE_UP, Accept, Call
from router.sim.verifier import FAIL, PASS

LAMBDA = 0.1      # USD weight of the reward (2606.22902)


def _train_texts(train: OutcomeMatrix, ctx: Context) -> pd.Series:
    if ctx.content is None:
        raise ValueError("text baselines need Context.content")
    return ctx.content.reindex(train.tasks)


class TextFeatures:
    def __init__(self, texts, n_components: int = 16):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                    sublinear_tf=True)
        X = self._vec.fit_transform(list(texts)).toarray()
        _, s, vt = np.linalg.svd(X, full_matrices=False)
        d = int(min(n_components, (s > 1e-10).sum()))
        self._P = vt[:d].T
        self.dim = d + 1

    def __call__(self, texts) -> np.ndarray:
        Z = self._vec.transform(list(texts)).toarray() @ self._P
        return np.hstack([Z, np.ones((len(Z), 1))])


class _LinUCBRouter:
    more_information = ()

    def __init__(self, models, feats, A, b, alpha, unit_cost, fallback):
        self.models, self.feats, self.A, self.b = models, feats, A, b
        self.alpha, self.unit_cost, self.fallback = alpha, unit_cost, fallback
        self._x: dict[str, np.ndarray] = {}
        self.n_no_text = 0

    def decide(self, task, available, history):
        if history:
            return Accept()
        if not available:
            return GIVE_UP
        if task.content_text is None:
            self.n_no_text += 1
            return self.fallback if self.fallback in available else available[0]
        x = self.feats([task.content_text])[0]
        self._x[task.key] = x
        best, best_v = None, -np.inf
        for m in available:
            Ainv = np.linalg.inv(self.A[m])
            v = x @ Ainv @ self.b[m] + self.alpha * np.sqrt(x @ Ainv @ x)
            if v > best_v:
                best, best_v = m, v
        return best

    def feedback(self, obs) -> None:
        if obs.verdict not in (PASS, FAIL) or obs.task not in self._x:
            return
        x = self._x[obs.task]
        r = float(obs.verdict == PASS) - LAMBDA * self.unit_cost[obs.model]
        self.A[obs.model] += np.outer(x, x)
        self.b[obs.model] += r * x


@dataclass
class LinUCB:
    alpha: float = 0.5
    n_components: int = 16
    name: str = "linucb"
    learned: bool = True
    more_information: tuple = ()
    definition: str = (f"{OUR_DEFINITION}: TF-IDF-SVD features of content_text, warm-started on "
                       "the training folds, online updates from verdicts only")

    def fit(self, train: OutcomeMatrix, ctx: Context):
        texts = _train_texts(train, ctx)
        keep = texts.notna().to_numpy()
        feats = TextFeatures(texts[keep], self.n_components)
        X = feats(texts[keep])
        ok = success(train).to_numpy(float)[keep]
        cost = train.cost.to_numpy(float)[keep]
        av = train.available.astype(bool).to_numpy()[keep]
        A, b = {}, {}
        for j, m in enumerate(train.models):
            rows = av[:, j]
            Xm = X[rows]
            r = ok[rows, j] - LAMBDA * cost[rows, j]
            A[m] = np.eye(feats.dim) + Xm.T @ Xm
            b[m] = Xm.T @ r
        unit = train.cost.where(train.available.astype(bool)).mean().to_dict()
        return _LinUCBRouter(list(train.models), feats, A, b, self.alpha, unit,
                             best_single_model(train))


class _KNNRouter:
    more_information = ()

    def __init__(self, models, E, U, unit_cost, k, embed, fallback):
        self.models, self.E, self.U, self.k, self.embed = models, E, U, k, embed
        self.unit = np.array([unit_cost[m] for m in models])
        self.fallback = fallback
        self.n_no_text = 0

    def decide(self, task, available, history):
        if history:
            return Accept()
        if not available:
            return GIVE_UP
        if task.content_text is None:
            self.n_no_text += 1
            m = self.fallback if self.fallback in available else available[0]
            return Call(m, verify=False)
        q = self.embed([task.content_text])[0]
        nn = np.argsort(-(self.E @ q), kind="stable")[: self.k]
        with np.errstate(all="ignore"):
            util = np.nanmean(self.U[nn], axis=0)
        cols = [j for j, m in enumerate(self.models) if m in available and not np.isnan(util[j])]
        if not cols:
            m = self.fallback if self.fallback in available else available[0]
            return Call(m, verify=False)
        cols = np.array(cols)
        j = cols[np.lexsort((cols, self.unit[cols], -util[cols]))[0]]
        return Call(self.models[j], verify=False)

    def feedback(self, obs) -> None:
        pass


@dataclass
class KNN:
    k: int = 10
    name: str = "knn"
    learned: bool = True
    more_information: tuple = ()
    definition: str = (f"{OUR_DEFINITION}: all-MiniLM-L6-v2 embeddings of content_text, k = 10, "
                       "best mean training reward (acceptance - 0.1 x USD)")

    def fit(self, train: OutcomeMatrix, ctx: Context):
        if ctx.embed is None:
            raise ValueError("kNN needs Context.embed")
        texts = _train_texts(train, ctx)
        keep = texts.notna().to_numpy()
        E = ctx.embed(list(texts[keep]))
        ok = success(train).to_numpy(float)[keep]
        cost = train.cost.to_numpy(float)[keep]
        av = train.available.astype(bool).to_numpy()[keep]
        U = np.where(av, ok - LAMBDA * cost, np.nan)
        unit = train.cost.where(train.available.astype(bool)).mean().to_dict()
        return _KNNRouter(list(train.models), E, U, unit, self.k, ctx.embed,
                          best_single_model(train))
