"""S4.4: task difficulty b_q predicted from the prompt (plan §2 item 4, S4.4 [M3.2];
`docs/research/S4_spec.md` §6.2; S4.6).

- Target: the training tasks' b_q from `ability.fit` on the training folds.
- Features: the pinned MiniLM embedding (`baselines.embeddings.embed`) of `content_text` only (S1.5;
  never the wrapped prompt, whose instruction wording identifies the family). No source, family,
  dimension or other label is a feature: the functions here take text and nothing else about a
  task. `embed` is injectable so unit tests can run without the model.
- Model: ridge regression with an intercept; alpha from `ALPHAS` by cross-validation grouped by
  problem inside the training data (`GroupKFold` over problem ids, `N_CV` folds or the number of
  problems if fewer; lowest mean squared error, ties to the larger alpha), then refitted on all
  training tasks. A task with no text has no prediction (nan): the caller falls back to the group.
- S4.6: `fit` and `grouped_cv` call `check_non_transductive` (training task ids against the scored
  ids) before fitting.
- `FamilyClassifier`: an optional family (group) classifier from the same embedding, a multinomial
  logistic regression. It predicts a label and fits no item parameter (`guards.NOT_ITEM_FITS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold

from router.core.guards import check_non_transductive

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
N_CV = 5
FAMILY_C = 1.0


def _default_embed(texts: Sequence[str]) -> np.ndarray:
    from router.baselines.embeddings import embed
    return embed(list(texts))


def _with_text(ids, texts: pd.Series) -> list:
    return [t for t in ids if isinstance(texts.get(t), str) and texts.get(t)]


def grouped_folds(problems: Sequence, n: int = N_CV) -> np.ndarray:
    """Fold index per task; no problem is split across folds."""
    problems = np.asarray(list(problems))
    k = min(n, len(set(problems)))
    if k < 2:
        raise ValueError("grouped cross-validation needs at least two problems")
    out = np.zeros(len(problems), dtype=int)
    for f, (_, te) in enumerate(GroupKFold(n_splits=k).split(problems, groups=problems)):
        out[te] = f
    return out


@dataclass
class DifficultyModel:
    alpha: float
    coef: np.ndarray
    intercept: float
    embed: Callable
    n_train: int

    def predict(self, texts: Sequence) -> np.ndarray:
        """b for each text; nan where there is no text."""
        texts = list(texts)
        ok = [i for i, x in enumerate(texts) if isinstance(x, str) and x]
        out = np.full(len(texts), np.nan)
        if ok:
            out[ok] = self.embed([texts[i] for i in ok]) @ self.coef + self.intercept
        return out


def _choose_alpha(X, y, problems, guard) -> float:
    check_non_transductive(*guard)
    folds = grouped_folds(problems)
    mse = []
    for a in ALPHAS:
        err = 0.0
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            pred = Ridge(alpha=a).fit(X[tr], y[tr]).predict(X[te])
            err += float(((pred - y[te]) ** 2).sum())
        mse.append(err / len(y))
    best = min(mse)
    return max(a for a, e in zip(ALPHAS, mse) if e <= best + 1e-12)


def fit(train_ids, b, texts: pd.Series, *, scored_ids, problem_of: pd.Series,
        embed: Callable | None = None) -> DifficultyModel:
    """Ridge from the embedding of `texts[t]` (content_text) to b[t] over the training tasks with
    text. `b` is indexed by task id."""
    embed = embed or _default_embed
    guard = (list(train_ids), scored_ids, problem_of)
    check_non_transductive(*guard)
    ids = _with_text(train_ids, texts)
    X = embed([texts[t] for t in ids])
    y = np.asarray(pd.Series(b).reindex(ids), dtype=float)
    alpha = _choose_alpha(X, y, problem_of.reindex(ids).tolist(), guard)
    r = Ridge(alpha=alpha).fit(X, y)
    return DifficultyModel(alpha, np.asarray(r.coef_, dtype=float), float(r.intercept_), embed,
                           len(ids))


def grouped_cv(train_ids, b, texts: pd.Series, *, scored_ids, problem_of: pd.Series,
               embed: Callable | None = None) -> dict:
    """Out-of-fold predictions of `fit` (folds grouped by problem), with R^2 and Spearman rho: the
    S5 diagnostic. Each fold's model chooses its own alpha inside its training part."""
    embed = embed or _default_embed
    check_non_transductive(list(train_ids), scored_ids, problem_of)
    ids = _with_text(train_ids, texts)
    folds = grouped_folds(problem_of.reindex(ids).tolist())
    y = pd.Series(b).reindex(ids).astype(float)
    pred = pd.Series(np.nan, index=ids)
    for f in np.unique(folds):
        tr = [t for t, k in zip(ids, folds) if k != f]
        te = [t for t, k in zip(ids, folds) if k == f]
        m = fit(tr, y, texts, scored_ids=list(scored_ids) + te, problem_of=problem_of, embed=embed)
        pred[te] = m.predict([texts[t] for t in te])
    ss = float(((y - y.mean()) ** 2).sum())
    return {"pred": pred, "folds": pd.Series(folds, index=ids),
            "r2": 1.0 - float(((y - pred) ** 2).sum()) / ss if ss > 0 else float("nan"),
            "spearman": float(spearmanr(y, pred)[0])}


class FamilyClassifier:
    """P(family | prompt) from the embedding of `content_text` (optional, plan S4.4)."""

    def __init__(self, embed: Callable | None = None, C: float = FAMILY_C):
        self.embed, self.C = embed or _default_embed, C

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "FamilyClassifier":
        self.clf = LogisticRegression(C=self.C, max_iter=5000).fit(self.embed(list(texts)),
                                                                     list(labels))
        return self

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        return self.clf.predict(self.embed(list(texts)))


__all__ = ["ALPHAS", "DifficultyModel", "FamilyClassifier", "fit", "grouped_cv", "grouped_folds"]
