"""The S2.5 cell predictor: dimension x family cell from the task text (S0.3(c), O3).

CellBest with predicted cells, the S0.3(c) comparator, routes on the cell this predictor reads
from `content_text`, the router's field (S1.5). The model is P0.6's, re-implemented from its stated
settings (`p0_6_cellbest_bound.make_classifier`; fixed in advance, not tuned): TF-IDF on word 1-2
grams and character 2-5 grams, then logistic regression (C = 10, random_state 0).

Training population (user decision 2026-11-21): the headline's development split. Training on all
eligible development tasks is a sensitivity check. The population is a parameter (`Context.
cell_pool`), so the evaluation run trains it once on the development split and freezes it with the
candidate list (O3). Inside development-split folds it is trained on the training folds only.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

import numpy as np
import pandas as pd

CLF_SEED = 0          # P0.6's random_state


def make_classifier():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline, make_union
    feats = make_union(
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                        max_features=200_000))
    return make_pipeline(feats, LogisticRegression(C=10.0, max_iter=5000, random_state=CLF_SEED))


class CellPredictor:
    """Fitted on (text, cell) pairs. With one cell in training it always predicts that cell."""

    def __init__(self, texts: Sequence[str], cells: Sequence[str]):
        texts, cells = list(texts), list(cells)
        if not texts:
            raise ValueError("no training texts")
        if any(t is None or (isinstance(t, float) and np.isnan(t)) for t in texts):
            raise ValueError("every training task needs a text")
        self.cells = sorted(set(cells))
        self.n_train = len(texts)
        self._clf = None if len(self.cells) == 1 else make_classifier().fit(texts, cells)
        self._cache: dict[str, str] = {}

    def predict(self, texts: Sequence[str]) -> np.ndarray:
        if self._clf is None:
            return np.array([self.cells[0]] * len(texts), dtype=object)
        return self._clf.predict(list(texts))

    def predict_one(self, text: str | None) -> str | None:
        if text is None:
            return None
        if text not in self._cache:
            self._cache[text] = str(self.predict([text])[0])
        return self._cache[text]


def training_pool(train_cells: pd.Series, ctx) -> pd.Series:
    """The predictor's (task -> cell) training set for one fit: `ctx.cell_pool`, or the training
    matrix's own cells, minus every task of a scored-fold problem."""
    pool = train_cells if ctx.cell_pool is None else ctx.cell_pool
    if ctx.exclude_problems:
        if ctx.problem_of is None:
            raise ValueError("exclude_problems needs problem_of")
        prob = ctx.problem_of.reindex(pool.index)
        if prob.isna().any():
            raise KeyError("a pool task has no problem id")
        pool = pool[~prob.isin(ctx.exclude_problems)]
    return pool


def fit_predictor(train_cells: pd.Series, ctx) -> CellPredictor:
    pool = training_pool(train_cells, ctx)
    if ctx.content is None:
        raise ValueError("the cell predictor needs task texts (Context.content)")
    texts = ctx.content.reindex(pool.index)
    keep = texts.notna()
    key = hashlib.sha256(repr((list(pool.index[keep]), list(pool[keep]))).encode()).hexdigest()
    if key not in _FITTED:          # the same training set is refitted by several baselines
        if len(_FITTED) > 64:
            _FITTED.clear()
        _FITTED[key] = CellPredictor(texts[keep].tolist(), pool[keep].tolist())
    return _FITTED[key]


_FITTED: dict[str, CellPredictor] = {}


def out_of_fold_accuracy(texts: pd.Series, cells: pd.Series, folds: pd.Series,
                         pool_texts: pd.Series | None = None, pool_cells: pd.Series | None = None,
                         problem_of: pd.Series | None = None) -> dict:
    """Held-out accuracy on the tasks of `cells`: for each fold, fit on the pool's tasks whose
    problem is not in that fold (default pool: the other folds' tasks) and predict the fold."""
    pool_texts = texts if pool_texts is None else pool_texts
    pool_cells = cells if pool_cells is None else pool_cells
    pred = pd.Series(index=cells.index, dtype=object)
    for f in sorted(folds.unique()):
        held = folds.index[folds == f]
        if problem_of is not None:
            bad = set(problem_of.reindex(held))
            tr = pool_cells.index[~problem_of.reindex(pool_cells.index).isin(bad)]
        else:
            tr = folds.index[folds != f]
        tr = tr[pool_texts.reindex(tr).notna()]
        clf = CellPredictor(pool_texts.loc[tr].tolist(), pool_cells.loc[tr].tolist())
        pred.loc[held] = clf.predict(texts.loc[held].tolist())
    right = pred == cells
    return {"n": int(len(cells)), "accuracy": float(right.mean()), "n_wrong": int((~right).sum()),
            "per_cell": {c: {"tasks": int((cells == c).sum()),
                             "accuracy": float(right[cells == c].mean())}
                         for c in sorted(cells.unique())},
            "confusion": pd.crosstab(cells, pred).to_dict()}
