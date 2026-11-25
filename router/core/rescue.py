"""S4.5: the rescue classifier, P(a later rung of the ladder passes | the rungs tried so far did not
end the task) (plan §2 item 5, S4.5; `docs/research/S4_spec.md` §6.3; S3 spec D8).

- A decision point is a rung j whose verdict did not end the full ladder (FAIL, or UNKNOWN under
  escalate) with a later rung available. Target y = 1 when any later available rung succeeds (S0.1,
  training folds only).
- Features, all observable when the decision is made (S0.13): the predicted group, one token per
  rung tried (`model:verdict`) and the task's `content_text` as TF-IDF (as `cells.make_classifier`
  and D8). Logistic regression, C = 10 (as D8). No released score, family or dimension.
- Calibration: Platt scaling (a logistic regression of y on the logit of the raw prediction), fitted
  on out-of-fold predictions inside the training folds (each inner fold predicted by a classifier
  fitted on the other training folds, as D8). The final classifier is fitted on all training rows.
- t: the largest t whose acceptance loss against the full ladder, on the calibrated out-of-fold
  predictions, is at most `MAX_LOSS` points (open point P11; D8's value). Give up when P < t.

Differences from D8: Platt calibration; `model:verdict` tokens instead of the rung position, so one
classifier serves any ladder; a general ladder; deterministic inner folds, so stream 22 is unused.
Nothing here fits an item parameter (`guards.NOT_ITEM_FITS`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from router.sim.verifier import PASS, UNKNOWN

MAX_LOSS = 2.0          # points of acceptance (P11): D8's value
C_RESCUE = 10.0         # as D8
C_PLATT = 1e6           # Platt scaling: effectively unregularised
P_CLIP = 1e-6
CLF_SEED = 0


@dataclass
class Walk:
    """The full ladder's walk on one task. `models` are the ladder's available rungs in order;
    `verdicts` and `step_cost` cover the rungs reached; `ok` is the S0.1 success of every available
    rung, reached or not; `stop_at` is the rung where the full ladder ends (-1: none available)."""
    models: list
    verdicts: list
    step_cost: list
    ok: list
    stop_at: int

    @classmethod
    def walk(cls, models: Sequence[str], verdict: dict, cost: dict, ok: dict,
             on_unknown_accept: bool = False) -> "Walk":
        """Walk an escalate (or accept) ladder: stop at the first PASS (or UNKNOWN in accept mode),
        else at the last rung."""
        w = cls(list(models), [], [], [bool(ok[m]) for m in models], -1)
        for i, m in enumerate(models):
            w.verdicts.append(verdict[m])
            w.step_cost.append(float(cost[m]))
            ends = verdict[m] == PASS or (on_unknown_accept and verdict[m] == UNKNOWN)
            if ends or i == len(models) - 1:
                w.stop_at = i
                break
        return w

    def full(self) -> tuple[bool, float]:
        """(accepted, cost) of the full ladder: the output of the rung it ends on is accepted."""
        return (self.ok[self.stop_at] if self.stop_at >= 0 else False), float(sum(self.step_cost))

    def stopped_at(self, j: int) -> tuple[bool, float]:
        """(accepted, cost) when the policy gives up after rung j."""
        return False, float(sum(self.step_cost[:j + 1]))


def decision_rows(walks: dict) -> pd.DataFrame:
    """One row per decision point: task, j, the history tokens, y."""
    rows = []
    for t, w in walks.items():
        for j in range(max(w.stop_at, 0)):
            hist = " ".join(f"{w.models[i]}:{w.verdicts[i]}" for i in range(j + 1))
            rows.append({"task_id": t, "j": j, "history": hist, "y": int(any(w.ok[j + 1:]))})
    return pd.DataFrame(rows, columns=["task_id", "j", "history", "y"])


def outcomes(walks: dict, stop: dict) -> tuple[float, float]:
    """(acceptance in points, total cost) when each task gives up at its first decision point with
    stop[(task, j)] True."""
    acc, cost = [], 0.0
    for t, w in walks.items():
        a, c = w.full()
        for j in range(max(w.stop_at, 0)):
            if stop.get((t, j), False):
                a, c = w.stopped_at(j)
                break
        acc.append(a)
        cost += c
    return 100.0 * float(np.mean(acc)) if acc else 0.0, cost


def choose_t(walks: dict, rows: pd.DataFrame, p: np.ndarray, max_loss: float = MAX_LOSS) -> float:
    """The largest t whose acceptance loss against the full ladder on these tasks is at most
    `max_loss` points (give up when p < t). Candidates: 0, every predicted value, and just above 1."""
    keys = list(zip(rows.task_id, rows.j))
    full, _ = outcomes(walks, {})
    best = 0.0
    for t in sorted({0.0, *map(float, p), 1.0 + 1e-9}):
        acc, _ = outcomes(walks, {k: bool(q < t) for k, q in zip(keys, p)})
        if full - acc <= max_loss + 1e-12:
            best = max(best, t)
    return best


class RescueClassifier:
    """Raw P(y = 1) from the predicted group, the `model:verdict` history and the task text."""

    def fit(self, rows: pd.DataFrame, texts: pd.Series, groups: pd.Series) -> "RescueClassifier":
        from scipy.sparse import hstack
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_union
        self._hstack = hstack
        y = rows.y.to_numpy()
        self.const = None
        if len(rows) == 0 or len(set(y)) < 2:
            self.const = float(y.mean()) if len(rows) else 0.5
            return self
        self.text = make_union(
            TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True,
                            max_features=200_000))
        self.meta = DictVectorizer()
        X = self._hstack([self.text.fit_transform(self._texts(rows, texts)),
                          self.meta.fit_transform(self._meta(rows, groups))]).tocsr()
        self.clf = LogisticRegression(C=C_RESCUE, max_iter=5000, random_state=CLF_SEED).fit(X, y)
        return self

    @staticmethod
    def _texts(rows, texts):
        return [x if isinstance(x, str) else "" for x in texts.reindex(rows.task_id)]

    @staticmethod
    def _meta(rows, groups):
        out = []
        for r in rows.itertuples():
            d = {f"group={groups.get(r.task_id)}": 1.0}
            for tok in r.history.split():
                d[tok] = 1.0
            out.append(d)
        return out

    def predict(self, rows: pd.DataFrame, texts: pd.Series, groups: pd.Series) -> np.ndarray:
        if len(rows) == 0:
            return np.zeros(0)
        if self.const is not None:
            return np.full(len(rows), self.const)
        X = self._hstack([self.text.transform(self._texts(rows, texts)),
                          self.meta.transform(self._meta(rows, groups))]).tocsr()
        return self.clf.predict_proba(X)[:, 1]


@dataclass
class Platt:
    """P = sigmoid(a * logit(p_raw) + c), fitted on held-out raw predictions."""
    a: float = 1.0
    c: float = 0.0

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "Platt":
        from sklearn.linear_model import LogisticRegression
        y = np.asarray(y, dtype=int)
        if len(set(y)) < 2:
            self.a, self.c = 0.0, float(logit(np.clip(y.mean() if len(y) else 0.5, P_CLIP, 1 - P_CLIP)))
            return self
        x = logit(np.clip(np.asarray(p_raw, dtype=float), P_CLIP, 1 - P_CLIP))[:, None]
        lr = LogisticRegression(C=C_PLATT, max_iter=5000).fit(x, y)
        self.a, self.c = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def __call__(self, p_raw) -> np.ndarray:
        x = logit(np.clip(np.asarray(p_raw, dtype=float), P_CLIP, 1 - P_CLIP))
        return expit(self.a * x + self.c)


@dataclass
class Rescue:
    clf: RescueClassifier
    platt: Platt
    t: float
    oof: pd.Series = field(repr=False, default=None)

    def predict(self, rows, texts, groups) -> np.ndarray:
        return self.platt(self.clf.predict(rows, texts, groups))

    def give_up(self, rows, texts, groups) -> np.ndarray:
        return self.predict(rows, texts, groups) < self.t


def fit_rescue(walks: dict, inner_fold: pd.Series, texts: pd.Series, groups: pd.Series,
               max_loss: float = MAX_LOSS) -> Rescue:
    """The rescue classifier on training walks: out-of-fold raw predictions over `inner_fold`
    (task -> fold, problem-grouped), Platt on them, t on the calibrated out-of-fold predictions,
    and the final classifier on every training row."""
    rows = decision_rows(walks)
    oof = pd.Series(np.nan, index=rows.index)
    fold_of_row = inner_fold.reindex(rows.task_id).to_numpy()
    for f in sorted(set(inner_fold.reindex(list(walks)).dropna())):
        tr, te = rows[fold_of_row != f], rows[fold_of_row == f]
        if len(te):
            oof.loc[te.index] = RescueClassifier().fit(tr, texts, groups).predict(te, texts, groups)
    platt = Platt().fit(oof.to_numpy(), rows.y.to_numpy())
    t = choose_t(walks, rows, platt(oof.to_numpy()), max_loss)
    return Rescue(RescueClassifier().fit(rows, texts, groups), platt, t, oof)


__all__ = ["MAX_LOSS", "Platt", "Rescue", "RescueClassifier", "Walk", "choose_t", "decision_rows",
           "fit_rescue", "outcomes"]
