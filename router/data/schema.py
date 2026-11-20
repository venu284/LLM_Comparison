"""The outcome matrix every router component reads (S1.2).

Storage is long format: one row per (task, model, run) with the columns in `LONG_COLUMNS`. The
wide task x model views (score, cost, availability) every router component reads are derived from
it by `OutcomeMatrix.from_long`. The original wide constructor `OutcomeMatrix(score, cost,
dimension, source, name)` still works; such a matrix has every cell available and no long table.

Scores are in [0, 1]; costs are USD per task. `source` holds the S1.1 family (`family` is an
alias). A cell with `available == False` is missing, not a failure (CLAIMS S0.11): its score stays
as released, and `unavailable_reason` says why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LONG_COLUMNS = ["task_id", "model", "run", "score", "cost_usd", "input_tokens", "output_tokens",
                "problem_id", "family", "dimension", "split", "score_type", "eligible",
                "available", "unavailable_reason"]

MISSING_TOKEN_RECORD = "missing_token_record"   # CodeRouterBench cost_source == missing_token_record_zero_total
EMPTY_RESPONSE = "empty_response"               # empty X13 response (CLAIMS S0.11), supplied by the verdict loader
UNAVAILABLE_REASONS = (MISSING_TOKEN_RECORD, EMPTY_RESPONSE)

# Columns filled after loading: score_type and eligible by scoring.annotate (S1.4, done);
# problem_id by S1.6. Left null when absent.
DEFERRED_COLUMNS = ("problem_id", "score_type", "eligible")   # S1.6, S1.4, S1.4


def long_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a long table to `LONG_COLUMNS`: add the deferred columns as nulls, default
    `available` to True and `unavailable_reason` to None, and check the reasons."""
    df = df.copy()
    for col in DEFERRED_COLUMNS:
        if col not in df:
            df[col] = None
    df["eligible"] = df["eligible"].astype("boolean")
    if "available" not in df:
        df["available"] = True
    if "unavailable_reason" not in df:
        df["unavailable_reason"] = None
    df["unavailable_reason"] = df["unavailable_reason"].astype(object).where(
        df["unavailable_reason"].notna(), None)
    missing = [c for c in LONG_COLUMNS if c not in df]
    if missing:
        raise ValueError(f"long table lacks columns {missing}")
    df = df[LONG_COLUMNS]
    df["available"] = df["available"].astype(bool)
    _check_reasons(df["available"], df["unavailable_reason"])
    return df.reset_index(drop=True)


def _check_reasons(available, reason) -> None:
    has_reason = reason.notna()
    if (has_reason == available).any(axis=None):
        raise ValueError("a cell is unavailable exactly when it has an unavailable_reason")
    bad = set(reason.to_numpy()[has_reason.to_numpy(dtype=bool)]) - set(UNAVAILABLE_REASONS)
    if bad:
        raise ValueError(f"unknown unavailable_reason {sorted(bad)}")


@dataclass
class OutcomeMatrix:
    score: pd.DataFrame          # task x model
    cost: pd.DataFrame           # task x model, USD
    dimension: pd.Series         # task -> dimension
    source: pd.Series            # task -> S1.1 family
    name: str = ""
    available: pd.DataFrame | None = field(default=None, kw_only=True)            # task x model, bool
    unavailable_reason: pd.DataFrame | None = field(default=None, kw_only=True)   # task x model, str | None
    long: pd.DataFrame | None = field(default=None, kw_only=True)                 # LONG_COLUMNS rows

    def __post_init__(self) -> None:
        if not self.score.index.equals(self.cost.index):
            raise ValueError("score and cost must share the task index")
        if list(self.score.columns) != list(self.cost.columns):
            raise ValueError("score and cost must share the model columns")
        self.dimension = self.dimension.reindex(self.score.index)
        self.source = self.source.reindex(self.score.index)
        if self.available is None:
            self.available = pd.DataFrame(True, index=self.score.index, columns=self.score.columns)
        if self.unavailable_reason is None:
            self.unavailable_reason = pd.DataFrame(None, index=self.score.index,
                                                   columns=self.score.columns, dtype=object)
        reason = self.unavailable_reason.astype(object)
        self.unavailable_reason = reason.where(reason.notna(), None)
        for label, frame in (("available", self.available),
                             ("unavailable_reason", self.unavailable_reason)):
            if not frame.index.equals(self.score.index) or list(frame.columns) != self.models:
                raise ValueError(f"{label} must share the score frame's tasks and models")
        _check_reasons(self.available, self.unavailable_reason)

    @classmethod
    def from_long(cls, long: pd.DataFrame, name: str = "", run: int | None = None) -> "OutcomeMatrix":
        """Wide views from a long table. With `run=None` every (task, model) must have one row."""
        long = long_frame(long)
        if run is not None:
            long = long[long["run"] == run].reset_index(drop=True)
        if long.duplicated(["task_id", "model"]).any():
            raise ValueError("more than one run per (task, model); pass run=")

        def wide(col):
            return long.pivot(index="task_id", columns="model", values=col)

        score = wide("score").astype(float)
        models = list(score.columns)
        cost = wide("cost_usd").reindex(index=score.index, columns=models).astype(float)
        available = wide("available").reindex(index=score.index, columns=models)
        if available.isna().any(axis=None):
            raise ValueError("the long table does not cover every task x model cell")
        available = available.astype(bool)
        reason = wide("unavailable_reason").reindex(index=score.index, columns=models).astype(object)
        reason = reason.where(reason.notna(), None)
        meta = long.drop_duplicates("task_id").set_index("task_id").reindex(score.index)
        return cls(score, cost, meta["dimension"], meta["family"], name,
                   available=available, unavailable_reason=reason, long=long)

    @property
    def tasks(self) -> pd.Index:
        return self.score.index

    @property
    def models(self) -> list[str]:
        return list(self.score.columns)

    @property
    def family(self) -> pd.Series:
        return self.source

    def __len__(self) -> int:
        return len(self.score)

    def _replace(self, tasks=None, models=None, available=None, reason=None, long=None) -> "OutcomeMatrix":
        tasks = self.tasks if tasks is None else pd.Index(tasks)
        models = self.models if models is None else models
        available = self.available if available is None else available
        reason = self.unavailable_reason if reason is None else reason
        long = self.long if long is None else long
        if long is not None:
            long = long[long["task_id"].isin(tasks) & long["model"].isin(models)].reset_index(drop=True)
        return OutcomeMatrix(self.score.loc[tasks, models], self.cost.loc[tasks, models],
                             self.dimension.loc[tasks], self.source.loc[tasks], self.name,
                             available=available.loc[tasks, models],
                             unavailable_reason=reason.loc[tasks, models], long=long)

    def subset(self, tasks) -> "OutcomeMatrix":
        return self._replace(tasks=tasks)

    def without(self, model: str) -> "OutcomeMatrix":
        """Leave-one-model-out view."""
        return self._replace(models=[m for m in self.models if m != model])

    def mark_unavailable(self, cells, reason: str) -> "OutcomeMatrix":
        """A copy with the given (task, model) cells marked unavailable for `reason`.

        Cells outside this matrix are skipped (a verdict set spans both splits). A cell that is
        already unavailable keeps its first reason. Scores are left as they are.
        """
        if reason not in UNAVAILABLE_REASONS:
            raise ValueError(f"unknown unavailable_reason {reason!r}")
        tasks, models = set(self.tasks), set(self.models)
        hit = pd.DataFrame(False, index=self.tasks, columns=self.models)
        for t, m in cells:
            if t in tasks and m in models:
                hit.loc[t, m] = True
        new = hit & self.available
        available = self.available & ~new
        out_reason = self.unavailable_reason.mask(new, reason)
        long = self.long
        if long is not None:
            flag = new.stack()
            flag = flag[flag]
            key = pd.MultiIndex.from_frame(long[["task_id", "model"]])
            sel = key.isin(flag.index)
            long = long.copy()
            long.loc[sel, "available"] = False
            long.loc[sel, "unavailable_reason"] = reason
        return self._replace(available=available, reason=out_reason, long=long)
