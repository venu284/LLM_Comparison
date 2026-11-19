"""The outcome matrix every router component reads.

One row per task, one column per model. Scores are in [0, 1]; costs are USD per task.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class OutcomeMatrix:
    score: pd.DataFrame          # task x model
    cost: pd.DataFrame           # task x model, USD
    dimension: pd.Series         # task -> dimension
    source: pd.Series            # task -> source benchmark family
    name: str = ""

    def __post_init__(self) -> None:
        if not self.score.index.equals(self.cost.index):
            raise ValueError("score and cost must share the task index")
        if list(self.score.columns) != list(self.cost.columns):
            raise ValueError("score and cost must share the model columns")
        self.dimension = self.dimension.reindex(self.score.index)
        self.source = self.source.reindex(self.score.index)

    @property
    def tasks(self) -> pd.Index:
        return self.score.index

    @property
    def models(self) -> list[str]:
        return list(self.score.columns)

    def __len__(self) -> int:
        return len(self.score)

    def subset(self, tasks) -> "OutcomeMatrix":
        tasks = pd.Index(tasks)
        return OutcomeMatrix(self.score.loc[tasks], self.cost.loc[tasks],
                             self.dimension.loc[tasks], self.source.loc[tasks], self.name)

    def without(self, model: str) -> "OutcomeMatrix":
        """Leave-one-model-out view."""
        keep = [m for m in self.models if m != model]
        return OutcomeMatrix(self.score[keep], self.cost[keep], self.dimension, self.source, self.name)
