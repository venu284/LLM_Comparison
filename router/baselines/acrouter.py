"""The released ACRouter decisions (S2.5; 2606.22902's `routing_decisions` file in the
CodeRouterBench release).

The release holds one chosen model per ID-test task and nothing else: there are no decisions for
probing tasks, so on the development split only the ID-test tasks are covered. The router looks
the decision up by the opaque task key; a task with no decision gets GIVE_UP and is reported as
uncovered. Score the covered subset only (`covered`). Not fitted: the decisions are fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from router.baselines.common import Context
from router.data import coderouterbench as crb
from router.data.schema import OutcomeMatrix
from router.sim.stream import GIVE_UP, Accept, Call, task_key


def covered(tasks, decisions: pd.Series | None = None) -> pd.Index:
    d = crb.released_acrouter_decisions() if decisions is None else decisions
    return pd.Index([t for t in tasks if t in d.index])


class _Lookup:
    more_information = ()

    def __init__(self, by_key: dict[str, str]):
        self._by_key = by_key
        self.uncovered: list[str] = []

    def decide(self, task, available, history):
        if history:
            return Accept()
        mdl = self._by_key.get(task.key)
        if mdl is None:
            self.uncovered.append(task.key)
            return GIVE_UP
        return Call(mdl, verify=False) if mdl in available else GIVE_UP

    def feedback(self, obs) -> None:
        pass


@dataclass
class ReleasedACRouter:
    decisions: pd.Series | None = None
    name: str = "acrouter_released"
    learned: bool = False
    more_information: tuple = ()
    definition: str = "the release's ID-test decisions; scored on covered tasks only"

    def fit(self, train: OutcomeMatrix, ctx: Context):
        d = crb.released_acrouter_decisions() if self.decisions is None else self.decisions
        return _Lookup({task_key(t): m for t, m in d.items()})
