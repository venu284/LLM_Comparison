"""Always-X, the always-best single model and Random (S2.5, R2)."""

from __future__ import annotations

from dataclasses import dataclass

from router.baselines.common import Context, best_column, success
from router.data.schema import OutcomeMatrix
from router.sim.stats import rng
from router.sim.stream import GIVE_UP, Accept, Call, SingleChoice


def _pick(model):
    return lambda task, available: model if model in available else GIVE_UP


@dataclass
class Always:
    """Always the same model, by direct dispatch."""
    model: str
    learned: bool = False
    more_information: tuple = ()
    definition: str | None = None

    @property
    def name(self) -> str:
        return f"always:{self.model}"

    def fit(self, train: OutcomeMatrix, ctx: Context):
        return SingleChoice(_pick(self.model))


def best_single_model(train: OutcomeMatrix, cruxeval_lenient: bool = False) -> str:
    """The best single model on the training tasks (O1): highest acceptance under S0.1, ties to
    the lower training mean cost."""
    acc = success(train, cruxeval_lenient).mean().to_numpy()
    cost = train.cost.where(train.available.astype(bool)).mean().to_numpy()
    return train.models[best_column(acc, cost)]


@dataclass
class BestSingle:
    """Always-best: the best single model, chosen on the training folds (O1)."""
    name: str = "best_single"
    learned: bool = True
    more_information: tuple = ()
    definition: str | None = None

    def fit(self, train: OutcomeMatrix, ctx: Context):
        best = best_single_model(train)
        r = SingleChoice(_pick(best))
        r.chosen = best
        return r


class _RandomRouter:
    more_information = ()

    def __init__(self, seed: int):
        self._g = rng(seed, "random_baseline")

    def decide(self, task, available, history):
        if history:
            return Accept()
        if not available:
            return GIVE_UP
        return Call(available[int(self._g.integers(len(available)))], verify=False)

    def feedback(self, obs) -> None:
        pass


@dataclass
class RandomChoice:
    """A uniformly random available model per task, from stream `random_baseline`."""
    name: str = "random"
    learned: bool = False
    more_information: tuple = ()
    definition: str | None = None

    def fit(self, train: OutcomeMatrix, ctx: Context):
        return _RandomRouter(ctx.seed)
