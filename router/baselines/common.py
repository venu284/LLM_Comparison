"""Shared pieces of the S2.5 baselines: the fit context, policies (direct dispatch and fixed
ladders), their per-task outcomes on training data, and the router that executes them.

Every baseline is fitted with `fit(train, ctx) -> Router`:
- `train` is an `OutcomeMatrix` of training tasks only. Fitting may read its released scores and
  costs (training folds of the scored population, O1).
- The returned router runs in `sim.stream.replay` on the scored tasks, where it sees only what
  every router sees (S0.13): an opaque task handle, the available models and its own verdicts.
  It never sees a scored task's score or realised cost. The garbage tests in
  `tests/router/test_baselines.py` check this for every baseline.

Policies are the per-task plans the lookup baselines choose among (P0.5, P0.6):
- direct dispatch to one model, with no verifier run and no verifier cost;
- a fixed ladder of 2 or 3 models in increasing training mean cost. Each rung is verified. A PASS
  stops the ladder, and so does an UNKNOWN in mode `accept`; in mode `escalate` an UNKNOWN goes on
  to the next rung. The last rung's output is always accepted. Unavailable rungs are skipped
  (S0.11). P0.5's `bound` mode reads the released score and is not offered (R1).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace
from typing import Callable, Protocol, Sequence

import numpy as np
import pandas as pd

from router.data import scoring
from router.data.schema import OutcomeMatrix
from router.settings import VERIFIER_RUN_USD
from router.sim.stream import GIVE_UP, Accept, Call, Router, TaskHandle
from router.sim.verifier import PASS, UNKNOWN, Unavailable

ACCEPT, ESCALATE = "accept", "escalate"
MODES = (ACCEPT, ESCALATE)
TOL = 1e-12                 # tolerance of P0.6's rule tables
OUR_DEFINITION = "our definition"


# ------------------------------------------------------------------------------- fit context
@dataclass(frozen=True)
class Context:
    """What a baseline may use when it is fitted, besides the training matrix.

    - `verifier`: the replay verifier, for the training tasks' verdicts (ladder outcomes).
    - `content`: task id -> `content_text` (S1.5) for every task a text-based baseline may train
      on; routers read the scored task's text from its handle, never from here.
    - `cell_pool`: task id -> true cell for the cell predictor's training population (user
      decision 2026-11-21: the headline's development split; all eligible development tasks as a
      sensitivity). None: the training matrix's own tasks.
    - `exclude_problems`: problems of the scored fold; no pool task of these problems is used.
    - `problem_of`: task id -> problem id (`problems.csv`).
    - `embed`: texts -> unit vectors, for kNN (`baselines/embeddings.py`; tests pass a stub).
    - `seed`: the run's seed, for the baselines' own random streams.
    - `verifier_usd`: verifier cost per run used when a ladder's training outcomes are computed
      (A1 by default; S3's verifier-cost sweep refits with it scaled, spec D11).
    """
    verifier: object
    content: pd.Series | None = None
    cell_pool: pd.Series | None = None
    exclude_problems: frozenset = frozenset()
    problem_of: pd.Series | None = None
    embed: Callable[[Sequence[str]], np.ndarray] | None = None
    seed: int = 0
    verifier_usd: float = VERIFIER_RUN_USD

    def for_fold(self, scored_problems) -> "Context":
        return replace(self, exclude_problems=frozenset(scored_problems))


class Baseline(Protocol):
    name: str
    learned: bool                          # fitted on training outcomes (shuffled-label test)
    more_information: tuple[str, ...]      # labels the router may not use (source, dimension)
    definition: str | None                 # "our definition" note, or None

    def fit(self, train: OutcomeMatrix, ctx: Context) -> Router: ...


# ---------------------------------------------------------------------------------- data views
def true_cells(m: OutcomeMatrix) -> pd.Series:
    """The cell of each task: dimension x S1.1 family. On the headline the four families are the
    four cells. Route Anything's 21 cells use the task id's source token instead (P0.5); on the
    headline the two coincide."""
    return (m.dimension.astype(str) + "|" + m.family.astype(str)).rename("cell")


def success(m: OutcomeMatrix, cruxeval_lenient: bool = False) -> pd.DataFrame:
    """Per-cell success under S0.1 (score 1.0; CRUXEval 0.8 only as the sensitivity check).
    Unavailable cells are False."""
    fam = list(m.family)
    ok = pd.DataFrame({mod: np.asarray(scoring.success(m.score[mod].to_numpy(), fam,
                                                        cruxeval_lenient), dtype=bool)
                       for mod in m.models}, index=m.tasks)
    return ok & m.available.astype(bool)


def verdicts(m: OutcomeMatrix, verifier) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verdict and verifier-run flag of every available cell of `m`, from the replay verifier.
    Unavailable cells hold None and False."""
    v = pd.DataFrame(None, index=m.tasks, columns=m.models, dtype=object)
    ran = pd.DataFrame(False, index=m.tasks, columns=m.models)
    av = m.available.astype(bool)
    for t in m.tasks:
        for mod in m.models:
            if not av.at[t, mod]:
                continue
            try:
                c = verifier.check(t, mod)
            except Unavailable:
                continue
            v.at[t, mod], ran.at[t, mod] = c.verdict, c.ran
    return v, ran


def cost_order(train: OutcomeMatrix) -> list[str]:
    """Models by training mean cost, cheapest first (ties by name)."""
    mc = train.cost.where(train.available.astype(bool)).mean()
    return sorted(train.models, key=lambda x: (mc[x], x))


# -------------------------------------------------------------------------------------- policies
@dataclass(frozen=True)
class Policy:
    rungs: tuple[str, ...]
    verify: bool = True
    on_unknown: str = ESCALATE

    @property
    def name(self) -> str:
        if not self.verify:
            return f"direct:{self.rungs[0]}"
        return f"{'>'.join(self.rungs)}@{self.on_unknown}"


def direct(model: str) -> Policy:
    return Policy((model,), verify=False)


def ladders(train: OutcomeMatrix, mode: str) -> list[Policy]:
    """P0.5's fixed ladders: 2- and 3-rung permutations of the models (matrix order) whose
    training mean cost is non-decreasing along the ladder. With 8 models: 84."""
    if mode not in MODES:
        raise ValueError(mode)
    mc = train.cost.where(train.available.astype(bool)).mean()
    out = []
    for k in (2, 3):
        for lad in itertools.permutations(train.models, k):
            if all(mc[a] <= mc[b] for a, b in zip(lad, lad[1:])):
                out.append(Policy(tuple(lad), True, mode))
    return out


def candidates(train: OutcomeMatrix, mode: str) -> list[Policy]:
    """P0.6's candidate columns in one mode: direct dispatch to each model, then the ladders."""
    return [direct(x) for x in train.models] + ladders(train, mode)


def policy_outcomes(train: OutcomeMatrix, verifier, policies: Sequence[Policy], *,
                    cruxeval_lenient: bool = False,
                    verifier_usd: float = VERIFIER_RUN_USD) -> tuple[np.ndarray, np.ndarray]:
    """Per-task acceptance (0/1) and cost (USD) of each policy on the training tasks, computed
    the way `replay` + `metrics.task_outcomes` would (a test checks the two agree)."""
    ok = success(train, cruxeval_lenient).to_numpy()
    cost = train.cost.to_numpy(float)
    av = train.available.astype(bool).to_numpy()
    v, ran = verdicts(train, verifier)
    passed = (v == PASS).to_numpy()
    unknown = (v == UNKNOWN).to_numpy()
    ranv = ran.to_numpy(bool)
    col = {mod: j for j, mod in enumerate(train.models)}
    n = len(train)
    A, C = np.zeros((n, len(policies))), np.zeros((n, len(policies)))
    for p, pol in enumerate(policies):
        js = [col[x] for x in pol.rungs]
        acc, cst, done = np.zeros(n), np.zeros(n), np.zeros(n, bool)
        if not pol.verify:
            j = js[0]
            A[:, p] = np.where(av[:, j], ok[:, j], 0.0)
            C[:, p] = np.where(av[:, j], cost[:, j], 0.0)
            continue
        # the last available rung of each task, for "the last rung is always accepted"
        avail = av[:, js]
        last = np.where(avail.any(1), len(js) - 1 - np.argmax(avail[:, ::-1], axis=1), -1)
        for i, j in enumerate(js):
            reach = ~done & avail[:, i]
            cst += np.where(reach, cost[:, j] + verifier_usd * ranv[:, j], 0.0)
            stop = passed[:, j] | (unknown[:, j] & (pol.on_unknown == ACCEPT)) | (last == i)
            take = reach & stop
            acc = np.where(take, ok[:, j], acc)
            done |= take
        A[:, p], C[:, p] = acc, cst
    return A, C


# ------------------------------------------------------------------------------- plan executor
class PlanRouter:
    """Executes one policy per task: `plan(handle) -> Policy`. Direct dispatch calls without the
    verifier and accepts; a ladder follows `sim.stream.Ladder`'s rules. A task with no available
    rung gets GIVE_UP (counted as not accepted)."""

    def __init__(self, plan: Callable[[TaskHandle], Policy], more_information=()):
        self._plan = plan
        self.more_information = tuple(more_information)
        self._current: tuple[str, Policy] | None = None
        self.plans: dict[str, str] = {}          # task key -> policy name, for reports

    def _policy(self, task: TaskHandle) -> Policy:
        if self._current is None or self._current[0] != task.key:
            pol = self._plan(task)
            self._current = (task.key, pol)
            self.plans[task.key] = pol.name
        return self._current[1]

    def decide(self, task, available, history):
        pol = self._policy(task)
        rungs = [x for x in pol.rungs if x in available]
        if not rungs:
            return Accept() if history else GIVE_UP
        if not pol.verify:
            return Accept() if history else Call(rungs[0], verify=False)
        if history:
            last = history[-1]
            stop = last.verdict == PASS or (last.verdict == UNKNOWN and pol.on_unknown == ACCEPT)
            if stop or len(history) == len(rungs):
                return Accept()
        return rungs[len(history)]

    def feedback(self, obs) -> None:
        pass


# ------------------------------------------------------------------------------ small helpers
def best_column(acc: np.ndarray, cost: np.ndarray) -> int:
    """Highest acceptance, ties to the cheaper column, then the lower index."""
    return int(np.lexsort((np.arange(len(acc)), cost, -acc))[0])


@dataclass
class Fitted:
    """What a lookup baseline chose, for the run report."""
    table: dict = field(default_factory=dict)
    fallback: str | None = None
