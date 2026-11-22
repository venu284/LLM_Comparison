"""Seeded task-stream replay with bandit feedback (S2.1; CLAIMS S0.13, R1, R16).

Replay is score-free by construction. It runs on a `ReplayView`, built from an `OutcomeMatrix` by
reading only its task and model lists, costs, availability and (for routers labelled as given
more information) family and dimension. The released score is never read here; `sim/metrics.py`
joins the decision trace to it afterwards.

What a router sees:
- a `TaskHandle`: an opaque key (not the CodeRouterBench task id, whose prefix encodes the family
  and dimension), the task's `content_text` when the caller supplies it (S1.5), and `info`, which
  is empty unless the router declares `more_information` (e.g. `("family",)`); such a router is
  recorded as **given more information**;
- the models available for the task: arrived (model-arrival events) and with a response;
- its own calls on this task so far, as `Observation`s: model, verdict (PASS / FAIL / UNKNOWN,
  S0.13) and call index. Never another model's outcome, never a score, and not the realised cell
  cost either: Qwen3-Max's released cost is a per-dimension constant (S1.7 data note), so the
  cost would reveal the dimension. Cost-aware routers use published prices (`data/prices.py`) or
  what they fitted on training folds.

What a router does on each task, one decision at a time (`decide`):
- name a model to call (each model at most once per task: replay holds one run per cell, so a
  retry would return the same output; retries are therefore not simulated and count as 0). A bare
  model name is a verified call: the verifier runs and its verdict comes back. `Call(model,
  verify=False)` calls without the verifier (direct dispatch): no verifier run is charged and the
  observation is UNKNOWN with state `not_run`;
- `Accept(model)` to submit one of the outputs it has called (default: the last one);
- `GIVE_UP` to submit nothing (C2's give-up action): the task is not accepted, and nothing more is
  spent on it.
After every call the router also gets `feedback(observation)`.

Unavailable cells (empty X13 responses and missing token records, S0.11) are never offered: the
rung is skipped and the skip is counted per task. Probe calls happen outside the stream, on
training-pool tasks; their cost and count enter the metrics directly (S0.2, S0.13).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, Union

import pandas as pd

from router.sim.stats import rng
from router.sim.verifier import UNKNOWN, Check, Unavailable

INFO_FIELDS = ("family", "dimension")
NOT_RUN = "not_run"          # state of a call made without the verifier


# ------------------------------------------------------------------------ router-facing objects
@dataclass(frozen=True)
class TaskHandle:
    key: str                                   # opaque, stable per task id
    content_text: str | None = None            # the router field (S1.5), when supplied
    info: Mapping[str, str] = field(default_factory=dict)   # only for "given more information"


@dataclass(frozen=True)
class Observation:
    task: str          # the TaskHandle key
    model: str
    verdict: str       # PASS, FAIL or UNKNOWN (S0.13)
    call_index: int    # 0 for the first call on the task


@dataclass(frozen=True)
class Call:
    """Call `model`; with `verify=False` the verifier does not run (direct dispatch)."""
    model: str
    verify: bool = True


@dataclass(frozen=True)
class Accept:
    model: str | None = None     # None: the last model called


class GiveUp:
    def __repr__(self) -> str:
        return "GIVE_UP"


GIVE_UP = GiveUp()
Decision = Union[str, Call, Accept, GiveUp]


class Router(Protocol):
    def decide(self, task: TaskHandle, available: tuple[str, ...],
               history: tuple[Observation, ...]) -> Decision: ...

    def feedback(self, obs: Observation) -> None: ...


def task_key(task_id: str) -> str:
    """Opaque router-facing key of a task id."""
    return hashlib.sha256(task_id.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------- replay view
@dataclass(frozen=True)
class ReplayView:
    """Everything replay may read. There is no score here."""
    tasks: pd.Index
    models: tuple[str, ...]
    cost: pd.DataFrame                 # task x model, USD per call
    available: pd.DataFrame            # task x model, bool
    content: pd.Series | None = None   # task -> content_text
    info: pd.DataFrame | None = None   # task -> family, dimension (given more information only)

    @classmethod
    def from_matrix(cls, m, content: pd.Series | None = None) -> "ReplayView":
        """Reads `m.tasks`, `m.models`, `m.cost`, `m.available`, `m.family`, `m.dimension`.
        Never `m.score`."""
        tasks = pd.Index(m.tasks)
        info = pd.DataFrame({"family": m.family.reindex(tasks),
                             "dimension": m.dimension.reindex(tasks)}, index=tasks)
        return cls(tasks, tuple(m.models), m.cost.copy(), m.available.astype(bool).copy(),
                   None if content is None else content.reindex(tasks), info)


# --------------------------------------------------------------------------------------- replay
@dataclass
class ReplayResult:
    calls: pd.DataFrame       # one row per call: task_id, position, call_index, model, verdict,
                              #   state, verifier_ran, model_cost_usd
    tasks: pd.DataFrame       # one row per task (index task_id): position, action, final_model,
                              #   n_calls, n_verifier_runs, model_cost_usd, n_skipped_unavailable,
                              #   skipped_models, has_record
    events: list[dict]        # model arrivals: position, model
    seed: int
    verifier_mode: str
    sensitivity: bool         # the verifier is a sensitivity mode, never claim-facing
    given_more_information: tuple[str, ...]
    n_retries: int = 0        # not simulated: one run per cell


def task_order(tasks: Sequence[str], seed: int) -> pd.Index:
    """The seeded stream order: a permutation of the sorted task ids from stream `stream_order`."""
    ordered = pd.Index(sorted(tasks))
    return ordered[rng(seed, "stream_order").permutation(len(ordered))]


def replay(router: Router, view: ReplayView, verifier, *, seed: int,
           arrivals: Mapping[str, int] | None = None) -> ReplayResult:
    """Run `router` over the view's tasks once, in the order of `task_order(seed)`.

    `arrivals` maps a model to the stream position from which it is available (default 0).
    """
    arrivals = dict(arrivals or {})
    unknown = set(arrivals) - set(view.models)
    if unknown:
        raise ValueError(f"arrivals name models not in the view: {sorted(unknown)}")
    empty = verifier.empty_cells()
    for t, mdl in empty:
        if t in view.available.index and mdl in view.available.columns and view.available.at[t, mdl]:
            raise ValueError(f"empty X13 cell ({t}, {mdl}) is available in the matrix; build it "
                             "with audit.load_release (S0.11)")
    reveal = tuple(getattr(router, "more_information", ()) or ())
    bad = set(reveal) - set(INFO_FIELDS)
    if bad:
        raise ValueError(f"cannot reveal {sorted(bad)}")

    order = task_order(view.tasks, seed)
    events = [{"position": int(p), "model": mdl} for mdl, p in sorted(arrivals.items(),
                                                                      key=lambda x: (x[1], x[0]))
              if p > 0]
    call_rows, task_rows = [], []
    for pos, tid in enumerate(order):
        arrived = [mdl for mdl in view.models if arrivals.get(mdl, 0) <= pos]
        avail = tuple(mdl for mdl in arrived if bool(view.available.at[tid, mdl]))
        skipped = tuple(mdl for mdl in arrived if mdl not in avail)
        info = {f: str(view.info.at[tid, f]) for f in reveal} if reveal else {}
        content = None if view.content is None else view.content.get(tid)
        handle = TaskHandle(task_key(tid), None if pd.isna(content) else content, info)
        history: list[Observation] = []
        called: set[str] = set()
        runs, spent = 0, 0.0
        while True:
            d = router.decide(handle, avail, tuple(history))
            if isinstance(d, GiveUp):
                action, final = "give_up", None
                break
            if isinstance(d, Accept):
                if not history:
                    raise ValueError("Accept before any call")
                final = d.model if d.model is not None else history[-1].model
                if final not in called:
                    raise ValueError(f"Accept names {final}, which was not called on this task")
                action = "accept"
                break
            verify = True
            if isinstance(d, Call):
                d, verify = d.model, d.verify
            if not isinstance(d, str):
                raise TypeError(f"unknown decision {d!r}")
            if d not in avail:
                raise ValueError(f"{d} is not available for this task")
            if d in called:
                raise ValueError(f"{d} was already called on this task (retries are not simulated)")
            if verify:
                try:
                    c: Check = verifier.check(tid, d)
                except Unavailable as e:          # guarded above; kept as a hard stop
                    raise AssertionError(str(e)) from e
            else:
                c = Check(UNKNOWN, False, NOT_RUN)
            cost = float(view.cost.at[tid, d])
            obs = Observation(handle.key, d, c.verdict, len(history))
            router.feedback(obs)
            history.append(obs)
            called.add(d)
            runs += int(c.ran)
            spent += cost
            call_rows.append({"task_id": tid, "position": pos, "call_index": obs.call_index,
                              "model": d, "verdict": c.verdict, "state": c.state,
                              "verifier_ran": c.ran, "model_cost_usd": cost})
        task_rows.append({"task_id": tid, "position": pos, "action": action, "final_model": final,
                          "n_calls": len(history), "n_verifier_runs": runs,
                          "model_cost_usd": spent, "n_skipped_unavailable": len(skipped),
                          "skipped_models": skipped, "has_record": verifier.has_record(tid)})
    calls = pd.DataFrame(call_rows, columns=["task_id", "position", "call_index", "model",
                                             "verdict", "state", "verifier_ran", "model_cost_usd"])
    tasks = pd.DataFrame(task_rows).set_index("task_id")
    return ReplayResult(calls, tasks, events, seed, verifier.mode, verifier.sensitivity, reveal)


# ------------------------------------------------------------------------------ simple routers
class SingleChoice:
    """Adapter: a policy that picks one model per task, calls it and accepts its output. By
    default the call is direct dispatch (no verifier, no verifier cost); `verify=True` runs it.

    `choose(task, available) -> model` gets the TaskHandle and the available models only."""

    def __init__(self, choose, more_information: tuple[str, ...] = (), verify: bool = False):
        self._choose = choose
        self.more_information = more_information
        self.verify = verify

    def decide(self, task, available, history):
        if history:
            return Accept()
        if not available:
            return GIVE_UP
        d = self._choose(task, available)
        return Call(d, verify=self.verify) if isinstance(d, str) else d

    def feedback(self, obs) -> None:
        pass


class Ladder:
    """Fixed cheap-to-strong ladder: call each available rung in order; accept the first PASS;
    after the last rung accept its output. Unavailable rungs are skipped (S0.11). `on_unknown`
    says what an UNKNOWN does: "accept" (stop and submit) or "escalate" (call the next rung)."""

    def __init__(self, ladder: Sequence[str], on_unknown: str = "escalate"):
        if on_unknown not in ("accept", "escalate"):
            raise ValueError(on_unknown)
        self.ladder, self.on_unknown = tuple(ladder), on_unknown
        self.more_information: tuple[str, ...] = ()

    def decide(self, task, available, history):
        rungs = [m for m in self.ladder if m in available]
        if not rungs:
            return GIVE_UP
        if history:
            last = history[-1]
            stop = last.verdict == "PASS" or (last.verdict == "UNKNOWN" and self.on_unknown == "accept")
            if stop or len(history) == len(rungs):
                return Accept()
        return rungs[len(history)]

    def feedback(self, obs) -> None:
        pass
