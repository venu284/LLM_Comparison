"""Run-time verifier for replay (S2.2; CLAIMS S0.11, S0.13, R1).

Three modes, each its own class:

- `ReplayVerifier` (**the default and the only claim-facing mode**): Route Anything's real X13
  visible-test verdicts (`router.data.verdicts.load_verdicts`). Per cell:
  - visible tests exist (`w1_n > 0`): the recorded verdict, PASS or FAIL, and one verifier run is
    charged;
  - no visible tests (`w1_n = 0`): UNKNOWN, and no verifier run is charged (A1);
  - empty X13 response: the model is unavailable for the task. Availability is read from the
    `OutcomeMatrix` (which also carries the 148 Qwen3-Max missing token records), so the stream
    never offers that model; calling it raises `Unavailable`;
  - the task has no X13 record at all (tasks outside the 2,911 Route Anything made runnable): the
    router sees UNKNOWN, no verifier run is charged, and the state `no_record` is kept apart from
    `no_signal` so the counts stay separate. Cascade claims never score these tasks (S0.11).
- `HeuristicVerifier`, **SENSITIVITY ONLY**: as replay, but on no-signal cells the X13
  syntax/quality heuristic (`w1_accept`) gives PASS or FAIL. The heuristic runs no tests, so no
  verifier run is charged for it.
- `NoiseVerifier`, **SENSITIVITY ONLY (E3)**: simulated noise around the S0.1 truth, with separate
  false-pass and false-reject rates. It is the one mode that reads the released score. Each cell's
  noise is drawn once, up front, from stream `verifier_noise`, so every router sees the same noise
  on the same cell whatever order it calls models in.

None of these objects is handed to a router; the stream turns a `Check` into an `Observation`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from router.data import scoring, verdicts
from router.sim.stats import rng

FAIL, PASS, UNKNOWN = verdicts.FAIL, verdicts.PASS, verdicts.UNKNOWN
VERDICTS = (PASS, FAIL, UNKNOWN)

REPLAY, HEURISTIC, NOISE = "replay", "heuristic", "noise"
SIGNAL, NO_SIGNAL, NO_RECORD, HEURISTIC_STATE, SIMULATED = (
    "signal", "no_signal", "no_record", "heuristic", "simulated")


class Unavailable(LookupError):
    """The model has no response for this task (empty X13 response, S0.11)."""


@dataclass(frozen=True)
class Check:
    """The verifier's result on one (task, model) cell. Internal to the stream."""
    verdict: str          # PASS, FAIL or UNKNOWN
    ran: bool             # a visible-test run happened (and is charged)
    state: str            # signal, no_signal, no_record, heuristic or simulated


class ReplayVerifier:
    """The real X13 visible-test verdicts (S0.11). Claim-facing."""
    mode = REPLAY
    sensitivity = False

    def __init__(self, table: pd.DataFrame):
        need = {"task_id", "model", "observation", "signal", "unavailable"}
        if need - set(table.columns):
            raise ValueError(f"verdict table lacks {sorted(need - set(table.columns))}")
        t = table.set_index(["task_id", "model"])
        if t.index.has_duplicates:
            raise ValueError("duplicate (task, model) verdicts")
        self._obs = t["observation"].to_dict()
        self._signal = t["signal"].astype(bool).to_dict()
        self._empty = {k for k, v in t["unavailable"].astype(bool).items() if v}
        self._tasks = set(t.index.get_level_values(0))

    @classmethod
    def from_x13(cls, **kwargs) -> "ReplayVerifier":
        return cls(verdicts.load_verdicts(**kwargs))

    def has_record(self, task_id: str) -> bool:
        return task_id in self._tasks

    def empty_cells(self) -> set[tuple[str, str]]:
        """(task, model) cells with an empty X13 response: unavailable (S0.11)."""
        return set(self._empty)

    def check(self, task_id: str, model: str) -> Check:
        key = (task_id, model)
        if key in self._empty:
            raise Unavailable(f"{model} has no response for {task_id}")
        if key not in self._obs:
            if task_id in self._tasks:
                raise KeyError(f"X13 has task {task_id} but no row for {model}")
            return Check(UNKNOWN, False, NO_RECORD)
        if self._signal[key]:
            return Check(self._obs[key], True, SIGNAL)
        return Check(UNKNOWN, False, NO_SIGNAL)


class HeuristicVerifier(ReplayVerifier):
    """SENSITIVITY ONLY (S0.11): the X13 syntax/quality heuristic decides no-signal cells."""
    mode = HEURISTIC
    sensitivity = True

    def __init__(self, table: pd.DataFrame, heuristic: pd.DataFrame):
        super().__init__(table)
        h = heuristic.set_index(["task_id", "model"])["heuristic_accept"].astype(bool)
        self._heuristic = h.to_dict()

    @classmethod
    def from_x13(cls, **kwargs) -> "HeuristicVerifier":
        path = kwargs.get("path", verdicts.X13_VERDICTS)
        return cls(verdicts.load_verdicts(**kwargs), verdicts.load_heuristic_accept(path))

    def check(self, task_id: str, model: str) -> Check:
        c = super().check(task_id, model)
        if c.state != NO_SIGNAL:
            return c
        return Check(PASS if self._heuristic[(task_id, model)] else FAIL, False, HEURISTIC_STATE)


class NoiseVerifier:
    """SENSITIVITY ONLY (E3): simulated noise around the S0.1 truth.

    `truth` is a task x model boolean frame (S0.1 success). A true success is reported FAIL with
    probability `false_reject`; a true failure is reported PASS with probability `false_pass`.
    Cells marked True in `no_signal` (optional) give UNKNOWN, as the real verifier would. Every
    other cell is charged one verifier run.
    """
    mode = NOISE
    sensitivity = True

    def __init__(self, truth: pd.DataFrame, *, false_pass: float, false_reject: float, seed: int,
                 no_signal: pd.DataFrame | None = None):
        for name, p in (("false_pass", false_pass), ("false_reject", false_reject)):
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        truth = truth.astype(bool)
        # One uniform per cell, in sorted (task, model) order, so the draw does not depend on the
        # order a router calls models in, nor on the order of the frame.
        cells = truth.sort_index().sort_index(axis=1)
        u = pd.DataFrame(rng(seed, "verifier_noise").random(cells.shape),
                         index=cells.index, columns=cells.columns)
        flip = np.where(cells.to_numpy(), u.to_numpy() < false_reject, u.to_numpy() < false_pass)
        observed = cells.to_numpy() ^ flip
        self._pass = pd.DataFrame(observed, index=cells.index, columns=cells.columns)
        self._no_signal = None if no_signal is None else no_signal.astype(bool)
        self.false_pass, self.false_reject, self.seed = false_pass, false_reject, seed

    @classmethod
    def from_matrix(cls, m, *, false_pass: float, false_reject: float, seed: int,
                    cruxeval_lenient: bool = False, no_signal: pd.DataFrame | None = None
                    ) -> "NoiseVerifier":
        """Truth from the released scores under S0.1, family by family."""
        fam = m.family.reindex(m.tasks)
        if fam.isna().any():
            raise ValueError("every task needs a family (S0.1)")
        truth = pd.DataFrame({mod: scoring.success(m.score[mod].to_numpy(), list(fam),
                                                   cruxeval_lenient)
                              for mod in m.models}, index=m.tasks)
        return cls(truth, false_pass=false_pass, false_reject=false_reject, seed=seed,
                   no_signal=no_signal)

    def has_record(self, task_id: str) -> bool:
        return task_id in self._pass.index

    def empty_cells(self) -> set[tuple[str, str]]:
        return set()

    def check(self, task_id: str, model: str) -> Check:
        if self._no_signal is not None and bool(self._no_signal.at[task_id, model]):
            return Check(UNKNOWN, False, NO_SIGNAL)
        return Check(PASS if bool(self._pass.at[task_id, model]) else FAIL, True, SIMULATED)
