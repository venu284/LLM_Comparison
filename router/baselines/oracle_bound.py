"""BOUND ONLY: the oracle-on-no-signal feasibility bound for DP1b (S3 spec D4; plan §1).

In the bound, a ladder rung with visible tests uses its real X13 verdict; a rung without visible
tests (no signal) is accepted exactly when the ground truth succeeds, at no verifier cost, as in
P0.5 and P0.6. The last available rung is always accepted. The bound reads the released score
through the truth, so **it is not buildable and is never a router**. It is labelled
`verifier_mode = "oracle_on_no_signal"` in every result.

Only `router/experiments/s3_dp1b.py` may import this module (S3 spec D4 ruling, 2026-11-23); a
test walks the import graph and enforces it, as for `router/sim/paper.py`. The spec placed the
`oracle` mode in `baselines.common`; it lives here instead so that the restriction can be enforced
at module level (spec §7, 2026-11-23). Behaviour is the same: `common.policy_outcomes` and
`common.PlanRouter` already escalate on anything but PASS outside mode `accept`, so with this
verifier a no-signal rung stops exactly when the truth succeeds.

- `OracleOnNoSignal`: wraps the replay verifier; on a no-signal cell it returns PASS or FAIL from
  the truth with `ran=False` (no verifier cost); every other cell is the replay verdict. A task
  with no X13 record is refused (cascade claims score X13 tasks only, S0.11).
- `ORACLE`: the ladder mode label of the bound; `oracle_candidates` are P0.6's 92 columns (direct
  dispatch to each model, then the 84 fixed ladders) in that mode.
- `BoundRule`: per-cell Rule A or B (`lookup.rule_a_column`, `rule_b_column`) over those columns,
  fitted on training folds with the bound's training outcomes; held-out tasks are routed by their
  predicted (or true) cell. Rule A in bound mode with predicted cells is the statistic that decides
  DP1b (spec D4 (a)).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from router.baselines.common import (ESCALATE, OUR_DEFINITION, Context, PlanRouter, direct,
                                     ladders, policy_outcomes, success, true_cells)
from router.baselines.lookup import (TRUE_CELL_INFO, _cell_reader, cellbest_column, group_means,
                                     rule_a_column, rule_b_column)
from router.baselines.static import best_single_model
from router.data.schema import OutcomeMatrix
from router.sim.verifier import FAIL, NO_RECORD, NO_SIGNAL, PASS, Check

ORACLE = "oracle"
VERIFIER_MODE = "oracle_on_no_signal"
ORACLE_STATE = "oracle"
DEFINITION = (f"{OUR_DEFINITION}: oracle-on-no-signal bound (S3 spec D4); not buildable, reads "
              "the truth on no-signal cells")


class OracleOnNoSignal:
    """The replay verifier with the truth on no-signal cells (BOUND ONLY)."""
    mode = VERIFIER_MODE
    sensitivity = True            # never a router claim

    def __init__(self, base, truth: pd.DataFrame):
        self.base = base
        self._truth = truth.astype(bool)

    @classmethod
    def from_matrix(cls, base, m: OutcomeMatrix, cruxeval_lenient: bool = False) -> "OracleOnNoSignal":
        """Truth = S0.1 success on `m` (CRUXEval >= 0.8 when `cruxeval_lenient`)."""
        return cls(base, success(m, cruxeval_lenient))

    def has_record(self, task_id: str) -> bool:
        return self.base.has_record(task_id)

    def empty_cells(self):
        return self.base.empty_cells()

    def check(self, task_id: str, model: str) -> Check:
        c = self.base.check(task_id, model)
        if c.state == NO_RECORD:
            raise ValueError(f"{task_id} has no X13 record; the bound scores X13 tasks only (S0.11)")
        if c.state != NO_SIGNAL:
            return c
        return Check(PASS if bool(self._truth.at[task_id, model]) else FAIL, False, ORACLE_STATE)


def oracle_ladders(train: OutcomeMatrix):
    """P0.5's 84 fixed ladders in the bound's mode."""
    return [replace(p, on_unknown=ORACLE) for p in ladders(train, ESCALATE)]


def oracle_candidates(train: OutcomeMatrix):
    """P0.6's 92 columns in the bound's mode: direct dispatch to each model, then the ladders."""
    return [direct(x) for x in train.models] + oracle_ladders(train)


def require_oracle(verifier) -> None:
    if not isinstance(verifier, OracleOnNoSignal):
        raise TypeError("the bound needs the OracleOnNoSignal verifier for fitting and scoring")


@dataclass
class BoundRule:
    """Per-cell Rule A or B over the bound's columns (BOUND ONLY; spec D4)."""
    rule: str                      # "A" or "B"
    cells: str = "pred"
    learned: bool = True
    definition: str = DEFINITION

    @property
    def name(self) -> str:
        return f"bound_rule{self.rule}_{self.cells}"

    @property
    def more_information(self) -> tuple:
        return TRUE_CELL_INFO if self.cells == "true" else ()

    def fit(self, train: OutcomeMatrix, ctx: Context):
        if self.rule not in ("A", "B"):
            raise ValueError(self.rule)
        require_oracle(ctx.verifier)
        pols = oracle_candidates(train)
        A, C = policy_outcomes(train, ctx.verifier, pols, verifier_usd=ctx.verifier_usd)
        groups = true_cells(train)
        names = sorted(groups.unique())
        v, _ = group_means(groups.to_numpy(), A, names)
        c, _ = group_means(groups.to_numpy(), C, names)
        M = len(train.models)
        pick = rule_a_column if self.rule == "A" else rule_b_column
        table = {}
        for i, g in enumerate(names):
            k = cellbest_column(v[i], c[i], M)
            table[g] = pols[pick(v[i], c[i], k)]
        read, info = _cell_reader(self.cells, train, ctx)
        fallback = direct(best_single_model(train))
        router = PlanRouter(lambda h: table.get(read(h), fallback), info)
        router.table = {g: p.name for g, p in table.items()}
        router.fallback = fallback.name
        return router
