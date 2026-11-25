"""Descriptive definitions for S3.6 and the descriptive part of S3.3 (`docs/research/S3_analysis_spec.md`
§5). Nothing here is fitted: every function counts on the cells it is given.

Inputs are task x model frames with the same index and columns:

- `ok`: success under S0.1 (bool), False on unavailable cells;
- `available`: the cell has a response (not an empty X13 response, not a missing token record);
- `verdict`: the X13 visible-test observation, "PASS", "FAIL", "UNKNOWN" (no visible tests) or
  None (unavailable or no X13 record);
- `cost`: released cost in USD.

Every rate is returned with its numerator and denominator. A rate with a zero denominator is NaN
here; the report writes it as "no signal" or "none" (spec §5.1).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import beta

from router.settings import CI_LEVEL
from router.sim.verifier import FAIL, PASS

NO_SIGNAL, NONE = "no signal", "none"


# ---------------------------------------------------------------------------------- intervals
def clopper_pearson(k: int, n: int, level: float = CI_LEVEL) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials; (nan, nan) when n == 0."""
    if n == 0:
        return float("nan"), float("nan")
    a = (1 - level) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def ratio(num: str, den: str, scale: float = 100.0):
    """Bootstrap statistic: weighted sum of `num` over weighted sum of `den`, in points."""
    def stat(f: pd.DataFrame, w: np.ndarray) -> float:
        d = float(np.sum(w * f[den].to_numpy(float)))
        return scale * float(np.sum(w * f[num].to_numpy(float))) / d if d else float("nan")
    return stat


def rate(num: int, den: int) -> float:
    return num / den if den else float("nan")


# ------------------------------------------------------------------------------- S3.6 verifier
def verifier_cells(ok: pd.DataFrame, verdict: pd.DataFrame, available: pd.DataFrame) -> pd.DataFrame:
    """One row per available cell with an X13 record: task_id, model, signal, passed (verdict
    PASS), success. Cells without a record (verdict None) are left out."""
    v = verdict.reindex(index=ok.index, columns=ok.columns)
    av = available.reindex(index=ok.index, columns=ok.columns).astype(bool)
    rec = v.notna() & av
    long = pd.DataFrame({"verdict": v.stack(future_stack=True), "success": ok.stack(future_stack=True),
                         "keep": rec.stack(future_stack=True)})
    long = long[long.keep.astype(bool)].drop(columns="keep")
    long.index.names = ["task_id", "model"]
    long = long.reset_index()
    long["signal"] = long.verdict.isin([PASS, FAIL])
    long["passed"] = long.verdict.eq(PASS)
    long["success"] = long.success.astype(bool)
    return long[["task_id", "model", "signal", "passed", "success"]]


def confusion(cells: pd.DataFrame) -> dict:
    """Counts and rates on the signal cells of `cells` (verifier_cells rows).

    false-pass rate = #(PASS, not success) / #(not success);
    false-reject rate = #(FAIL, success) / #(success)."""
    s = cells[cells.signal]
    pass_ok = int((s.passed & s.success).sum())
    pass_bad = int((s.passed & ~s.success).sum())
    fail_ok = int((~s.passed & s.success).sum())
    fail_bad = int((~s.passed & ~s.success).sum())
    n_bad, n_ok = pass_bad + fail_bad, pass_ok + fail_ok
    return {"cells": len(cells), "signal_cells": len(s), "no_signal_cells": int((~cells.signal).sum()),
            "pass_success": pass_ok, "pass_failure": pass_bad, "fail_success": fail_ok,
            "fail_failure": fail_bad, "failures": n_bad, "successes": n_ok,
            "false_pass_rate": rate(pass_bad, n_bad), "false_reject_rate": rate(fail_ok, n_ok)}


def confusion_frame(cells: pd.DataFrame) -> pd.DataFrame:
    """Per signal cell, the indicator columns a cluster bootstrap of the two rates needs."""
    s = cells[cells.signal]
    return pd.DataFrame({"task_id": s.task_id.to_numpy(),
                         "fp_num": (s.passed & ~s.success).astype(float).to_numpy(),
                         "fp_den": (~s.success).astype(float).to_numpy(),
                         "fr_num": (~s.passed & s.success).astype(float).to_numpy(),
                         "fr_den": s.success.astype(float).to_numpy()})


def no_signal_share(cells: pd.DataFrame, tasks: Sequence[str]) -> dict:
    """Cells with no signal among available recorded cells, and tasks with no signal for any
    available model among `tasks` (a task with no available recorded cell counts as no signal)."""
    sig_tasks = set(cells.loc[cells.signal, "task_id"])
    n_tasks = len(tasks)
    n_ns = sum(t not in sig_tasks for t in tasks)
    return {"cells": len(cells), "no_signal_cells": int((~cells.signal).sum()),
            "no_signal_cell_share": rate(int((~cells.signal).sum()), len(cells)),
            "tasks": n_tasks, "no_signal_tasks": n_ns, "no_signal_task_share": rate(n_ns, n_tasks)}


# ------------------------------------------------------------------------------- S3.3 unsolved
def unsolved(ok: pd.DataFrame, available: pd.DataFrame) -> pd.Series:
    """Per task: no available model succeeds (`ok` is already False on unavailable cells)."""
    return ~(ok & available.astype(bool)).any(axis=1)


# --------------------------------------------------------------------------- S3.3 rescue pairs
def rescue_pairs(ok: pd.DataFrame, verdict: pd.DataFrame, order_price: Sequence[str],
                 order_cost: Sequence[str]) -> pd.DataFrame:
    """Spec §5.3 for every ordered pair (l, h) of distinct models, on the tasks of `ok` (pass a
    complete-case frame). `verdict` is PASS / FAIL / UNKNOWN per cell."""
    models = list(ok.columns)
    v = verdict.reindex(index=ok.index, columns=models)
    sig = v.isin([PASS, FAIL])
    vfail, vpass = v.eq(FAIL), v.eq(PASS)
    rp = {m: i for i, m in enumerate(order_price)}
    rc = {m: i for i, m in enumerate(order_cost)}
    rows = []
    for lo in models:
        for hi in models:
            if lo == hi:
                continue
            l_fail = ~ok[lo]
            n_fail = int(l_fail.sum())
            n_resc = int((l_fail & ok[hi]).sum())
            n_vfail = int(vfail[lo].sum())
            n_vfail_resc = int((vfail[lo] & ok[hi]).sum())
            both = vfail[lo] & sig[hi]
            n_both = int(both.sum())
            n_vv = int((both & vpass[hi]).sum())
            n_seen = int((vfail[lo] & l_fail).sum())
            rows.append({"lower": lo, "higher": hi,
                         "lower_cheaper_by_price": rp[lo] < rp[hi],
                         "lower_cheaper_by_cost": rc[lo] < rc[hi],
                         "tasks": len(ok), "lower_fails": n_fail, "true_rescues": n_resc,
                         "true_rescue_rate": rate(n_resc, n_fail),
                         "lower_signal": int(sig[lo].sum()), "lower_verdict_fail": n_vfail,
                         "observable_rescues": n_vfail_resc,
                         "observable_rescue_rate": rate(n_vfail_resc, n_vfail),
                         "both_signal_after_fail": n_both, "verifier_only_rescues": n_vv,
                         "verifier_only_rescue_rate": rate(n_vv, n_both),
                         "lower_fails_shown": n_seen, "observability": rate(n_seen, n_fail)})
    return pd.DataFrame(rows)


def rescue_frame(ok: pd.DataFrame, verdict: pd.DataFrame, lo: str, hi: str) -> pd.DataFrame:
    """Per task, the indicator columns a cluster bootstrap of one pair's true and observable
    rescue rates needs."""
    vfail = verdict.reindex(index=ok.index)[lo].eq(FAIL)
    return pd.DataFrame({"t_num": (~ok[lo] & ok[hi]).astype(float),
                         "t_den": (~ok[lo]).astype(float),
                         "o_num": (vfail & ok[hi]).astype(float),
                         "o_den": vfail.astype(float)}, index=ok.index)


# ---------------------------------------------------------------------- S3.3 historical ladder
def mean_cost_order(cost: pd.DataFrame, available: pd.DataFrame) -> list[str]:
    """Models by mean released cost over available cells, cheapest first (ties by name)."""
    mc = cost.where(available.astype(bool)).mean()
    return sorted(cost.columns, key=lambda m: (mc[m], m))


def ladder_costs(ok: pd.DataFrame, cost: pd.DataFrame, available: pd.DataFrame,
                 order: Sequence[str], verifier_usd: float = 0.0) -> pd.DataFrame:
    """Spec §5.4: a cheapest-first ladder over `order` with a perfect verifier.

    Per task: rungs reached, solved, `ladder` (every available rung up to and including the first
    success; all available rungs when none succeeds) and `perfect_stop` (as `ladder`, except an
    unsolved task pays only its first available rung). Unavailable rungs are skipped and never
    charged. `verifier_usd` is charged once per rung reached (0 = the old, uncharged figure)."""
    order = list(order)
    O = ok[order].to_numpy(bool)
    A = available.reindex(index=ok.index, columns=order).astype(bool).to_numpy()
    C = cost.reindex(index=ok.index, columns=order).to_numpy(float)
    O = O & A
    n = len(ok)
    solved = O.any(axis=1)
    first_ok = np.where(solved, O.argmax(axis=1), len(order) - 1)
    pos = np.arange(len(order))[None, :]
    reached = A & (pos <= first_ok[:, None])
    per_rung = np.where(reached, C + verifier_usd, 0.0)
    ladder = per_rung.sum(axis=1)
    has = A.any(axis=1)
    first_av = np.where(has, A.argmax(axis=1), 0)
    first_cost = np.where(has, C[np.arange(n), first_av] + verifier_usd, 0.0)
    stop = np.where(solved, ladder, first_cost)
    return pd.DataFrame({"rungs_reached": reached.sum(axis=1), "solved": solved,
                         "ladder": ladder, "perfect_stop": stop}, index=ok.index)


def ladder_summary(lc: pd.DataFrame, always: pd.Series) -> dict:
    """Totals of `ladder_costs`, the always-X cost beside them, and the share of the ladder's cost
    spent on unsolved tasks (RQ-B)."""
    n = len(lc)
    tot = float(lc.ladder.sum())
    uns = float(lc.ladder[~lc.solved].sum())
    return {"tasks": n, "unsolved": int((~lc.solved).sum()),
            "unsolved_share": rate(int((~lc.solved).sum()), n),
            "ladder_usd": tot, "always_usd": float(always.sum()),
            "perfect_stop_usd": float(lc.perfect_stop.sum()),
            "ladder_musd_per_task": 1e3 * tot / n if n else float("nan"),
            "always_musd_per_task": 1e3 * float(always.sum()) / n if n else float("nan"),
            "perfect_stop_musd_per_task": 1e3 * float(lc.perfect_stop.sum()) / n if n else float("nan"),
            "ladder_share_on_unsolved": rate(uns, tot) if tot else float("nan"),
            "mean_rungs_reached": float(lc.rungs_reached.mean()) if n else float("nan")}
