"""Claim-facing metrics on a replay trace (S2.3; CLAIMS S0.1, S0.2, S0.11, R3, R4, R15).

`task_outcomes` joins a `ReplayResult` (which holds no score) to the released scores of an
`OutcomeMatrix` once, after replay, and gives one row per task. Everything else works on that
frame:

- **acceptance** (S0.1): the submitted output's released score is a success under
  `scoring.success(score, family)`, family by family. `family` always comes from the matrix, so the
  CRUXEval sensitivity (`cruxeval_lenient=True`) can never fall back to the strict rule by
  accident. A give-up is not accepted;
- **cost** (S0.2): tokens x published price of every call (the released `cost_usd`, which equals
  tokens x the `models.json` prices on every token-priced row, S1.10), plus verifier runs at
  `VERIFIER_RUN_USD` (swept x1, x10, x100) **only where a verifier ran**, plus probe calls spread
  evenly over the tasks scored. Retries are not simulated (one run per cell) and count as 0. The
  price sweep scales one model's cost at a time (`PRICE_SWEEP`);
- **CPAT** only with an acceptance floor (R3): `cpat_at_floor(frame, floor)` has no default floor;
- **cost at matched acceptance** on a cost-acceptance curve;
- **regret** (our definition; the plan does not define it): against the cost-aware oracle, which
  for each task calls the cheapest available model that succeeds, and gives up at no cost when
  none does. Reported in acceptance points and in dollars; the dollar figure includes verifier and
  probe cost, and the probe count is reported beside it.

Unavailable cells are never scored as failures: `complete_case` drops every task with an
unavailable cell for every policy and reports the counts (gate scoring, S0.11, O2), and
`cascade_population` drops tasks with no X13 record (S0.11).

The paper's AvgPerf (mean released score, O10) is **not** here: it lives in `router/sim/paper.py`
and is used only to reproduce published numbers. These functions refuse the zero-filled paper view.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from router.data import scoring
from router.data import splits as splits_mod
from router.settings import VERIFIER_RUN_USD
from router.sim.stream import ReplayResult

OUR_DEFINITION = "our definition"


def _refuse_paper_view(m) -> None:
    if str(getattr(m, "name", "")).endswith("/paper"):
        raise ValueError("claim metrics never run on the zero-filled paper view; use sim.paper")


def task_outcomes(result: ReplayResult, m, *, cruxeval_lenient: bool = False,
                  verifier_usd: float = VERIFIER_RUN_USD, verifier_scale: float = 1.0,
                  price_scale: Mapping[str, float] | None = None, probe_usd: float = 0.0,
                  n_probes: int = 0) -> pd.DataFrame:
    """One row per replayed task: family, problem_id, action, final_model, accepted, cost parts
    and population flags (`complete_case`, `has_record`)."""
    _refuse_paper_view(m)
    t = result.tasks.copy()
    missing = t.index.difference(m.tasks)
    if len(missing):
        raise KeyError(f"{len(missing)} replayed tasks are not in the matrix")
    fam = m.family.reindex(t.index)
    if fam.isna().any():
        raise ValueError("every task needs a family (S0.1)")
    accept = t.action.eq("accept").to_numpy()
    score = np.zeros(len(t))
    rows = np.flatnonzero(accept)
    if len(rows):
        cols = m.score.columns.get_indexer(t.final_model.iloc[rows])
        score[rows] = m.score.loc[t.index[rows]].to_numpy()[np.arange(len(rows)), cols]
    ok = np.asarray(scoring.success(score, list(fam), cruxeval_lenient), dtype=bool)
    t["family"] = fam
    t["accepted"] = ok & accept

    scale = dict(price_scale or {})
    if set(scale) - set(m.models):
        raise ValueError(f"price_scale names unknown models {sorted(set(scale) - set(m.models))}")
    calls = result.calls
    factor = calls.model.map(lambda x: scale.get(x, 1.0)).astype(float)
    model_cost = (calls.model_cost_usd * factor).groupby(calls.task_id).sum()
    t["model_cost_usd"] = model_cost.reindex(t.index).fillna(0.0)
    t["verifier_usd"] = t.n_verifier_runs * verifier_usd * verifier_scale
    t["probe_usd"] = probe_usd / len(t) if len(t) else 0.0
    t["cost_usd"] = t.model_cost_usd + t.verifier_usd + t.probe_usd
    t.attrs.update(n_probes=int(n_probes), probe_usd=float(probe_usd), n_retries=result.n_retries,
                   verifier_mode=result.verifier_mode, sensitivity=result.sensitivity,
                   given_more_information=result.given_more_information,
                   cruxeval_lenient=cruxeval_lenient, verifier_scale=verifier_scale,
                   price_scale=scale)
    t["complete_case"] = m.available.loc[t.index].all(axis=1)
    problems = splits_mod.load_problems().set_index("task_id")["problem_id"]
    t["problem_id"] = problems.reindex(t.index)
    return t


def _drop(o: pd.DataFrame, keep: pd.Series, reason: str) -> tuple[pd.DataFrame, dict]:
    dropped = o[~keep]
    info = {"reason": reason, "n_kept": int(keep.sum()), "n_dropped": int((~keep).sum()),
            "dropped_by_family": dropped.family.value_counts().sort_index().to_dict()}
    out = o[keep].copy()
    out.attrs.update(o.attrs)
    # S0.2: probe cost is spread over the tasks scored, so re-spread it over the tasks kept.
    total = float(o.attrs.get("probe_usd", 0.0))
    out["probe_usd"] = total / len(out) if len(out) else 0.0
    out["cost_usd"] = out.model_cost_usd + out.verifier_usd + out.probe_usd
    return out, info


def complete_case(o: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tasks with no unavailable cell (gate scoring, S0.11, O2), and the counts dropped."""
    return _drop(o, o.complete_case.astype(bool), "unavailable cell")


def cascade_population(o: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tasks with an X13 record (cascade claims, S0.11), and the counts dropped."""
    return _drop(o, o.has_record.astype(bool), "no X13 record")


def acceptance(o: pd.DataFrame) -> float:
    """Acceptance rate under S0.1, in points."""
    return 100.0 * float(o["accepted"].mean())


def acceptance_by_family(o: pd.DataFrame) -> pd.DataFrame:
    g = o.groupby("family")["accepted"]
    return pd.DataFrame({"n": g.size(), "accepted": g.sum().astype(int),
                         "acceptance": 100.0 * g.mean()})


def cost_breakdown(o: pd.DataFrame) -> dict:
    """Totals of every S0.2 cost part, as counts and dollars."""
    return {"n_tasks": len(o), "model_usd": float(o.model_cost_usd.sum()),
            "n_calls": int(o.n_calls.sum()), "n_verifier_runs": int(o.n_verifier_runs.sum()),
            "verifier_usd": float(o.verifier_usd.sum()), "n_probes": o.attrs.get("n_probes", 0),
            "probe_usd": float(o.probe_usd.sum()), "n_retries": o.attrs.get("n_retries", 0),
            "total_usd": float(o.cost_usd.sum())}


def cpat_at_floor(o: pd.DataFrame, floor: float) -> dict:
    """Cost per accepted task, only when the acceptance rate (points) meets `floor` (R3).
    Below the floor the CPAT is NaN: an unconstrained CPAT is never quoted."""
    acc = acceptance(o)
    n_acc = int(o["accepted"].sum())
    meets = acc >= floor
    cpat = float(o.cost_usd.sum()) / n_acc if meets and n_acc else float("nan")
    return {"acceptance": acc, "floor": float(floor), "meets_floor": bool(meets), "cpat": cpat,
            "n_accepted": n_acc, "total_usd": float(o.cost_usd.sum())}


def cost_at_matched_acceptance(curve: pd.DataFrame, target: float) -> float:
    """Total cost at acceptance `target` (points) on a curve of operating points (columns
    `acceptance`, `cost`), by linear interpolation between neighbouring points after keeping the
    cheapest point per acceptance value. NaN outside the curve's range: never extrapolated."""
    c = curve.groupby("acceptance", as_index=False).cost.min().sort_values("acceptance")
    x, y = c.acceptance.to_numpy(float), c.cost.to_numpy(float)
    if len(x) == 0 or target < x[0] or target > x[-1]:
        return float("nan")
    return float(np.interp(target, x, y))


def oracle_outcomes(m, tasks, *, cruxeval_lenient: bool = False) -> pd.DataFrame:
    """The cost-aware oracle per task (our definition): the cheapest available model that
    succeeds under S0.1; if none does, it gives up at no cost."""
    _refuse_paper_view(m)
    tasks = pd.Index(tasks)
    fam = m.family.reindex(tasks)
    sc, co, av = (m.score.loc[tasks], m.cost.loc[tasks], m.available.loc[tasks].astype(bool))
    ok = pd.DataFrame({mod: scoring.success(sc[mod].to_numpy(), list(fam), cruxeval_lenient)
                       for mod in m.models}, index=tasks) & av
    masked = co.where(ok)
    accepted = ok.any(axis=1)
    cost = masked.min(axis=1).where(accepted, 0.0)
    choice = masked.fillna(np.inf).idxmin(axis=1).where(accepted, None)
    return pd.DataFrame({"accepted": accepted, "cost_usd": cost, "model": choice})


def regret(o: pd.DataFrame, m) -> dict:
    """Regret against the cost-aware oracle (our definition), on the tasks of `o`."""
    lenient = bool(o.attrs.get("cruxeval_lenient", False))
    orc = oracle_outcomes(m, o.index, cruxeval_lenient=lenient)
    return {"definition": OUR_DEFINITION,
            "acceptance": acceptance(o), "oracle_acceptance": 100.0 * float(orc["accepted"].mean()),
            "acceptance_regret_points": 100.0 * float(orc["accepted"].mean() - o["accepted"].mean()),
            "cost_usd": float(o.cost_usd.sum()), "oracle_cost_usd": float(orc.cost_usd.sum()),
            "cost_regret_usd": float(o.cost_usd.sum() - orc.cost_usd.sum()),
            "probe_usd": float(o.probe_usd.sum()), "n_probes": o.attrs.get("n_probes", 0)}


def paired_frame(a: pd.DataFrame, b: pd.DataFrame, names=("a", "b")) -> pd.DataFrame:
    """Two policies' outcomes on the same tasks, side by side, for a paired bootstrap:
    `accepted_<name>`, `cost_<name>`, plus `problem_id` and `family`."""
    if not a.index.sort_values().equals(b.index.sort_values()):
        raise ValueError("a paired comparison needs the same tasks on both sides")
    b = b.loc[a.index]
    out = pd.DataFrame({f"accepted_{names[0]}": a["accepted"].astype(float),
                        f"accepted_{names[1]}": b["accepted"].astype(float),
                        f"cost_{names[0]}": a.cost_usd, f"cost_{names[1]}": b.cost_usd,
                        "problem_id": a.problem_id, "family": a.family}, index=a.index)
    return out
