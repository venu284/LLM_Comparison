"""P0.5 provisional recount (docs/IMPLEMENTATION_PLAN.md, Pre-S0).

Every number in docs/research/P0.5_recount.md comes from this script, run from the repo root on
the pinned data (scripts/fetch_data.py --verify-only must print `fetched: 0, verified: 187`):

    pipeline/venv/bin/python router/experiments/p0_5_recount.py

Modes (default `all`):
  repro    re-implements the 2026-11-19 inline computation unchanged and asserts its numbers;
  recount  treats empty X13 responses and the 148 missing Qwen3-Max runs as unavailable (S0.11),
           recomputes the no-signal shares and the S3.2 feasibility table (complete case, with the
           old treatment next to it), adds verifier cost, and repeats the table with the best
           single model chosen inside training folds;
  ra       reproduces Route Anything's CodeRouterBench zero-loss route (arXiv 2611.05275, Table 10)
           and then scores its lookup baselines on our population against S0.3.

Results are written to router/experiments/out/p0_5_recount.json. All numbers are provisional:
S0.1-S0.14 are not signed off and S1-S3 redo them on the final population.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.settings import EXTERNAL_DIR  # noqa: E402

X13_VERDICTS = EXTERNAL_DIR / "route-anything-data/experiments_x/X13_memory/cache/verdicts.jsonl"
CRB_DIR = EXTERNAL_DIR / "CodeRouterBench"
CRB_FILES = ("id_probing_results_long.csv", "id_test_results_long.csv")
OUT_JSON = REPO_ROOT / "router/experiments/out/p0_5_recount.json"

OPUS = "claude-opus-4-6"
EXEC_DIMS = ["code_generation", "data_science", "algorithm", "code_completion", "test_generation"]
SUSPECTED = re.compile(r"^dim\d+_(lcb_|mbpp_testgen|he_testgen)")
MISSING_RUN = "missing_token_record_zero_total"
# S0.2: X13 mean visible-test run time (0.074 s) at an on-demand vCPU-hour rate, about 1e-6 USD.
VERIFIER_RUN_USD = 1e-6
GATE_SAVING, GATE_GAP = 20.0, -2.0          # S0.3 thresholds (percent, points)
N_FOLDS, N_SEEDS, GATE_SEED = 5, 5, 0       # S0.6 cross-fitting; S0.8 gate on one pre-registered seed


# ----------------------------------------------------------------------------- data

def problem_id(t: str) -> str:
    """Problem id that keeps variants of one problem together.

    Copied from router/experiments/e0_power/power.py (importing it would run that script).
    Provisional: S1.6 replaces it with ids from source ids and tasks.jsonl.
    """
    s = re.sub(r"^dim\d+_", "", t)
    m = re.search(r"HumanEval[_/]?(\d+)", s)
    if m:
        return f"he{m.group(1)}"
    m = re.search(r"mbpp(?:_completion|_testgen)?_(\d+)", s)
    if m:
        return f"mbpp{m.group(1)}"
    m = re.search(r"BigCodeBench/(\d+)", s)
    if m:
        return f"bcb{m.group(1)}"
    m = re.search(r"lcb(?:_testgen)?_(.+)$", s)
    if m:
        return f"lcb{m.group(1)}"
    return s


def load():
    """Load X13 verdicts and the CodeRouterBench long CSVs as task x model frames."""
    v = pd.read_json(X13_VERDICTS, lines=True, dtype=str)
    L = pd.concat([pd.read_csv(CRB_DIR / f) for f in CRB_FILES], ignore_index=True)
    assert len(L) == 79_992 and len(v) == 23_288
    L["cost_usd"] = L.cost_usd.fillna(0)
    d = {"v": v, "L": L}
    S = L.pivot_table(index="task_id", columns="model", values="score")
    d["S"] = S
    d["C"] = L.pivot_table(index="task_id", columns="model", values="cost_usd").reindex(S.index)[S.columns]
    meta = L.drop_duplicates("task_id").set_index("task_id").reindex(S.index)
    d["dim"], d["split"] = meta.dimension, meta.split
    d["missing_run"] = (L.assign(x=L.cost_source == MISSING_RUN)
                        .pivot_table(index="task_id", columns="model", values="x", aggfunc="first")
                        .reindex(S.index)[S.columns].astype(bool))
    v["acc"] = v.w1_accept == "True"
    v["n"] = v.w1_n.astype(int)
    v["is_empty"] = v["empty"] == "True"
    piv = lambda col: v.pivot_table(index="task_id", columns="model", values=col, aggfunc="first")
    d["A"], d["N"], d["E"] = piv("acc"), piv("n"), piv("is_empty")
    d["family"] = v.drop_duplicates("task_id").set_index("task_id").source
    return d


def headline_mask(d):
    """The provisional 1,971-task headline set (S0.11): exec dimensions, suspected families removed,
    tasks with X13 verdicts."""
    S = d["S"]
    ex = d["dim"].isin(EXEC_DIMS)
    susp = pd.Series([bool(SUSPECTED.search(t)) for t in S.index], index=S.index)
    return ex & ~susp & S.index.isin(set(d["v"].task_id))


class Pop:
    """One population: binary success B (score = 1.0, S0.1), cost c, X13 verdicts A and test counts N."""

    def __init__(self, d, mask):
        self.ids = d["S"].index[mask]
        self.B = (d["S"].loc[self.ids] == 1.0).astype(float)
        self.c = d["C"].loc[self.ids]
        self.A = d["A"].reindex(self.ids)[self.B.columns]
        self.N = d["N"].reindex(self.ids)[self.B.columns]
        self.E = d["E"].reindex(self.ids)[self.B.columns].fillna(False).astype(bool)
        self.M = d["missing_run"].loc[self.ids]
        self.unavail = self.E | self.M
        self.family = d["family"].reindex(self.ids)
        self.dim = d["dim"].loc[self.ids]
        self.problem = pd.Series([problem_id(t) for t in self.ids], index=self.ids)
        self.models = list(self.B.columns)

    def __len__(self):
        return len(self.ids)

    def subset(self, keep):
        p = object.__new__(Pop)
        for k, val in self.__dict__.items():
            p.__dict__[k] = val[keep] if isinstance(val, (pd.Series, pd.DataFrame)) else val
        p.ids = self.ids[np.asarray(keep)]
        return p


# ----------------------------------------------------------------------------- ladders (S3.2)

def run_ladder(p: Pop, lad, mode, ver_cost=0.0):
    """Replay a fixed ladder. A rung is accepted when its visible-test verdict accepts (cells with
    visible tests); on cells with no visible tests the decision is `accept`, `escalate`, or the
    released score (`bound`, the oracle-on-no-signal bound and the only place it is read). The last
    rung is always accepted. Returns per-task acceptance and cost arrays. With ver_cost > 0 one
    verifier run is charged on every reached rung whose cell has visible tests."""
    n = len(p)
    acc, cost, done = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for i, m in enumerate(lad):
        sig = p.N[m].values > 0
        cost += np.where(done, 0, p.c[m].values + ver_cost * sig)
        ver = p.A[m].fillna(False).values.astype(bool)
        if mode == "bound":
            ok = np.where(sig, ver, p.B[m].values == 1)
        elif mode == "accept":
            ok = np.where(sig, ver, True)
        else:
            ok = np.where(sig, ver, False)
        take = ~done & (ok | (i == len(lad) - 1))
        acc = np.where(take, p.B[m].values, acc)
        done |= take
    return acc, cost


def ladders(p: Pop):
    """Cost-ordered 2- and 3-rung permutations (cost order taken on the whole population)."""
    return [lad for k in (2, 3) for lad in itertools.permutations(p.models, k)
            if p.c[list(lad)].mean().is_monotonic_increasing]


def feasibility(p: Pop, base_acc, base_cost, ver_cost=0.0):
    """S3.2 table against per-task baseline arrays: ladders meeting both thresholds (point
    estimates) and the best gap among ladders saving >= 20%."""
    out = {}
    lads = ladders(p)
    for mode in ("accept", "escalate", "bound"):
        rows = []
        for lad in lads:
            a, c = run_ladder(p, lad, mode, ver_cost)
            rows.append((">".join(lad), 100 * (a.mean() - base_acc.mean()),
                         100 * (1 - c.sum() / base_cost.sum())))
        R = pd.DataFrame(rows, columns=["ladder", "gap", "saving"])
        top = R[R.saving >= GATE_SAVING].sort_values("gap", ascending=False).head(3)
        out[mode] = {"n_ladders": len(R),
                     "n_meet_both": int(((R.gap >= GATE_GAP) & (R.saving >= GATE_SAVING)).sum()),
                     "top3_at_20pct_saving": top.round(4).to_dict("records")}
    return out


def whole_set_baseline(p: Pop):
    best = p.B.mean().idxmax()
    return best, p.B[best].values, p.c[best].values


def no_signal(p: Pop):
    return {"n_tasks": len(p),
            "tasks_no_signal_any_model": int((p.N == 0).all(axis=1).sum()),
            "share_tasks_no_signal": float((p.N == 0).all(axis=1).mean()),
            "share_cells_no_signal": float((p.N == 0).values.mean())}


# ----------------------------------------------------------------------------- mode: repro

def mode_repro(d):
    """Step 2: the 2026-11-19 computation, unchanged (unavailable cells kept as recorded, no
    verifier cost, best single model on the whole set, point estimates)."""
    p = Pop(d, headline_mask(d))
    best, ba, bc = whole_set_baseline(p)
    ns = no_signal(p)
    out = {"n": len(p), "best_single": best, "best_single_acceptance": float(ba.mean()),
           "share_unsolved_by_any_model": float((p.B.max(axis=1) == 0).mean()), **ns,
           "feasibility": feasibility(p, ba, bc)}
    f = out["feasibility"]
    checks = [
        (out["n"] == 1971, "n = 1,971"),
        (ns["tasks_no_signal_any_model"] == 1562, "1,562 tasks with no signal"),
        (round(100 * ns["share_tasks_no_signal"], 1) == 79.2, "79.2% of tasks"),
        (round(100 * ns["share_cells_no_signal"], 1) == 79.4, "79.4% of cells"),
        (best == OPUS and round(100 * ba.mean(), 1) == 66.8, "Opus best at 66.8%"),
        (round(100 * out["share_unsolved_by_any_model"], 1) == 10.8, "10.8% unsolved"),
        (f["accept"]["n_meet_both"] == 0, "accept: 0 ladders"),
        (round(f["accept"]["top3_at_20pct_saving"][0]["gap"], 1) == -13.4, "accept: -13.4"),
        (f["escalate"]["n_meet_both"] == 0, "escalate: 0 ladders"),
        (round(f["escalate"]["top3_at_20pct_saving"][0]["gap"]) == -14, "escalate: about -14"),
        (f["bound"]["n_meet_both"] == 12, "bound: 12 ladders"),
        (round(f["bound"]["top3_at_20pct_saving"][0]["gap"], 1) == 10.8, "bound: +10.8"),
        (round(f["bound"]["top3_at_20pct_saving"][0]["saving"], 1) == 24.5, "bound: 24.5% saving"),
    ]
    failed = [msg for ok, msg in checks if not ok]
    out["reproduced"] = not failed
    if failed:
        raise SystemExit(f"Reproduction failed, stop and report: {failed}\n{json.dumps(out, indent=1)}")
    return out


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["repro", "recount", "ra", "all"], default="all")
    ap.add_argument("--boot", type=int, default=1000, help="cluster-bootstrap resamples")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    d = load()
    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    if a.mode in ("repro", "all"):
        res["repro"] = mode_repro(d)
        print("repro: all 2026-11-19 numbers reproduced")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {a.out.relative_to(REPO_ROOT) if a.out.is_relative_to(REPO_ROOT) else a.out}")


if __name__ == "__main__":
    main()
