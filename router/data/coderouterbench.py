"""Loaders for the public CodeRouterBench release (HF `Lance1573/CodeRouterBench`, rev 2026-06-23).

Two views of the in-distribution data:

- `complete`: the release CSVs, where every task has a score for all eight models;
- `paper`: the uncleaned `obs_matrix.json` scores with missing cells scored 0 and costs from the
  release CSVs. This is the view the paper's Table 3 AvgPerf numbers match (Report 8, R-c).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from router.data.schema import MISSING_TOKEN_RECORD, OutcomeMatrix, long_frame
from router.data.scoring import annotate
from router.data.sources import family_of
from router.settings import EXTERNAL_DIR

CRB_DIR = EXTERNAL_DIR / "CodeRouterBench"
# Paths inside the repo mirror that scripts/fetch_data.py writes to external/CodeRouterBench/.
OBS_MATRIX = "raw_matrices/phase1_acrouter_v2/obs_matrix.json"
DECISIONS = "outputs/current/id_decisions.jsonl"
SPLIT_FILES = {"probing": "id_probing_results_long.csv", "id_test": "id_test_results_long.csv"}
VIEWS = ("complete", "paper")
MISSING_TOKEN_COST_SOURCE = "missing_token_record_zero_total"


def source_of(task_id: str) -> str:
    """Deprecated: use `router.data.sources.family_of`. Kept for the quick-check callers.

    The S1.1 family of a task id, e.g. dim1_mbpp_631 -> mbpp.
    """
    return family_of(task_id)


def load_long(split: str, root: Path = CRB_DIR) -> pd.DataFrame:
    return pd.read_csv(root / SPLIT_FILES[split])


def long_table(split: str, view: str = "complete", root: Path = CRB_DIR) -> pd.DataFrame:
    """One split in the S1.2 long format (`schema.LONG_COLUMNS`), one run (run 0) per cell.

    `complete` keeps the release CSV scores; `paper` takes `obs_matrix.json` scores with missing
    cells scored 0, as the paper's Table 3 does. Costs come from the release CSVs in both views.
    The rows with `cost_source == "missing_token_record_zero_total"` are flagged unavailable
    (CLAIMS S0.11) in both views; their score is left as released.
    """
    if view not in VIEWS:
        raise ValueError(view)
    raw = load_long(split, root)
    long = pd.DataFrame({
        "task_id": raw["task_id"], "model": raw["model"], "run": 0,
        "score": raw["score"].astype(float), "cost_usd": raw["cost_usd"].astype(float),
        "input_tokens": raw["input_tokens"], "output_tokens": raw["output_tokens"],
        "dimension": raw["dimension"], "split": split,
    })
    long["family"] = long["task_id"].map(family_of)
    missing = (raw["cost_source"] == MISSING_TOKEN_COST_SOURCE).to_numpy()
    long["available"] = ~missing
    long["unavailable_reason"] = pd.Series(None, index=long.index, dtype=object).mask(
        missing, MISSING_TOKEN_RECORD)
    if view == "paper":
        obs = json.load(open(root / OBS_MATRIX))
        perf = {(t, m): v["perf"] for t, mm in obs.items() for m, v in mm.items()
                if v.get("perf") is not None}
        long["score"] = [float(perf.get(k, 0.0)) for k in zip(long["task_id"], long["model"])]
    # S1.4: score_type and eligible come from the family rule only (R16).
    return long_frame(annotate(long))


def load(split: str, view: str = "complete", root: Path = CRB_DIR) -> OutcomeMatrix:
    return OutcomeMatrix.from_long(long_table(split, view, root), name=f"crb/{split}/{view}")


def missing_cells(split: str, root: Path = CRB_DIR) -> int:
    obs = json.load(open(root / OBS_MATRIX))
    tasks = set(load_long(split, root).task_id)
    n_models = 8
    return sum(n_models - len(v) for k, v in obs.items() if k in tasks)


def released_acrouter_decisions(root: Path = CRB_DIR) -> pd.Series:
    d = pd.read_json(root / DECISIONS, lines=True)
    return d.set_index("task_id")["chosen_model"]
