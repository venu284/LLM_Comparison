"""Loaders for the public CodeRouterBench release (HF `Lance1573/CodeRouterBench`, rev 2026-06-23).

Two views of the in-distribution data:

- `complete`: the release CSVs, where every task has a score for all eight models;
- `paper`: the uncleaned `obs_matrix.json` scores with missing cells scored 0 and costs from the
  release CSVs. This is the view the paper's Table 3 AvgPerf numbers match (Report 8, R-c).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from router.data.schema import OutcomeMatrix
from router.settings import EXTERNAL_DIR

CRB_DIR = EXTERNAL_DIR / "CodeRouterBench"
SPLIT_FILES = {"probing": "id_probing_results_long.csv", "id_test": "id_test_results_long.csv"}


def source_of(task_id: str) -> str:
    """Source benchmark family from an ID task id, e.g. dim1_mbpp_631 -> mbpp."""
    s = re.sub(r"^dim\d+_", "", task_id)
    if s.startswith("swebench_"):
        return "swebench"
    s = re.sub(r"/\d+$", "", s)          # BigCodeBench/767
    s = re.sub(r"_\d+$", "", s)
    if s.startswith("lcb_") and not s.startswith("lcb_testgen"):
        s = "lcb"
    if s.startswith("bcb"):
        s = s.replace("_BigCodeBench", "")
    return s


def load_long(split: str, root: Path = CRB_DIR) -> pd.DataFrame:
    return pd.read_csv(root / SPLIT_FILES[split])


def load(split: str, view: str = "complete", root: Path = CRB_DIR) -> OutcomeMatrix:
    if view not in ("complete", "paper"):
        raise ValueError(view)
    long = load_long(split, root)
    long["cost_usd"] = long["cost_usd"].fillna(0.0)
    score = long.pivot_table(index="task_id", columns="model", values="score").sort_index()
    cost = long.pivot_table(index="task_id", columns="model", values="cost_usd").reindex(score.index)
    cost = cost[score.columns]
    if view == "paper":
        obs = json.load(open(root / "obs_matrix.json"))
        cells = pd.DataFrame([(t, m, v.get("perf")) for t, mm in obs.items() for m, v in mm.items()],
                             columns=["task_id", "model", "perf"]).dropna()
        paper = cells.pivot_table(index="task_id", columns="model", values="perf")
        score = paper.reindex(index=score.index, columns=score.columns).fillna(0.0)
    meta = long.drop_duplicates("task_id").set_index("task_id")
    dimension = meta["dimension"]
    source = pd.Series({t: source_of(t) for t in score.index})
    return OutcomeMatrix(score, cost, dimension, source, name=f"crb/{split}/{view}")


def missing_cells(split: str, root: Path = CRB_DIR) -> int:
    obs = json.load(open(root / "obs_matrix.json"))
    tasks = set(load_long(split, root).task_id)
    n_models = 8
    return sum(n_models - len(v) for k, v in obs.items() if k in tasks)


def released_acrouter_decisions(root: Path = CRB_DIR) -> pd.Series:
    d = pd.read_json(root / "id_decisions.jsonl", lines=True)
    return d.set_index("task_id")["chosen_model"]
