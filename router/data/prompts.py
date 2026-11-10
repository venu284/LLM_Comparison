"""Rebuild prompt text for CodeRouterBench ID tasks, which ship without it (Report 8, R-a).

Each adapter maps a task id to the public source benchmark it came from. Adapters exist for five
families so far: DS-1000, CRUXEval, MBPP, HumanEval (also used for the HumanEval-X variants,
which share the problem statement) and BigCodeBench. Other families return None.

Source parquet files live in `external/sources/` (git-ignored):
  ds1000.parquet      xlangai/DS-1000 test
  cruxeval.parquet    cruxeval-org/cruxeval test
  mbpp_*.parquet      google-research-datasets/mbpp full (prompt, train, validation, test)
  humaneval.parquet   openai/openai_humaneval test
  bcb.parquet         bigcode/bigcodebench v0.1.4
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

from router.settings import EXTERNAL_DIR

SRC = EXTERNAL_DIR / "sources"


@lru_cache(maxsize=None)
def _tables(root: Path = SRC) -> dict[str, dict]:
    t: dict[str, dict] = {}
    ds = pd.read_parquet(root / "ds1000.parquet")
    t["ds1000"] = {i: p for i, p in enumerate(ds["prompt"])}
    cx = pd.read_parquet(root / "cruxeval.parquet")
    t["cruxeval"] = {int(r.id.split("_")[1]): f"{r.code}\n# input: {r.input}" for r in cx.itertuples()}
    mb = pd.concat([pd.read_parquet(root / f"mbpp_{s}.parquet") for s in ("prompt", "train", "validation", "test")])
    # Problem text only: the reference solution would leak the answer into the features.
    t["mbpp"] = {int(r.task_id): r.text for r in mb.itertuples()}
    he = pd.read_parquet(root / "humaneval.parquet")
    t["humaneval"] = {int(r.task_id.split("/")[1]): r.prompt for r in he.itertuples()}
    bcb = pd.read_parquet(root / "bcb.parquet")
    t["bcb"] = {int(r.task_id.split("/")[1]): r.complete_prompt for r in bcb.itertuples()}
    return t


def prompt_for(task_id: str) -> str | None:
    t = _tables()
    body = re.sub(r"^dim\d+_", "", task_id)
    he = re.search(r"HumanEval_(\d+)", body)      # multiple_rs_HumanEval_162_string_to_md5
    m = he or re.search(r"(\d+)$", body)
    if not m:
        return None
    n = int(m.group(1))
    if body.startswith("ds1000_"):
        return t["ds1000"].get(n)
    if body.startswith("cruxeval_"):
        return t["cruxeval"].get(n)
    if body.startswith("mbpp"):
        return t["mbpp"].get(n)
    if "HumanEval" in body or body.startswith("hex_"):
        return t["humaneval"].get(n)
    if body.startswith("bcb"):
        return t["bcb"].get(n)
    return None


def join(task_ids) -> pd.Series:
    """Prompt per task id; None where no adapter or no match."""
    return pd.Series({tid: prompt_for(tid) for tid in task_ids}, dtype=object)
