"""Real verifier verdicts for replay: Route Anything's X13 records (S1.11; CLAIMS S0.11, S0.13).

Source: route-anything-data `experiments_x/X13_memory/cache/verdicts.jsonl`, 23,288 task x model
rows on 2,911 CodeRouterBench ID tasks (both splits). X13 carries no split; it is joined here
straight from the CodeRouterBench release CSVs.

Fields in the X13 file (route-anything-data README, "Memory verdicts"):

- `w1_*` is the visible-test verifier. `w1_n` visible tests exist for the task, `w1_p` of them
  passed, `w1_sec` is the run time. Where `w1_n > 0`, `w1_accept` is exactly `w1_p == w1_n`.
  Where `w1_n == 0`, `w1_accept` is Route Anything's syntax/quality heuristic (`w1_syntax`,
  `w1_quality`, and the `w1b_*` tool score). CLAIMS S0.11 excludes that heuristic from the main
  path; it is exposed only by `load_heuristic_accept()` for the sensitivity check.
- `w2_*` is the hidden-test run. The README calls these "hidden-test outcomes". Route Anything
  (arXiv 2611.05275 v1, App. C.4, Table 14) uses hidden tests as "an oracle-like control" verifier
  (a sandbox run of about 0.15 s), and App. C.6 reports that hidden-test writes have TPR/FPR
  0.960/0.205 against the released score. Neither the paper nor the README defines the `w2_kind`
  categories (passed, assertion, extraction_failure, harness_mismatch, empty_response,
  runtime_error, timeout, missing_lib). X13 stores no error text. Route Anything's X7b attempt
  records (`experiments_x/X7b_inloop_live/attempts.jsonl`) use the same `failure_kind` labels with
  the error string: there, every `harness_mismatch` is a `NameError` for the entry-point function
  the test harness calls (the response defined a differently named function), `missing_lib` is a
  `ModuleNotFoundError`, `extraction_failure` a `SyntaxError` in the extracted code. That reading
  of X13's labels is inferred from X7b, not documented. In X13, `empty_response` coincides exactly
  with `empty`. CLAIMS S0.1 makes the hidden-test outcome a sensitivity check only: it is never
  ground truth. Exposed only by `load_hidden_tests()`.
- `empty` marks a cell with no model response: the model is unavailable for that task (S0.11).
- `truth` is the released CodeRouterBench score. The router never sees it (S0.13, R16): it is not in
  `load_verdicts()`, and only `truth_matches_release()` and `hidden_vs_release()` read it.

Observations (S0.13) in `load_verdicts()`:

- PASS / FAIL where the cell has visible tests (`signal`), from `visible_pass`;
- UNKNOWN where it has none: never PASS, never a training label;
- missing (None) where the cell is unavailable: the rung is skipped and the cell is not scored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from router.settings import EXTERNAL_DIR

X13_VERDICTS = EXTERNAL_DIR / "route-anything-data/experiments_x/X13_memory/cache/verdicts.jsonl"
CRB_DIR = EXTERNAL_DIR / "CodeRouterBench"
SPLIT_FILES = {"probing": "id_probing_results_long.csv", "id_test": "id_test_results_long.csv"}

PASS, FAIL, UNKNOWN = "PASS", "FAIL", "UNKNOWN"
KEY = ["task_id", "model"]


def load_raw(path: Path = X13_VERDICTS) -> pd.DataFrame:
    """The X13 file as released, every field a string."""
    return pd.read_json(path, lines=True, dtype=str)


def task_split(root: Path = CRB_DIR) -> pd.Series:
    """CodeRouterBench split per task id, read from the release CSVs (not from A1's loader)."""
    parts = []
    for split, f in SPLIT_FILES.items():
        ids = pd.read_csv(root / f, usecols=["task_id"]).task_id.drop_duplicates()
        parts.append(pd.Series(split, index=ids.values))
    s = pd.concat(parts)
    if s.index.duplicated().any():
        raise ValueError("a task id appears in both CodeRouterBench splits")
    return s


def _flag(col: pd.Series) -> pd.Series:
    return col.astype(str).eq("True")


def load_verdicts(path: Path = X13_VERDICTS, root: Path = CRB_DIR) -> pd.DataFrame:
    """One row per (task_id, model): the router-side view of the X13 visible-test verifier.

    Columns: task_id, model, split, signal, visible_pass, n_visible_tests, unavailable,
    verifier_seconds, observation. No released score, no heuristic, no hidden-test field.
    """
    raw = load_raw(path)
    n = raw.w1_n.astype(int)
    p = raw.w1_p.astype(int)
    signal = n > 0
    unavailable = _flag(raw["empty"])
    visible_pass = pd.Series(np.where(signal, p == n, None), index=raw.index, dtype=object)
    observation = pd.Series(np.where(p == n, PASS, FAIL), index=raw.index, dtype=object)
    observation[~signal] = UNKNOWN
    observation[unavailable] = None
    split = task_split(root)
    out = pd.DataFrame({
        "task_id": raw.task_id,
        "model": raw.model,
        "split": raw.task_id.map(split),
        "signal": signal,
        "visible_pass": visible_pass,
        "n_visible_tests": n,
        "unavailable": unavailable,
        "verifier_seconds": raw.w1_sec.astype(float),
        "observation": observation,
    })
    if out.split.isna().any():
        raise ValueError(f"{int(out.split.isna().sum())} X13 rows do not join to CodeRouterBench")
    if out.duplicated(KEY).any():
        raise ValueError("duplicate (task_id, model) rows in X13")
    return out


def load_heuristic_accept(path: Path = X13_VERDICTS) -> pd.DataFrame:
    """SENSITIVITY ONLY (S0.11): Route Anything's `w1_accept`, which on no-signal cells is its
    syntax/quality heuristic. Never used on the main path."""
    raw = load_raw(path)
    return pd.DataFrame({"task_id": raw.task_id, "model": raw.model,
                         "heuristic_accept": _flag(raw.w1_accept),
                         "w1_syntax": _flag(raw.w1_syntax),
                         "w1_quality": pd.to_numeric(raw.w1_quality, errors="coerce")})


def load_hidden_tests(path: Path = X13_VERDICTS) -> pd.DataFrame:
    """SENSITIVITY ONLY (S0.1): the X13 hidden-test (`w2_*`) outcomes. Not ground truth."""
    raw = load_raw(path)
    return pd.DataFrame({"task_id": raw.task_id, "model": raw.model,
                         "w2_pass": _flag(raw.w2_pass),
                         "w2_frac": pd.to_numeric(raw.w2_frac, errors="coerce"),
                         "w2_sec": pd.to_numeric(raw.w2_sec, errors="coerce"),
                         "w2_kind": raw.w2_kind})


def compare_with_p0_5(v: pd.DataFrame, d: dict, ids) -> dict:
    """Mismatches between `load_verdicts()` and P0.5's pivots A (w1_accept), N (w1_n) and E (empty)
    on the tasks `ids`, over every signal cell and every empty cell."""
    sub = v[v.task_id.isin(set(ids))]
    checked = sub[sub.signal | sub.unavailable]

    def p05(name):
        s = d[name].stack(future_stack=True).rename(name)
        s.index.names = KEY
        return s

    j = checked.set_index(KEY).join([p05("A"), p05("N"), p05("E")], how="left")
    sig = j[j.signal]
    return {
        "n_tasks": int(sub.task_id.nunique()),
        "n_signal_cells": int(j.signal.sum()),
        "n_empty_cells": int(j.unavailable.sum()),
        "A_mismatch": int((sig.A.astype(bool) != sig.visible_pass.astype(bool)).sum()
                          + sig.A.isna().sum()),
        "N_mismatch": int((j.N.astype(float) != j.n_visible_tests.astype(float)).sum()),
        "E_mismatch": int((j.E.astype(bool) != j.unavailable).sum() + j.E.isna().sum()),
    }


# ---------------------------------------------------------------------------------------------
# Checks that read the released score. Never call these from router code (S0.13, R16).
# ---------------------------------------------------------------------------------------------

def _release_scores(root: Path = CRB_DIR) -> pd.DataFrame:
    cols = ["task_id", "model", "score"]
    return pd.concat([pd.read_csv(root / f, usecols=cols) for f in SPLIT_FILES.values()],
                     ignore_index=True)


def truth_matches_release(path: Path = X13_VERDICTS, root: Path = CRB_DIR) -> dict:
    """CHECK ONLY: does X13 `truth` equal the CodeRouterBench released score on every joined cell?"""
    raw = load_raw(path)
    x = pd.DataFrame({"task_id": raw.task_id, "model": raw.model,
                      "truth": pd.to_numeric(raw.truth, errors="coerce"),
                      "empty": _flag(raw["empty"])})
    j = x.merge(_release_scores(root), on=KEY, how="left", validate="one_to_one")
    joined = j.score.notna()
    diff = joined & ~np.isclose(j.truth, j.score, rtol=0, atol=1e-9)
    succ_diff = joined & ((j.truth == 1.0) != (j.score == 1.0))
    return {
        "n_rows": len(j),
        "n_joined": int(joined.sum()),
        "n_truth_missing": int(j.truth.isna().sum()),
        "n_mismatch": int(diff.sum()),
        "n_mismatch_success": int(succ_diff.sum()),
        "n_mismatch_on_empty": int((diff & j["empty"]).sum()),
        "examples": j[diff].head(10)[KEY + ["truth", "score", "empty"]].to_dict("records"),
    }


def hidden_vs_release(path: Path = X13_VERDICTS, root: Path = CRB_DIR) -> pd.DataFrame:
    """CHECK ONLY: hidden-test kind x released success (score = 1.0), counts of cells."""
    h = load_hidden_tests(path).merge(_release_scores(root), on=KEY, how="left")
    h["release_success"] = h.score == 1.0
    return pd.crosstab(h.w2_kind, h.release_success, margins=True)
