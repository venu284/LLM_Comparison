"""S1.7 checkpoint 2: summaries computed from the recorded cell rows (out/s1_7_cells_<part>.csv),
after every row is written. Interim for 2a and 2b: labels come after 2c (Amendment 2).

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_summary bcb py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "router/experiments/out"
Z = 1.959963984540054


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def load(parts) -> pd.DataFrame:
    return pd.concat([pd.read_csv(OUT / f"s1_7_cells_{p}.csv").assign(part=p) for p in parts],
                     ignore_index=True)


def executed(c: pd.DataFrame) -> pd.Series:
    """§3.1 n_exec: our run ended in pass or fail (an `our_harness` error or timeout excluded)."""
    return c.rerun.isin(["pass", "fail"])


def by_sample(c: pd.DataFrame) -> pd.DataFrame:
    g = c[c.key_role == "graded"]
    rows = []
    for (smp, ver), s in g.groupby(["sample", g.test_version.fillna("-")]):
        ex = s[executed(s)]
        k, n = int(ex.agree.astype(bool).sum()), len(ex)
        rows.append({"sample": smp, "test_version": ver, "drawn": len(s),
                     "no_response": int((~s.has_response.astype(bool)).sum()),
                     "no_tests": int((~s.has_tests.astype(bool)).sum()),
                     "harness_error_or_timeout": int(s.rerun.isin(["error", "timeout"]).sum()),
                     "n_exec": n, "agree": k, "a": round(k / n, 4) if n else None,
                     "a_ci": wilson(k, n),
                     "released_pass_rerun_fail": int((ex.released_pass & (ex.rerun == "fail")).sum()),
                     "released_fail_rerun_pass": int((~ex.released_pass.astype(bool) & (ex.rerun == "pass")).sum()),
                     "flaky": int(ex.flaky.astype(bool).sum())})
    return pd.DataFrame(rows)


def by_model(c: pd.DataFrame) -> pd.DataFrame:
    g = c[(c.key_role == "graded") & executed(c) & (c.test_version.fillna("plus") == "plus")]
    g = g[~g["sample"].str.startswith(("allzero", "identical", "cruxeval_0.8", "humaneval_version"))]
    rows = []
    for m, s in g.groupby("model"):
        k, n = int(s.agree.astype(bool).sum()), len(s)
        rows.append({"model": m, "unconfirmed": bool(s.unconfirmed.iloc[0]), "n_exec": n,
                     "agree": k, "a": round(k / n, 4), "a_ci": wilson(k, n)})
    return pd.DataFrame(rows)


def mapping_interim(bm: pd.DataFrame) -> dict:
    """Amendment 1, interim only (before mechanisms are assigned; the test runs after 2c)."""
    conf = bm[~bm.unconfirmed]
    a_c = conf.agree.sum() / conf.n_exec.sum()
    return {"a_C": round(float(a_c), 4),
            **{m: {"a_U": float(r.a), "n": int(r.n_exec), "gap": round(float(a_c - r.a), 4)}
               for r in bm[bm.unconfirmed].itertuples() for m in [r.model]}}


MAIN_SAMPLES = ("bcb", "bcb_v014", "bcb_algo2", "bcb_completion", "bcb_ds", "ds1000", "humaneval",
                "hex_python", "mbpp", "mbpp_completion", "lcb", "cruxeval", "hex_cpp", "hex_go",
                "hex_java", "hex_js", "multiple_rs", "multiple_ts", "swebench")


def capped_share(c: pd.DataFrame, stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Amendment 6: share of each main sample's graded cells whose stored text is exactly 2,000
    characters, with its Wilson interval; a lower bound >= 0.05 bars `cleared`."""
    if stats is None:
        from router.data import audit
        stats = audit.response_stats()
    n = stats.set_index(["task_id", "key"])["n_chars"]
    g = c[(c.key_role == "graded") & c["sample"].isin(MAIN_SAMPLES)
          & (c.test_version.fillna("plus") == "plus")].drop_duplicates(["sample", "task_id", "key"])
    rows = []
    for smp, s in g.groupby("sample"):
        capped = int(sum(n.get((t, k), -1) == 2000 for t, k in zip(s.task_id, s.key)))
        lo, hi = wilson(capped, len(s))
        rows.append({"sample": smp, "cells": len(s), "capped": capped,
                     "share": round(capped / len(s), 4), "wilson_lo": lo, "wilson_hi": hi,
                     "bars_cleared": bool(lo >= 0.05)})
    return pd.DataFrame(rows)


def main(parts) -> None:
    c = load(parts)
    smp = by_sample(c)
    bm = by_model(c)
    res = {"by_sample": smp.to_dict("records"), "by_model": bm.to_dict("records"),
           "mapping_interim": mapping_interim(bm)}
    name = "_".join(parts)
    (OUT / f"s1_7_summary_{name}.json").write_text(json.dumps(res, indent=2, default=str) + "\n")
    pd.set_option("display.width", 220)
    print(smp.to_string(index=False))
    print(bm.to_string(index=False))
    print(res["mapping_interim"])


if __name__ == "__main__":
    main(sys.argv[1:])
