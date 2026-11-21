"""S1.7 checkpoint 2a: assign a mechanism (criteria §1.1) to every disagreeing cell, by fixed
evidence rules written after the cell rows were recorded, then compute the interim §3 numbers.

Rules (each with the evidence it rests on; see docs/research/S1.7_audit.md, checkpoint 2a):
  R-BCB-LIB   BigCodeBench, released fail, our run passes, task imports a library in SCI_LIBS:
              `toolchain`. Evidence: across all 1,097 BigCodeBench tasks the release passes 0.18%
              of cells on the 500 tasks importing these libraries against 41% on the rest, while
              the pinned evaluation image runs their canonical solutions.
  R-LCB-CALL  lcb, released fail, our run passes, and the `lcb_stdin` probe (every task run as a
              stdin program, LeetCode `Solution()` never called) reproduces the release:
              `harness_timeout` (call convention).
  R-DS-RAW    ds1000, released fail, our run passes, and the `ds_raw` probe (response executed
              with no extraction) reproduces the release: `extraction`.
  R-CX-LIT    cruxeval, released pass, our run fails with NameError/SyntaxError on the extracted
              literal (an unquoted string answer): `our_harness`.
  R-HE-TESTS  humaneval, released pass, our run fails under both HumanEval+ and base tests:
              `test_version` (the release's tests were neither documented suite).
  R-FLAKY     the three runs of the cell differ: `flaky`.
  otherwise   `unexplained`.
Mechanisms on our side or neutral (`our_harness`, `flaky`, `release_correct`, `mapping`) leave
the cell out of `d`; `our_harness` cells also leave `n_exec` (§3.1).

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_mechanisms
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.experiments.s1_7_cp2 import _bcb_tests  # noqa: E402
from router.experiments.s1_7_summary import wilson  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out"
SCI_LIBS = {"matplotlib", "sklearn", "seaborn", "scipy", "wordcloud"}
RELEASE_SIDE = {"extraction", "toolchain", "harness_timeout", "truncation", "test_version", "wrong_field"}
MAIN = ("bcb", "bcb_v014", "bcb_algo2", "bcb_completion", "bcb_ds", "ds1000", "humaneval",
        "hex_python", "mbpp", "mbpp_completion", "lcb", "cruxeval")


def _runs(part):
    return {json.loads(l)["run_id"]: json.loads(l) for l in open(OUT / f"s1_7_runs_{part}.jsonl")}


def assign() -> pd.DataFrame:
    c = pd.concat([pd.read_csv(OUT / "s1_7_cells_bcb.csv").assign(part="bcb"),
                   pd.read_csv(OUT / "s1_7_cells_py.csv").assign(part="py")], ignore_index=True)
    c = c[c.key_role == "graded"].copy()
    bcb = _bcb_tests()
    probes = {n: pd.read_csv(OUT / f"s1_7_probe_{n}.csv").drop_duplicates(["task_id", "key"])
              .set_index(["task_id", "key"])["probe_agrees_release"].to_dict()
              for n in ("lcb_stdin", "ds_raw")}
    runs = _runs("py")
    he_base = (c[(c.family == "humaneval") & (c.test_version == "base")]
               .drop_duplicates(["task_id", "key"]).set_index(["task_id", "key"])["rerun"].to_dict())

    def mech(r):
        if r.rerun not in ("pass", "fail") or r.agree is None or pd.isna(r.agree) or bool(r.agree):
            return None
        if bool(r.flaky):
            return "flaky"
        rel_fail_run_pass = (not r.released_pass) and r.rerun == "pass"
        if r.part == "bcb" and rel_fail_run_pass:
            libs = set(ast.literal_eval(bcb[r.upstream]["libs"]))
            return "toolchain" if libs & SCI_LIBS else "unexplained"
        if r.family == "lcb" and rel_fail_run_pass:
            return "harness_timeout" if bool(probes["lcb_stdin"].get((r.task_id, r.key), False)) else "unexplained"
        if r.family == "ds1000" and rel_fail_run_pass:
            return "extraction" if bool(probes["ds_raw"].get((r.task_id, r.key), False)) else "unexplained"
        if r.family == "cruxeval" and r.released_pass and r.rerun == "fail":
            err = ((runs.get(f"{r.task_id}|{r.key}|None|1") or {}).get("cell") or {}).get("err", "")
            return "our_harness" if ("NameError" in err or "SyntaxError" in err) else "unexplained"
        if r.family == "humaneval" and r.released_pass and r.rerun == "fail":
            return "test_version" if he_base.get((r.task_id, r.key)) == "fail" else "unexplained"
        return "unexplained"

    c["mechanism"] = [mech(r) for r in c.itertuples()]
    return c


def interim(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    g = c[c["sample"].isin(MAIN) & (c.test_version.fillna("plus") == "plus")]
    for fam, s in g.groupby("sample"):
        ok = s.rerun.isin(["pass", "fail"]) & (s.mechanism != "our_harness")
        n = int(ok.sum())
        agree = int((s[ok].agree.astype(bool)).sum())
        d = int(s[ok].mechanism.isin(RELEASE_SIDE).sum())
        u = int((s[ok].mechanism == "unexplained").sum())
        lo_r = wilson(d, n)[0]
        harness_err = int((~s.rerun.isin(["pass", "fail"]) | (s.mechanism == "our_harness")).sum())
        drawn = len(s)
        if harness_err > 0.10 * drawn:
            label = "undecided (harness, §3.4)"
        elif lo_r >= 0.05:
            label = "confirmed defect"
        elif n and agree / n >= 0.95 and u == 0:
            label = "cleared"
        else:
            label = "undecided"
        mech = s[ok & s.mechanism.notna()].mechanism.value_counts().to_dict()
        rows.append({"family": fam, "drawn": drawn, "n_exec": n, "agree": agree,
                     "a": round(agree / n, 4) if n else None, "d": d,
                     "r": round(d / n, 4) if n else None, "r_wilson_lo": lo_r, "u": u,
                     "our_harness_or_error": harness_err, "mechanisms": mech,
                     "interim_label": label})
    return pd.DataFrame(rows)


def main():
    c = assign()
    c.to_csv(OUT / "s1_7_mechanisms_2a.csv", index=False, lineterminator="\n")
    t = interim(c)
    t.to_csv(OUT / "s1_7_interim_2a.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    other = c[~c["sample"].isin(MAIN) & c.mechanism.notna()]
    print(other.groupby(["sample", "mechanism"]).size().to_string())


if __name__ == "__main__":
    main()
