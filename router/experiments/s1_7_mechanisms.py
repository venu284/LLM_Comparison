"""S1.7 checkpoint 2a: assign a mechanism (criteria §1.1) to every disagreeing cell, by fixed
evidence rules written after the cell rows were recorded, then compute the interim §3 numbers.

Rules (each with the evidence it rests on; see docs/research/S1.7_audit.md, checkpoint 2a):
  R-BCB-LIB   BigCodeBench, released fail, our run passes, and the `block_sci` probe (the task's
              own code cannot import matplotlib, sklearn, seaborn, scipy or wordcloud) reproduces
              the release: `toolchain`. Supporting population pattern: the release passes 0.18% of
              cells on the 500 tasks importing these libraries against 41% on the rest. The
              library list was chosen from those release pass shares.
  R-LCB-CALL  lcb, released fail, our run passes, and the `lcb_stdin` probe (every task run as a
              stdin program, LeetCode `Solution()` never called) reproduces the release:
              `harness_timeout` (call convention).
  R-DS-RAW    ds1000, released fail, our run passes, and the `ds_raw` probe (response executed
              with no extraction) reproduces the release: `extraction`.
  R-CX-LIT    cruxeval, released pass, our run fails with NameError/SyntaxError on the extracted
              literal (an unquoted string answer): `our_harness`.
  R-HE-TESTS  humaneval, released pass, our run fails under both HumanEval+ and base tests:
              `test_version` (the release's tests were neither documented suite).
  R-MX-NONEXEC  hex_cpp/go/java/js (2b), released fail with a released score strictly between 0
              and 1, our run passes: `wrong_field` (the released value is not an execution
              result: the documented harness gives pass/fail only; 1,286-1,306 of each family's
              1,312 released cells are fractional, and in the 2b samples the fractions rank with
              text similarity to the canonical solution, Spearman 0.31-0.83 for cpp/go/js).
  R-MX-CONST  multiple_rs/multiple_ts (2b), every released cell of the family is 0 (1,248 and
              1,080 cells, all models), our run passes: `wrong_field` (a constant independent of
              the response).
  R-FLAKY     the three runs of the cell differ: `flaky`.
  otherwise   `unexplained`.
Then criteria Amendment 11 (s1_7_entry_point): `extraction` with `mechanism_detail`
`entry_point_unstated` on cells meeting its conditions (i)-(v), including cells where our re-run
agrees with the release. `main` also writes the Amendment 11 funnel and before/after table and
recomputes Amendment 1's mapping test.
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
        "hex_python", "mbpp", "mbpp_completion", "lcb", "cruxeval",
        "hex_cpp", "hex_go", "hex_java", "hex_js", "multiple_rs", "multiple_ts", "lcb_kept", "swebench")
MX_HEX = ("hex_cpp", "hex_go", "hex_java", "hex_js")
MX_CONST = ("multiple_rs", "multiple_ts")


def _runs(part):
    return {json.loads(l)["run_id"]: json.loads(l) for l in open(OUT / f"s1_7_runs_{part}.jsonl")}


def assign(apply_d: bool = True, amend11: bool = True) -> pd.DataFrame:
    parts = [("bcb", "s1_7_cells_bcb.csv"), ("py", "s1_7_cells_py.csv"), ("mx", "s1_7_cells_mx.csv"),
             ("lcbk", "s1_7_cells_lcbk.csv"), ("swe", "s1_7_cells_swe.csv")]
    c = pd.concat([pd.read_csv(OUT / f).assign(part=p) for p, f in parts if (OUT / f).exists()],
                  ignore_index=True)
    c = c[c.key_role == "graded"].copy()
    bcb = _bcb_tests()
    probes = {n: pd.read_csv(OUT / f"s1_7_probe_{n}.csv").drop_duplicates(["task_id", "key"])
              .set_index(["task_id", "key"])["probe_agrees_release"].to_dict()
              for n in ("lcb_stdin", "ds_raw", "block_sci")}
    runs = _runs("py")
    if (OUT / "s1_7_runs_lcbk.jsonl").exists():
        runs.update(_runs("lcbk"))
    he_base = (c[(c.family == "humaneval") & (c.test_version == "base")]
               .drop_duplicates(["task_id", "key"]).set_index(["task_id", "key"])["rerun"].to_dict())

    from router.data import audit
    long = audit.released_long()
    const_zero = {f: bool((long[(long.family == f) & long.available].score == 0).all()) for f in MX_CONST}

    def mech(r):
        if r.rerun not in ("pass", "fail") or r.agree is None or pd.isna(r.agree) or bool(r.agree):
            return None
        if bool(r.flaky):
            return "flaky"
        rel_fail_run_pass = (not r.released_pass) and r.rerun == "pass"
        if r.part == "bcb" and rel_fail_run_pass:
            return "toolchain" if bool(probes["block_sci"].get((r.task_id, r.key), False)) else "unexplained"
        if r.family == "lcb" and rel_fail_run_pass:
            return "harness_timeout" if bool(probes["lcb_stdin"].get((r.task_id, r.key), False)) else "unexplained"
        if r.family == "ds1000" and rel_fail_run_pass:
            return "extraction" if bool(probes["ds_raw"].get((r.task_id, r.key), False)) else "unexplained"
        if r.family == "cruxeval" and r.released_pass and r.rerun == "fail":
            err = ((runs.get(f"{r.task_id}|{r.key}|None|1") or {}).get("cell") or {}).get("err", "")
            return "our_harness" if ("NameError" in err or "SyntaxError" in err) else "unexplained"
        if r.family in MX_HEX and rel_fail_run_pass and 0 < r.released_score < 1:
            return "wrong_field"
        if r.family in MX_CONST and rel_fail_run_pass and const_zero.get(r.family, False):
            return "wrong_field"
        if r.family == "humaneval" and r.released_pass and r.rerun == "fail":
            return "test_version" if he_base.get((r.task_id, r.key)) == "fail" else "unexplained"
        return "unexplained"

    c["mechanism"] = [mech(r) for r in c.itertuples()]
    d_file = OUT / "s1_7_d_reassignments.csv"
    if apply_d and d_file.exists():          # decision D (s1_7_decision_d): unexplained cells only
        d = pd.read_csv(d_file)
        d = d[d["to"] != "unexplained"].set_index(["sample", "task_id", "key"])["to"].to_dict()
        c["mechanism"] = [d.get((r.sample, r.task_id, r.key), r.mechanism) if r.mechanism == "unexplained"
                          else r.mechanism for r in c.itertuples()]
    c["mechanism_detail"] = None
    if amend11:
        c = apply_amendment_11(c)
    return c


def apply_amendment_11(c: pd.DataFrame) -> pd.DataFrame:
    """Criteria Amendment 11: `extraction` / `entry_point_unstated` on cells meeting (i)-(v),
    including cells where our re-run agrees with the release. Such a cell is always a released
    fail whose re-run also failed, so no earlier rule assigned it a mechanism; that is checked."""
    from router.experiments import s1_7_entry_point as ep
    c = ep.apply(c)
    hit = c.a11 == ep.SUBTYPE
    clash = c[hit & c.mechanism.notna()]
    if len(clash):
        raise ValueError(f"Amendment 11 would replace an earlier mechanism on {len(clash)} cells")
    c.loc[hit, "mechanism"] = ep.MECHANISM
    c.loc[hit, "mechanism_detail"] = ep.SUBTYPE
    return c


_NCHARS = None


def _capped(s: pd.DataFrame) -> tuple[int, float]:
    """Amendment 6: capped cells (stored text exactly 2,000 characters) and the Wilson lower
    bound of their share among the drawn cells."""
    global _NCHARS
    if _NCHARS is None:
        from router.data import audit
        _NCHARS = audit.response_stats().set_index(["task_id", "key"])["n_chars"]
    k = int(sum(_NCHARS.get((t, key), -1) == 2000 for t, key in zip(s.task_id, s.key)))
    return k, wilson(k, len(s))[0]


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
        capped, cap_lo = _capped(s)
        if harness_err > 0.10 * drawn:
            label = "undecided (harness, §3.4)"
        elif lo_r >= 0.05:
            label = "confirmed defect"
        elif n and agree / n >= 0.95 and u == 0:
            label = "cleared" if cap_lo < 0.05 else "undecided (capped, Amendment 6)"
        else:
            label = "undecided"
        mech = s[ok & s.mechanism.notna()].mechanism.value_counts().to_dict()
        rows.append({"family": fam, "drawn": drawn, "n_exec": n, "agree": agree,
                     "a": round(agree / n, 4) if n else None, "d": d,
                     "r": round(d / n, 4) if n else None, "r_wilson_lo": lo_r, "u": u,
                     "our_harness_or_error": harness_err, "capped": capped, "capped_wilson_lo": cap_lo,
                     "mechanisms": mech,
                     "interim_label": label})
    return pd.DataFrame(rows)


def main():
    c = assign()
    a2 = c[c.part.isin(["bcb", "py"])]
    a2.to_csv(OUT / "s1_7_mechanisms_2a.csv", index=False, lineterminator="\n")
    interim(a2).to_csv(OUT / "s1_7_interim_2a.csv", index=False, lineterminator="\n")
    b = c[c.part == "mx"]
    if len(b):
        b.to_csv(OUT / "s1_7_mechanisms_2b.csv", index=False, lineterminator="\n")
        interim(b).to_csv(OUT / "s1_7_interim_2b.csv", index=False, lineterminator="\n")
    k = c[c.part == "lcbk"]
    if len(k):
        k.to_csv(OUT / "s1_7_mechanisms_lcbk.csv", index=False, lineterminator="\n")
        interim(k).to_csv(OUT / "s1_7_interim_lcbk.csv", index=False, lineterminator="\n")
    w = c[c.part == "swe"]
    if len(w):
        w.to_csv(OUT / "s1_7_mechanisms_2c.csv", index=False, lineterminator="\n")
        interim(w).to_csv(OUT / "s1_7_interim_2c.csv", index=False, lineterminator="\n")
    t = interim(c)
    pd.set_option("display.width", 250)
    print(t.to_string(index=False))
    other = c[~c["sample"].isin(MAIN) & c.mechanism.notna()]
    print(other.groupby(["sample", "mechanism"]).size().to_string())

    # Amendment 11 reporting: funnel per sample, before/after per family, mapping test
    from router.experiments import s1_7_entry_point as ep
    fun = ep.funnel(c, list(MAIN) + ["allzero_mbpp"])
    fun.to_csv(OUT / "s1_7_amend11_funnel.csv", index=False, lineterminator="\n")
    ba = before_after(c)
    ba.to_csv(OUT / "s1_7_amend11_labels.csv", index=False, lineterminator="\n")
    mt = {"after": mapping_test(c), "before": mapping_test(without_amendment_11(c))}
    (OUT / "s1_7_mapping_test.json").write_text(json.dumps(mt["after"], indent=2) + "\n")
    print(fun.to_string(index=False))
    print(ba.to_string(index=False))
    print(json.dumps(mt, indent=1))


def without_amendment_11(c: pd.DataFrame) -> pd.DataFrame:
    """The cell rows as they were before Amendment 11 (its cells had no mechanism)."""
    c = c.copy()
    hit = c.mechanism_detail == "entry_point_unstated"
    c.loc[hit, "mechanism"] = None
    c.loc[hit, "mechanism_detail"] = None
    return c


def before_after(c: pd.DataFrame) -> pd.DataFrame:
    """Per main sample: the §3 quantities and label without and with Amendment 11."""
    keys = ["n_exec", "a", "d", "r", "r_wilson_lo", "u", "interim_label"]
    old = interim(without_amendment_11(c)).set_index("family")[keys]
    new = interim(c).set_index("family")[keys]
    out = old.add_suffix("_before").join(new.add_suffix("_after"))
    out["changed"] = [bool((old.loc[f] != new.loc[f]).any()) for f in out.index]
    return out.reset_index()


MAPPING_SAMPLES = tuple(s for s in MAIN if s != "lcb_kept")     # streams 100-116, 130, 131
MAPPING_GAP, MAPPING_MIN_CELLS = 0.10, 30                       # Amendment 1


def mapping_test(c: pd.DataFrame) -> dict:
    """Amendment 1's mapping test over the main samples, excluding cells with a release-side
    mechanism, `our_harness` or `release_correct` (the recipe that reproduces the committed
    s1_7_mapping_test.json, criteria Amendment 11)."""
    from router.data import audit
    g = c[c["sample"].isin(MAPPING_SAMPLES) & (c.key_role == "graded") & c.rerun.isin(["pass", "fail"])
          & (c.test_version.fillna("plus") == "plus")]
    g = g[~g.mechanism.isin(RELEASE_SIDE | {"our_harness", "release_correct"})]
    unc = g.model.isin(audit.UNCONFIRMED)
    conf = g[~unc]
    a_c = float(conf.agree.astype(bool).mean())
    out = {"a_C": round(a_c, 4), "n_C": int(len(conf))}
    for m in sorted(audit.UNCONFIRMED, key=str.lower):
        s = g[g.model == m]
        a_u = float(s.agree.astype(bool).mean())
        out[m] = {"a_U": round(a_u, 4), "n": int(len(s)), "gap": round(a_c - a_u, 4),
                  "fails": bool(a_c - a_u >= MAPPING_GAP and len(s) >= MAPPING_MIN_CELLS)}
    return out


if __name__ == "__main__":
    main()
