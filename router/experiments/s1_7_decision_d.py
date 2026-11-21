"""S1.7 decision D (reviewer check of 2a, 2026-11-21): investigate every `unexplained` cell in the
main family samples and reassign it only where a log line or response excerpt names the
mechanism. No threshold changes. Every reassignment is listed in out/s1_7_d_reassignments.csv,
which s1_7_mechanisms applies (to cells still `unexplained` only).

Rules (each with its evidence; written after the cell rows, as investigation):
  D-BCB-LIB     BigCodeBench, released fail, our run passes: the task imports a third-party
                package (not in sys.stdlib_module_names) whose release pass share over all
                BigCodeBench cells is <= 2%, and blocking it (probe block_libs) reproduces the
                release: `toolchain`.
  D-DS-LIB      ds1000, released fail, our run passes: the problem's DS-1000 library has a release
                pass share of 0.000 (Matplotlib 1,240 cells, Pytorch 544, Tensorflow 360), and
                blocking it reproduces the release: `toolchain`.
  D-LCB-PUBLIC  lcb / lcb_kept, released pass, our run fails: the first failing test is a private
                one (every public test passes). Across the lcb samples no cell that fails a public
                test is a released pass (0 of 147): `test_version` (public tests only).
  D-HE-BASE     humaneval, released pass, our HumanEval+ run fails, our base-test run passes (the
                release's notes name base tests): `test_version`.
  D-HARNESS     our image is at fault (the canonical solution also fails): `our_harness`.
  D-EXTRACT     our extraction misread a correct answer, shown by the log and the response; the
                released outcome is right: `release_correct` (§1.1, "our run is wrong for a
                stated reason").

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_decision_d
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import audit  # noqa: E402
from router.data.prompts import upstream_id  # noqa: E402
from router.experiments.s1_7_cp2 import SOURCES, _bcb_tests, _jsonl  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out"
DS_LIB_SHARE_MAX = 0.0          # D-DS-LIB: libraries the release never passes
BCB_LIB_SHARE_MAX = 0.02        # D-BCB-LIB

# D-HARNESS cells, each with the log line and the excerpt that show our side is at fault.
HARNESS = {
    ("bcb", "dim1_bcb_BigCodeBench/734", "gpt-5.4"):
        "canonical solution also fails: nltk LookupError 'tokenizers/punkt/PY3/english.pickle' "
        "(the image has no nltk data)",
    ("hex_python", "dim7_hex_python_Python/50", "qwen3.5-plus"):
        "NameError: name 'encode_shift' is not defined; the response defines encode_shift, so "
        "our extraction (EvalPlus sanitize with the entry point decode_shift) lost it",
    ("mbpp_completion", "dim4_mbpp_completion_103", "glm-5"):
        "NameError: name 'eulerian_num' is not defined; the response defines it (under "
        "@lru_cache(maxsize=None)), so our extraction (EvalPlus sanitize) lost it",
    ("cruxeval", "dim4_cruxeval_505", "gpt-5.4"):
        "our 2a rule took the first assert line (== '--4/0-209'); the response's final line is "
        "'' which equals CRUXEval's output ''",
}


def _runs(part):
    p = OUT / f"s1_7_runs_{part}.jsonl"
    return {json.loads(l)["run_id"]: json.loads(l) for l in open(p)} if p.exists() else {}


def build() -> pd.DataFrame:
    from router.experiments.s1_7_mechanisms import MAIN, assign
    c = assign(apply_d=False)
    u = c[(c.mechanism == "unexplained") & c["sample"].isin(MAIN)
          & (c.test_version.fillna("plus") == "plus")]
    long = audit.released_long()

    # BigCodeBench library shares and the block_libs probe
    bt = _bcb_tests()
    b = long[long.family.str.startswith("bcb") & long.available].copy()
    b["up"] = [upstream_id(t, f) for t, f in zip(b.task_id, b.family)]
    libs = lambda up: sorted(set(ast.literal_eval(bt[up]["libs"]) if isinstance(bt[up]["libs"], str)
                                 else bt[up]["libs"]))
    b["libs"] = [libs(x) for x in b.up]
    bshare = b.assign(ok=b.score >= 1).explode("libs").groupby("libs").ok.mean()
    bprobe = pd.read_csv(OUT / "s1_7_probe_block_libs.csv").set_index(["task_id", "key"])

    # DS-1000 library shares and probe
    dsl = {}
    for r in _jsonl(SOURCES / "ds1000_tests.jsonl"):
        pid = r.get("metadata.problem_id", (r.get("metadata") or {}).get("problem_id"))
        dsl[str(pid)] = r.get("metadata.library", (r.get("metadata") or {}).get("library"))
    d = long[(long.family == "ds1000") & long.available]
    dshare = d.assign(lib=[dsl.get(upstream_id(t, "ds1000")) for t in d.task_id],
                      ok=d.score >= 1).groupby("lib").ok.mean()
    dprobe = pd.read_csv(OUT / "s1_7_probe_block_libs_ds1000.csv").set_index(["task_id", "key"])

    # LCB public-test counts and first failing test
    npub = {}
    for n in ("livecodebench_tests", "livecodebench_tests_140"):
        if (SOURCES / f"{n}.jsonl").exists():
            for r in _jsonl(SOURCES / f"{n}.jsonl"):
                npub[r["question_id"]] = len(json.loads(r["public_test_cases"] or "[]"))
    runs = {**_runs("py"), **_runs("lcbk")}
    he = c[(c.family == "humaneval") & (c.test_version == "base")].set_index(["task_id", "key"])["rerun"]

    rows = []
    for r in u.itertuples():
        to, rule, ev = None, None, None
        key = (r.sample, r.task_id, r.model)
        if key in HARNESS:
            to, rule, ev = ("our_harness", "D-HARNESS", HARNESS[key]) if key[0] == "bcb" \
                else ("release_correct", "D-EXTRACT", HARNESS[key])
        elif r.part == "bcb" and not r.released_pass and r.rerun == "pass":
            cand = [l for l in libs(r.upstream) if l not in sys.stdlib_module_names
                    and bshare[l] <= BCB_LIB_SHARE_MAX]
            pr = bprobe["probe"].get((r.task_id, r.key)) if cand else None
            if cand and pr == "fail":
                to, rule = "toolchain", "D-BCB-LIB"
                ev = ("release pass share " + ", ".join(f"{l} {bshare[l]:.3f}" for l in cand)
                      + f"; probe block_libs({','.join(cand)}) fails like the release")
        elif r.family == "ds1000" and not r.released_pass and r.rerun == "pass":
            lib = dsl.get(r.upstream.split("::")[1])
            pr = dprobe["probe"].get((r.task_id, r.key))
            if lib in dshare and dshare[lib] <= DS_LIB_SHARE_MAX and pr == "fail":
                to, rule = "toolchain", "D-DS-LIB"
                ev = f"DS-1000 library {lib}: release pass share {dshare[lib]:.3f}; probe block_libs fails like the release"
        elif r.family == "lcb" and r.released_pass and r.rerun == "fail":
            run = runs.get(f"{r.task_id}|{r.key}|None|1") or {}
            m = re.match(r"test (\d+)", (run.get("cell") or {}).get("err") or "")
            q = r.upstream.split("::")[1]
            if m and q in npub and int(m.group(1)) >= npub[q]:
                to, rule = "test_version", "D-LCB-PUBLIC"
                ev = f"all {npub[q]} public tests pass; first failure is test {m.group(1)} (private)"
        elif r.family == "humaneval" and r.released_pass and r.rerun == "fail":
            if he.get((r.task_id, r.key)) == "pass":
                to, rule = "test_version", "D-HE-BASE"
                ev = "fails HumanEval+; passes the base tests (log |base|1)"
        rows.append({"part": r.part, "sample": r.sample, "task_id": r.task_id, "model": r.model,
                     "key": r.key, "released_score": r.released_score, "rerun": r.rerun,
                     "from": "unexplained", "to": to or "unexplained", "rule": rule or "",
                     "evidence": ev or "no mechanism found", "log": r.log})
    out = pd.DataFrame(rows).sort_values(["sample", "task_id", "model"])
    out.to_csv(OUT / "s1_7_d_reassignments.csv", index=False, lineterminator="\n")
    return out


if __name__ == "__main__":
    t = build()
    pd.set_option("display.width", 250)
    print(t.groupby(["sample", "to"]).size().to_string())
