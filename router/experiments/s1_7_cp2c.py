"""S1.7 checkpoint 2c: SWE-bench Lite (criteria §11, Amendment 10).

    SWE_PY=<python with swebench 4.0.4> PYTHONPATH=.:pipeline pipeline/venv/bin/python \
        -m router.experiments.s1_7_cp2c gate <instance_id>
    ... -m router.experiments.s1_7_cp2c run
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.audit_sandbox import runner, swe  # noqa: E402
from router.data import audit  # noqa: E402
from router.experiments.s1_7_cp2 import (OUT, PLATFORM, SOURCES, Cache, _jsonl, _with_keys,  # noqa: E402
                                         load_responses)
from router.settings import SUCCESS_SCORE  # noqa: E402

EXTRACT = SOURCES / "swebench_lite_tests.jsonl"
TOOL = REPO_ROOT / "router/audit_sandbox/swe_tool.py"
SWE_PY = os.environ.get("SWE_PY", "")
FLAKY_RUNS = 3
_ENV = (("PATH", "/opt/miniconda3/envs/testbed/bin:/opt/miniconda3/bin:/usr/local/sbin:/usr/local/bin:"
         "/usr/sbin:/usr/bin:/sbin:/bin"), ("HOME", "/work"), ("TMPDIR", "/tmp"),
        ("MPLCONFIGDIR", "/tmp"), ("PYTHONDONTWRITEBYTECODE", "1"))


def tool(*args) -> dict:
    r = subprocess.run([SWE_PY, str(TOOL), *args], capture_output=True, text=True, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


def image_id(key: str) -> str | None:
    r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", key],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def spec_for(iid: str) -> runner.Spec:
    return runner.Spec(image=image_id(tool("spec", str(EXTRACT), iid)["image_key"]), platform=PLATFORM,
                       cpus=2.0, memory="6g", pids=1024, tmpfs_size="3g", wall_s=1800.0, env=_ENV)


def run_patch(iid: str, patch: str | None, how: str) -> dict:
    """One cell: apply `patch` and run the instance's tests in the sandbox; grade on the host."""
    if patch is None:
        return {"status": "fail", "phase": "no_patch", "extraction": how}
    sp = tool("spec", str(EXTRACT), iid)
    run_sh, eval_sh = swe.run_script(sp["eval_script"])
    spec = spec_for(iid)
    res = runner.run(spec, {"files": {"run.sh": run_sh, "eval.sh": eval_sh, "patch.diff": patch},
                            "cmd": ["bash", "/work/run.sh"], "timeout": spec.wall_s - 60})
    log = (res.get("stdout") or "") + "\n" + (res.get("stderr") or "")
    out = {"extraction": how, "patch_chars": len(patch), "elapsed_s": res.get("elapsed_s"),
           "log_tail": log[-3000:]}
    if res.get("timed_out") or res.get("host_timeout"):
        return {**out, "status": "timeout", "phase": "run"}
    if swe.APPLY_FAIL in log or swe.APPLY_PASS not in log:
        return {**out, "status": "fail", "phase": "apply"}
    with tempfile.TemporaryDirectory() as d:
        lp, pp = Path(d) / "test_output.txt", Path(d) / "patch.diff"
        lp.write_text(log)
        pp.write_text(patch)
        g = tool("grade", str(EXTRACT), iid, str(lp), str(pp))
    rep = g["report"].get("tests_status", {})
    out["f2p_fail"] = rep.get("FAIL_TO_PASS", {}).get("failure", [])[:10]
    out["p2p_fail"] = rep.get("PASS_TO_PASS", {}).get("failure", [])[:10]
    return {**out, "status": "pass" if g["resolved"] else "fail", "phase": "tests"}


def gate(iid: str) -> dict:
    """Amendment 10 gate on one instance (no response): the gold patch resolves; a patch that
    applies but changes no behaviour does not; no patch fails."""
    gold = next(r for r in _jsonl(EXTRACT) if r["instance_id"] == iid)["patch"]
    first = gold.split("\n", 1)[0]                                # "diff --git a/X b/X"
    path = first.split(" b/", 1)[1]
    noop = (f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1,0 +1,1 @@\n"
            "+# audit no-op line\n")
    cache = Cache(OUT / "s1_7_runs_swe.jsonl")
    rows = {}
    for name, patch, exp in (("gold", gold, "pass"), ("noop", noop, "fail"), ("none", None, "fail")):
        rid = f"control:{name}|{iid}"
        if cache.get(rid) is None:
            cache.put({"run_id": rid, **run_patch(iid, patch, name)})
        got = cache.get(rid)
        rows[name] = {"expected": exp, "got": got["status"], "phase": got.get("phase"),
                      "elapsed_s": got.get("elapsed_s")}
    print(json.dumps(rows, indent=1))
    return rows


if __name__ == "__main__" and sys.argv[1] == "gate":
    gate(sys.argv[2])


def _free_gib() -> float:
    st = os.statvfs("/")
    return st.f_bavail * st.f_frsize / 2**30


def _rmi(key: str) -> None:
    subprocess.run(["docker", "rmi", "-f", key], capture_output=True)


def run() -> pd.DataFrame:
    from router.audit_sandbox.reading import read_cell
    long, stats = audit.released_long(), audit.response_stats()
    fam = long[long.family == "swebench"]
    s = audit.draw(audit.frame(long, stats, fam.task_id.unique()), 108).assign(sample="swebench")
    s = _with_keys(s)
    s.drop(columns=["cell_id"]).to_csv(OUT / "s1_7_samples_swe.csv", index=False, lineterminator="\n")
    resp = load_responses(s)
    s["iid"] = s.task_id.str.replace("dim3_swebench_", "", regex=False)
    pd.DataFrame([{"cell_id": r.cell_id, **read_cell(resp.get((r.task_id, r.key)), "python", None)}
                  for r in s.itertuples()]).sort_values("cell_id").to_csv(
        OUT / "s1_7_reading_swe.csv", index=False, lineterminator="\n")
    ext = {r.cell_id: swe.extract_patch(resp.get((r.task_id, r.key)) or "") for r in s.itertuples()}
    cache = Cache(OUT / "s1_7_runs_swe.jsonl")
    lite = {r["instance_id"]: r for r in _jsonl(EXTRACT)}
    for r in s.itertuples():                              # no patch: fails without running
        if ext[r.cell_id][0] is None and cache.get(f"{r.cell_id}|1") is None:
            cache.put({"run_id": f"{r.cell_id}|1", "status": "fail", "phase": "no_patch",
                       "extraction": "no_patch"})
    todo = s[[ext[c][0] is not None for c in s.cell_id]]
    order = sorted(set(todo.iid), key=lambda i: (lite[i]["repo"], lite[i]["version"], i))
    rel = long.set_index(["task_id", "model"])["score"]
    prev_env = None
    for iid in ([] if "--rows-only" in sys.argv else order):
        cells = todo[todo.iid == iid]
        if all(cache.get(f"{c}|1") for c in cells.cell_id) and cache.get(f"control:gold|{iid}"):
            continue
        if _free_gib() < 3.0:
            print("disk below 3 GiB: stop building", iid, flush=True)
            for c in cells.cell_id:
                if cache.get(f"{c}|1") is None:
                    cache.put({"run_id": f"{c}|1", "status": "not_run", "phase": "disk"})
            continue
        print("instance", iid, len(cells), "cells; free", round(_free_gib(), 1), "GiB", flush=True)
        subprocess.run([SWE_PY, str(TOOL), "build", str(EXTRACT), iid], capture_output=True, text=True)
        key = tool("spec", str(EXTRACT), iid)["image_key"]
        if image_id(key) is None:
            for c in cells.cell_id:
                cache.put({"run_id": f"{c}|1", "status": "error", "phase": "build"})
            continue
        if cache.get(f"control:gold|{iid}") is None:
            cache.put({"run_id": f"control:gold|{iid}", **run_patch(iid, lite[iid]["patch"], "gold")})
        for r in cells.itertuples():
            patch, how = ext[r.cell_id]
            if cache.get(f"{r.cell_id}|1") is None:
                cache.put({"run_id": f"{r.cell_id}|1", **run_patch(iid, patch, how)})
            first = cache.get(f"{r.cell_id}|1")["status"]
            if (first == "pass") != (rel[(r.task_id, r.model)] >= SUCCESS_SCORE):
                for k in range(2, FLAKY_RUNS + 1):
                    if cache.get(f"{r.cell_id}|{k}") is None:
                        cache.put({"run_id": f"{r.cell_id}|{k}", **run_patch(iid, patch, how)})
        _rmi(key)
        env = subprocess.run([SWE_PY, "-c", "import sys,json;sys.path.insert(0,%r);" % str(TOOL.parent)
                              + "from swe_tool import _instance;"
                              + "from swebench.harness.test_spec.test_spec import make_test_spec;"
                              + f"print(make_test_spec(_instance({str(EXTRACT)!r},{iid!r}),namespace=None).env_image_key)"],
                             capture_output=True, text=True).stdout.strip()
        nxt = order[order.index(iid) + 1] if order.index(iid) + 1 < len(order) else None
        if nxt is None or (lite[nxt]["repo"], lite[nxt]["version"]) != (lite[iid]["repo"], lite[iid]["version"]):
            _rmi(env)                                      # environment no longer needed
        subprocess.run(["docker", "builder", "prune", "-f"], capture_output=True)
    rows = []
    for r in s.itertuples():
        runs = [cache.get(f"{r.cell_id}|{k}") for k in range(1, FLAKY_RUNS + 1)]
        runs = [x for x in runs if x]
        st = [x["status"] for x in runs]
        score = float(rel[(r.task_id, r.model)])
        gold = cache.get(f"control:gold|{r.iid}")
        rerun = st[0] if st else "not_run"
        no_patch = bool(runs) and runs[0].get("phase") == "no_patch"
        if gold is not None and gold["status"] != "pass" and rerun in ("pass", "fail") and not no_patch:
            rerun = "error"                                 # instance control failed: our_harness
        rows.append({"sample": "swebench", "stream": 108, "pair": None, "pair_side": None,
                     "task_id": r.task_id, "upstream": r.iid, "family": "swebench", "model": r.model,
                     "key": r.key, "key_role": r.key_role, "test_version": None,
                     "unconfirmed": r.model in audit.UNCONFIRMED, "released_score": score,
                     "released_pass": score >= SUCCESS_SCORE, "has_response": True, "has_tests": True,
                     "canonical": gold["status"] if gold else None, "rerun": rerun,
                     "rerun_all": "/".join(st), "flaky": len(set(st)) > 1,
                     "agree": (rerun == "pass") == (score >= SUCCESS_SCORE) if rerun in ("pass", "fail") else None,
                     "phase": runs[0].get("phase") if runs else None,
                     "extraction": ext[r.cell_id][1], "log": f"s1_7_runs_swe.jsonl#{r.cell_id}|1"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "s1_7_cells_swe.csv", index=False, lineterminator="\n")
    print("cell rows written:", len(out))
    return out


if __name__ == "__main__" and sys.argv[1] == "run":
    run()
