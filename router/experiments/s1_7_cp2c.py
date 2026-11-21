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
