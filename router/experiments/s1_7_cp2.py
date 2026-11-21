"""S1.7 checkpoint 2: draw the pre-registered samples, read them, re-run them in the sandbox.

Rules: docs/research/S1.7_audit_criteria.md (§4, §8, Amendments 1-3). Order per part:
  1. draw the samples (seeded; written to out/s1_7_samples_<part>.csv);
  2. read every sampled response with the fixed categories (out/s1_7_reading_<part>.csv),
     with no released score joined;
  3. run harness controls (canonical solutions) and every sampled cell in the sandbox;
     re-run each disagreeing cell twice more (flaky check);
  4. only then join released scores and write the cell rows (out/s1_7_cells_<part>.csv).

Sandbox runs are cached in out/s1_7_runs_<part>.jsonl (append-only, keyed by run id) and are not
part of any byte-for-byte check.

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_cp2 bcb
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.audit_sandbox import runner  # noqa: E402
from router.audit_sandbox.reading import read_cell  # noqa: E402
from router.data import audit  # noqa: E402
from router.settings import EXTERNAL_DIR, SUCCESS_SCORE  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out"
SANDBOX = REPO_ROOT / "router/audit_sandbox"
SOURCES = EXTERNAL_DIR / "sources"
WORKERS = 4
FLAKY_RUNS = 3                      # §8.1: each disagreeing cell runs three times in all

BCB_IMAGE = ("bigcodebench/bigcodebench-evaluate@sha256:"
             "a3cd34ec3840a49d6b7afb240f4bdd47c350bc5991043fd0a91773830f7cd405")   # v0.2.4
BCB_FAMILIES = ("bcb", "bcb_v014", "bcb_algo2", "bcb_completion", "bcb_ds")
BCB_SPEC = runner.Spec(image=BCB_IMAGE, cpus=2.0, memory="4g", pids=512, tmpfs_size="1g",
                       wall_s=330.0)   # 240 s BigCodeBench task limit + start-up under emulation


# --------------------------------------------------------------------------- data
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def bcb_tasks() -> dict[str, dict]:
    """BigCodeBench/<n> -> test fields (pinned extracts)."""
    tests = {r["task_id"]: r for r in _jsonl(SOURCES / "bigcodebench_tests.jsonl")}
    for r in _jsonl(SOURCES / "bigcodebench.jsonl"):
        tests[r["task_id"]]["complete_prompt"] = r["complete_prompt"]
    return tests


def upstream_bcb_id(task_id: str) -> str:
    """CodeRouterBench task id -> BigCodeBench/<n> (the id's own suffix)."""
    i = task_id.index("BigCodeBench/")
    return task_id[i:]


def load_responses(cells: pd.DataFrame) -> dict[tuple[str, str], str]:
    want = set(zip(cells.task_id, cells.key))
    with open(audit.RESPONSE_MATRIX) as fh:
        m = json.load(fh)
    return {(t, k): m[t][k] for t, k in want if k in m.get(t, {})}


# --------------------------------------------------------------------------- samples
def bcb_samples(long: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for fam in BCB_FAMILIES:
        tasks = long.loc[long.family == fam, "task_id"].unique()
        parts.append(audit.draw(audit.frame(long, stats, tasks), audit.MAIN_STREAMS[fam])
                     .assign(sample=fam))
    z = audit.all_zero_tasks(long, BCB_FAMILIES)
    parts.append(audit.draw(audit.frame(long, stats, z), 120).assign(sample="allzero_bcb"))
    e0 = json.loads((OUT / "e0_audit.json").read_text())
    ident = audit.draw(audit.identical_frame(e0["identical_prompt_bcb"]["differing_cells"]), 129)
    fam = long.drop_duplicates("task_id").set_index("task_id")["family"]
    for side in ("bcb_task", "other_task"):
        parts.append(pd.DataFrame({"task_id": ident[side], "model": ident["model"],
                                   "family": ident[side].map(fam), "stream": 129,
                                   "sample": "identical", "pair": ident.index,
                                   "pair_side": side}))
    s = pd.concat(parts, ignore_index=True)
    s["key"] = s.model.map(audit.response_key)
    q = s[(s.model == "Qwen3-Max") & s.family.isin(audit.QWEN_CODEGEN)].copy()
    q["key"] = "qwen3.6-plus"                                   # Amendment 1, decision B
    s = pd.concat([s, q], ignore_index=True)
    s["key_role"] = (s.key == s.model.map(audit.response_key)).map({True: "graded", False: "alternate"})
    s["cell_id"] = s.apply(lambda r: f"{r.task_id}|{r.key}", axis=1)
    return s[["sample", "stream", "pair", "pair_side", "task_id", "family", "model", "key",
              "key_role", "cell_id"]]


# --------------------------------------------------------------------------- sandbox
class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.rows = {r["run_id"]: r for r in _jsonl(path)} if path.exists() else {}

    def get(self, run_id):
        return self.rows.get(run_id)

    def put(self, row):
        self.rows[row["run_id"]] = row
        with open(self.path, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


HARNESS_BCB = (SANDBOX / "harness_bcb.py").read_text()


def run_bcb(payload: dict) -> dict:
    files = {"cell.json": json.dumps(payload), "harness.py": HARNESS_BCB}
    res = runner.run(BCB_SPEC, {"files": files, "cmd": ["python", "harness.py"], "timeout": 300})
    line = next((l for l in (res.get("stdout") or "").splitlines() if l.startswith("@@CELL@@")), None)
    cell = json.loads(line[len("@@CELL@@"):]) if line else None
    status = cell["stat"] if cell else ("timeout" if res.get("timed_out") or res.get("host_timeout")
                                         else "error")
    return {"status": status, "cell": cell,
            "stderr": (res.get("stderr") or res.get("log") or "")[-3000:],
            "container_s": res.get("container_s") or res.get("elapsed_s")}


def run_many(jobs: list[tuple[str, dict]], cache: Cache, fn) -> None:
    todo = [(rid, p) for rid, p in jobs if cache.get(rid) is None]
    print(f"{len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)

    def one(job):
        rid, p = job
        r = fn(p)
        r["run_id"] = rid
        cache.put(r)
        return r["status"]

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, st in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done (last: {st})", flush=True)


# --------------------------------------------------------------------------- part: BigCodeBench
def part_bcb() -> None:
    long = audit.released_long()
    stats = audit.response_stats()
    s = bcb_samples(long, stats)
    s.to_csv(OUT / "s1_7_samples_bcb.csv", index=False, lineterminator="\n")

    tests = bcb_tasks()
    resp = load_responses(s)
    s["upstream"] = s.task_id.map(upstream_bcb_id)
    s["has_response"] = [isinstance(resp.get((t, k)), str) and bool(resp[(t, k)].strip())
                         for t, k in zip(s.task_id, s.key)]

    # 2. reading, blind to released scores
    reading = []
    for r in s.drop_duplicates("cell_id").itertuples():
        rd = read_cell(resp.get((r.task_id, r.key)), "python", tests[r.upstream]["entry_point"])
        reading.append({"cell_id": r.cell_id, **rd})
    pd.DataFrame(reading).sort_values("cell_id").to_csv(OUT / "s1_7_reading_bcb.csv", index=False,
                                                         lineterminator="\n")

    # 3a. harness control: canonical solution of every sampled task
    cache = Cache(OUT / "s1_7_runs_bcb.jsonl")
    ups = sorted(set(s.upstream))
    run_many([(f"canonical|{u}", {"mode": "canonical", "entry_point": tests[u]["entry_point"],
                                   "complete_prompt": tests[u]["complete_prompt"],
                                   "canonical_solution": tests[u]["canonical_solution"],
                                   "test": tests[u]["test"]}) for u in ups], cache, run_bcb)
    gt = {u: (cache.get(f"canonical|{u}")["cell"] or {}).get("elapsed")
          if cache.get(f"canonical|{u}")["status"] == "pass" else None for u in ups}

    # 3b. every sampled cell with a response
    def payload(r):
        t = tests[r.upstream]
        return {"mode": "response", "response": resp[(r.task_id, r.key)],
                "entry_point": t["entry_point"], "code_prompt": t["code_prompt"],
                "test": t["test"], "gt_time": gt[r.upstream]}

    cells = s[s.has_response].drop_duplicates("cell_id")
    run_many([(f"{r.cell_id}|1", payload(r)) for r in cells.itertuples()], cache, run_bcb)

    # 3c. flaky check: disagreeing cells twice more. Needs the released outcome, which is joined
    # here only to select re-runs; the reading table above is already written.
    rel = long.set_index(["task_id", "model"])["score"]
    first = {r.cell_id: cache.get(f"{r.cell_id}|1")["status"] for r in cells.itertuples()}
    dis = [r for r in cells.itertuples()
           if (first[r.cell_id] == "pass") != (rel[(r.task_id, r.model)] >= SUCCESS_SCORE)]
    run_many([(f"{r.cell_id}|{k}", payload(r)) for r in dis for k in range(2, FLAKY_RUNS + 1)],
             cache, run_bcb)

    # 4. cell rows
    rows = []
    for r in s.itertuples():
        score = float(rel[(r.task_id, r.model)])
        runs = [cache.get(f"{r.cell_id}|{k}") for k in range(1, FLAKY_RUNS + 1)]
        runs = [x["status"] for x in runs if x]
        can = cache.get(f"canonical|{r.upstream}")
        rows.append({"sample": r.sample, "stream": r.stream, "pair": r.pair, "pair_side": r.pair_side,
                     "task_id": r.task_id, "upstream": r.upstream, "family": r.family,
                     "model": r.model, "key": r.key, "key_role": r.key_role,
                     "unconfirmed": r.model in audit.UNCONFIRMED,
                     "released_score": score, "released_pass": score >= SUCCESS_SCORE,
                     "has_response": r.has_response, "canonical": can["status"] if can else None,
                     "rerun": runs[0] if runs else "not_run",
                     "rerun_all": "/".join(runs), "flaky": len(set(runs)) > 1,
                     "agree": (runs[0] == "pass") == (score >= SUCCESS_SCORE) if runs else None,
                     "log": f"s1_7_runs_bcb.jsonl#{r.cell_id}|1" if runs else ""})
    pd.DataFrame(rows).to_csv(OUT / "s1_7_cells_bcb.csv", index=False, lineterminator="\n")
    print("cell rows written:", len(rows))


if __name__ == "__main__":
    {"bcb": part_bcb}[sys.argv[1]]()
