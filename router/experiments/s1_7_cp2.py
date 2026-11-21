"""S1.7 checkpoint 2: draw the pre-registered samples, read them, re-run them in the sandbox.

Rules: docs/research/S1.7_audit_criteria.md (§4, §8, Amendments 1-3). Order per part:
  1. draw the samples (seeded; out/s1_7_samples_<part>.csv);
  2. read every sampled response with the fixed categories (out/s1_7_reading_<part>.csv), with
     no released score joined;
  3. run harness controls (reference solutions) and every sampled cell in the sandbox; re-run
     each disagreeing cell twice more (flaky check);
  4. only then join released scores and write the cell rows (out/s1_7_cells_<part>.csv).

Sandbox runs are cached in out/s1_7_runs_<part>.jsonl (append-only, keyed by run id) and are not
part of any byte-for-byte check. All Python families run on the pinned BigCodeBench evaluation
image (it carries numpy, pandas and the scientific stack the tests import).

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_cp2 bcb
    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_7_cp2 py
"""

from __future__ import annotations

import json
import re
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
from router.data.prompts import upstream_id  # noqa: E402
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
PY_SPEC = runner.Spec(image=BCB_IMAGE, cpus=2.0, memory="4g", pids=512, tmpfs_size="1g",
                      wall_s=600.0)
HARNESS = {"bcb": (SANDBOX / "harness_bcb.py").read_text(),
           "py": (SANDBOX / "harness_py.py").read_text()}


# --------------------------------------------------------------------------- data
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _index(name: str, key: str) -> dict:
    return {str(r[key]): r for r in _jsonl(SOURCES / f"{name}.jsonl")}


def load_responses(cells: pd.DataFrame) -> dict[tuple[str, str], str]:
    want = set(zip(cells.task_id, cells.key))
    with open(audit.RESPONSE_MATRIX) as fh:
        m = json.load(fh)
    return {(t, k): m[t][k] for t, k in want if k in m.get(t, {})}


# --------------------------------------------------------------------------- samples
def _with_keys(s: pd.DataFrame) -> pd.DataFrame:
    s = s.copy()
    s["key"] = s.model.map(audit.response_key)
    q = s[(s.model == "Qwen3-Max") & s.family.isin(audit.QWEN_CODEGEN)].copy()
    q["key"] = "qwen3.6-plus"                                   # Amendment 1, decision B
    s = pd.concat([s, q], ignore_index=True)
    s["key_role"] = (s.key == s.model.map(audit.response_key)).map({True: "graded", False: "alternate"})
    s["cell_id"] = s.task_id + "|" + s.key
    for c in ("pair", "pair_side"):
        if c not in s:
            s[c] = None
    return s


def family_sample(long, stats, fam, stream=None, sample=None, exclude=None):
    fr = audit.frame(long, stats, long.loc[long.family == fam, "task_id"].unique())
    if exclude is not None:
        fr = fr[~(fr.task_id + "|" + fr.model).isin(exclude)].reset_index(drop=True)
    return audit.draw(fr, stream or audit.MAIN_STREAMS[fam]).assign(sample=sample or fam)


def stratum_sample(long, stats, families, stream, sample):
    fr = audit.frame(long, stats, audit.all_zero_tasks(long, families))
    return audit.draw(fr, stream).assign(sample=sample)


def bcb_samples(long, stats) -> pd.DataFrame:
    parts = [family_sample(long, stats, f) for f in BCB_FAMILIES]
    parts.append(stratum_sample(long, stats, BCB_FAMILIES, 120, "allzero_bcb"))
    e0 = json.loads((OUT / "e0_audit.json").read_text())
    ident = audit.draw(audit.identical_frame(e0["identical_prompt_bcb"]["differing_cells"]), 129)
    fam = long.drop_duplicates("task_id").set_index("task_id")["family"]
    for side in ("bcb_task", "other_task"):
        parts.append(pd.DataFrame({"task_id": ident[side], "model": ident["model"],
                                   "family": ident[side].map(fam), "stream": 129,
                                   "sample": "identical", "pair": ident.index, "pair_side": side}))
    return _with_keys(pd.concat(parts, ignore_index=True))


def py_samples(long, stats) -> pd.DataFrame:
    parts = []
    for fam in ("ds1000", "humaneval", "hex_python", "mbpp", "mbpp_completion", "lcb", "cruxeval"):
        parts.append(family_sample(long, stats, fam))
    main_he = parts[1]
    parts.append(family_sample(long, stats, "humaneval", 117, "humaneval_version_id",
                               exclude=set(main_he.task_id + "|" + main_he.model)))
    for fams, stream, name in ((["ds1000"], 121, "allzero_ds1000"), (["lcb"], 122, "allzero_lcb"),
                               (["mbpp"], 123, "allzero_mbpp"), (["humaneval"], 124, "allzero_humaneval")):
        parts.append(stratum_sample(long, stats, fams, stream, name))
    cx = audit.frame(long, stats, long.loc[long.family == "cruxeval", "task_id"].unique())
    sc = long.set_index(["task_id", "model"])["score"]
    cx = cx[[abs(sc[(t, m)] - 0.8) < 1e-9 for t, m in zip(cx.task_id, cx.model)]].reset_index(drop=True)
    parts.append(audit.draw(cx, 125).assign(sample="cruxeval_0.8"))
    for fam, stream in (("lcb_testgen", 126), ("he_testgen", 127), ("mbpp_testgen", 128)):
        parts.append(family_sample(long, stats, fam, stream, f"read_{fam}"))
    return _with_keys(pd.concat(parts, ignore_index=True))


# --------------------------------------------------------------------------- payloads
def _bcb_tests():
    t = _index("bigcodebench_tests", "task_id")
    for r in _jsonl(SOURCES / "bigcodebench.jsonl"):
        t[r["task_id"]]["complete_prompt"] = r["complete_prompt"]
    return t


class Bcb:
    kind = "bcb"
    spec = BCB_SPEC

    def __init__(self):
        self.t = _bcb_tests()

    def up(self, task_id, family):
        return upstream_id(task_id, family)

    def entry(self, up):
        return self.t[up]["entry_point"]

    def has_tests(self, up):
        return up in self.t

    def canonical(self, up):
        t = self.t[up]
        return {"mode": "canonical", "entry_point": t["entry_point"], "test": t["test"],
                "complete_prompt": t["complete_prompt"], "canonical_solution": t["canonical_solution"]}

    def payload(self, up, response, gt, version=None):
        t = self.t[up]
        return {"mode": "response", "response": response, "entry_point": t["entry_point"],
                "code_prompt": t["code_prompt"], "test": t["test"], "gt_time": gt}


class Py:
    """humaneval (HumanEval+ documented; base for §2), hex_python, mbpp, mbpp_completion,
    ds1000, cruxeval, lcb."""
    kind = "py"
    spec = PY_SPEC

    def __init__(self):
        self.hep = _index("humanevalplus_tests", "task_id")
        self.hex = _index("humaneval_x_python_tests", "task_id")
        self.mbp = _index("mbppplus_tests", "task_id")
        self.ds = {str(r["metadata"]["problem_id"]) if isinstance(r.get("metadata"), dict)
                   else str(r["metadata.problem_id"]): r for r in _jsonl(SOURCES / "ds1000_tests.jsonl")}
        self.cx = _index("cruxeval_tests", "id")
        self.lcb = _index("livecodebench_tests", "question_id") if (SOURCES / "livecodebench_tests.jsonl").exists() else {}

    @staticmethod
    def fam_up(task_id, family):
        return family, upstream_id(task_id, family)

    def up(self, task_id, family):
        return f"{family}::{upstream_id(task_id, family)}"

    def _split(self, up):
        fam, u = up.split("::", 1)
        return fam, u

    def has_tests(self, up):
        fam, u = self._split(up)
        return {"humaneval": self.hep, "hex_python": self.hex, "mbpp": self.mbp,
                "mbpp_completion": self.mbp, "ds1000": self.ds, "cruxeval": self.cx,
                "lcb": self.lcb}.get(fam, {}).__contains__(u)

    def entry(self, up):
        fam, u = self._split(up)
        if fam == "humaneval":
            return self.hep[u]["entry_point"]
        if fam == "hex_python":
            return re.findall(r"def\s+(\w+)\s*\(", self.hex[u]["declaration"])[-1]
        if fam in ("mbpp", "mbpp_completion"):
            m = re.search(r"assertion\((\w+)\(", self.mbp[u]["test"])
            return m.group(1) if m else None
        return None

    def versions(self, up):
        return ("plus", "base") if up.startswith("humaneval::") else (None,)

    def _body(self, up, version):
        fam, u = self._split(up)
        ep = self.entry(up)
        if fam == "humaneval" and version == "plus":
            t = self.hep[u]
            return {"family_kind": "humaneval", "entry_point": ep, "prompt": t["prompt"],
                    "test": t["test"], "post": f"check({ep})",
                    "reference": t["prompt"] + t["canonical_solution"]}
        if fam in ("humaneval", "hex_python"):
            hid = "Python/" + u.split("/")[-1]
            t = self.hex[hid]
            return {"family_kind": "humaneval", "entry_point": ep, "prompt": t["prompt"],
                    "test": t["test"], "post": f"check({ep})",
                    "reference": t["prompt"] + t["canonical_solution"]}
        if fam in ("mbpp", "mbpp_completion"):
            t = self.mbp[u]
            return {"family_kind": "mbpp", "entry_point": ep, "prompt": None,
                    "pre": "\n".join(t.get("test_imports") or []), "test": t["test"],
                    "reference": t["code"]}
        if fam == "ds1000":
            t = self.ds[u]
            return {"family_kind": "ds1000", "code_context": t["code_context"],
                    "reference": t["reference_code"], "timeout_s": 120}
        if fam == "cruxeval":
            t = self.cx[u]
            return {"family_kind": "cruxeval", "code": t["code"], "input": t["input"],
                    "output": t["output"]}
        if fam == "lcb":
            t = self.lcb[u]
            return {"family_kind": "lcb", "public_test_cases": t["public_test_cases"],
                    "private_test_cases": t["private_test_cases"], "metadata": t["metadata"],
                    "reference": None, "per_test_s": 10}
        raise KeyError(up)

    def canonical(self, up, version=None):
        b = self._body(up, version)
        return None if b.get("family_kind") == "lcb" else {**b, "mode": "canonical"}

    def payload(self, up, response, gt, version=None):
        return {**self._body(up, version), "mode": "response", "response": response}


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


def sandbox(kind: str, spec: runner.Spec, payload: dict) -> dict:
    files = {"cell.json": json.dumps(payload), "harness.py": HARNESS[kind]}
    res = runner.run(spec, {"files": files, "cmd": ["python", "harness.py"],
                            "timeout": spec.wall_s - 20})
    line = next((l for l in (res.get("stdout") or "").splitlines() if l.startswith("@@CELL@@")), None)
    cell = json.loads(line[len("@@CELL@@"):]) if line else None
    status = cell["stat"] if cell else ("timeout" if res.get("timed_out") or res.get("host_timeout")
                                         else "error")
    return {"status": status, "cell": cell,
            "stderr": (res.get("stderr") or res.get("log") or "")[-3000:],
            "container_s": res.get("container_s") or res.get("elapsed_s")}


def run_many(jobs, cache: Cache, kind: str, spec: runner.Spec) -> None:
    todo = [(rid, p) for rid, p in jobs if cache.get(rid) is None]
    print(f"{len(jobs) - len(todo)} cached, {len(todo)} to run", flush=True)

    def one(job):
        rid, p = job
        r = sandbox(kind, spec, p)
        r["run_id"] = rid
        cache.put(r)
        return r["status"]

    with ThreadPoolExecutor(WORKERS) as ex:
        for i, st in enumerate(ex.map(one, todo), 1):
            if i % 25 == 0:
                print(f"  {i}/{len(todo)} done (last: {st})", flush=True)


# --------------------------------------------------------------------------- one part
READ_ONLY = ("read_lcb_testgen", "read_he_testgen", "read_mbpp_testgen")


def run_part(part: str, s: pd.DataFrame, fam_obj, long: pd.DataFrame) -> pd.DataFrame:
    s.drop(columns=["cell_id"]).to_csv(OUT / f"s1_7_samples_{part}.csv", index=False,
                                       lineterminator="\n")
    resp = load_responses(s)
    s = s.copy()
    s["upstream"] = [fam_obj.up(t, f) if not smp.startswith("read_") else upstream_id(t, f)
                     for t, f, smp in zip(s.task_id, s.family, s["sample"])]
    s["has_response"] = [isinstance(resp.get((t, k)), str) and bool(resp[(t, k)].strip())
                         for t, k in zip(s.task_id, s.key)]
    s["has_tests"] = [(not smp.startswith("read_")) and fam_obj.has_tests(u)
                      for u, smp in zip(s.upstream, s["sample"])]

    # 2. reading, blind to released scores
    reading = []
    for r in s.drop_duplicates("cell_id").itertuples():
        ep = fam_obj.entry(r.upstream) if r.has_tests else None
        reading.append({"cell_id": r.cell_id, **read_cell(resp.get((r.task_id, r.key)), "python", ep)})
    pd.DataFrame(reading).sort_values("cell_id").to_csv(OUT / f"s1_7_reading_{part}.csv",
                                                         index=False, lineterminator="\n")

    cache = Cache(OUT / f"s1_7_runs_{part}.jsonl")
    ex = s[~s["sample"].isin(READ_ONLY) & s.has_tests]
    versions = {u: fam_obj.versions(u) if hasattr(fam_obj, "versions") else (None,) for u in set(ex.upstream)}

    # 3a. harness controls
    jobs = []
    for u in sorted(set(ex.upstream)):
        for v in versions[u]:
            c = fam_obj.canonical(u, v) if v else fam_obj.canonical(u)
            if c is not None:
                jobs.append((f"canonical|{u}|{v}", c))
    run_many(jobs, cache, fam_obj.kind, fam_obj.spec)

    def gt(u, v):
        c = cache.get(f"canonical|{u}|{v}")
        return (c["cell"] or {}).get("elapsed") if c and c["status"] == "pass" else None

    # 3b. every sampled cell with a response and pinned tests
    cells = ex[ex.has_response].drop_duplicates("cell_id")
    run_many([(f"{r.cell_id}|{v}|1", fam_obj.payload(r.upstream, resp[(r.task_id, r.key)], gt(r.upstream, v), v))
              for r in cells.itertuples() for v in versions[r.upstream]], cache, fam_obj.kind, fam_obj.spec)

    # 3c. flaky check on disagreeing cells (released outcome used here only to select re-runs)
    rel = long.set_index(["task_id", "model"])["score"]
    dis = [(r, v) for r in cells.itertuples() for v in versions[r.upstream]
           if (cache.get(f"{r.cell_id}|{v}|1")["status"] == "pass") != (rel[(r.task_id, r.model)] >= SUCCESS_SCORE)]
    run_many([(f"{r.cell_id}|{v}|{k}", fam_obj.payload(r.upstream, resp[(r.task_id, r.key)], gt(r.upstream, v), v))
              for r, v in dis for k in range(2, FLAKY_RUNS + 1)], cache, fam_obj.kind, fam_obj.spec)

    # 4. cell rows (one per sampled cell and test version)
    rows = []
    for r in s.itertuples():
        score = float(rel[(r.task_id, r.model)])
        for v in (versions.get(r.upstream, (None,)) if r.has_tests else (None,)):
            runs = [cache.get(f"{r.cell_id}|{v}|{k}") for k in range(1, FLAKY_RUNS + 1)]
            runs = [x["status"] for x in runs if x]
            can = cache.get(f"canonical|{r.upstream}|{v}")
            rows.append({"sample": r.sample, "stream": r.stream, "pair": r.pair, "pair_side": r.pair_side,
                         "task_id": r.task_id, "upstream": r.upstream, "family": r.family,
                         "model": r.model, "key": r.key, "key_role": r.key_role, "test_version": v,
                         "unconfirmed": r.model in audit.UNCONFIRMED,
                         "released_score": score, "released_pass": score >= SUCCESS_SCORE,
                         "has_response": r.has_response, "has_tests": r.has_tests,
                         "canonical": can["status"] if can else None,
                         "rerun": runs[0] if runs else "not_run", "rerun_all": "/".join(runs),
                         "flaky": len(set(runs)) > 1,
                         "agree": (runs[0] == "pass") == (score >= SUCCESS_SCORE) if runs else None,
                         "log": f"s1_7_runs_{part}.jsonl#{r.cell_id}|{v}|1" if runs else ""})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"s1_7_cells_{part}.csv", index=False, lineterminator="\n")
    print("cell rows written:", len(out))
    return out


def main(part: str) -> None:
    long = audit.released_long()
    stats = audit.response_stats()
    if part == "bcb":
        run_part("bcb", bcb_samples(long, stats), Bcb(), long)
    elif part == "py":
        run_part("py", py_samples(long, stats), Py(), long)
    else:
        raise SystemExit(part)


if __name__ == "__main__":
    main(sys.argv[1])
