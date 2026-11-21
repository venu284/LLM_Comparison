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
             "ff56521c31aa9a9d31679ed94a0869875abf68cb7eba352d0c2b1d0b3cb1758b")   # v0.2.1, arm64 (Amendment 4)
PLATFORM = "linux/arm64"
GATE_MAX_CANONICAL_FAIL = 0.10                                                  # Amendment 4
BCB_FAMILIES = ("bcb", "bcb_v014", "bcb_algo2", "bcb_completion", "bcb_ds")
BCB_SPEC = runner.Spec(image=BCB_IMAGE, platform=PLATFORM, cpus=2.0, memory="4g", pids=512, tmpfs_size="1g",
                       wall_s=330.0)   # 240 s BigCodeBench task limit + start-up
PY_SPEC = runner.Spec(image=BCB_IMAGE, platform=PLATFORM, cpus=2.0, memory="4g", pids=512, tmpfs_size="1g",
                      wall_s=600.0)
DS_IMAGE_FILE = SANDBOX / "ds1000" / "IMAGE_ID"      # written after the local build (Amendment 5)
DS_SPEC = runner.Spec(image=DS_IMAGE_FILE.read_text().strip() if DS_IMAGE_FILE.exists() else "unbuilt",
                      platform=PLATFORM, cpus=2.0, memory="4g", pids=512, tmpfs_size="1g",
                      wall_s=600.0,
                      env=runner.Spec.env + (("TF_CPP_MIN_LOG_LEVEL", "3"), ("CUDA_VISIBLE_DEVICES", "-1")))
HARNESS = {"bcb": (SANDBOX / "harness_bcb.py").read_text(),
           "py": (SANDBOX / "harness_py.py").read_text(),
           "mx": (SANDBOX / "harness_mx.py").read_text()}


def _tc_image(tc: str) -> str:
    f = SANDBOX / "toolchains" / tc / "IMAGE_ID"
    return f.read_text().strip() if f.exists() else "unbuilt"


_BASE_PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_TC_ENV = {   # Amendment 9: fixed per-toolchain variables (env -i inside the container)
    "cpp": (), "node": (), "java": (("PATH", "/opt/java/openjdk/bin:" + _BASE_PATH),),
    "go": (("PATH", "/usr/local/go/bin:" + _BASE_PATH), ("GOCACHE", "/tmp/gocache"),
           ("GOPATH", "/tmp/gopath"), ("GOPROXY", "off"), ("GOTOOLCHAIN", "local"),
           ("GOFLAGS", "-mod=vendor"), ("CGO_ENABLED", "0")),
    "rust": (("PATH", "/usr/local/cargo/bin:" + _BASE_PATH), ("RUSTUP_HOME", "/usr/local/rustup"),
             ("CARGO_HOME", "/usr/local/cargo")),
}
LANG_TC = {"cpp": "cpp", "go": "go", "java": "java", "js": "node", "ts": "node", "rs": "rust"}


def _mx_spec(tc: str) -> runner.Spec:
    base = dict(runner.Spec.env)
    base.update(dict(_TC_ENV[tc]))
    return runner.Spec(image=_tc_image(tc), platform=PLATFORM, cpus=2.0, memory="4g", pids=1024,
                       tmpfs_size="1g", wall_s=300.0, env=tuple(base.items()))


MX_SPECS = {lang: _mx_spec(tc) for lang, tc in LANG_TC.items()}


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


# --------------------------------------------------------------------------- 2b: other languages
MX_FAMILIES = ("hex_cpp", "hex_go", "hex_java", "hex_js", "multiple_rs", "multiple_ts")
FAM_LANG = {"hex_cpp": "cpp", "hex_go": "go", "hex_java": "java", "hex_js": "js",
            "multiple_rs": "rs", "multiple_ts": "ts"}
READ_LANG = {"cpp": "cpp", "go": "go", "java": "java", "js": "javascript", "rs": "rust",
             "ts": "typescript"}
FENCE_TAG = {"cpp": "cpp", "go": "go", "java": "java", "js": "javascript", "rs": "rust",
             "ts": "typescript"}
# Amendment 9 controls, written before any 2b response is read.
MAIN_STUB = {"cpp": "\nint main(){\n    return 0;\n}\n", "go": "\nfunc main() {\n}\n",
             "java": "\npublic class Main {\n    public static void main(String[] args) {\n    }\n}\n",
             "js": "\n", "rs": "\nfn main() {\n}\n", "ts": "\n"}
FAIL_BODY = {"cpp": "    abort();\n}\n", "go": "    panic(\"stub\")\n}\n",
             "java": "        throw new RuntimeException(\"stub\");\n    }\n}\n",
             "js": "  return undefined;\n}\n", "rs": "    todo!()\n", "ts": "    throw new Error(\"stub\");\n}\n"}
# Hand-written MultiPL-E references (bodies after the prompt), the only positive controls for
# Rust and TypeScript (MultiPL-E ships no reference solutions).
MX_REFS = {
    ("rs", "HumanEval_0_has_close_elements"): "    for i in 0..numbers.len() {\n        for j in 0..numbers.len() {\n            if i != j && (numbers[i] - numbers[j]).abs() < threshold {\n                return true;\n            }\n        }\n    }\n    false\n",
    ("rs", "HumanEval_2_truncate_number"): "    number % 1.0\n",
    ("rs", "HumanEval_4_mean_absolute_deviation"): "    let n = numbers.len() as f64;\n    let m: f64 = numbers.iter().sum::<f64>() / n;\n    numbers.iter().map(|x| (x - m).abs()).sum::<f64>() / n\n",
    ("rs", "HumanEval_23_strlen"): "    string.len() as isize\n",
    ("ts", "HumanEval_0_has_close_elements"): "  for (let i = 0; i < numbers.length; i++)\n    for (let j = 0; j < numbers.length; j++)\n      if (i !== j && Math.abs(numbers[i] - numbers[j]) < threshold) return true;\n  return false;\n}\n",
    ("ts", "HumanEval_2_truncate_number"): "  return number % 1.0;\n}\n",
    ("ts", "HumanEval_4_mean_absolute_deviation"): "  const m = numbers.reduce((a, b) => a + b, 0) / numbers.length;\n  return numbers.reduce((a, b) => a + Math.abs(b - m), 0) / numbers.length;\n}\n",
    ("ts", "HumanEval_23_strlen"): "  return string.length;\n}\n",
}
GATE_MX = SANDBOX.parent / "experiments/out/s1_7_gate_mx.json"


class Mx:
    """hex_cpp, hex_go, hex_java, hex_js (HumanEval-X, CodeGeeX harness) and multiple_rs,
    multiple_ts (MultiPL-E harness), criteria Amendment 9."""
    kind = "mx"
    spec = MX_SPECS["cpp"]                    # replaced per payload in sandbox()

    def __init__(self):
        self.hx = _index("humaneval_x_tests", "task_id")
        self.hx.update(_index("humaneval_x_go_tests", "task_id"))
        self.me = {f"{r['language']}::{r['name']}": r for r in _jsonl(SOURCES / "multipl_e_tests.jsonl")}

    def up(self, task_id, family):
        return f"{family}::{upstream_id(task_id, family)}"

    @staticmethod
    def _split(up):
        fam, u = up.split("::", 1)
        return fam, u, FAM_LANG[fam]

    def _row(self, up):
        fam, u, lang = self._split(up)
        return self.hx.get(u) if fam.startswith("hex_") else self.me.get(f"{lang}::{u}")

    def has_tests(self, up):
        return self._row(up) is not None

    def read_lang(self, up):
        return READ_LANG[self._split(up)[2]]

    def entry(self, up):
        fam, u, lang = self._split(up)
        r = self._row(up)
        if lang == "go":
            return re.findall(r"func\s+(\w+)", r["declaration"])[-1]
        if lang == "js":
            return re.findall(r"const\s+(\w+)\s*=", r["declaration"])[-1]
        if lang in ("cpp", "java"):   # last signature followed by "{" (CPP/38 ends inside a body)
            sigs = [n for n in re.findall(r"(\w+)\s*\([^(){};]*\)\s*(?:throws[\w.,\s]+)?\{", r["declaration"])
                    if n not in ("if", "for", "while", "switch", "catch")]
            return sigs[-1]
        if lang == "rs":
            return re.findall(r"fn\s+(\w+)", r["prompt"])[-1]
        return re.findall(r"function\s+(\w+)", r["prompt"])[-1]

    def _body(self, up):
        fam, u, lang = self._split(up)
        r = self._row(up)
        b = {"lang": lang, "task_id": u, "entry_point": self.entry(up), "prompt": r["prompt"]}
        if fam.startswith("hex_"):
            b.update(test=r["test"], declaration=r["declaration"], import_=None)
            if lang == "go":
                b.update({"import": r["import"], "test_setup": r["test_setup"]})
        else:
            b.update(test=r["tests"], declaration="")
        b.pop("import_", None)
        return b

    def canonical(self, up, version=None):
        r = self._row(up)
        if not self._split(up)[0].startswith("hex_"):
            return None
        return {**self._body(up), "mode": "canonical", "code": r["canonical_solution"]}

    def controls(self, up) -> list[tuple[str, dict, str]]:
        """(name, payload, expected status). Uses task fields only, never a response."""
        fam, u, lang = self._split(up)
        b, r = self._body(up), self._row(up)
        fence = lambda code: f"Here is the function.\n\n```{FENCE_TAG[lang]}\n{code}\n```\n"
        out = [("negative", {**b, "mode": "canonical", "code": FAIL_BODY[lang]}, "fail")]
        if fam.startswith("hex_"):
            full = r["prompt"] + r["canonical_solution"]
            out.append(("synthetic_full_main", {**b, "mode": "response",
                                                "response": fence(full + MAIN_STUB[lang])}, "pass"))
        else:
            stub = r["prompt"] + FAIL_BODY[lang] + ("}\n" if lang == "rs" else "")
            out.append(("synthetic_stub_main", {**b, "mode": "response",
                                                "response": fence(stub + MAIN_STUB[lang])}, "fail"))
            ref = MX_REFS.get((lang, u))
            if ref:
                out.append(("reference", {**b, "mode": "canonical", "code": ref
                                         }, "pass"))
                closed = ref + ("}\n" if lang == "rs" else "")
                out.append(("reference_full_main", {**b, "mode": "response",
                                                    "response": fence(r["prompt"] + closed + MAIN_STUB[lang])}, "pass"))
        return out

    def payload(self, up, response, gt, version=None):
        return {**self._body(up), "mode": "response", "response": response}


def mx_samples(long, stats) -> pd.DataFrame:
    return _with_keys(pd.concat([family_sample(long, stats, f) for f in MX_FAMILIES], ignore_index=True))


def mx_gate(s: pd.DataFrame, mx: "Mx") -> bool:
    """Amendment 9 gate, before any 2b response is loaded: on each family's sampled tasks, the
    documented canonical (HumanEval-X) and every control must give its expected status; more
    than 10% of tasks with a failed control stops that family."""
    cache = Cache(OUT / "s1_7_runs_mx.jsonl")
    ups = sorted(set(mx.up(t, f) for t, f in zip(s.task_id, s.family)))
    jobs, expect = [], {}
    extra = [f"{f}::{n}" for (lang, n) in MX_REFS for f in ("multiple_rs", "multiple_ts")
             if FAM_LANG[f] == lang]
    for u in ups + [x for x in extra if x not in ups]:
        c = mx.canonical(u)
        if c is not None:
            jobs.append((f"canonical|{u}|None", c)); expect[f"canonical|{u}|None"] = (u, "pass")
        for name, pl, exp in mx.controls(u):
            jobs.append((f"control:{name}|{u}", pl)); expect[f"control:{name}|{u}"] = (u, exp)
    run_many(jobs, cache, "mx", MX_SPECS["cpp"])
    gate, sampled = {}, set(ups)
    for rid, (u, exp) in expect.items():
        fam = u.split("::")[0]
        got = (cache.get(rid) or {}).get("status")
        g = gate.setdefault(fam, {"tasks": set(), "bad_tasks": set(), "bad": []})
        if u in sampled:
            g["tasks"].add(u)
        if got != exp:
            g["bad"].append({"run": rid, "expected": exp, "got": got})
            if u in sampled:
                g["bad_tasks"].add(u)
    rep = {f: {"tasks": len(g["tasks"]), "tasks_with_failed_control": len(g["bad_tasks"]),
               "share": round(len(g["bad_tasks"]) / max(len(g["tasks"]), 1), 4), "failures": g["bad"]}
           for f, g in sorted(gate.items())}
    GATE_MX.write_text(json.dumps(rep, indent=2, sort_keys=True) + "\n")
    print({f: (v["tasks_with_failed_control"], v["tasks"]) for f, v in rep.items()})
    return all(v["share"] <= GATE_MAX_CANONICAL_FAIL for v in rep.values())


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
    if payload.get("family_kind") == "ds1000":
        spec = DS_SPEC
    if kind == "mx":
        spec = MX_SPECS[payload["lang"]]
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
    gate_only = "--gate-only" in sys.argv          # Amendment 4: no response is loaded
    resp = {} if gate_only else load_responses(s)
    s = s.copy()
    s["upstream"] = [fam_obj.up(t, f) if not smp.startswith("read_") else upstream_id(t, f)
                     for t, f, smp in zip(s.task_id, s.family, s["sample"])]
    s["has_response"] = [isinstance(resp.get((t, k)), str) and bool(resp[(t, k)].strip())
                         for t, k in zip(s.task_id, s.key)]
    s["has_tests"] = [(not smp.startswith("read_")) and fam_obj.has_tests(u)
                      for u, smp in zip(s.upstream, s["sample"])]

    # 2. reading, blind to released scores
    reading = [] if not gate_only else None
    for r in ([] if gate_only else s.drop_duplicates("cell_id").itertuples()):
        ep = fam_obj.entry(r.upstream) if r.has_tests else None
        reading.append({"cell_id": r.cell_id, **read_cell(resp.get((r.task_id, r.key)),
                                                       (fam_obj.read_lang(r.upstream) if hasattr(fam_obj, "read_lang")
                                                        else "python"), ep)})
    if not gate_only:
        pd.DataFrame(reading).sort_values("cell_id").to_csv(OUT / f"s1_7_reading_{part}.csv",
                                                             index=False, lineterminator="\n")

    cache = Cache(OUT / f"s1_7_runs_{part}.jsonl")
    ex = s[~s["sample"].isin(READ_ONLY) & s.has_tests]
    if gate_only and fam_obj.kind == "bcb":
        ex = ex[ex["sample"].isin(BCB_FAMILIES)]
    versions = {u: fam_obj.versions(u) if hasattr(fam_obj, "versions") else (None,) for u in set(ex.upstream)}

    # 3a. harness controls
    jobs = []
    for u in sorted(set(ex.upstream)):
        for v in versions[u]:
            c = fam_obj.canonical(u, v) if v else fam_obj.canonical(u)
            if c is not None:
                jobs.append((f"canonical|{u}|{v}", c))
    run_many(jobs, cache, fam_obj.kind, fam_obj.spec)

    if fam_obj.kind == "bcb":                                    # Amendment 4 harness gate
        gate = {}
        main = ex[ex["sample"].isin(BCB_FAMILIES)]
        for fam, g in main.groupby("family"):
            ups = sorted(set(g.upstream))
            bad = [u for u in ups if (cache.get(f"canonical|{u}|None") or {}).get("status") != "pass"]
            gate[fam] = {"tasks": len(ups), "canonical_fail": len(bad), "share": len(bad) / len(ups),
                         "failed": bad}
        (OUT / "s1_7_gate_bcb.json").write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n")
        print({f: round(v["share"], 3) for f, v in gate.items()})
        if any(v["share"] > GATE_MAX_CANONICAL_FAIL for v in gate.values()) or "--gate-only" in sys.argv:
            print("STOP: harness gate (Amendment 4)" if "--gate-only" not in sys.argv else "gate only")
            return None

    if gate_only:                                                # controls only, no response
        ctl = pd.DataFrame([{"upstream": u, "version": v,
                             "status": (cache.get(f"canonical|{u}|{v}") or {}).get("status")}
                            for u in sorted(set(ex.upstream)) for v in versions[u]])
        ctl["family"] = ctl.upstream.str.split("::").str[0]
        print(ctl.groupby(["family", "version"], dropna=False).status.value_counts().to_string())
        return None

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
    elif part == "mx":
        s, mx = mx_samples(long, stats), Mx()
        s.drop(columns=["cell_id"]).to_csv(OUT / "s1_7_samples_mx.csv", index=False, lineterminator="\n")
        if not mx_gate(s, mx) or "--gate-only" in sys.argv:
            print("STOP: Amendment 9 gate" if "--gate-only" not in sys.argv else "gate only")
            return
        run_part("mx", s, mx, long)
    else:
        raise SystemExit(part)


if __name__ == "__main__" and sys.argv[1] not in ("probe", "cruxeval_a2"):
    main(sys.argv[1])


# --------------------------------------------------------------------------- mechanism probes
PROBES = {"lcb_stdin": ("lcb", "allzero_lcb"), "ds_raw": ("ds1000", "allzero_ds1000")}


def probe(name: str) -> pd.DataFrame:
    """Run one fault model on every executed cell of its samples; report how often the probe's
    outcome equals the released S0.1 outcome, split by the main re-run's outcome."""
    c = pd.read_csv(OUT / "s1_7_cells_py.csv")
    c = c[c["sample"].isin(PROBES[name]) & (c.key_role == "graded") & c.rerun.isin(["pass", "fail"])]
    resp = load_responses(c.assign(key=c.key))
    py = Py()
    cache = Cache(OUT / "s1_7_runs_py.jsonl")
    jobs = [(f"probe:{name}|{r.task_id}|{r.key}",
             {**py.payload(r.upstream, resp[(r.task_id, r.key)], None), "probe": name})
            for r in c.itertuples()]
    run_many(jobs, cache, "py", PY_SPEC)
    c = c.assign(probe=[cache.get(f"probe:{name}|{t}|{k}")["status"] for t, k in zip(c.task_id, c.key)])
    c["probe_agrees_release"] = (c.probe == "pass") == c.released_pass
    c[["sample", "task_id", "model", "key", "released_pass", "rerun", "probe",
       "probe_agrees_release"]].to_csv(OUT / f"s1_7_probe_{name}.csv", index=False, lineterminator="\n")
    t = pd.crosstab([c["sample"], c.rerun], c.probe_agrees_release)
    print(t.to_string())
    return c


def probe_bcb(name: str = "block_sci") -> pd.DataFrame:
    """BigCodeBench fault model: the five scientific libraries cannot be imported."""
    c = pd.read_csv(OUT / "s1_7_cells_bcb.csv")
    c = c[c["sample"].isin(BCB_FAMILIES + ("allzero_bcb",)) & (c.key_role == "graded")
          & c.rerun.isin(["pass", "fail"])].drop_duplicates(["task_id", "key"])
    resp = load_responses(c)
    b = Bcb()
    cache = Cache(OUT / "s1_7_runs_bcb.jsonl")

    def gt(u):
        r = cache.get(f"canonical|{u}|None")
        return (r["cell"] or {}).get("elapsed") if r and r["status"] == "pass" else None

    jobs = [(f"probe:{name}|{r.task_id}|{r.key}",
             {**b.payload(r.upstream, resp[(r.task_id, r.key)], gt(r.upstream)), "probe": name})
            for r in c.itertuples()]
    run_many(jobs, cache, "bcb", BCB_SPEC)
    c = c.assign(probe=[cache.get(f"probe:{name}|{t}|{k}")["status"] for t, k in zip(c.task_id, c.key)])
    c["probe_agrees_release"] = (c.probe == "pass") == c.released_pass
    c[["sample", "task_id", "upstream", "model", "key", "released_pass", "rerun", "probe",
       "probe_agrees_release"]].to_csv(OUT / f"s1_7_probe_{name}.csv", index=False, lineterminator="\n")
    return c


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "probe" and sys.argv[2] == "block_sci":
    probe_bcb()
elif __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[1] == "probe":
    probe(sys.argv[2])


# --------------------------------------------------------------------------- CRUXEval under A2
A2_FORMATS = {   # synthetic answers built from CRUXEval's `output` field (controls, no response)
    "fenced": "Let me trace the function step by step.\n\n```python\n{o}\n```\n",
    "label": "Tracing the loop gives the result.\n\n**Output:** {o}\n",
    "assert": "```python\nassert f({i}) == {o}\n```\n",
    "inline": "So the call returns `{o}`.",
}


def cruxeval_a2() -> pd.DataFrame:
    """Criteria Amendment 8: CRUXEval value extraction, judged against released >= 0.8 (CLAIMS
    A2); the 2a literal rule against released = 1.0 is the strict sensitivity."""
    c = pd.read_csv(OUT / "s1_7_cells_py.csv")
    c = c[c["sample"].isin(["cruxeval", "cruxeval_0.8"]) & (c.key_role == "graded")
          & c.has_response.astype(bool) & c.has_tests.astype(bool)]
    py, cache = Py(), Cache(OUT / "s1_7_runs_py.jsonl")
    ctl = []
    for u in sorted(set(c[c["sample"] == "cruxeval"].upstream)):
        b = py._body(u, None)
        for name, tpl in A2_FORMATS.items():
            ctl.append((f"a2ctl:{name}|{u}", {**b, "mode": "response", "rule": "a2",
                                              "response": tpl.format(o=b["output"], i=b["input"])}, "pass"))
        wrong = "```python\n" + repr(("not", "the", "output")) + "\n```\n"
        ctl.append((f"a2ctl:wrong|{u}", {**b, "mode": "response", "rule": "a2", "response": wrong}, "fail"))
    run_many([(rid, pl) for rid, pl, _ in ctl], cache, "py", PY_SPEC)
    bad = [(rid, exp, cache.get(rid)["status"]) for rid, _, exp in ctl if cache.get(rid)["status"] != exp]
    (OUT / "s1_7_cruxeval_a2_controls.json").write_text(json.dumps(
        {"controls": len(ctl), "unexpected": bad}, indent=2) + "\n")
    print(f"A2 controls: {len(ctl)}, unexpected {len(bad)}")
    if len(bad) > 0.10 * len(ctl):
        print("STOP: A2 extraction controls")
        return None
    resp = load_responses(c)
    rid = lambda r, k: f"a2:{r.task_id}|{r.key}|{k}"
    run_many([(rid(r, 1), {**py.payload(r.upstream, resp[(r.task_id, r.key)], None), "rule": "a2"})
              for r in c.itertuples()], cache, "py", PY_SPEC)
    rel_a2 = c.released_score >= 0.8 - 1e-9
    dis = [r for r, ra in zip(c.itertuples(), rel_a2) if (cache.get(rid(r, 1))["status"] == "pass") != ra]
    run_many([(rid(r, k), {**py.payload(r.upstream, resp[(r.task_id, r.key)], None), "rule": "a2"})
              for r in dis for k in range(2, FLAKY_RUNS + 1)], cache, "py", PY_SPEC)
    rows = []
    for r, ra in zip(c.itertuples(), rel_a2):
        runs = [cache.get(rid(r, k)) for k in range(1, FLAKY_RUNS + 1)]
        runs = [x for x in runs if x]
        cell = runs[0].get("cell") or {}
        rows.append({"sample": r.sample, "task_id": r.task_id, "upstream": r.upstream, "model": r.model,
                     "key": r.key, "unconfirmed": r.unconfirmed, "released_score": r.released_score,
                     "released_pass_a2": bool(ra), "released_pass_strict": bool(r.released_pass),
                     "rerun_strict": r.rerun, "agree_strict": r.agree,
                     "rerun_a2": runs[0]["status"], "rerun_a2_all": "/".join(x["status"] for x in runs),
                     "flaky_a2": len({x["status"] for x in runs}) > 1, "a2_rule": cell.get("a2_rule"),
                     "agree_a2": (runs[0]["status"] == "pass") == bool(ra),
                     "log": f"s1_7_runs_py.jsonl#{rid(r, 1)}"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "s1_7_cruxeval_a2.csv", index=False, lineterminator="\n")
    for smp, g in out.groupby("sample"):
        ex = g[g.rerun_a2.isin(["pass", "fail"])]
        print(smp, "n_exec", len(ex), "agree_a2", int(ex.agree_a2.sum()),
              "| strict agree", int(g.agree_strict.astype(bool).sum()), "of", len(g))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "cruxeval_a2":
    cruxeval_a2()
