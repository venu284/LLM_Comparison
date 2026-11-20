"""Task text for CodeRouterBench ID tasks (S1.5).

CodeRouterBench ships no prompt text for its ID tasks, and publishes no prompt template for
them. Two text fields are built per task:

- `raw_text`, the **router field**: the upstream benchmark's task text with no instruction
  wrapper (`raw_prompt` in Route Anything's `tasks.jsonl`, or the same field read from the
  pinned upstream release). The router, every baseline and the cell predictor read this field
  (user decision, 2026-11-20): the wrapper is not evidence of what the models saw, and it works
  as a family label, which S4.4 forbids the difficulty predictor.
- `wrapped_prompt`: Route Anything's reconstruction of the prompt (`prompt` in `tasks.jsonl`),
  "Reconstructed replay prompts ... from the original local benchmark snapshots"
  (`inventory.json`). It is kept only to reproduce P0.6. `wrap()` rebuilds it from the raw text
  for every family, so tasks outside `tasks.jsonl` get the same wrapper.

Sources, in priority order: `tasks.jsonl` (3,297 tasks), then adapters over the pinned extracts
in `external/sources/` (`scripts/fetch_data.py`, EXTRACTS). Families with no source are gaps;
`GAPS` records why.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from router.data.sources import family_of, load_table
from router.settings import EXTERNAL_DIR

TASKS_JSONL = EXTERNAL_DIR / "route-anything-data" / "experiments_x" / "common" / "tasks.jsonl"
SOURCES_DIR = EXTERNAL_DIR / "sources"
TASKS_JSONL_SOURCE = "route-anything-data tasks.jsonl"

# --- Route Anything's wrappers, derived from tasks.jsonl (the builders) ---------------------------

_COMPLETE = "Complete the following Python function."
_BCB = _COMPLETE + " Return the complete function including the imports."
_HE_TESTGEN = ("Write a comprehensive set of unit tests (plain Python `assert` statements or a "
               "unittest class) for the following function. Do not implement the function itself.")
_LCB_STDIN = ("Solve the following problem in Python. Read the input from standard input and write "
              "the answer to standard output.")
_LCB_METHOD = ("Solve the following problem in Python. Use the starter code below and implement the "
               "method `{method}`; return the complete `Solution` class.")
_LCB_TESTGEN = ("Write test cases for the following programming problem: give several "
                "input/expected-output pairs, including edge cases, in the problem's I/O format. "
                "Do not solve the problem.")
_CRUXEVAL = ("You are given a Python function and an input. Predict the exact return value of the "
             "call. Give the output as a Python literal inside a code block.")

# Families whose wrapper is an instruction, then the raw text in a code fence. The raw text's
# trailing whitespace is dropped before the closing fence (hex_cpp CPP/21 ends in "{ \n").
FENCED: dict[str, tuple[str, str]] = {
    "humaneval": (_COMPLETE, "python"),
    "he_completion": (_COMPLETE + " Return the complete function.", "python"),
    "he_testgen": (_HE_TESTGEN, "python"),
    "hex_python": ("Complete the following python function.", "python"),
    "hex_cpp": ("Complete the following cpp function.", "cpp"),
    "hex_go": ("Complete the following go function.", "go"),
    "hex_java": ("Complete the following java function.", "java"),
    "hex_js": ("Complete the following javascript function.", "javascript"),
    **{f: (_BCB, "python") for f in ("bcb", "bcb_algo2", "bcb_completion", "bcb_ds", "bcb_v014")},
}
IDENTITY = {"ds1000", "mbpp"}


def _fence(lang: str, body: str) -> str:
    return f"```{lang}\n{body.rstrip()}\n```"


def solution_method(starter_code: str) -> str:
    """The method a LeetCode-form starter asks for: the first `def` inside `class Solution`."""
    m = re.search(r"class Solution\b.*?\n\s+def (\w+)\(", starter_code, re.S)
    if not m:
        raise ValueError(f"no Solution method in starter code: {starter_code[:80]!r}")
    return m.group(1)


def template_name(family: str, starter_code: str = "") -> str | None:
    if family in IDENTITY:
        return "identity"
    if family in FENCED:
        return f"fenced:{family}"
    if family in ("lcb", "lcb_testgen"):
        return f"{family}:{'method' if starter_code else 'stdin'}"
    if family in ("cruxeval", "mbpp_completion"):
        return family
    return None


def wrap(family: str, raw: str, *, starter_code: str = "", call_input: str | None = None,
         def_header: str | None = None) -> str:
    """Route Anything's wrapped prompt for a family, rebuilt from the raw task text.

    `starter_code` is LiveCodeBench's starter (lcb, lcb_testgen). `call_input` (cruxeval, the
    `f(...)` call) and `def_header` (mbpp_completion, a `def` line) are task fields that no
    pinned source here provides; both families are fully covered by tasks.jsonl, so they are
    only needed to check the template.
    """
    if family in IDENTITY:
        return raw
    if family in FENCED:
        instruction, lang = FENCED[family]
        return f"{instruction}\n\n{_fence(lang, raw)}"
    if family in ("lcb", "lcb_testgen"):
        if family == "lcb":
            head = _LCB_METHOD.format(method=solution_method(starter_code)) if starter_code else _LCB_STDIN
        else:
            head = _LCB_TESTGEN
        text = f"{head}\n\n{raw}"
        return f"{text}\n\n{_fence('python', starter_code)}" if starter_code else text
    if family == "cruxeval":
        if call_input is None:
            raise ValueError("cruxeval needs call_input")
        return f"{_CRUXEVAL}\n\n{_fence('python', raw)}\n\nInput: `{call_input}`\nOutput:"
    if family == "mbpp_completion":
        if def_header is None:
            raise ValueError("mbpp_completion needs def_header")
        return f"{_COMPLETE}\n\n{raw}\n\n```python\n{def_header}\n    ...\n```"
    raise ValueError(f"no wrapper for family {family!r}")


# --- task text -------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TaskText:
    task_id: str
    family: str
    raw_text: str
    wrapped_prompt: str | None   # None where no template evidence exists (template "none")
    prompt_source: str           # source and field of raw_text
    upstream_id: str | None
    template: str                # builder name, or "none"


def upstream_id(task_id: str, family: str) -> str | None:
    """The source benchmark's own id for a task, where the CodeRouterBench id encodes one."""
    body = re.sub(r"^dim\d+_", "", task_id)
    if family in ("lcb", "lcb_testgen"):
        return re.sub(r"^lcb_(testgen_)?", "", body)
    if (m := re.search(r"BigCodeBench/\d+$", body)) and family.startswith("bcb"):
        return m.group(0)
    if family.startswith("hex_") and (m := re.search(r"[A-Za-z+]+/\d+$", body)):
        return m.group(0)                       # HumanEval-X: CPP/0, Go/0, Python/0, ...
    if family in ("humaneval", "he_completion", "he_testgen") and (m := re.search(r"HumanEval_(\d+)$", body)):
        return f"HumanEval/{m.group(1)}"
    if family.startswith("multiple_") and (m := re.search(r"HumanEval_\d+_\w+$", body)):
        return m.group(0)                       # MultiPL-E `name`
    if family == "cruxeval":
        return f"sample_{body.rsplit('_', 1)[1]}"
    if family in ("ds1000", "mbpp", "mbpp_completion", "mbpp_testgen") and (m := re.search(r"_(\d+)$", body)):
        return m.group(1)
    return None


def rebuild_tasks_jsonl_prompt(row: dict) -> str:
    """`wrap()` applied to one tasks.jsonl row, to compare with its `prompt`.

    lcb starter code comes from the LiveCodeBench extract. For cruxeval and mbpp_completion the
    call input and the def header are read back from the prompt itself, so for those two
    families equality shows only that the template around them is constant.
    """
    fam, p = family_of(row["task_id"]), row["prompt"]
    kw: dict = {}
    if fam in ("lcb", "lcb_testgen"):
        lcb = _index_cached("livecodebench", "question_id")
        kw["starter_code"] = lcb[upstream_id(row["task_id"], fam)]["starter_code"]
    elif fam == "cruxeval":
        kw["call_input"] = re.search(r"\nInput: `(.*)`\nOutput:$", p, re.S).group(1)
    elif fam == "mbpp_completion":
        kw["def_header"] = re.search(r"```python\n(.*)\n    \.\.\.\n```$", p, re.S).group(1)
    return wrap(fam, row["raw_prompt"], **kw)


def _tasks_jsonl_rows() -> list[dict]:
    with open(TASKS_JSONL, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def from_tasks_jsonl() -> dict[str, TaskText]:
    """Every tasks.jsonl row: raw_prompt is the router field, prompt the wrapped one."""
    out: dict[str, TaskText] = {}
    for r in _tasks_jsonl_rows():
        tid, fam = r["task_id"], family_of(r["task_id"])
        if tid in out:
            raise ValueError(f"tasks.jsonl repeats {tid}")
        if not r["found"] or not r["raw_prompt"] or not r["prompt"]:
            raise ValueError(f"tasks.jsonl row {tid} is not found or has empty text")
        # An lcb prompt ends in a code fence exactly when the task has starter code.
        has_starter = "starter" if r["prompt"].endswith("```") else ""
        out[tid] = TaskText(tid, fam, r["raw_prompt"], r["prompt"], f"{TASKS_JSONL_SOURCE}:raw_prompt",
                            upstream_id(tid, fam), template_name(fam, has_starter))
    return out


@lru_cache(maxsize=1)
def task_texts() -> dict[str, TaskText]:
    """Task text for every task with a source: tasks.jsonl first, then the adapters."""
    out = dict(from_tasks_jsonl())
    for tid, t in adapter_texts().items():
        out.setdefault(tid, t)
    return out


# --- adapters over the pinned extracts ------------------------------------------------------------

# Extract name -> (repo@revision, field read as raw text). Matches scripts/fetch_data.py EXTRACTS;
# tests check the two agree.
SOURCE_FIELDS: dict[str, tuple[str, str]] = {
    "ds1000": ("xlangai/DS-1000@4416080ac5cb80bdf7576aefb8f9a0b4d5426a44", "prompt"),
    "bigcodebench": ("bigcode/bigcodebench@b74c0d0bf70d2c0bc459be537895cca163007f1a v0.1.4", "complete_prompt"),
    "humaneval_x": ("THUDM/humaneval-x@62c78627f3072a1454fa0cb0184737cafe5e4198", "prompt"),
    "humanevalplus": ("evalplus/humanevalplus@d32357cf319e50e9c8d8dab5ea876c72b0fd321b", "prompt"),
    "multipl_e": ("nuprl/MultiPL-E@28441b6024e71d4a1c1c0f6bf171c935cd5a43f2", "prompt"),
    "livecodebench": ("livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505", "question_content"),
}

# Family -> (extract, upstream key field, how the key appears in upstream_id()).
ADAPTERS: dict[str, tuple[str, str]] = {
    "ds1000": ("ds1000", "metadata.problem_id"),
    **{f: ("bigcodebench", "task_id") for f in ("bcb", "bcb_algo2", "bcb_completion", "bcb_ds", "bcb_v014")},
    **{f: ("humaneval_x", "task_id") for f in ("hex_cpp", "hex_go", "hex_java", "hex_js")},
    "he_testgen": ("humanevalplus", "task_id"),
    "lcb": ("livecodebench", "question_id"),
    "lcb_testgen": ("livecodebench", "question_id"),
    "multiple_rs": ("multipl_e", "name"),
    "multiple_ts": ("multipl_e", "name"),
}

# Families with no task text: what is missing and the evidence that would close the gap. None
# has a tasks.jsonl row, and CodeRouterBench publishes no ID prompts (only OOD ones). Not
# chased further in S1.5: these families are probably ineligible (S1.4 decides).
GAPS: dict[str, str] = {
    "mbpp_testgen": (
        "the test-generation prompt is not published; the raw MBPP text describes the function, "
        "not the test-writing task (user decision, 2026-11-20). Needs: the prompt construction "
        "from arXiv 2606.22902 or the CodeRouterBench card"),
    "swebench": ("needs: what was given with each issue (repository files, retrieval, patch format); "
                 "the instance ids are SWE-bench's, the prompt is not"),
    "debugbench": "needs: the mapping from dim3_debugbench_<n> to a DebugBench record, and the prompt format",
    "debugbench_sup": "needs: the base of the _sup numbering (1000+) and the source of the extra tasks",
    "canitedit": "needs: which CanItEdit instruction (descriptive or lazy) was given, and the id mapping",
    "bugs2fix": "needs: the CodeXGLUE code-refinement split and row behind dim5_bugs2fix_<n>, and the prompt format",
    "bugs2fix_sup": "needs: the base of the _sup numbering (7000+) and the source of the extra tasks",
    "codexglue": "needs: the CodeXGLUE task, split and row behind dim8_codexglue_<n>, and the prompt format",
    "codexglue_sup": "needs: the base of the _sup numbering (15000+) and the source of the extra tasks",
}

_MULTIPL_E_LANG = {"multiple_rs": "rs", "multiple_ts": "ts"}


@lru_cache(maxsize=None)
def _index_cached(name: str, key: str) -> dict[str, dict]:
    return _index(name, key)


@lru_cache(maxsize=None)
def read_extract(name: str) -> tuple[dict, ...]:
    """Rows of one pinned extract. Missing extract: empty (run scripts/fetch_data.py)."""
    path = SOURCES_DIR / f"{name}.jsonl"
    if not path.is_file():
        return ()
    with open(path, encoding="utf-8") as f:
        return tuple(json.loads(line) for line in f if line.strip())


def _index(name: str, key: str, rows=None) -> dict[str, dict]:
    """Upstream rows by key; a key that appears twice makes the mapping ambiguous."""
    out: dict[str, dict] = {}
    for r in rows if rows is not None else read_extract(name):
        k = str(r[key])
        if k in out:
            raise ValueError(f"{name}: upstream key {key}={k} is not unique")
        out[k] = r
    return out


def adapter_text(task_id: str, family: str, row: dict) -> TaskText:
    """TaskText for one task from its upstream row."""
    name = ADAPTERS[family][0]
    source, field = SOURCE_FIELDS[name]
    raw = row[field]
    if family in _MULTIPL_E_LANG:
        wrapped, template = None, "none"
    else:
        starter = row.get("starter_code") or ""
        wrapped, template = wrap(family, raw, starter_code=starter), template_name(family, starter)
    return TaskText(task_id, family, raw, wrapped, f"{source}:{field}", upstream_id(task_id, family), template)


def adapter_texts() -> dict[str, TaskText]:
    """Task text from the pinned upstream extracts, for every task an adapter covers
    (including tasks that tasks.jsonl also covers, so the two can be compared)."""
    out: dict[str, TaskText] = {}
    table = load_table()
    for family, (name, key) in ADAPTERS.items():
        rows = read_extract(name)
        if not rows:
            continue
        if family in _MULTIPL_E_LANG:
            rows = [r for r in rows if r["language"] == _MULTIPL_E_LANG[family]]
        index = _index(name, key, rows)
        for tid in table.task_id[table.family == family]:
            row = index.get(upstream_id(tid, family))
            if row is not None:
                out[tid] = adapter_text(tid, family, row)
    return out


def join(task_ids) -> pd.Series:
    """Raw task text (the router field) per task id; None where no source covers the task."""
    texts = task_texts()
    return pd.Series({tid: (texts[tid].raw_text if tid in texts else None) for tid in task_ids}, dtype=object)


def sha256_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


TABLE_COLUMNS = ["task_id", "family", "split_kind", "split", "prompt_source", "raw_sha256",
                 "prompt_sha256", "joined", "upstream_id", "template"]


def prompt_table() -> pd.DataFrame:
    """One row per task in the family table: where its text comes from and the hashes of both
    fields. `prompt_sha256` hashes the wrapped prompt (P0.6 reproduction only)."""
    texts = task_texts()
    rows = []
    for r in load_table().itertuples(index=False):
        t = texts.get(r.task_id)
        rows.append({
            "task_id": r.task_id, "family": r.family, "split_kind": r.split_kind, "split": r.split,
            "prompt_source": t.prompt_source if t else None,
            "raw_sha256": sha256_text(t.raw_text) if t else None,
            "prompt_sha256": sha256_text(t.wrapped_prompt) if t else None,
            "joined": t is not None,
            "upstream_id": t.upstream_id if t else upstream_id(r.task_id, r.family),
            "template": t.template if t else None,
        })
    return pd.DataFrame(rows, columns=TABLE_COLUMNS)
