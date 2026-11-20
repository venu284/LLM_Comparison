"""S1.5 task text: builders, the tasks.jsonl join, adapters and the prompt table."""

import inspect
import json
import re

import pandas as pd
import pytest

from router.data import prompts as P
from router.data.sources import family_of

HAVE_TASKS = P.TASKS_JSONL.is_file()
needs_tasks = pytest.mark.skipif(not HAVE_TASKS, reason="external data not fetched")


def _extract(name):
    path = P.SOURCES_DIR / f"{name}.jsonl"
    if not path.is_file():
        pytest.skip(f"extract {name} not built (scripts/fetch_data.py)")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


# --- builders (no data) -----------------------------------------------------------------------

def test_fenced_wrapper_drops_trailing_whitespace_before_the_fence():
    assert P.wrap("hex_cpp", "int f(){ \n") == "Complete the following cpp function.\n\n```cpp\nint f(){\n```"
    assert P.wrap("ds1000", "x\n") == "x\n"


def test_lcb_wrappers_follow_starter_code():
    starter = "class Solution:\n    def maxSum(self, nums: List[int]) -> int:\n        "
    assert P.solution_method(starter) == "maxSum"
    method = P.wrap("lcb", "Q", starter_code=starter)
    assert "implement the method `maxSum`" in method and method.endswith("-> int:\n```")
    assert P.wrap("lcb", "Q").endswith("standard output.\n\nQ")
    assert P.wrap("lcb_testgen", "Q").endswith("Do not solve the problem.\n\nQ")
    assert P.template_name("lcb", starter) == "lcb:method" and P.template_name("lcb") == "lcb:stdin"


def test_families_without_template_evidence_have_no_wrapper():
    for fam in ("multiple_rs", "multiple_ts", "mbpp_testgen", "swebench"):
        assert P.template_name(fam) is None
        with pytest.raises(ValueError):
            P.wrap(fam, "x")


def test_join_keeps_its_signature():
    assert list(inspect.signature(P.join).parameters) == ["task_ids"]


# --- tasks.jsonl ------------------------------------------------------------------------------

@needs_tasks
def test_builders_rebuild_every_tasks_jsonl_prompt():
    """The wrapper is a deterministic function of the raw text on all 3,297 rows. For cruxeval
    and mbpp_completion the call input and the def header are read back from the prompt, so
    there the test shows only that the template around them is constant."""
    _extract("livecodebench")
    rows = P._tasks_jsonl_rows()
    assert len(rows) == 3297
    for r in rows:
        assert P.rebuild_tasks_jsonl_prompt(r) == r["prompt"], r["task_id"]


def test_content_is_raw_text_plus_payload_without_fence_or_label():
    assert P.content("cruxeval", "def f(x):\n    return x\n", call_input="f(1)") == \
        "def f(x):\n    return x\n\nf(1)"
    assert P.content("mbpp_completion", "Write a function.", def_header="def g(a):") == \
        "Write a function.\n\ndef g(a):"
    starter = "class Solution:\n    def m(self) -> int:\n        "
    assert P.content("lcb", "Q\n", starter_code=starter) == "Q\n\nclass Solution:\n    def m(self) -> int:"
    assert P.content("lcb", "Q\n") == "Q\n"
    for fam in ("humaneval", "bcb", "ds1000", "mbpp", "hex_cpp", "multiple_rs"):
        assert P.content(fam, "x \n") == "x \n"
    with pytest.raises(ValueError):
        P.content("cruxeval", "def f(): pass")


@needs_tasks
def test_tasks_jsonl_payload_inversion_round_trips_on_every_row():
    """The payload read back from `prompt` rebuilds `prompt` exactly (addendum A, step 2)."""
    rows = P._tasks_jsonl_rows()
    assert len(rows) == 3297
    kinds = {}
    for r in rows:
        payload = P.invert_tasks_jsonl_payload(r)
        fam = family_of(r["task_id"])
        assert P.wrap(fam, r["raw_prompt"], **payload) == r["prompt"], r["task_id"]
        kinds.setdefault(fam, set()).update(payload)
    assert kinds["cruxeval"] == {"call_input"} and kinds["mbpp_completion"] == {"def_header"}
    assert kinds["lcb"] == {"starter_code"} and kinds["lcb_testgen"] == {"starter_code"}


@needs_tasks
def test_tasks_jsonl_rows_are_complete_and_content_is_the_router_field():
    texts = P.from_tasks_jsonl()
    assert len(texts) == 3297
    t = texts["dim4_cruxeval_0"]
    assert t.raw_text.startswith("def f(nums):") and "Predict" in t.wrapped_prompt
    assert t.content_text == t.raw_text.rstrip() + "\n\nf([1, 1, 3, 1, 3, 1])"
    joined = P.join(["dim4_cruxeval_0", "no_such_task"])
    assert isinstance(joined, pd.Series) and joined["dim4_cruxeval_0"] == t.content_text
    assert joined["no_such_task"] is None


@needs_tasks
def test_content_carries_the_payload_and_no_instruction_sentence():
    _all_extracts()
    texts = P.task_texts()
    assert P.INSTRUCTIONS and all(len(s) > 20 for s in P.INSTRUCTIONS)
    for t in texts.values():
        for s in P.INSTRUCTIONS:
            assert s not in t.content_text, (t.task_id, s)
        if t.family == "cruxeval":
            call = re.search(r"\nInput: `(.*)`\nOutput:$", t.wrapped_prompt, re.S).group(1)
            assert t.content_text.endswith("\n\n" + call) and call.startswith("f(")
            assert "Input:" not in t.content_text and "Output:" not in t.content_text
        elif t.family == "mbpp_completion":
            header = re.search(r"```python\n(.*)\n    \.\.\.\n```$", t.wrapped_prompt, re.S).group(1)
            assert header.startswith("def ") and t.content_text.endswith("\n\n" + header.rstrip()), t.task_id
        elif t.template and t.template.endswith(":method"):
            assert t.content_text.startswith(t.raw_text.rstrip() + "\n\nclass Solution"), t.task_id
        else:
            assert t.content_text == t.raw_text, t.task_id


# --- adapters and pins ------------------------------------------------------------------------

def _fetch_data():
    import importlib.util
    import sys
    from pathlib import Path

    if "fetch_data" in sys.modules:
        return sys.modules["fetch_data"]
    spec = importlib.util.spec_from_file_location(
        "fetch_data", Path(__file__).resolve().parents[2] / "scripts" / "fetch_data.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_data"] = mod
    spec.loader.exec_module(mod)
    return mod


def _all_extracts():
    for name in {n for n, _ in P.ADAPTERS.values()}:
        _extract(name)


def test_every_text_field_is_pinned_and_allowlisted_in_fetch_data():
    """prompts.py reads only fields that fetch_data.py pins, at the revision it names."""
    from router.data.splits import DS1000_ORIGIN_EXTRACT

    ex = _fetch_data().EXTRACTS
    # ds1000_origin holds DS-1000 metadata only (S1.6 problem ids), no task text.
    assert set(P.SOURCE_FIELDS) == set(ex) - {DS1000_ORIGIN_EXTRACT}
    for name, (source, field) in P.SOURCE_FIELDS.items():
        repo, rev = source.split()[0].split("@")
        assert (ex[name].repo_id, ex[name].revision) == (repo, rev), name
        assert field in ex[name].fields, name
    for family, (name, key) in P.ADAPTERS.items():
        assert key in ex[name].fields, family
    assert "starter_code" in ex["livecodebench"].fields and "language" in ex["multipl_e"].fields


def test_gap_families_have_no_adapter_and_say_what_would_close_them():
    assert not set(P.GAPS) & set(P.ADAPTERS)
    for fam, why in P.GAPS.items():
        assert "needs" in why.lower(), fam


# Overlap of each adapter with tasks.jsonl, as reported by router/experiments/s1_5_prompt_join.py.
ADAPTER_OVERLAP = {
    "bcb": 238, "bcb_algo2": 35, "bcb_completion": 29, "bcb_ds": 14, "bcb_v014": 31,
    "ds1000": 297, "he_testgen": 81, "hex_cpp": 50, "hex_go": 42, "hex_java": 33, "hex_js": 48,
    "lcb": 776, "lcb_testgen": 112,
}


@needs_tasks
def test_adapters_match_tasks_jsonl_on_the_overlap_for_both_fields():
    _all_extracts()
    jl, ad = P.from_tasks_jsonl(), P.adapter_texts()
    seen = {}
    for tid, a in ad.items():
        if tid in jl:
            assert a.content_text == jl[tid].content_text, tid
            assert a.raw_text == jl[tid].raw_text, tid
            assert a.wrapped_prompt == jl[tid].wrapped_prompt, tid
            seen[a.family] = seen.get(a.family, 0) + 1
    assert seen == ADAPTER_OVERLAP


@needs_tasks
def test_upstream_keys_map_one_to_one():
    """Each adapter family's tasks hit distinct upstream rows, and each key is unique upstream
    (`_index` raises otherwise). Covers the LiveCodeBench question_id and MultiPL-E name maps."""
    _all_extracts()
    table = P.prompt_table()
    for family, (name, key) in P.ADAPTERS.items():
        rows = P.read_extract(name)
        if family.startswith("multiple_"):
            rows = [r for r in rows if r["language"] == family.split("_")[1]]
        index = P._index(name, key, rows)
        ids = table.upstream_id[table.family == family]
        assert ids.notna().all() and ids.is_unique, family
        assert set(ids) <= set(index), family


# Joined / total per ID family, as reported by router/experiments/s1_5_prompt_join.py on the
# pinned data.
JOINED = {
    "bcb": (675, 675), "bcb_algo2": (111, 111), "bcb_completion": (89, 89), "bcb_ds": (111, 111),
    "bcb_v014": (111, 111), "bugs2fix": (0, 980), "bugs2fix_sup": (0, 111), "canitedit": (0, 20),
    "codexglue": (0, 1000), "codexglue_sup": (0, 111), "cruxeval": (800, 800),
    "debugbench": (0, 929), "debugbench_sup": (0, 111), "ds1000": (1000, 1000),
    "he_completion": (98, 98), "he_testgen": (131, 131), "hex_cpp": (164, 164),
    "hex_go": (164, 164), "hex_java": (164, 164), "hex_js": (164, 164), "hex_python": (164, 164),
    "humaneval": (100, 100), "lcb": (1000, 1000), "lcb_testgen": (869, 869), "mbpp": (225, 225),
    "mbpp_completion": (124, 124), "mbpp_testgen": (0, 111), "multiple_rs": (156, 156),
    "multiple_ts": (135, 135), "swebench": (0, 71),
}


@needs_tasks
def test_join_counts_per_id_family_are_pinned():
    _all_extracts()
    t = P.prompt_table()
    idt = t[t.split_kind == "id"]
    got = {f: (int(g.joined.sum()), len(g)) for f, g in idt.groupby("family")}
    assert got == JOINED
    assert {f for f, (j, n) in JOINED.items() if j < n} == set(P.GAPS)
    assert (idt.joined == idt.raw_sha256.notna()).all()
    assert (idt.joined == idt.content_sha256.notna()).all()
    # The wrapped field is absent only where no template evidence exists.
    no_wrap = idt[idt.joined & idt.prompt_sha256.isna()]
    assert set(no_wrap.family) == {"multiple_rs", "multiple_ts"}
    assert set(no_wrap.template) == {"none"}


@needs_tasks
def test_eligible_pool_join_rate_meets_the_gate():
    """Addendum A: the S1 gate's 95% is measured on S1.4's eligible pool; swebench (no source)
    is the only gap there."""
    from router.data.scoring import eligible_families

    _all_extracts()
    t = P.prompt_table()
    el = t[(t.split_kind == "id") & t.family.isin(eligible_families())]
    assert (int(el.joined.sum()), len(el)) == (5555, 5626)
    assert set(el.family[~el.joined]) == {"swebench"}
