"""S1.8: LLMRouterBench code subsets, a pinned loader and a descriptive profile (no splits, no audit).

LLMRouterBench (arXiv 2601.07206v1; Hugging Face `NPULH/LLMRouterBench`, pinned in
`scripts/fetch_data.py` ARCHIVES) ships one JSON file per (dataset, split, model) under
`external/llmrouterbench/bench-release/<dataset>/<split>/<model dir>/`. Each file holds
`performance`, `cost`, `counts`, `model_name` and `records`; each record holds `index` (a 1-based
list position), `origin_query`, `prompt`, `prompt_tokens`, `completion_tokens`, `cost`, `score`,
`prediction`, `ground_truth` and `raw_output` (SWE-bench records add `instance_id`).

Eligibility follows documented scoring rules only, never observed scores (CLAIMS S0.12, R16). The
rule table below (`RULES`) was written and committed before any score distribution of the corpus
was computed; its commit is the R16 evidence. Each subset carries two citations, as in
`scoring.RULES`:

- `lrb_rule`: how LLMRouterBench produced the score: the paper's Table 9 and, where it exists,
  the evaluator code in the repository (github.com/ynulihao/LLMRouterBench at 811df08, read
  2026-11-21);
- `source_rule`: the source benchmark's own rule (the quotes `scoring.py` already holds).

This module delivers a loader and a profile only. If LLMRouterBench ever becomes a primary
corpus, it needs the S1.4-S1.7 steps again (eligibility review, task text, problem ids and
splits across both corpora, an artefact audit, and pricing for the zero-cost models).
"""

from __future__ import annotations

from dataclasses import dataclass

from router.data.scoring import _HE, _HE_TESTS, _LCB, _LCB_TESTS, _MBPP, _MBPP_TESTS, _SWE_TESTS

# --------------------------------------------------------------------------- citations
LRB_PAPER = "arXiv:2601.07206v1"
LRB_REPO = "github.com/ynulihao/LLMRouterBench@811df08b1b37646064eeb8cb6acfd498bf99885a"
LRB_ACCESSED = "2026-11-21"

_T9 = (f'{LRB_PAPER} App. B, Table 9 ("Detailed information of the datasets"), row {{row}}: '
       '"{metric}", {n} tasks')
_RUNNER = (f'{LRB_REPO} data_collector/runner.py l.260-261: "elif isinstance(is_correct_value, '
           'bool):" / "score = 1.0 if is_correct_value else 0.0"')
_PROMPT_TESTS = ('the prompt template ends "Your code should pass these tests:\\n{test}", so the '
                 'model is shown the test code it is graded on')

_LRB_HE = (_T9.format(row="HumanEval", metric="Code Pass@1, 0-shot", n=164) + "; "
           f'{LRB_REPO} evaluation/HumanEval/humaneval.py l.68: "is_correct = check_correctness('
           "task_id=data['task_id'], completion_id=0, solution=prediction, time_out=10)['passed']\"; "
           'l.61: the checked program is the extracted code + "\\n" + test + "\\n" + '
           '"check({entry_point})"; evaluation/HumanEval/execution.py l.46-47: '
           '"exec(check_program, exec_globals)" then "result.append(\\"passed\\")". ' + _RUNNER)
_LRB_MBPP = (_T9.format(row="MBPP", metric="Code Pass@1, 0-shot", n=974) + "; "
             f'{LRB_REPO} evaluation/MBPP/mbpp.py l.65: "is_correct = check_correctness('
             "task_id=data['task_id'], completion_id=0, solution=prediction, time_out=10)['passed']\"; "
             'l.52: the checked program is the extracted code + "\\n" + "\\n".join(test_list). '
             + _RUNNER)
_LRB_LCB = (_T9.format(row="LiveCodeBench", metric="Code Pass@1, 0-shot", n=1055) + "; "
            f'{LRB_REPO} evaluation/LiveCodeBench/livecodebench.py l.142: "all_tests = '
            'public_test_cases + private_test_cases"; l.148: "evaluate_generation(generations='
            '[prediction], sample=sample, debug=False, timeout=6)"; l.155: "is_correct = True if '
            'correct_number == len(real_results) else False". ' + _RUNNER)
_LRB_SWE = (f'{LRB_PAPER} App. B, Table 9 (performance-cost datasets), row SWE-Bench: "Code '
            'Pass@1, 0-shot", 500 tasks. The repository at 811df08 has no SWE-bench evaluator: '
            'evaluation/ has no SWE-bench module, and external_bench/README.md l.3 calls SWE-bench '
            'one of the "complex third-party benchmarks ... that are too difficult to port into '
            "LLMRouterBench's standard evaluation system\".")
_LRB_ARENA = (f'{LRB_PAPER} App. B, Table 9: ArenaHard "Instruction Following (IF)", "LLM as judge, '
              '0-shot", 750; caption: "\\u201cLLM as judge\\u201d means that we use an auxiliary LLM '
              'to score model outputs with the official prompts: ... for ArenaHard we adopt '
              f'deepseek-v3-0324"; {LRB_REPO} evaluation/ArenaHard/arenahard.py l.286: '
              '"\\"is_correct\\": score  # 0.0, 0.5, or 1.0".')

_SWE_SOURCE = ('arXiv:2310.06770v3 (SWE-bench) §2.2: "If the patch applies successfully and all of '
               'these tests pass we consider the proposed solution to have successfully resolved '
               'the issue." SWE-bench Verified is the 500-instance subset pinned as the '
               '`swebench_verified_ids` extract (princeton-nlp/SWE-bench_Verified@c104f84).')
_ARENA_SOURCE = ('Arena-Hard-Auto (arXiv:2406.11939) was not read for S1.8 and is not quoted. '
                 'Ineligibility rests on LLMRouterBench\'s own judge rule, quoted in lrb_rule.')


@dataclass(frozen=True)
class SubsetRule:
    score_type: str          # "exec" or "judge"
    partial_credit: str      # "no" (binary by code), "yes", or "unstated"
    test_checkable: bool     # the source benchmark ships tests a verifier can run
    lrb_rule: str            # how LLMRouterBench scored it, with quotes
    source_rule: str         # the source benchmark's own rule, with a quote
    test_checkable_cite: str
    notes: str = ""

    @property
    def eligible(self) -> bool:
        """S0.12: execution-scored and test-checkable."""
        return self.score_type == "exec" and self.test_checkable


# --------------------------------------------------------------------------- rule table
# Frozen 2026-11-21 before any score distribution of the corpus was computed (R16). Changing it
# is a plan change. Keys are the subset directories inside bench-release/.
RULES: dict[str, SubsetRule] = {
    "humaneval/test": SubsetRule(
        "exec", "no", True, _LRB_HE, _HE, _HE_TESTS,
        "Base HumanEval tests (the `test` field of data/HumanEval/HumanEval.jsonl), not HumanEval+; "
        "CodeRouterBench used HumanEval+ variants. " + _PROMPT_TESTS + " (humaneval.py l.14-19)."),
    "mbpp/test": SubsetRule(
        "exec", "no", True, _LRB_MBPP, _MBPP, _MBPP_TESTS,
        "Base MBPP `test_list` (three asserts), not MBPP+; 974 tasks is MBPP's full set, not its "
        "500-task test split, although the split is named 'test'. " + _PROMPT_TESTS
        + " (mbpp.py l.14-19)."),
    "livecodebench/test": SubsetRule(
        "exec", "no", True, _LRB_LCB, _LCB, _LCB_TESTS,
        "Public and private tests. If the private tests fail to decode, livecodebench.py l.138-139 "
        "falls back to an empty list (public tests only), silently. Which LiveCodeBench release "
        "files the 1,055 tasks come from is not stated (load_data globs data/LiveCodeBench/*.jsonl)."),
    "swe-bench/verified": SubsetRule(
        "exec", "unstated", True, _LRB_SWE, _SWE_SOURCE, _SWE_TESTS,
        "Eligible by the paper's Pass@1 label and SWE-bench's own resolved rule, but how the "
        "released scores were produced (harness version, single-shot or agentic) is not documented "
        "and no scorer exists in the repository: open question for the user. The record prompts "
        "begin 'You will be provided with a partial code base and an issue statement', which is "
        "SWE-bench's retrieval (single-shot patch) prompt format, not an agent scaffold."),
    "arenahard_coding": SubsetRule(
        "judge", "yes", False, _LRB_ARENA, _ARENA_SOURCE,
        "No executable tests: answers are compared by a judge model.",
        "Judge-scored, so ineligible as a whole (S0.12). Not extracted (fetch_data ARCHIVES)."),
}

# The subsets the loader reads: the eligible ones, which are also the only ones extracted.
CODE_SUBSETS: tuple[str, ...] = tuple(k for k, r in RULES.items() if r.eligible)

# Family name per subset. Prefixed, because CodeRouterBench also has `humaneval` and `mbpp`.
FAMILY: dict[str, str] = {
    "humaneval/test": "lrb_humaneval",
    "mbpp/test": "lrb_mbpp",
    "livecodebench/test": "lrb_livecodebench",
    "swe-bench/verified": "lrb_swebench_verified",
}

# Model-axis rule, from documentation only. The `openrouter` files are OpenRouter's own router,
# a routing baseline in the paper, not a model: App. B.3 "OpenRouter": "We use the official API
# provided by OpenRouter with the openroute/auto model, which is currently an advanced commercial
# LLM router available." They are kept out of the model axis and counted separately.
ROUTER_BASELINES: frozenset[str] = frozenset({"openrouter"})


def rule(subset: str) -> SubsetRule:
    return RULES[subset]


def eligible_subsets() -> list[str]:
    return sorted(CODE_SUBSETS)


# --------------------------------------------------------------------------- loader
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from functools import lru_cache  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from router.data.schema import NULL_SCORE, OutcomeMatrix, long_frame  # noqa: E402
from router.settings import EXTERNAL_DIR  # noqa: E402

ROOT = EXTERNAL_DIR / "llmrouterbench" / "bench-release"
SOURCES = EXTERNAL_DIR / "sources"
DIMENSION = "code"   # LLMRouterBench's own category for all four subsets (Table 9)

# How each subset's records were mapped to source keys (the `splits.source_key` namespace). No
# record carries a source id except SWE-bench's `instance_id`; `index` is a 1-based list position
# (runner.py l.280 "index=index + 1,  # 1-based indexing") and is never used as an id.
MAPPING = {
    "humaneval/test": "he/N: origin_query equals the `prompt` of exactly one task in the pinned "
                      "openai_humaneval extract (HumanEval/N).",
    "mbpp/test": "mbpp/N: the record prompt equals, after collapsing whitespace, LLMRouterBench's "
                 "MBPP template (mbpp.py l.14-19) filled with the `text` and `test_list` of exactly "
                 "one task in the pinned mbpp_full extract. The text alone is shared by four "
                 "pairs of MBPP tasks; the tests separate them.",
    "livecodebench/test": "lcb/<question_id>: origin_query equals the `question_content` of "
                          "exactly one problem in the pinned livecodebench extract (release v1-v6).",
    "swe-bench/verified": "swe/<instance_id>: the record's own `instance_id`, checked against the "
                          "pinned swebench_verified_ids extract.",
}
_MBPP_TEMPLATE = ("You are an expert Python programmer, and here is your task:\n{question}\n\n"
                  "Your code should pass these tests:\n{test}")


def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def sha256_text(text: str | None) -> str | None:
    return None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()


def _extract(name: str) -> list[dict]:
    with open(SOURCES / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _unique(pairs) -> dict:
    """{key: value} for keys that occur once; keys seen twice map to None (ambiguous)."""
    out: dict = {}
    for k, v in pairs:
        out[k] = None if k in out else v
    return out


@lru_cache(maxsize=1)
def _source_indexes() -> dict[str, dict]:
    he = _unique((r["prompt"], f"he/{int(r['task_id'].split('/')[1])}") for r in _extract("openai_humaneval"))
    mbpp = _unique((_ws(_MBPP_TEMPLATE.format(question=r["text"], test="\n".join(r["test_list"]))),
                    f"mbpp/{int(r['task_id'])}") for r in _extract("mbpp_full"))
    lcb = _unique((r["question_content"], f"lcb/{r['question_id']}") for r in _extract("livecodebench"))
    swe = {r["instance_id"] for r in _extract("swebench_verified_ids")}
    return {"he": he, "mbpp": mbpp, "lcb": lcb, "swe": swe}


def source_key(subset: str, record: dict) -> str | None:
    """The `splits.source_key`-namespace key of a record, or None when no unique match exists."""
    ix = _source_indexes()
    if subset == "humaneval/test":
        return ix["he"].get(record["origin_query"])
    if subset == "mbpp/test":
        return ix["mbpp"].get(_ws(record["prompt"]))
    if subset == "livecodebench/test":
        return ix["lcb"].get(record["origin_query"])
    if subset == "swe-bench/verified":
        iid = record.get("instance_id")
        return f"swe/{iid}" if iid in ix["swe"] else None
    raise KeyError(subset)


def task_id(subset: str, index: int) -> str:
    return f"{FAMILY[subset]}_{int(index)}"


@dataclass
class LrbData:
    """Everything the loader reads. `long` is in `schema.LONG_COLUMNS` form and holds only the
    cells a model ran (ragged across subsets, never zero-filled); `matrices` has one dense
    OutcomeMatrix per subset; `cells` adds per-cell fields the schema has no column for."""

    long: pd.DataFrame
    cells: pd.DataFrame
    tasks: pd.DataFrame
    files: pd.DataFrame
    cost_flags: pd.DataFrame
    matrices: dict[str, OutcomeMatrix]

    @property
    def zero_cost(self) -> dict[str, list[str]]:
        """{family: models whose every recorded cost is 0}: excluded from cost metrics, kept for accuracy."""
        z = self.cost_flags[self.cost_flags.zero_cost]
        return {f: sorted(g.model) for f, g in z.groupby("family")}


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load(root: Path = ROOT, include_router_baselines: bool = False) -> LrbData:
    """Read the extracted code subsets. Files of ROUTER_BASELINES are listed in `files` and left out
    of every other table unless `include_router_baselines` (for raw counts only)."""
    files, cells, tasks = [], [], {}
    for subset in CODE_SUBSETS:
        fam = FAMILY[subset]
        for path in sorted((root / subset).glob("*/*.json")):
            d = _read(path)
            model = d["model_name"]
            router = model in ROUTER_BASELINES
            files.append({"subset": subset, "family": fam, "model_dir": path.parent.name,
                          "model": model, "file": str(path.relative_to(root)),
                          "records": len(d["records"]), "counts": d["counts"],
                          "file_cost": d["cost"], "performance": d["performance"],
                          "router_baseline": router})
            if router and not include_router_baselines:
                continue
            for r in d["records"]:
                tid = task_id(subset, r["index"])
                if tid not in tasks:
                    tasks[tid] = {"task_id": tid, "family": fam, "subset": subset, "index": r["index"],
                                  "source_key": source_key(subset, r),
                                  "instance_id": r.get("instance_id"),
                                  "origin_query": r["origin_query"], "prompt": r["prompt"],
                                  "origin_query_sha256": sha256_text(r["origin_query"]),
                                  "prompt_sha256": sha256_text(r["prompt"])}
                elif tasks[tid]["origin_query_sha256"] != sha256_text(r["origin_query"]):
                    raise ValueError(f"{path}: index {r['index']} holds a different task than in other files")
                raw = r.get("raw_output") or ""
                cells.append({"task_id": tid, "model": model, "family": fam, "subset": subset,
                              "score": r["score"], "recorded_cost": r["cost"],
                              "input_tokens": r["prompt_tokens"], "output_tokens": r["completion_tokens"],
                              "raw_output_chars": len(raw), "raw_output_empty": not raw.strip(),
                              "processing_failed": raw.startswith("Processing failed")})
    files = pd.DataFrame(files)
    cells = pd.DataFrame(cells)
    tasks = pd.DataFrame(list(tasks.values()))
    if cells.duplicated(["task_id", "model"]).any():
        raise ValueError("a (task, model) cell appears twice")

    flags = (cells.groupby(["family", "model"])
             .agg(cells=("recorded_cost", "size"), zero_cost_cells=("recorded_cost", lambda c: int((c == 0).sum())),
                  recorded_cost_total=("recorded_cost", "sum"))
             .reset_index())
    flags["zero_cost"] = flags.zero_cost_cells == flags.cells
    flags["partial_zero_cost"] = (flags.zero_cost_cells > 0) & ~flags.zero_cost
    zero = set(zip(flags.family[flags.zero_cost], flags.model[flags.zero_cost]))
    cells["cost_excluded"] = [(f, m) in zero for f, m in zip(cells.family, cells.model)]

    available = cells.score.notna()
    long = pd.DataFrame({
        "task_id": cells.task_id, "model": cells.model, "run": 1,
        "score": cells.score.astype(float),
        # Zero-cost models: excluded from cost metrics (plan S1.8), so no cost, not a cost of 0.
        "cost_usd": cells.recorded_cost.where(~cells.cost_excluded).astype(float),
        "input_tokens": cells.input_tokens, "output_tokens": cells.output_tokens,
        "problem_id": None, "family": cells.family, "dimension": DIMENSION, "split": None,
        "score_type": [RULES[s].score_type for s in cells.subset],
        "eligible": [RULES[s].eligible for s in cells.subset],
        "available": available,
        "unavailable_reason": [None if a else NULL_SCORE for a in available],
    })
    long = long_frame(long)
    matrices = {fam: OutcomeMatrix.from_long(long[long.family == fam], name=fam)
                for fam in sorted(long.family.unique())}
    return LrbData(long, cells, tasks, files, flags, matrices)


@lru_cache(maxsize=1)
def load_cached() -> LrbData:
    return load()
