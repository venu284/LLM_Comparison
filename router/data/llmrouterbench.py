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
_ARENA_SOURCE = ('Arena-Hard-Auto (arXiv:2406.11939) scores by pairwise LLM-judge comparison against '
                 'a baseline answer; there is nothing to execute.')


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
