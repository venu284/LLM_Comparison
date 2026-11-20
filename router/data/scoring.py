"""S1.4: how each family's released score was produced, which families are eligible, and the
success rule.

Eligibility follows documented scoring rules only, never the observed scores (CLAIMS S0.12, R16).
Each family in `RULES` carries two citations with verbatim quotes:

- `crb_rule`: how CodeRouterBench produced the released score. The paper (arXiv 2606.22902v3)
  documents the scorer per dimension, in the Eval column of Table 5 (App. B.1), not per source.
  The ACRouter code repository (github.com/LanceZPF/agent-as-a-router) returned 404 on
  2026-11-20, so no per-source scorer code could be read.
- `source_rule`: the source benchmark's own documented rule.

Where the two differ, the released score means what CodeRouterBench did (`score_type` follows
`crb_rule`), and the difference is recorded in `notes`.

The rule table was frozen before any observed score was looked at. `score_types()` below is
descriptive only and is never used for eligibility.

The 2026-11-10 quick-check threshold survives as `LEGACY_ACCEPT_THRESHOLD` for the m3 and
`router/sim` code that S2 rebuilds; nothing else may import it (tests/router/test_scoring.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from router.data.sources import FAMILIES
from router.settings import SUCCESS_SCORE

SCORE_TYPES = ("exec", "similarity", "judge", "undocumented")
PARTIAL_CREDIT = ("yes", "no", "unstated")


@dataclass(frozen=True)
class Rule:
    score_type: str          # one of SCORE_TYPES, from crb_rule
    partial_credit: str      # one of PARTIAL_CREDIT: can the released score lie in (0, 1) by rule
    partial_credit_cite: str
    test_checkable: bool     # the task ships tests a verifier can run
    test_checkable_cite: str
    crb_rule: str            # how CodeRouterBench scored it, with a quote
    source_rule: str         # the source benchmark's own rule, with a quote
    notes: str = ""
    id_pool: bool = True     # False for OOD families, which are outside the evaluation population

    @property
    def eligible(self) -> bool:
        """S0.12: execution-scored and test-checkable, in the ID pool."""
        return self.score_type == "exec" and self.test_checkable and self.id_pool


# --------------------------------------------------------------------------- citations
# CodeRouterBench: arXiv 2606.22902v3 ("Agent-as-a-Router"), read 2026-11-20 at arxiv.org/html.
_CRB = "arXiv:2606.22902v3"
_CRB_EXEC_CAPTION = (f'{_CRB} App. B.1, Table 5 caption: "Eval: Exec = execution-based pass@1"; '
                     '§4.1: "Seven dimensions use execution-based scoring (pass@1 in sandboxed '
                     'environments) for evaluations, while three dimensions use proxy metrics '
                     'supplemented by LLM-as-Judge."')
_CRB_PROXY_CAPTION = (f'{_CRB} App. B.1, Table 5 caption: "Proxy+ = proxy metric supplemented by '
                      'LLM-as-Judge"; §4.1: "three dimensions use proxy metrics supplemented by '
                      'LLM-as-Judge."')
_SEVEN_VS_SIX = ("§4.1's 'seven' execution-scored dimensions are Table 5's six 'Exec' rows plus the "
                 "OOD agentic dimension ('Sandbox'); the three 'Proxy+' rows make ten.")


def _crb_exec(row: str, sources: str) -> str:
    return f'{_CRB} App. B.1 Table 5, row "{row}" (sources listed: {sources}), Eval "Exec". ' \
           + _CRB_EXEC_CAPTION


def _crb_proxy(row: str, sources: str) -> str:
    return f'{_CRB} App. B.1 Table 5, row "{row}" (sources listed: {sources}), Eval "Proxy+". ' \
           + _CRB_PROXY_CAPTION


_PC_EXEC = (f'{_CRB} names the metric "execution-based pass@1" (App. B.1) but gives no sample count '
            'or per-task aggregation, so whether a released per-task score can be fractional is '
            'not stated. Third-party: Route Anything (github.com/InfRec-Cardinal/route-anything '
            '@e137f3f, experiments_x/X8_operating_point/README.md) describes its V2 verifier as '
            '"`common.sandbox.verify_response` pass fraction (hidden tests; the released score\'s '
            'own mechanism)".')
_PC_PROXY = (f'{_CRB} App. B.1 names only a "proxy metric supplemented by LLM-as-Judge"; the proxy '
             'metric, its range and its aggregation are not stated.')

# Source benchmarks. arXiv versions are the ones read on 2026-11-20 (arxiv.org/html).
# Local dataset cards are the upstream cards pinned in external/route-anything-data/provenance/
# (Route Anything release, checksums in external/CHECKSUMS).
_HE = ('arXiv:2107.03374v2 (HumanEval) §2.1: "a sample is considered correct if it passes a set of '
       'unit tests"')
_HE_TESTS = ('arXiv:2107.03374v2 §2.2: "Each problem includes a function signature, docstring, '
             'body, and several unit tests, with an average of 7.7 tests per problem."')
_EVALPLUS = ('arXiv:2305.01210v3 (EvalPlus, HumanEval+/MBPP+) §2.3: "comparing the output of the '
             'generated code against that of the ground-truth solution"')
_MBPP = ('arXiv:2108.07732v1 (MBPP) §3: "we check whether the code passes a set of test cases when '
         'executed"')
_MBPP_TESTS = ('arXiv:2108.07732v1 §2.1: "three test cases that check for semantic correctness of '
               'the function"; MBPP+ card (provenance/evalplus--mbppplus-README.md) lists features '
               '"test_list" and "test".')
_BCB = ('arXiv:2406.15877v5 (BigCodeBench) §4: "we report Pass@1 with greedy decoding for the main '
        'experiments in the zero-shot setting."; §2.3: "the test cases are automatically run '
        'against the task implementations in a configured sandbox environment."')
_BCB_TESTS = ('arXiv:2406.15877v5 §2: tasks are "paired with at least five test cases encapsulated '
              'in the unittest framework".')
_LCB = ('arXiv:2403.07974v2 (LiveCodeBench) §4: "programs must pass all tests to be considered '
        'correct."; §2: "the evaluation is performed based on functional correctness, using a set '
        'of unseen test cases."')
_LCB_TESTS = ('LiveCodeBench code_generation_lite card (provenance/livecodebench--code_generation_lite'
              '-README.md): "Each problem instance consists of a problem description, input/output '
              'examples, and hidden test cases."')
_LCB_TOP = ('arXiv:2403.07974v2 §4 (test output prediction): "parse the generated response to extract '
            'the answer and use equivalence checks for grading as specified in Section 2." Whether '
            'the check executes code or matches strings is not stated.')
_CRUX = ('arXiv:2401.03065v1 (CRUXEval) §3: "for output prediction, a generated output passes if '
         'assert f(input) == generated_output passes."; "For both tasks, we use an execution-based '
         'correctness metric."; §4: "We use N=100 samples for all non-GPT models and N=10 samples '
         'for GPT models."')
_CRUX_TESTS = ('CRUXEval card (provenance/cruxeval-org--cruxeval-README.md): "a benchmark of 800 '
               'Python functions and input-output pairs"; the check is the assert in arXiv:'
               '2401.03065v1 §3.')
_DS1000 = ('arXiv:2211.11501v1 (DS-1000) §2.3: "To evaluate a predicted program, we execute it on '
           'the test inputs and compare the outputs with the ground truth."; "we introduced '
           'additional surface-form constraints that require the presence/absence of specific APIs '
           'for keywords."')
_DS1000_TESTS = ('DS-1000 card (provenance/xlangai--DS-1000-README.md): "See testing code and more '
                 'information ... in the DS-1000 repo"; arXiv:2211.11501v1 Introduction: '
                 '"carefully written and reviewed test cases and constraints".')
_HEX = ('arXiv:2303.17568v2 (CodeGeeX, HumanEval-X) §3.2: "we use test cases to evaluate the exact '
        'functional correctness of the generated code"; "c is the number of samples that pass all '
        'test cases."')
_HEX_TESTS = ('HumanEval-X card (provenance/THUDM--humaneval-x-README.md): "820 high-quality '
              'human-crafted data samples (each with test cases)"; field "test": "hidden test '
              'samples, used for evaluation."')
_MULTIPLE = ('arXiv:2208.08227v4 (MultiPL-E) §II-C: "We judge a generated function correct if it '
             'passes all tests."')
_MULTIPLE_TESTS = ('arXiv:2208.08227v4 §II-C: "We translate two code generation benchmarks that '
                   'include unit tests for every problem."; §III-D: "for certain typed languages, '
                   'we fail to compile up to 5 benchmarks with untranslatable types."')
_DEBUG = ('arXiv:2401.04621v3 (DebugBench) §3 Metric: "The metric for DebugBench is based on the test '
          'suites provided by [LeetCode]."; Pass Rate counts an instance only when all its test '
          'cases pass.')
_DEBUG_TESTS = ('arXiv:2401.04621v3 §3 Metric, footnote 3: "Users of the benchmark will require a '
                'LeetCode account to access these test suites."')
_SWE = ('arXiv:2310.06770v3 (SWE-bench) §2.2: "If the patch applies successfully and all of these '
        'tests pass we consider the proposed solution to have successfully resolved the issue."; '
        'swebench.com/lite.html: "SWE-bench Lite provides a smaller, carefully selected subset of '
        '300 tasks from the full benchmark."')
_SWE_TESTS = ('arXiv:2310.06770v3 §2.2: the patch is applied, then the harness will "execute the unit '
              'and system tests associated with the task instance".')
_CANITEDIT = ('arXiv:2312.12450v6 (CanItEdit) §5.1: "pass@k is the likelihood that at least one '
              'successful edit was made from k attempts, as assessed by the test suite."')
_CANITEDIT_TESTS = ('arXiv:2312.12450v6 §3: each problem comes with "a hidden test suite"; §6: "Each '
                    'problem has a comprehensive test suite."')
_B2F = ('arXiv:2102.04664v2 (CodeXGLUE) §5.8, code repair (Bugs2Fix): "We report the exact match '
        'accuracy, the BLEU score"; "The exact match accuracy is used as the overall evaluation '
        'metric."; arXiv:1812.08693v2 (Tufano et al.) §1: "we do not need to execute test cases '
        'during patch generation".')
_B2F_TESTS = ('arXiv:2102.04664v2 §3.7 and §5.8 describe no executable tests; arXiv:1812.08693v2 '
              '§3.2: "We do not assess the compilability of the patches."')
_CXG = ('arXiv:2102.04664v2 (CodeXGLUE) §5.9, code summarization: "who use smoothed BLEU score" '
        '"as evaluation metric, because this is suitable for evaluating short documents."')
_CXG_TESTS = ('arXiv:2102.04664v2 §5.9: the target is natural-language text scored by smoothed BLEU; '
              'there is nothing to execute.')
_TESTGEN_TESTS = ('Route Anything tasks.jsonl (external/route-anything-data/experiments_x/common/'
                  'tasks.jsonl) notes for this family: "dim9 test generation: ... no '
                  'execution-based verification". No source document describes a runnable check '
                  'of generated tests.')
_SUP = (f"{_CRB} does not mention '_sup' tasks; the source is read from the task-id prefix, and "
        "the dimension's Table 5 rule covers every task in the dimension (Table 5 caption: \"The 9 "
        "single-turn dimensions each contain 1,111 tasks\").")
_RA_BCB = ("Route Anything tasks.jsonl notes this cell as 'BigCodeBench v0.1.4 "
           "complete_prompt/test/entry_point/libs'.")
_OOD = (f'{_CRB} Table 14 caption: "Resolved%: fraction of tasks where the model produces a patch '
        'that the SWE-Bench harness grades as correct."; App. B.2: "Evaluation uses the SWE-Bench '
        'Docker harness [18] with mini-swe-agent, capped at 40 steps per task."')

# --------------------------------------------------------------------------- rule table
# Frozen 2026-11-20 before any observed score was read (R16). Changing it is a plan change.
_CG = _crb_exec("Code Generation", "HumanEval+, MBPP+, BigCodeBench")
_AL = _crb_exec("Algorithm Design", "LiveCodeBench, BigCodeBench")
_BF = _crb_exec("Bug Fixing", "DebugBench, SWE-bench Lite")
_CC = _crb_exec("Code Completion", "CRUXEval, HumanEval+ variants")
_DS = _crb_exec("Data Science", "DS-1000, BigCodeBench")
_ML = _crb_exec("Multi-Language", "HumanEval-X, MultiPL-E")
_CR = _crb_proxy("Code Refactoring", "Bugs2Fix, CanItEdit")
_CU = _crb_proxy("Code Understanding", "CodeXGLUE Summarization")
_TG = _crb_proxy("Test Generation", "LiveCodeBench, HumanEval+ variants")


def _exec(crb, source, tests, notes="", checkable=True, tests_cite=None) -> Rule:
    return Rule("exec", "unstated", _PC_EXEC, checkable, tests_cite or tests, crb, source, notes)


def _proxy(crb, source, checkable, tests, notes="") -> Rule:
    return Rule("judge", "unstated", _PC_PROXY, checkable, tests, crb, source, notes)


_NO_TOOLCHAIN = ("Route Anything's sandbox has no {lang} toolchain, so its tasks.jsonl marks every "
                 "task runnable=False; that is an environment limit, not missing tests.")
_PROXY_NOTE = ("CodeRouterBench's Proxy+ rule wins over the source rule (the released score means "
               "what CodeRouterBench did); the proxy metric is not named, so 'judge' is used for "
               "'proxy metric supplemented by LLM-as-Judge'.")

RULES: dict[str, Rule] = {
    # code_generation
    "humaneval": _exec(_CG, f"{_HE}; {_EVALPLUS}", _HE_TESTS),
    "mbpp": _exec(_CG, f"{_MBPP}; {_EVALPLUS}", _MBPP_TESTS),
    "bcb": _exec(_CG, _BCB, _BCB_TESTS, _RA_BCB),
    "bcb_v014": _exec(_CG, _BCB, _BCB_TESTS, f"{_SUP} {_RA_BCB}"),
    # algorithm
    "lcb": _exec(_AL, _LCB, _LCB_TESTS),
    "bcb_algo2": _exec(_AL, _BCB, _BCB_TESTS, f"{_SUP} {_RA_BCB}"),
    # code_completion
    "cruxeval": _exec(_CC, _CRUX, _CRUX_TESTS,
                      "CRUXEval's own pass@1 is estimated from N=10 or N=100 samples, so its "
                      "published per-task values are fractional; how many samples CodeRouterBench "
                      "drew is not stated."),
    "he_completion": _exec(_CC, f"{_HE}; {_EVALPLUS}", _HE_TESTS,
                           "Table 5 lists 'HumanEval+ variants' for Code Completion; the variant "
                           "itself is not described."),
    "mbpp_completion": _exec(_CC, f"{_MBPP}; {_EVALPLUS}", _MBPP_TESTS,
                             "Table 5 does not list MBPP under Code Completion; the dimension's "
                             "Exec rule is applied."),
    "bcb_completion": _exec(_CC, _BCB, _BCB_TESTS,
                            f"Table 5 does not list BigCodeBench under Code Completion; the "
                            f"dimension's Exec rule is applied. {_SUP} {_RA_BCB}"),
    # data_science
    "ds1000": _exec(_DS, _DS1000, _DS1000_TESTS),
    "bcb_ds": _exec(_DS, _BCB, _BCB_TESTS, f"{_SUP} {_RA_BCB}"),
    # multi_language
    "hex_python": _exec(_ML, _HEX, _HEX_TESTS),
    "hex_cpp": _exec(_ML, _HEX, _HEX_TESTS, _NO_TOOLCHAIN.format(lang="C++")),
    "hex_go": _exec(_ML, _HEX, _HEX_TESTS, _NO_TOOLCHAIN.format(lang="Go")),
    "hex_java": _exec(_ML, _HEX, _HEX_TESTS, _NO_TOOLCHAIN.format(lang="Java")),
    "hex_js": _exec(_ML, _HEX, _HEX_TESTS, _NO_TOOLCHAIN.format(lang="JavaScript")),
    "multiple_rs": _exec(_ML, _MULTIPLE, _MULTIPLE_TESTS, "Not in Route Anything's tasks.jsonl."),
    "multiple_ts": _exec(_ML, _MULTIPLE, _MULTIPLE_TESTS, "Not in Route Anything's tasks.jsonl."),
    # bug_fixing
    "debugbench": _exec(_BF, _DEBUG, _DEBUG_TESTS, checkable=False,
                        notes="Execution-scored by both rules, but the source's tests sit behind "
                              "LeetCode and do not ship with the benchmark, so it is not "
                              "test-checkable. How CodeRouterBench ran 'pass@1 in sandboxed "
                              "environments' on it is not documented: open question for the user."),
    "debugbench_sup": _exec(_BF, _DEBUG, _DEBUG_TESTS, checkable=False,
                            notes=f"As debugbench. {_SUP}"),
    "swebench": _exec(_BF, _SWE, _SWE_TESTS,
                      notes="Tests ship with each instance (FAIL_TO_PASS / PASS_TO_PASS) but run "
                            "only in the per-repository SWE-bench harness. Not in Route Anything's "
                            "tasks.jsonl."),
    # code_refactoring (Proxy+)
    "bugs2fix": _proxy(_CR, _B2F, False, _B2F_TESTS,
                       "Source rule is similarity (exact match, BLEU). " + _PROXY_NOTE),
    "bugs2fix_sup": _proxy(_CR, _B2F, False, _B2F_TESTS,
                           f"As bugs2fix. {_SUP} {_PROXY_NOTE}"),
    "canitedit": _proxy(_CR, _CANITEDIT, True, _CANITEDIT_TESTS,
                        "Rules differ: the source is execution-scored on hidden test suites, but "
                        "CodeRouterBench scored the whole Code Refactoring dimension Proxy+. "
                        + _PROXY_NOTE),
    # code_understanding (Proxy+)
    "codexglue": _proxy(_CU, _CXG, False, _CXG_TESTS,
                        "Source rule is similarity (smoothed BLEU). " + _PROXY_NOTE),
    "codexglue_sup": _proxy(_CU, _CXG, False, _CXG_TESTS, f"As codexglue. {_SUP} {_PROXY_NOTE}"),
    # test_generation (Proxy+)
    "lcb_testgen": _proxy(_TG, _LCB_TOP, False, _TESTGEN_TESTS,
                          "Assumed to be LiveCodeBench's test output prediction scenario (Table 5 "
                          "lists LiveCodeBench under Test Generation). " + _PROXY_NOTE),
    "he_testgen": _proxy(_TG, "NOT DOCUMENTED: no source document describes scoring for the "
                              "HumanEval test-generation variant ('HumanEval+ variants' in Table 5).",
                         False, _TESTGEN_TESTS, _PROXY_NOTE),
    "mbpp_testgen": _proxy(_TG, "NOT DOCUMENTED: no source document describes scoring for an MBPP "
                                "test-generation variant; Table 5 does not list MBPP under Test "
                                "Generation.",
                           False, "No source document describes a runnable check of generated "
                                  "tests; the family is not in Route Anything's tasks.jsonl.",
                           _PROXY_NOTE),
}

# OOD families: recorded for completeness; outside the evaluation population (S0.12).
OOD_RULES: dict[str, Rule] = {
    fam: Rule("exec", "no", f"{_OOD} A task is resolved or not.", True, _OOD, _OOD, _OOD,
              "OOD: outside the evaluation population; never eligible for the ID pool.",
              id_pool=False)
    for fam in ("featurebench", "longcli", "old112", "swe_ci")
}


def rule(family: str) -> Rule:
    return RULES[family] if family in RULES else OOD_RULES[family]


def eligible_families() -> list[str]:
    return sorted(f for f, r in RULES.items() if r.eligible)


# --------------------------------------------------------------------------- success rule
def success(score):
    """S0.1: success means the released score is 1.0 (every test passes). Partial credit below 1.0
    is not success, whatever the family."""
    return score >= SUCCESS_SCORE


# --------------------------------------------------------------------------- long table hook
def annotate(long: pd.DataFrame) -> pd.DataFrame:
    """Fill `score_type` and `eligible` from each row's family. Reads only `family` (R16): the
    observed scores cannot change either column. OOD families are never eligible."""
    unknown = set(long["family"]) - set(RULES) - set(OOD_RULES)
    if unknown:
        raise KeyError(f"no scoring rule for families {sorted(unknown)}")
    out = long.copy()
    fam = out["family"]
    out["score_type"] = fam.map(lambda f: rule(f).score_type).astype(object)
    out["eligible"] = fam.map(lambda f: rule(f).eligible).astype("boolean")
    return out


# --------------------------------------------------------------------------- legacy (S2 removes)
LEGACY_ACCEPT_THRESHOLD = 0.5   # 2026-11-10 quick check; contradicts S0.1, kept for m3 and sim/*


def legacy_accepted(score: pd.DataFrame | pd.Series, threshold: float = LEGACY_ACCEPT_THRESHOLD):
    """The 2026-11-10 quick-check rule (score >= 0.5). Not S0.1; do not use in new code."""
    return score >= threshold


def score_types(score: pd.DataFrame, dimension: pd.Series) -> pd.DataFrame:
    """Observed, descriptive only; never used for eligibility (R16).

    Share of 0/1 scores and number of distinct values per dimension in an observed score matrix.
    """
    long = score.stack().rename("s").reset_index()
    long["dim"] = long.iloc[:, 0].map(dimension)
    g = long.groupby("dim")["s"]
    out = pd.DataFrame({"binary_share": g.apply(lambda s: s.isin([0.0, 1.0]).mean()),
                        "distinct": g.nunique()})
    out["type"] = pd.cut(out.binary_share, [-0.01, 0.5, 0.999, 1.0],
                         labels=["continuous", "mostly-binary", "binary"])
    return out


assert set(RULES) == {f for f, v in FAMILIES.items() if v.split_kind == "id"}
assert set(OOD_RULES) == {f for f, v in FAMILIES.items() if v.split_kind == "ood"}
