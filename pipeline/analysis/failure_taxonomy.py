"""Derived failure taxonomy for the Phase 5 runs.

`runs.failure_mode` is defined in `db/schema.sql`, in `models/run_result.py` and
in Report 5's metrics table, but it was never populated: it is NULL in all 325
rows. Without it the 63 zero-test runs cannot be split into harness artifacts and
genuine model failures, and that split is what decides whether the RQ1 result is
robust.

This module reconstructs as much of the taxonomy as the recorded columns allow.
It is a *derived* view: nothing is written back to `runs`. Repairing the Phase 5
rows in place would invalidate comparison with the already-published numbers, so
the taxonomy is computed at analysis time and the raw data is left alone.

What can and cannot be recovered:

- Truncation and defective tasks are recoverable, because `tokens_output` and the
  per-task zero-test pattern are both recorded.
- Extraction failures are recoverable from `extraction_success`.
- Everything else collapses into `no_tests_run`. A run where the harness crashed
  and a run where the model emitted code so broken that no test could be
  collected look identical in the stored columns. Those rows are labelled
  honestly as undifferentiable rather than guessed at; separating them needs
  Docker and a re-run.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

PASSED = "passed"
TESTS_FAILED = "tests_failed"
DEFECTIVE_TASK = "defective_task"
TRUNCATED_OUTPUT = "truncated_output"
EXTRACTION_FAILED = "extraction_failed"
NO_TESTS_RUN = "no_tests_run"

LABELS = (
    PASSED,
    TESTS_FAILED,
    DEFECTIVE_TASK,
    TRUNCATED_OUTPUT,
    EXTRACTION_FAILED,
    NO_TESTS_RUN,
)

#: Labels whose cause lies in the harness or the run configuration rather than in
#: the model's competence at the task.
#:
#: `truncated_output` is included deliberately. A model that never reached the end
#: of its answer was not given the chance to fail on the merits -- the 4,096-token
#: ceiling in `config.yaml` cut it off. Only Qwen3-32B ever hit that ceiling,
#: because it is the only model in the set that emits chain-of-thought into the
#: same budget as the code, so leaving these rows in place is what confounds the
#: published "chain-of-thought hurts code generation" claim with plain truncation.
ARTIFACTUAL_LABELS = frozenset({DEFECTIVE_TASK, TRUNCATED_OUTPUT, EXTRACTION_FAILED})

LABEL_DESCRIPTIONS: Dict[str, str] = {
    PASSED: "All collected tests passed.",
    TESTS_FAILED: "Tests were collected and ran; the model's code did not pass them.",
    DEFECTIVE_TASK: "Every model returned zero tests on this task, so the task or its test file is at fault.",
    TRUNCATED_OUTPUT: "Output hit the max_tokens ceiling, so the answer was cut off before it was complete.",
    EXTRACTION_FAILED: "No code block could be extracted from the model's response.",
    NO_TESTS_RUN: "Zero tests collected, cause not recoverable from the stored columns.",
}


def defective_task_ids(frame: pd.DataFrame) -> List[str]:
    """Tasks where *every* model returned zero tests.

    Derived rather than hard-coded. When no model in a five-model set can get a
    single test to run, the task or its test file is the more likely explanation
    than five simultaneous model failures. This is a strong signal, not a proof:
    confirming it means running the reference solution under Docker.
    """
    per_task_max = frame.groupby("task_id")["tests_total"].max()
    return sorted(per_task_max[per_task_max == 0].index.tolist())


def classify(frame: pd.DataFrame, max_tokens: Optional[int] = None) -> pd.Series:
    """Label every run. Returns a Series aligned to `frame.index`.

    Precedence among the zero-test causes matters, because a row can satisfy more
    than one test. Defective task wins first -- if no model could run a test on
    that task, the task explains the row regardless of what else is true of it.
    Truncation comes next, then extraction failure, then the residual bucket.
    """
    if max_tokens is None:
        from config import get_experiment_settings

        max_tokens = int(get_experiment_settings()["max_tokens"])

    passed = frame["pass_fail"].astype(bool)
    zero_tests = frame["tests_total"] == 0
    defective = frame["task_id"].isin(defective_task_ids(frame))
    truncated = frame["tokens_output"] >= max_tokens
    extraction_failed = ~frame["extraction_success"].astype(bool)

    labels = pd.Series(TESTS_FAILED, index=frame.index, dtype=object)
    labels[passed] = PASSED
    # Not gated on zero_tests: the run_eval failure builders record
    # tests_total=task.test_count even when no code was extracted, so an
    # extraction failure can carry a non-zero test total (FE-010 / Llama-3.3-70B).
    labels[~passed & extraction_failed] = EXTRACTION_FAILED
    labels[zero_tests & truncated] = TRUNCATED_OUTPUT
    labels[zero_tests & defective] = DEFECTIVE_TASK
    labels[zero_tests & ~defective & ~truncated & ~extraction_failed] = NO_TESTS_RUN
    return labels


def artifactual_mask(frame: pd.DataFrame, max_tokens: Optional[int] = None) -> pd.Series:
    """True where a run failed for a reason that is not the model's competence.

    A passing run is never artifactual even if it was truncated -- if the tests
    passed anyway, the ceiling did no harm.
    """
    labels = classify(frame, max_tokens=max_tokens)
    return labels.isin(ARTIFACTUAL_LABELS)


def summary(frame: pd.DataFrame, max_tokens: Optional[int] = None) -> pd.DataFrame:
    """Counts of each label per model. Missing labels appear as zero, not absent."""
    labels = classify(frame, max_tokens=max_tokens)
    table = pd.crosstab(frame["model_name"], labels)
    for label in LABELS:
        if label not in table.columns:
            table[label] = 0
    return table[list(LABELS)].sort_values(PASSED, ascending=False)


def report(frame: pd.DataFrame, max_tokens: Optional[int] = None) -> Dict[str, object]:
    """Machine-readable taxonomy block for `phase6_results.json`."""
    labels = classify(frame, max_tokens=max_tokens)
    zero_tests = frame["tests_total"] == 0
    artifactual = labels.isin(ARTIFACTUAL_LABELS)

    return {
        "label_descriptions": LABEL_DESCRIPTIONS,
        "counts": {label: int((labels == label).sum()) for label in LABELS},
        "by_model": summary(frame, max_tokens=max_tokens).to_dict(orient="index"),
        "defective_task_ids": defective_task_ids(frame),
        "zero_test_rows": int(zero_tests.sum()),
        "zero_test_explained": int((zero_tests & artifactual).sum()),
        "zero_test_undifferentiable": int((labels == NO_TESTS_RUN).sum()),
        "artifactual_rows": int(artifactual.sum()),
        "source_column_populated": bool(frame["failure_mode"].notna().any())
        if "failure_mode" in frame.columns
        else False,
        "note": (
            "Derived at analysis time from tests_total, tokens_output and "
            "extraction_success. The stored failure_mode column is empty; these "
            "labels are not written back to the database."
        ),
    }
