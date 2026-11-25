"""S4.6: the non-transductive rule (plan S4.6 [M4.3]; `docs/research/S4_spec.md` §5).

A scored task's difficulty, or any other item parameter, is never fitted from any model's outcomes
on that task or on any task of its problem. Item parameters are fitted on training-fold tasks only;
a scored task's difficulty comes only from a predictor applied to its `content_text` (S4.4).

The bug this fixes: `router/experiments/m3_irt.py` `a6_a7` fits `irt.fit` on the probe items and
the test items together (`items = probe.tasks.append(test.tasks)`), so each test item's difficulty
is fitted from the other models' outcomes on it.

Every item-parameter fit in `router/core/` calls `check_non_transductive` before fitting. A test
(`tests/router/test_core_guards.py`) scans `router/core/` with `ast`: each function named `fit` or
`fit_*`, and each function that calls a fitting routine (`.fit`, `partial_fit`, `fit_transform`,
`minimize`, `minimize_scalar`, `curve_fit`, `least_squares`), must either call the guard, be listed in
`NOT_ITEM_FITS` with the reason it fits no item parameter, or sit in a module of `EXEMPT_MODULES`.
A new fit therefore cannot skip the rule silently: it fails the scan until it is guarded or listed.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

# Functions the scan flags in router/core/ that fit no item parameter ("module.qualname").
NOT_ITEM_FITS: dict[str, str] = {
    "policy.RouterV1.fit": "router v1: a per-group lookup table over training tasks; no item parameter",
    "policy.RouterV1.fit_parts": "the parts of that lookup table (router v1 and v2); no item parameter",
    "policy.RouterV2.fit": "router v2: v1 plus a Beta posterior per (new model, group) from probes; "
                           "no item parameter",
}

# Modules not yet under the rule, with the reason. Empty since S4.3 rebuilt `irt` on the guarded
# ability model (the quick-check code is frozen in router/experiments/; S4 spec §6.1); kept so the
# scan's contract stays visible, and the S4.6 test asserts it stays empty.
EXEMPT_MODULES: dict[str, str] = {}


class TransductiveFitError(ValueError):
    """An item-parameter fit would read outcomes on a scored task or on a task of its problem."""


def check_non_transductive(fit_item_ids: Iterable[str], scored_ids: Iterable[str],
                           problem_of: pd.Series) -> None:
    """Raise `TransductiveFitError` if a fitted item is a scored task or shares a problem with one.
    `problem_of` maps task id -> problem id (`problems.csv`); a task missing from it is an error,
    since the rule cannot be checked for it."""
    fit_ids, scored = pd.Index(list(fit_item_ids)), pd.Index(list(scored_ids))
    both = fit_ids.intersection(scored)
    if len(both):
        raise TransductiveFitError(f"{len(both)} scored task(s) among the fitted items, "
                                   f"e.g. {sorted(both)[:3]}")
    need = fit_ids.append(scored)
    missing = need.difference(problem_of.index)
    if len(missing):
        raise KeyError(f"{len(missing)} task(s) have no problem id, e.g. {sorted(missing)[:3]}")
    shared = set(problem_of.reindex(fit_ids)) & set(problem_of.reindex(scored))
    if shared:
        raise TransductiveFitError(f"{len(shared)} problem(s) shared by fitted items and scored "
                                   f"tasks, e.g. {sorted(shared)[:3]}")
