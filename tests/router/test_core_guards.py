"""S4.6 non-transductive rule (docs/research/S4_spec.md §5): the guard refuses an item-parameter
fit that reads a scored task's outcomes (the m3_irt a6_a7 pattern) or another task of its problem,
and every item-parameter fit in router/core/ goes through it."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.core import guards, irt
from router.core.guards import TransductiveFitError, check_non_transductive

CORE = Path(guards.__file__).resolve().parent


def _toy(n_probe=12, n_test=6, n_models=5, seed=0):
    g = np.random.default_rng(seed)
    probe = [f"p{i}" for i in range(n_probe)]
    test = [f"t{i}" for i in range(n_test)]
    Y = (g.random((n_models, n_probe + n_test)) < 0.6).astype(float)
    problems = pd.Series({t: f"P_{t}" for t in probe + test})
    return probe, test, Y, [f"m{i}" for i in range(n_models)], problems


def guarded_item_fit(Y, items, models, scored, problem_of):
    """The 1PL entry point of router/core/, which runs the guard before fitting."""
    return irt.fit(Y, items, models, scored_ids=scored, problem_of=problem_of, maxiter=50)


def test_the_m3_irt_a6_a7_pattern_is_refused():
    """a6_a7 fits on probe + test items (`items = probe.tasks.append(test.tasks)`)."""
    probe, test, Y, models, problems = _toy()
    with pytest.raises(TransductiveFitError, match="scored task"):
        guarded_item_fit(Y, probe + test, models, test, problems)


def test_a_fit_sharing_a_problem_with_a_scored_task_is_refused():
    probe, test, Y, models, problems = _toy()
    problems[probe[0]] = problems[test[0]]          # a variant of a scored task's problem
    with pytest.raises(TransductiveFitError, match="problem"):
        guarded_item_fit(Y[:, :len(probe)], probe, models, test, problems)


def test_a_fit_on_training_items_only_passes():
    probe, test, Y, models, problems = _toy()
    params = guarded_item_fit(Y[:, :len(probe)], probe, models, test, problems)
    assert list(params.items) == probe


def test_unknown_problem_ids_are_an_error():
    with pytest.raises(KeyError):
        check_non_transductive(["a"], ["b"], pd.Series({"a": "P1"}))


FIT_CALLS = {"fit", "partial_fit", "fit_transform", "minimize", "minimize_scalar", "curve_fit",
             "least_squares"}


def _callee(n: ast.Call):
    f = n.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _fits_something(fn) -> bool:
    return any(isinstance(n, ast.Call) and _callee(n) in FIT_CALLS for n in ast.walk(fn))


def _fit_functions(path: Path):
    """(qualname, node) for every function, methods included, that is named fit or fit_*, or that
    calls a fitting routine (`.fit`, `minimize`, `curve_fit`, ...; `FIT_CALLS`)."""
    tree = ast.parse(path.read_text())
    out = []

    def walk(node, prefix):
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if ch.name == "fit" or ch.name.startswith("fit_") or _fits_something(ch):
                    out.append((prefix + ch.name, ch))
                walk(ch, prefix + ch.name + ".")
    walk(tree, "")
    return out


def _calls_guard(fn) -> bool:
    return any(isinstance(n, ast.Call) and _callee(n) == "check_non_transductive"
               for n in ast.walk(fn))


def test_every_item_fit_in_router_core_calls_the_guard():
    seen_not_item, unguarded = set(), []
    for path in sorted(CORE.glob("*.py")):
        mod = path.stem
        if mod in guards.EXEMPT_MODULES or mod == "guards":
            continue
        for qual, fn in _fit_functions(path):
            key = f"{mod}.{qual}"
            if key in guards.NOT_ITEM_FITS:
                seen_not_item.add(key)
            elif not _calls_guard(fn):
                unguarded.append(key)
    assert not unguarded, f"item fits without check_non_transductive: {unguarded}"
    stale = set(guards.NOT_ITEM_FITS) - seen_not_item
    assert not stale, f"NOT_ITEM_FITS names functions that no longer exist: {stale}"
    assert guards.EXEMPT_MODULES == {}, "S4.3 removed the last exemption; none may be added"


def test_the_scan_catches_an_unguarded_fit(tmp_path):
    src = ("class Ability:\n    def fit(self, Y, items):\n        return Y\n\n"
           "def fit_difficulty(x):\n    check_non_transductive(x, [], None)\n\n"
           "def train_difficulty(X, y):\n    return Ridge().fit(X, y)\n\n"
           "def estimate_b(f):\n    return minimize(f, 0.0)\n\n"
           "def helper(x):\n    return x + 1\n")
    p = tmp_path / "ability.py"
    p.write_text(src)
    fns = dict(_fit_functions(p))
    assert set(fns) == {"Ability.fit", "fit_difficulty", "train_difficulty", "estimate_b"}
    assert [q for q, f in fns.items() if not _calls_guard(f)] == ["Ability.fit", "train_difficulty",
                                                                  "estimate_b"]
