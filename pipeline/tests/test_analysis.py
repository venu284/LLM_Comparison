"""Tests for the Phase 6 analysis module.

The statistical functions are checked against distributions whose answers are
known in advance, not against the experimental data. A test that only asserts
"the number the code produced equals the number the code produced" would have
passed while the contingency table was silently wrong -- which is exactly the
defect these tests were written to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from analysis.recommender import (  # noqa: E402
    DEFAULT_WEIGHTS,
    Recommender,
    build_profiles,
    classify_prompt,
)
from analysis.stats import (  # noqa: E402
    chi_square_test,
    holm_correction,
    mcnemar_test,
)


def _frame(groups, outcomes):
    return pd.DataFrame({"g": groups, "pass_fail": outcomes})


class TestChiSquare:
    def test_no_association_gives_p_near_one(self):
        result = chi_square_test(
            _frame(["a"] * 100 + ["b"] * 100, ([True] * 50 + [False] * 50) * 2), "g", "null"
        )
        assert result.p_value > 0.9
        assert result.cramers_v < 0.05
        assert result.n == 200

    def test_perfect_separation_gives_tiny_p_and_max_effect(self):
        result = chi_square_test(
            _frame(["a"] * 100 + ["b"] * 100, [True] * 100 + [False] * 100), "g", "sep"
        )
        assert result.p_value < 1e-10
        assert result.cramers_v > 0.97

    def test_boolean_columns_are_not_read_as_a_row_mask(self):
        """Regression: `table[[False, True]]` silently masked rows in pandas."""
        result = chi_square_test(
            _frame(["a"] * 30 + ["b"] * 30, [True] * 20 + [False] * 40), "g", "shape"
        )
        assert result.n == 60, "all rows must survive contingency construction"
        assert result.dof == 1

    def test_degenerate_table_does_not_raise(self):
        result = chi_square_test(_frame(["a"] * 10 + ["b"] * 10, [True] * 20), "g", "allpass")
        assert not result.assumption_met
        assert "degenerate" in result.test_used


class TestMcNemar:
    def test_symmetric_disagreement_is_not_significant(self):
        a = np.array([True] * 10 + [False] * 10 + [True] * 40)
        b = np.array([False] * 10 + [True] * 10 + [True] * 40)
        *_, p_value = mcnemar_test(a, b)
        assert p_value == pytest.approx(1.0)

    def test_lopsided_disagreement_is_significant(self):
        a = np.array([True] * 12 + [False] + [True] * 20)
        b = np.array([False] * 12 + [True] + [True] * 20)
        *_, p_value = mcnemar_test(a, b)
        assert p_value < 0.01

    def test_identical_models_are_never_distinguishable(self):
        same = np.array([True, False, True, False])
        a_only, b_only, *_, p_value = mcnemar_test(same, same)
        assert (a_only, b_only) == (0, 0)
        assert p_value == 1.0

    def test_concordant_pairs_carry_no_information(self):
        """Adding tasks both models pass must not change the p-value."""
        a = np.array([True] * 8 + [False] * 2)
        b = np.array([False] * 8 + [True] * 2)
        *_, base = mcnemar_test(a, b)

        padded_a = np.concatenate([a, np.ones(50, dtype=bool)])
        padded_b = np.concatenate([b, np.ones(50, dtype=bool)])
        *_, padded = mcnemar_test(padded_a, padded_b)
        assert base == pytest.approx(padded)


class TestHolmCorrection:
    def test_adjusted_values_never_shrink_and_stay_monotonic(self):
        raw = [0.001, 0.02, 0.03, 0.04, 0.5]
        adjusted, _ = holm_correction(raw)
        assert all(new >= old for new, old in zip(adjusted, raw))
        assert adjusted == sorted(adjusted)

    def test_is_less_conservative_than_bonferroni(self):
        raw = [0.01, 0.02, 0.03]
        adjusted, _ = holm_correction(raw)
        bonferroni = [value * len(raw) for value in raw]
        assert adjusted[-1] <= bonferroni[-1]

    def test_empty_input(self):
        assert holm_correction([]) == ([], [])


class TestClassifier:
    @pytest.mark.parametrize(
        "prompt,expected",
        [
            ("Build a React component with useState and a data-testid", "frontend"),
            ("Implement an Express endpoint POST /login returning a JWT", "api"),
            ("Center a card using flexbox with responsive breakpoints", "css"),
            ("Define a TypeScript interface with generics", "typescript"),
            ("Fix the bug in the following code", "bugfix"),
        ],
    )
    def test_unambiguous_prompts(self, prompt, expected):
        assert classify_prompt(prompt) == expected

    def test_repair_intent_beats_embedded_language(self):
        """A bug-fix prompt contains more React words than repair words."""
        prompt = (
            "Fix the bug in the following code. import React, { useState } from 'react'; "
            "export default function Counter() { const [count] = useState(0); return "
            "<div data-testid='count'>{count}</div>; }"
        )
        assert classify_prompt(prompt) == "bugfix"

    def test_falls_back_rather_than_failing(self):
        assert classify_prompt("") in {
            "frontend", "api", "css", "typescript", "bugfix",
        }


class TestRecommenderScoring:
    @staticmethod
    def _runs():
        rows = []
        for model, rate, latency, tokens in [
            ("Fast", 0.2, 500, 200),
            ("Accurate", 0.8, 2000, 800),
        ]:
            for index in range(10):
                rows.append(
                    {
                        "task_id": f"T-{index:03d}",
                        "category": "frontend",
                        "difficulty": "easy",
                        "model_name": model,
                        "pass_fail": index < rate * 10,
                        "latency_total_ms": latency,
                        "tokens_output": tokens,
                    }
                )
        return pd.DataFrame(rows)

    def test_scoring_is_invariant_to_latency_units(self):
        """Milliseconds versus seconds must not change the ranking."""
        runs = self._runs()
        rescaled = runs.assign(
            latency_total_ms=runs["latency_total_ms"] / 1000.0,
            tokens_output=runs["tokens_output"] / 7.0,
        )

        first = Recommender(build_profiles(runs)).rank("frontend", "easy")
        second = Recommender(build_profiles(rescaled)).rank("frontend", "easy")
        assert [item.model_name for item in first] == [item.model_name for item in second]

    def test_pure_accuracy_weighting_picks_the_accurate_model(self):
        weights = {"category": 1.0, "difficulty": 0.0, "latency": 0.0, "tokens": 0.0}
        recommender = Recommender(build_profiles(self._runs()), weights)
        assert recommender.recommend("frontend", "easy") == "Accurate"

    def test_heavy_efficiency_weighting_picks_the_fast_model(self):
        """Documents the calibration failure found in Phase 6."""
        weights = {"category": 0.1, "difficulty": 0.0, "latency": 0.45, "tokens": 0.45}
        recommender = Recommender(build_profiles(self._runs()), weights)
        assert recommender.recommend("frontend", "easy") == "Fast"

    def test_default_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_ranking_is_deterministic_under_ties(self):
        runs = self._runs()
        tied = runs.assign(pass_fail=True, latency_total_ms=1000, tokens_output=400)
        recommender = Recommender(build_profiles(tied))
        first = [item.model_name for item in recommender.rank("frontend", "easy")]
        second = [item.model_name for item in recommender.rank("frontend", "easy")]
        assert first == second == sorted(first)
