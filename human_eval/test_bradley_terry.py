"""Tests for the Bradley-Terry estimator.

The central test plants a known set of model strengths, simulates votes from
them, and checks the fit recovers what it was never told. An estimator that
cannot do that cannot be trusted with real votes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HUMAN_EVAL_DIR = Path(__file__).resolve().parent
if str(HUMAN_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(HUMAN_EVAL_DIR))

from bradley_terry import build_win_matrix, fit  # noqa: E402

PLANTED = {"A": 0.50, "B": 0.25, "C": 0.15, "D": 0.10}


def _simulate(strengths, n_votes=8000, seed=7):
    rng = np.random.default_rng(seed)
    models = list(strengths)
    votes = []
    for _ in range(n_votes):
        a, b = rng.choice(models, 2, replace=False)
        probability = strengths[a] / (strengths[a] + strengths[b])
        votes.append((a, b, a if rng.random() < probability else b))
    return votes


class TestRecovery:
    def test_recovers_planted_ranking(self):
        result = fit(_simulate(PLANTED), models=list(PLANTED))
        expected = [model for model, _ in sorted(PLANTED.items(), key=lambda x: -x[1])]
        assert [model for model, _ in result.ranking()] == expected

    def test_recovers_planted_strengths(self):
        result = fit(_simulate(PLANTED), models=list(PLANTED))
        for model, true_strength in PLANTED.items():
            assert result.strengths[model] == pytest.approx(true_strength, abs=0.03)

    def test_converges(self):
        result = fit(_simulate(PLANTED), models=list(PLANTED))
        assert result.converged
        assert result.iterations < 500

    def test_pairwise_probability_round_trips(self):
        result = fit(_simulate(PLANTED), models=list(PLANTED))
        expected = PLANTED["A"] / (PLANTED["A"] + PLANTED["D"])
        assert result.probability("A", "D") == pytest.approx(expected, abs=0.05)


class TestEdgeCases:
    def test_symmetric_votes_give_equal_ratings(self):
        votes = [("X", "Y", "X"), ("X", "Y", "Y")] * 200
        result = fit(votes, models=["X", "Y"])
        assert result.ratings["X"] == pytest.approx(result.ratings["Y"], abs=1.0)

    def test_ties_count_as_half_wins(self):
        matrix = build_win_matrix([("X", "Y", "tie")], ["X", "Y"])
        assert matrix[0, 1] == 0.5
        assert matrix[1, 0] == 0.5

    def test_unknown_models_are_ignored(self):
        matrix = build_win_matrix([("X", "Z", "X")], ["X", "Y"])
        assert matrix.sum() == 0

    def test_models_inferred_from_votes_when_not_supplied(self):
        result = fit([("P", "Q", "P")] * 20 + [("Q", "P", "Q")] * 5)
        assert set(result.models) == {"P", "Q"}

    def test_a_model_that_never_wins_ranks_last(self):
        votes = [("A", "B", "A")] * 50 + [("B", "C", "B")] * 50
        result = fit(votes, models=["A", "B", "C"])
        assert result.ranking()[-1][0] == "C"

    def test_rating_scale_is_centred(self):
        result = fit(_simulate(PLANTED), models=list(PLANTED))
        assert np.mean(list(result.ratings.values())) == pytest.approx(1000.0, abs=1e-6)


class TestSyntheticGenerator:
    def test_correctness_dominates_the_preference_model(self):
        from synthetic_votes import _preference_probability

        # A passes, B fails -> A strongly preferred.
        assert _preference_probability(True, False, 0.15, 500, 500) == pytest.approx(0.85)
        assert _preference_probability(False, True, 0.15, 500, 500) == pytest.approx(0.15)

    def test_tiebreak_is_weaker_when_both_fail(self):
        from synthetic_votes import _preference_probability

        both_pass = _preference_probability(True, True, 0.15, 200, 800)
        both_fail = _preference_probability(False, False, 0.15, 200, 800)
        assert abs(both_pass - 0.5) > abs(both_fail - 0.5)

    def test_equal_tokens_give_a_coin_flip(self):
        from synthetic_votes import _preference_probability

        assert _preference_probability(True, True, 0.15, 400, 400) == pytest.approx(0.5)
