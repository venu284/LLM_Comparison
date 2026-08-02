"""Tests for the Phase 7 comparison scheduler.

Each test asserts a property the uniform-random sampler it replaced did not
have, so a regression to `random.choice` fails here rather than showing up as an
uncomputable Krippendorff's alpha after the evaluators have gone home.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

HUMAN_EVAL_DIR = Path(__file__).resolve().parent
if str(HUMAN_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(HUMAN_EVAL_DIR))

from sampler import (  # noqa: E402
    BOTH_FAIL,
    BOTH_PASS,
    SPLIT,
    UNKNOWN,
    ComparisonScheduler,
    make_cell,
)

MODELS = ["A", "B", "C", "D", "E"]


def _grid(n_tasks: int = 20, models=MODELS):
    return {f"T-{index:03d}": list(models) for index in range(n_tasks)}


def _drain(scheduler: ComparisonScheduler, evaluator: str, count: int):
    """Serve `count` comparisons to one evaluator, recording each as a vote."""
    served = []
    for _ in range(count):
        comparison = scheduler.next_comparison(evaluator)
        scheduler.record(*comparison.cell, evaluator)
        served.append(comparison)
    return served


class TestCells:
    def test_pair_order_does_not_create_two_cells(self):
        assert make_cell("T-001", "B", "A") == make_cell("T-001", "A", "B")

    def test_grid_is_every_task_by_every_model_pair(self):
        scheduler = ComparisonScheduler(_grid(4))
        assert scheduler.coverage()["cells_total"] == 4 * 10

    def test_a_task_with_one_model_contributes_nothing(self):
        scheduler = ComparisonScheduler({"T-001": ["A"], "T-002": ["A", "B"]})
        assert scheduler.coverage()["cells_total"] == 1

    def test_no_comparable_task_raises_rather_than_serving_nonsense(self):
        scheduler = ComparisonScheduler({})
        with pytest.raises(RuntimeError):
            scheduler.next_comparison("human_01")


class TestNoRepeatsWithinAnEvaluator:
    def test_eighty_comparisons_contain_no_duplicate(self):
        """The defect this replaced: p(repeat) was ~0.99 at this session length."""
        scheduler = ComparisonScheduler(_grid(65))
        served = _drain(scheduler, "human_01", 80)
        assert len(served) == 80
        assert len({comparison.cell for comparison in served}) == 80

    def test_exhausting_the_pool_falls_back_instead_of_raising(self):
        scheduler = ComparisonScheduler({"T-001": ["A", "B"], "T-002": ["A", "B"]})
        served = _drain(scheduler, "human_01", 5)
        assert len(served) == 5  # 2 distinct cells, then repeats rather than an error


class TestAnchors:
    def test_every_model_pair_is_anchored(self):
        """A pair missing from the anchor set would have no overlapping ratings."""
        scheduler = ComparisonScheduler(_grid(20))
        pairs = {(cell[1], cell[2]) for cell in scheduler._anchors}
        assert len(pairs) == 10

    def test_anchor_set_is_identical_across_instances(self):
        first = ComparisonScheduler(_grid(20))
        second = ComparisonScheduler(_grid(20))
        assert first._anchors == second._anchors

    def test_several_evaluators_converge_on_the_same_anchor_cells(self):
        """This is what makes Krippendorff's alpha computable at all."""
        scheduler = ComparisonScheduler(_grid(65))
        for index in range(6):
            _drain(scheduler, f"human_{index:02d}", 80)

        coverage = scheduler.coverage()
        assert coverage["anchor_cells"] == 20
        assert coverage["anchor_cells_multi_rated"] == 20

    def test_anchors_are_not_all_front_loaded(self):
        """Interleaved, so anchors are not rated only while calibrating."""
        scheduler = ComparisonScheduler(_grid(65))
        served = _drain(scheduler, "human_01", 80)
        anchor_positions = [i for i, item in enumerate(served) if item.is_anchor]
        assert len(anchor_positions) == 20
        assert max(anchor_positions) > 20


class TestCoverageBalance:
    def test_the_least_observed_cell_is_served_next(self):
        scheduler = ComparisonScheduler({"T-001": ["A", "B", "C"]}, anchors_per_pair=0)
        scheduler.record("T-001", "A", "B", "human_01")
        scheduler.record("T-001", "A", "B", "human_02")
        scheduler.record("T-001", "A", "C", "human_01")

        assert scheduler.next_comparison("human_03").cell == make_cell("T-001", "B", "C")

    def test_model_pairs_stay_within_one_vote_of_each_other(self):
        """Bradley-Terry intervals are set by the least-observed pair."""
        scheduler = ComparisonScheduler(_grid(20), anchors_per_pair=0)
        for index in range(5):
            _drain(scheduler, f"human_{index:02d}", 40)

        per_pair = scheduler.coverage()["votes_per_model_pair"]
        assert len(per_pair) == 10
        assert max(per_pair.values()) - min(per_pair.values()) <= 1


class TestStrata:
    @staticmethod
    def _outcomes():
        return {
            ("T-001", "A"): True,
            ("T-001", "B"): True,
            ("T-001", "C"): False,
            ("T-002", "A"): False,
            ("T-002", "B"): False,
        }

    def test_labels_match_the_recorded_outcomes(self):
        scheduler = ComparisonScheduler(
            {"T-001": ["A", "B", "C"], "T-002": ["A", "B"]}, outcomes=self._outcomes()
        )
        assert scheduler.stratum(make_cell("T-001", "A", "B")) == BOTH_PASS
        assert scheduler.stratum(make_cell("T-001", "A", "C")) == SPLIT
        assert scheduler.stratum(make_cell("T-002", "A", "B")) == BOTH_FAIL

    def test_missing_outcome_is_unknown_rather_than_assumed_failed(self):
        scheduler = ComparisonScheduler({"T-009": ["A", "B"]})
        assert scheduler.stratum(make_cell("T-009", "A", "B")) == UNKNOWN

    def test_all_fail_tasks_are_kept_in_the_pool(self):
        """Filtering them would bias H3 upward by only asking easy questions."""
        scheduler = ComparisonScheduler(
            {"T-001": ["A", "B", "C"], "T-002": ["A", "B"]}, outcomes=self._outcomes()
        )
        served = _drain(scheduler, "human_01", 4)
        assert any(item.stratum == BOTH_FAIL for item in served)


class TestLoadVotes:
    def test_restarting_does_not_reset_coverage(self):
        first = ComparisonScheduler(_grid(20))
        _drain(first, "human_01", 30)

        votes = [
            (cell[0], cell[1], cell[2], "human_01")
            for cell, count in first._counts.items()
            for _ in range(count)
        ]
        second = ComparisonScheduler(_grid(20))
        second.load_votes(votes)

        assert second.coverage()["votes_total"] == first.coverage()["votes_total"]
        assert second.coverage()["cells_rated"] == first.coverage()["cells_rated"]
