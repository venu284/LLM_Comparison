"""Comparison scheduling for Phase 7.

`app.py` originally drew a task uniformly at random, then two of that task's
models uniformly at random. At the sample sizes the protocol itself specifies,
that has three defects:

- **Repeats within one evaluator.** 65 tasks x 10 model pairs is 650 cells. An
  evaluator doing the protocol's 80 comparisons draws a repeat with probability
  ~0.99 by the birthday bound. One person rating the same pair twice corrupts
  the inter-rater agreement promised in protocol section 6.3, because
  Krippendorff's alpha treats those as two coders.
- **Overlap between evaluators left to chance.** Alpha needs units rated by more
  than one person. Under uniform sampling at the 300-vote minimum only ~51 of
  650 cells attract a second vote, and which ones is random.
- **Unequal coverage of the 10 model pairs.** A Bradley-Terry interval is set by
  the least-observed pair, so an unbalanced draw spends votes without buying
  precision.

The scheduler fixes all three without changing what a comparison *is*:

1. Every evaluator is shown the same **anchor set** first -- a deterministic,
   pair-stratified subset -- in a per-evaluator shuffled order so position in the
   session is not confounded with the anchor cells. This is the subset alpha is
   computed on, and it exists by construction rather than by luck.
2. After the anchors, the next cell is the one with the fewest votes so far that
   this evaluator has not already seen. Ties are broken randomly.
3. A cell an evaluator has already voted on is never offered to them again.

What the scheduler deliberately does **not** do is filter out tasks that no model
solved. 27 of the 65 tasks are failed by every model, and dropping them would
bias H3 upward by only ever asking about tasks the models can do. They are
labelled with an outcome stratum instead, so the analysis can report Spearman rho
overall and restricted to tasks with at least one passing model, and the reader
can see both.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: A comparison cell: one task and one unordered pair of models. The pair is
#: stored lexicographically so that (X, Y) and (Y, X) are the same cell -- which
#: side is shown on the left is decided per serve, not per cell.
Cell = Tuple[str, str, str]

BOTH_PASS = "both_pass"
SPLIT = "split"
BOTH_FAIL = "both_fail"
UNKNOWN = "unknown"

DEFAULT_ANCHORS_PER_PAIR = 2
DEFAULT_ANCHOR_RATE = 0.35
ANCHOR_SEED = 20260802


def make_cell(task_id: str, model_a: str, model_b: str) -> Cell:
    lo, hi = sorted((model_a, model_b))
    return (task_id, lo, hi)


@dataclass(frozen=True)
class Comparison:
    """One scheduled comparison, before sides are randomized."""

    cell: Cell
    stratum: str
    is_anchor: bool

    @property
    def task_id(self) -> str:
        return self.cell[0]

    @property
    def models(self) -> Tuple[str, str]:
        return self.cell[1], self.cell[2]


class ComparisonScheduler:
    """Decides which comparison an evaluator sees next.

    Holds its counts in memory and is updated on each recorded vote, so the
    caller must seed it with the votes already in the database at startup.
    """

    def __init__(
        self,
        task_models: Dict[str, Sequence[str]],
        outcomes: Optional[Dict[Tuple[str, str], bool]] = None,
        anchors_per_pair: int = DEFAULT_ANCHORS_PER_PAIR,
        anchor_rate: float = DEFAULT_ANCHOR_RATE,
        seed: int = ANCHOR_SEED,
    ) -> None:
        """`task_models` maps a task id to the models with extractable code for
        it. `outcomes` maps (task_id, model) to whether that run passed; it is
        used only to label strata and may be omitted.

        `anchor_rate` is the chance a comparison is drawn from the anchor set
        while any of it is unrated by this evaluator. At the default 0.35 an
        evaluator completing the protocol's 80 comparisons rates all 20 anchors
        within roughly the first 57, leaving the rest for coverage.
        """
        self._anchor_rate = anchor_rate
        self._outcomes = dict(outcomes or {})
        self._cells: List[Cell] = []
        for task_id, models in sorted(task_models.items()):
            unique = sorted(set(models))
            for index, model_a in enumerate(unique):
                for model_b in unique[index + 1 :]:
                    self._cells.append((task_id, model_a, model_b))

        self._counts: Counter = Counter()
        self._seen: Dict[str, Set[Cell]] = {}
        self._rng = random.Random(seed)
        self._anchors = self._choose_anchors(anchors_per_pair, seed)
        self._anchor_set = set(self._anchors)

    # ------------------------------------------------------------------ setup

    def _choose_anchors(self, anchors_per_pair: int, seed: int) -> List[Cell]:
        """A fixed, pair-stratified subset every evaluator rates.

        Stratified by model pair rather than drawn at random, so the anchor set
        cannot happen to omit a pair -- which would leave that pair's rating
        resting entirely on non-overlapping single votes.
        """
        if anchors_per_pair <= 0:
            return []

        by_pair: Dict[Tuple[str, str], List[Cell]] = {}
        for cell in self._cells:
            by_pair.setdefault((cell[1], cell[2]), []).append(cell)

        chooser = random.Random(seed)
        anchors: List[Cell] = []
        for pair in sorted(by_pair):
            candidates = sorted(by_pair[pair])
            take = min(anchors_per_pair, len(candidates))
            anchors.extend(chooser.sample(candidates, take))
        return sorted(anchors)

    def load_votes(self, votes: Iterable[Tuple[str, str, str, str]]) -> None:
        """Seed from existing votes as (task_id, model_a, model_b, evaluator_id)."""
        for task_id, model_a, model_b, evaluator_id in votes:
            self.record(task_id, model_a, model_b, evaluator_id)

    def record(self, task_id: str, model_a: str, model_b: str, evaluator_id: str) -> None:
        cell = make_cell(task_id, model_a, model_b)
        self._counts[cell] += 1
        self._seen.setdefault(evaluator_id, set()).add(cell)

    # --------------------------------------------------------------- serving

    def stratum(self, cell: Cell) -> str:
        """Whether both models passed this task, one did, or neither."""
        _, model_a, model_b = cell
        first = self._outcomes.get((cell[0], model_a))
        second = self._outcomes.get((cell[0], model_b))
        if first is None or second is None:
            return UNKNOWN
        if first and second:
            return BOTH_PASS
        if first or second:
            return SPLIT
        return BOTH_FAIL

    def next_comparison(self, evaluator_id: str) -> Comparison:
        """The next cell for this evaluator: unrated anchors first, then the
        globally least-observed cell they have not yet seen.
        """
        if not self._cells:
            raise RuntimeError("no comparison cells; no task has two extractable solutions")

        seen = self._seen.get(evaluator_id, set())

        pending_anchors = [cell for cell in self._anchors if cell not in seen]
        if pending_anchors and self._rng.random() < self._anchor_rate:
            # Interleaved rather than front-loaded, so an anchor is not always
            # rated while the evaluator is still calibrating.
            cell = self._rng.choice(pending_anchors)
            return Comparison(cell=cell, stratum=self.stratum(cell), is_anchor=True)

        remaining = [cell for cell in self._cells if cell not in seen]
        if not remaining:
            # This evaluator has rated every cell. Rather than refuse, fall back
            # to the least-observed cell overall; a second rating from the same
            # person is excluded from alpha at analysis time.
            remaining = self._cells

        fewest = min(self._counts[cell] for cell in remaining)
        candidates = [cell for cell in remaining if self._counts[cell] == fewest]
        cell = self._rng.choice(candidates)
        return Comparison(
            cell=cell, stratum=self.stratum(cell), is_anchor=cell in self._anchor_set
        )

    # ------------------------------------------------------------ reporting

    def coverage(self) -> Dict[str, object]:
        """Where the votes have landed. Surfaced by `/api/results`."""
        per_pair: Counter = Counter()
        for cell, count in self._counts.items():
            per_pair[f"{cell[1]} vs {cell[2]}"] += count

        strata: Counter = Counter()
        for cell, count in self._counts.items():
            strata[self.stratum(cell)] += count

        multi_rated = sum(1 for cell in self._anchors if self._counts[cell] >= 2)
        return {
            "cells_total": len(self._cells),
            "cells_rated": sum(1 for cell in self._cells if self._counts[cell]),
            "votes_total": sum(self._counts.values()),
            "votes_per_model_pair": dict(sorted(per_pair.items())),
            "votes_by_stratum": dict(sorted(strata.items())),
            "anchor_cells": len(self._anchors),
            "anchor_cells_multi_rated": multi_rated,
            "evaluators": len(self._seen),
            "note": (
                "anchor_cells_multi_rated is the base for Krippendorff's alpha. "
                "Strata are reported because tasks no model solved are kept in "
                "the pool rather than filtered, which would bias H3 upward."
            ),
        }
