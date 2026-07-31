"""Leave-one-task-out validation of the recommender (H4).

H4 predicts that task-aware routing improves code quality by more than 15% over
single-model strategies. This module measures that, and reports the answer
whichever way it comes out.

Every fold rebuilds model profiles from the other 64 tasks only. If profiles
were built once from all 65, the held-out task's own outcome would be baked
into the pass rate used to choose a model for it, and the result would be
inflated. `assert_no_leakage` enforces this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.data import ExperimentData
from analysis.recommender import (
    DEFAULT_WEIGHTS,
    Recommender,
    build_profiles,
    classify_prompt,
)


@dataclass
class FoldOutcome:
    task_id: str
    category: str
    difficulty: str
    recommended_model: str
    recommender_pass: bool
    best_overall_model: str
    best_overall_pass: bool
    random_expected: float
    oracle_pass: bool


@dataclass
class ValidationResult:
    folds: List[FoldOutcome]
    strategy_scores: Dict[str, float]
    per_category: pd.DataFrame
    h4_improvement_pct: float
    h4_supported: bool
    weights: Dict[str, float]
    notes: List[str] = field(default_factory=list)

    @property
    def n_folds(self) -> int:
        return len(self.folds)


def assert_no_leakage(training: pd.DataFrame, held_out_task: str) -> None:
    if (training["task_id"] == held_out_task).any():
        raise AssertionError(
            f"Leakage: held-out task {held_out_task} present in training rows"
        )


def leave_one_task_out(
    data: ExperimentData,
    weights: Optional[Dict[str, float]] = None,
    prompts: Optional[pd.DataFrame] = None,
) -> ValidationResult:
    """Run the full LOOCV comparison.

    `prompts` is optional. When supplied, the category for each held-out task is
    predicted by the keyword classifier rather than read from the task metadata,
    which measures the deployed pipeline end to end instead of assuming perfect
    task classification.
    """
    if not data.is_fully_paired():
        missing = int(data.missing_mask().to_numpy().sum())
        raise ValueError(
            f"LOOCV requires a complete task x model grid; {missing} cells are missing. "
            "This happens on a filtered frame (exclude_zero_tests=True), where absent "
            "runs would be scored as failures and the oracle bound would be meaningless."
        )

    runs = data.runs
    pass_matrix = data.pass_matrix()
    metadata = data.task_metadata()
    weights = dict(weights or DEFAULT_WEIGHTS)

    prompt_lookup: Dict[str, str] = {}
    if prompts is not None:
        prompt_lookup = dict(zip(prompts["task_id"], prompts["prompt"]))

    folds: List[FoldOutcome] = []

    for task_id in data.task_ids:
        training = runs[runs["task_id"] != task_id]
        assert_no_leakage(training, task_id)

        profiles = build_profiles(training)
        recommender = Recommender(profiles, weights)

        difficulty = str(metadata.loc[task_id, "difficulty"])
        if task_id in prompt_lookup:
            category = classify_prompt(prompt_lookup[task_id])
        else:
            category = str(metadata.loc[task_id, "category"])

        recommended = recommender.recommend(category, difficulty)

        # The single-best-model baseline is also fitted on training folds only,
        # otherwise it would enjoy an information advantage the recommender does
        # not have.
        training_rates = training.groupby("model_name")["pass_fail"].mean()
        best_overall = str(training_rates.idxmax())

        outcomes = pass_matrix.loc[task_id]
        folds.append(
            FoldOutcome(
                task_id=task_id,
                category=str(metadata.loc[task_id, "category"]),
                difficulty=difficulty,
                recommended_model=recommended,
                recommender_pass=bool(outcomes[recommended]),
                best_overall_model=best_overall,
                best_overall_pass=bool(outcomes[best_overall]),
                # Exact expectation of picking uniformly at random, which is
                # less noisy than sampling a finite number of seeds.
                random_expected=float(outcomes.mean()),
                oracle_pass=bool(outcomes.any()),
            )
        )

    frame = pd.DataFrame([vars(fold) for fold in folds])
    strategy_scores = {
        "recommender": float(frame["recommender_pass"].mean()),
        "best_single_model": float(frame["best_overall_pass"].mean()),
        "random": float(frame["random_expected"].mean()),
        "oracle": float(frame["oracle_pass"].mean()),
    }

    baseline = strategy_scores["best_single_model"]
    improvement = (
        (strategy_scores["recommender"] - baseline) / baseline * 100.0
        if baseline > 0
        else float("nan")
    )

    per_category = frame.groupby("category").agg(
        tasks=("task_id", "count"),
        recommender=("recommender_pass", "mean"),
        best_single=("best_overall_pass", "mean"),
        random=("random_expected", "mean"),
        oracle=("oracle_pass", "mean"),
    )

    notes: List[str] = []
    if strategy_scores["recommender"] > strategy_scores["oracle"]:
        notes.append("IMPOSSIBLE: recommender exceeded the oracle upper bound")
    if strategy_scores["random"] > strategy_scores["oracle"]:
        notes.append("IMPOSSIBLE: random exceeded the oracle upper bound")

    return ValidationResult(
        folds=folds,
        strategy_scores=strategy_scores,
        per_category=per_category,
        h4_improvement_pct=float(improvement),
        h4_supported=bool(improvement > 15.0),
        weights=weights,
        notes=notes,
    )


def weight_sensitivity(
    data: ExperimentData,
    grid: Optional[List[Dict[str, float]]] = None,
    prompts: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """How much does the conclusion depend on the chosen weights?

    The framework fixed w=(0.5, 0.2, 0.15, 0.15) before any data existed. If the
    result flips under mild reweighting, that fragility belongs in the report.
    """
    if grid is None:
        grid = [
            {"category": 0.5, "difficulty": 0.2, "latency": 0.15, "tokens": 0.15},
            {"category": 1.0, "difficulty": 0.0, "latency": 0.0, "tokens": 0.0},
            {"category": 0.7, "difficulty": 0.3, "latency": 0.0, "tokens": 0.0},
            {"category": 0.4, "difficulty": 0.1, "latency": 0.25, "tokens": 0.25},
            {"category": 0.25, "difficulty": 0.25, "latency": 0.25, "tokens": 0.25},
        ]

    rows = []
    for weights in grid:
        result = leave_one_task_out(data, weights, prompts)
        rows.append(
            {
                **{f"w_{key}": value for key, value in weights.items()},
                "recommender": result.strategy_scores["recommender"],
                "best_single": result.strategy_scores["best_single_model"],
                "improvement_pct": result.h4_improvement_pct,
                "h4_supported": result.h4_supported,
            }
        )
    return pd.DataFrame(rows)


def bootstrap_difference(
    result: ValidationResult, iterations: int = 10000, seed: int = 42
) -> Dict[str, float]:
    """Bootstrap CI for recommender minus best-single-model, over tasks.

    A point estimate on 65 tasks says little on its own; this shows whether the
    difference is distinguishable from zero.
    """
    rng = np.random.default_rng(seed)
    recommender = np.array([fold.recommender_pass for fold in result.folds], dtype=float)
    baseline = np.array([fold.best_overall_pass for fold in result.folds], dtype=float)

    differences = recommender - baseline
    count = len(differences)
    samples = rng.choice(differences, size=(iterations, count), replace=True).mean(axis=1)

    return {
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.percentile(samples, 2.5)),
        "ci_upper": float(np.percentile(samples, 97.5)),
        "p_two_sided": float(2 * min((samples <= 0).mean(), (samples >= 0).mean())),
    }
