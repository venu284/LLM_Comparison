"""Bradley-Terry model for pairwise human preference (H3).

Bradley-Terry assigns each model a latent strength p_i such that

    P(i beats j) = p_i / (p_i + p_j)

This is the model WebDev Arena and Chatbot Arena use to turn pairwise votes into
a ranking. Fitting is by the minorization-maximization iteration, which is
monotonic -- every step increases the log-likelihood -- and needs no step size
or gradient. Convergence is checked, not assumed.

Identifiability: only ratios of strengths are determined by the data, so the
scale is fixed by normalizing the strengths to sum to 1 before reporting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Same convention as Elo/Arena: 400 points is a 10:1 odds ratio.
RATING_SCALE = 400.0
RATING_ANCHOR = 1000.0


@dataclass
class BradleyTerryFit:
    models: List[str]
    strengths: Dict[str, float]
    ratings: Dict[str, float]
    iterations: int
    converged: bool
    log_likelihood: float
    n_comparisons: int
    win_matrix: np.ndarray
    ci_lower: Dict[str, float] = field(default_factory=dict)
    ci_upper: Dict[str, float] = field(default_factory=dict)

    def ranking(self) -> List[Tuple[str, float]]:
        """Models ordered strongest first, with their ratings."""
        return sorted(self.ratings.items(), key=lambda item: -item[1])

    def probability(self, model_a: str, model_b: str) -> float:
        strength_a = self.strengths[model_a]
        strength_b = self.strengths[model_b]
        total = strength_a + strength_b
        return 0.5 if total <= 0 else strength_a / total


def build_win_matrix(
    votes: Iterable[Tuple[str, str, str]], models: Sequence[str]
) -> np.ndarray:
    """Count wins into a matrix. votes are (model_a, model_b, winner) triples.

    A winner that is neither model_a nor model_b (a tie) contributes half a win
    to each side, which is the standard Bradley-Terry treatment of draws.
    """
    index = {model: position for position, model in enumerate(models)}
    wins = np.zeros((len(models), len(models)), dtype=float)

    for model_a, model_b, winner in votes:
        if model_a not in index or model_b not in index:
            continue
        position_a, position_b = index[model_a], index[model_b]
        if winner == model_a:
            wins[position_a, position_b] += 1.0
        elif winner == model_b:
            wins[position_b, position_a] += 1.0
        else:
            wins[position_a, position_b] += 0.5
            wins[position_b, position_a] += 0.5

    return wins


def _log_likelihood(strengths: np.ndarray, wins: np.ndarray) -> float:
    total = 0.0
    count = len(strengths)
    for i in range(count):
        for j in range(count):
            if i == j or wins[i, j] == 0:
                continue
            denominator = strengths[i] + strengths[j]
            if denominator > 0:
                total += wins[i, j] * math.log(strengths[i] / denominator)
    return total


def fit_bradley_terry(
    wins: np.ndarray,
    models: Sequence[str],
    max_iterations: int = 1000,
    tolerance: float = 1e-9,
) -> Tuple[np.ndarray, int, bool, float]:
    """Minorization-maximization fit. Returns (strengths, iterations, converged, ll)."""
    count = len(models)
    strengths = np.ones(count, dtype=float) / count
    comparisons = wins + wins.T
    total_wins = wins.sum(axis=1)

    iterations = 0
    converged = False

    for iterations in range(1, max_iterations + 1):
        updated = np.zeros(count, dtype=float)

        for i in range(count):
            denominator = 0.0
            for j in range(count):
                if i == j or comparisons[i, j] == 0:
                    continue
                denominator += comparisons[i, j] / (strengths[i] + strengths[j])
            # A model with no wins has strength 0; guard against 0/0.
            updated[i] = total_wins[i] / denominator if denominator > 0 else 0.0

        total = updated.sum()
        if total <= 0:
            break
        updated /= total

        if np.max(np.abs(updated - strengths)) < tolerance:
            strengths = updated
            converged = True
            break
        strengths = updated

    return strengths, iterations, converged, _log_likelihood(strengths, wins)


def strengths_to_ratings(
    strengths: np.ndarray, models: Sequence[str]
) -> Dict[str, float]:
    """Map latent strengths onto an Elo-like scale for readability."""
    floor = 1e-12
    logs = np.log10(np.maximum(strengths, floor))
    centered = logs - logs.mean()
    return {
        model: float(RATING_ANCHOR + RATING_SCALE * value)
        for model, value in zip(models, centered)
    }


def bootstrap_intervals(
    votes: Sequence[Tuple[str, str, str]],
    models: Sequence[str],
    iterations: int = 500,
    seed: int = 42,
    confidence: float = 0.95,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Percentile bootstrap over votes, resampled with replacement."""
    if not votes:
        return {}, {}

    rng = np.random.default_rng(seed)
    votes_array = list(votes)
    count = len(votes_array)
    samples: List[Dict[str, float]] = []

    for _ in range(iterations):
        indices = rng.integers(0, count, size=count)
        resampled = [votes_array[index] for index in indices]
        wins = build_win_matrix(resampled, models)
        strengths, _, _, _ = fit_bradley_terry(wins, models)
        samples.append(strengths_to_ratings(strengths, models))

    tail = (1.0 - confidence) / 2.0 * 100.0
    lower, upper = {}, {}
    for model in models:
        values = [sample[model] for sample in samples]
        lower[model] = float(np.percentile(values, tail))
        upper[model] = float(np.percentile(values, 100.0 - tail))

    return lower, upper


def fit(
    votes: Sequence[Tuple[str, str, str]],
    models: Optional[Sequence[str]] = None,
    bootstrap: int = 0,
) -> BradleyTerryFit:
    """Fit Bradley-Terry to a vote list, optionally with bootstrap intervals."""
    if models is None:
        discovered = set()
        for model_a, model_b, _ in votes:
            discovered.update((model_a, model_b))
        models = sorted(discovered)

    models = list(models)
    wins = build_win_matrix(votes, models)
    strengths, iterations, converged, log_likelihood = fit_bradley_terry(wins, models)
    ratings = strengths_to_ratings(strengths, models)

    lower, upper = ({}, {})
    if bootstrap > 0:
        lower, upper = bootstrap_intervals(votes, models, iterations=bootstrap)

    return BradleyTerryFit(
        models=models,
        strengths={model: float(value) for model, value in zip(models, strengths)},
        ratings=ratings,
        iterations=iterations,
        converged=converged,
        log_likelihood=log_likelihood,
        n_comparisons=len(votes),
        win_matrix=wins,
        ci_lower=lower,
        ci_upper=upper,
    )
