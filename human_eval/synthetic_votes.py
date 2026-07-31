"""Synthetic pairwise votes for validating the Phase 7 pipeline.

WHAT THIS IS FOR
----------------
These votes are generated, not collected. Their only purpose is to prove that
the path

    generate -> store -> fit Bradley-Terry -> rank

works, and that the fitting code recovers a ranking it was not told, before any
human time is spent on it.

WHAT THIS IS NOT
----------------
Not evidence for H3. H3 claims automated test-driven rankings correlate with
*human* preference at rho > 0.7. Testing that claim against votes generated
from the automated results would be circular: the correlation is baked in by
construction. Every function here tags its output `synthetic=True`, and the
report must label these results as synthetic wherever they appear.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SyntheticVote:
    task_id: str
    model_a: str
    model_b: str
    winner: str
    evaluator_id: str
    synthetic: bool = True


def _preference_probability(
    a_passed: bool,
    b_passed: bool,
    noise: float,
    a_tokens: float,
    b_tokens: float,
) -> float:
    """P(voter prefers A).

    Correctness dominates: a reviewer shown working code and broken code picks
    the working one most of the time, with `noise` controlling how often they
    do not. When both sides are equally correct the tie is broken by concision,
    which is a weak, plausible stand-in for perceived quality.
    """
    if a_passed and not b_passed:
        return 1.0 - noise
    if b_passed and not a_passed:
        return noise

    total = a_tokens + b_tokens
    if total <= 0:
        return 0.5
    concision_edge = (b_tokens - a_tokens) / total  # positive when A is shorter

    # The tiebreak has to be weak, and weakest when both solutions are broken.
    # Most task/model pairs in this benchmark fail, so a strong concision term
    # would decide the majority of comparisons and the simulated reviewer would
    # be ranking verbosity rather than quality. Both-pass carries a little more
    # signal than both-fail: a reviewer comparing two working solutions can
    # reasonably prefer the tighter one, whereas two broken solutions give them
    # very little to go on.
    weight = 0.15 if (a_passed and b_passed) else 0.05
    return float(np.clip(0.5 + weight * concision_edge, 0.05, 0.95))


def generate_votes(
    runs: pd.DataFrame,
    n_votes: int = 1000,
    noise: float = 0.15,
    seed: int = 42,
    models: Optional[Sequence[str]] = None,
) -> List[SyntheticVote]:
    """Sample pairwise comparisons from real per-task outcomes.

    `noise` is the probability that a simulated reviewer prefers the failing
    solution. 0.15 reflects that human reviewers are good but not perfect at
    spotting broken code from a short read.
    """
    rng = np.random.default_rng(seed)

    if models is None:
        models = sorted(runs["model_name"].unique())
    models = list(models)

    outcomes: Dict[Tuple[str, str], Tuple[bool, float]] = {}
    for row in runs.itertuples():
        outcomes[(row.task_id, row.model_name)] = (
            bool(row.pass_fail),
            float(row.tokens_output),
        )

    task_ids = sorted(runs["task_id"].unique())
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1 :]]
    if not pairs or not task_ids:
        return []

    votes: List[SyntheticVote] = []
    for index in range(n_votes):
        task_id = str(rng.choice(task_ids))
        model_a, model_b = pairs[int(rng.integers(0, len(pairs)))]

        record_a = outcomes.get((task_id, model_a))
        record_b = outcomes.get((task_id, model_b))
        if record_a is None or record_b is None:
            continue

        a_passed, a_tokens = record_a
        b_passed, b_tokens = record_b
        probability = _preference_probability(a_passed, b_passed, noise, a_tokens, b_tokens)
        winner = model_a if rng.random() < probability else model_b

        votes.append(
            SyntheticVote(
                task_id=task_id,
                model_a=model_a,
                model_b=model_b,
                winner=winner,
                evaluator_id=f"synthetic_{index % 12:02d}",
            )
        )

    return votes


def votes_to_triples(votes: Sequence[SyntheticVote]) -> List[Tuple[str, str, str]]:
    return [(vote.model_a, vote.model_b, vote.winner) for vote in votes]


def empirical_win_rates(votes: Sequence[SyntheticVote]) -> Dict[str, float]:
    """Raw win rate per model across the generated votes."""
    wins: Dict[str, float] = {}
    appearances: Dict[str, float] = {}

    for vote in votes:
        for model in (vote.model_a, vote.model_b):
            appearances[model] = appearances.get(model, 0.0) + 1.0
        wins[vote.winner] = wins.get(vote.winner, 0.0) + 1.0

    return {
        model: wins.get(model, 0.0) / count
        for model, count in appearances.items()
        if count > 0
    }


def recovery_check(
    fitted_ranking: Sequence[Tuple[str, float]], votes: Sequence[SyntheticVote]
) -> Dict[str, float]:
    """Does the Bradley-Terry fit reproduce the ordering present in the votes?

    Compared against the *empirical win rates in the generated votes*, which is
    what the fit is supposed to explain. Comparing against automated pass rates
    instead would conflate two different questions: whether the fitting code
    works, and whether the generator's preference model happens to agree with
    pass rate. Only the first is a validation.
    """
    from scipy import stats as scipy_stats

    win_rates = empirical_win_rates(votes)
    models = [model for model, _ in fitted_ranking]
    fitted_scores = [rating for _, rating in fitted_ranking]
    observed = [win_rates.get(model, 0.0) for model in models]

    if len(models) < 3:
        return {"spearman_rho": float("nan"), "n": len(models)}

    rho, p_value = scipy_stats.spearmanr(fitted_scores, observed)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n": len(models),
        "win_rates": win_rates,
        "note": "fit vs the votes it was fitted to; validates the estimator, not H3",
    }


def agreement_with_pass_rate(
    fitted_ranking: Sequence[Tuple[str, float]], pass_rates: pd.Series
) -> Dict[str, float]:
    """How closely the simulated preference ordering tracks automated pass rate.

    This is the quantity H3 is about -- but measured on synthetic votes it is
    circular, because the generator was seeded from these same pass rates. It is
    reported to show what the pipeline *would* compute once real votes exist,
    never as evidence.
    """
    from scipy import stats as scipy_stats

    models = [model for model, _ in fitted_ranking]
    fitted_scores = [rating for _, rating in fitted_ranking]
    automated = [float(pass_rates.get(model, 0.0)) for model in models]

    if len(models) < 3:
        return {"spearman_rho": float("nan"), "n": len(models)}

    rho, p_value = scipy_stats.spearmanr(fitted_scores, automated)
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_value),
        "n": len(models),
        "note": "CIRCULAR on synthetic votes; shown only to demonstrate the H3 computation",
    }
