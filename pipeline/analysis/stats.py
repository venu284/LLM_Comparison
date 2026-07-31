"""Statistical tests for RQ1 and RQ2.

Two properties of this dataset drive every choice here:

1. Every model attempts the same 65 tasks, so model-vs-model comparisons are
   *paired*. McNemar's test is correct; a two-sample proportion test is not.
2. n = 5 models, so any correlation on model-level aggregates (parameter count
   against pass rate) is descriptive. It is reported with that caveat attached
   rather than dressed up as an inferential result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from analysis.data import CATEGORIES, MODEL_PARAMS_B, ExperimentData

MIN_EXPECTED_COUNT = 5.0


@dataclass
class ChiSquareResult:
    label: str
    chi2: float
    p_value: float
    dof: int
    cramers_v: float
    n: int
    min_expected: float
    test_used: str
    assumption_met: bool

    def describe(self) -> str:
        return (
            f"{self.label}: {self.test_used} chi2={self.chi2:.3f}, df={self.dof}, "
            f"p={self.p_value:.4g}, Cramer's V={self.cramers_v:.3f}, n={self.n}"
        )


@dataclass
class McNemarResult:
    model_a: str
    model_b: str
    a_only: int          # tasks a passed and b failed
    b_only: int          # tasks b passed and a failed
    both: int
    neither: int
    p_value: float
    p_value_holm: float = float("nan")
    significant_holm: bool = False

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only


@dataclass
class StatsReport:
    overall_chi2: ChiSquareResult
    per_category_chi2: List[ChiSquareResult]
    difficulty_chi2: ChiSquareResult
    spearman_params: Dict[str, float]
    mcnemar: List[McNemarResult]
    notes: List[str] = field(default_factory=list)


def cramers_v(chi2: float, n: int, rows: int, cols: int) -> float:
    """Effect size for a contingency table. 0 = no association, 1 = perfect.

    Chi-squared answers "is there an association"; with n=325 even a trivial one
    reaches significance, so magnitude has to be reported alongside it.
    """
    smaller_dimension = min(rows - 1, cols - 1)
    if smaller_dimension <= 0 or n <= 0:
        return float("nan")
    return float(np.sqrt(chi2 / (n * smaller_dimension)))


def _contingency(frame: pd.DataFrame, index: str) -> pd.DataFrame:
    """Rows = levels of `index`, columns = [failed, passed] counts.

    Column selection goes through `reindex`, not `table[[False, True]]`: with
    boolean column labels the latter is interpreted as a boolean *row* mask and
    silently returns the wrong table.
    """
    table = pd.crosstab(frame[index], frame["pass_fail"])
    return table.reindex(columns=[False, True], fill_value=0)


def chi_square_test(frame: pd.DataFrame, index: str, label: str) -> ChiSquareResult:
    table = _contingency(frame, index)
    observed = table.to_numpy()

    # scipy raises if any expected frequency is zero, which happens whenever a
    # whole row or column is empty (e.g. a category where every model failed).
    degenerate = (
        observed.sum() == 0
        or (observed.sum(axis=1) == 0).any()
        or (observed.sum(axis=0) == 0).any()
    )
    if degenerate:
        return ChiSquareResult(
            label=label,
            chi2=float("nan"),
            p_value=float("nan"),
            dof=0,
            cramers_v=float("nan"),
            n=int(observed.sum()),
            min_expected=0.0,
            test_used="not applicable (degenerate table)",
            assumption_met=False,
        )

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    min_expected = float(expected.min())
    assumption_met = min_expected >= MIN_EXPECTED_COUNT
    test_used = "Pearson"

    # Sparse 2x2 tables violate the chi-squared approximation; Fisher is exact.
    if not assumption_met and observed.shape == (2, 2):
        _, p_value = stats.fisher_exact(observed)
        test_used = "Fisher exact"

    return ChiSquareResult(
        label=label,
        chi2=float(chi2),
        p_value=float(p_value),
        dof=int(dof),
        cramers_v=cramers_v(float(chi2), int(observed.sum()), *observed.shape),
        n=int(observed.sum()),
        min_expected=min_expected,
        test_used=test_used,
        assumption_met=assumption_met,
    )


def mcnemar_test(passes_a: np.ndarray, passes_b: np.ndarray) -> Tuple[int, int, int, int, float]:
    """Exact McNemar via a two-sided binomial test on the discordant pairs.

    Only tasks where the two models disagree carry information about which is
    better. Tasks both passed or both failed are uninformative by construction.
    """
    a_only = int(np.sum(passes_a & ~passes_b))
    b_only = int(np.sum(~passes_a & passes_b))
    both = int(np.sum(passes_a & passes_b))
    neither = int(np.sum(~passes_a & ~passes_b))

    discordant = a_only + b_only
    if discordant == 0:
        return a_only, b_only, both, neither, 1.0

    p_value = float(stats.binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue)
    return a_only, b_only, both, neither, p_value


def holm_correction(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Holm-Bonferroni. Uniformly more powerful than plain Bonferroni.

    Ten pairwise comparisons at alpha=0.05 would otherwise produce a false
    positive around 40% of the time.
    """
    count = len(p_values)
    if count == 0:
        return [], []

    order = np.argsort(p_values)
    adjusted = np.empty(count, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running_max = max(running_max, value)  # enforce monotonicity
        adjusted[index] = running_max

    return adjusted.tolist(), [value < alpha for value in adjusted]


def pairwise_mcnemar(data: ExperimentData, alpha: float = 0.05) -> List[McNemarResult]:
    """Pairwise McNemar across models. Requires a complete task x model grid.

    On a filtered frame the missing cells would be silently counted as failures,
    biasing every discordant count, so this refuses rather than misleads.
    """
    if not data.is_fully_paired():
        missing = int(data.missing_mask().to_numpy().sum())
        raise ValueError(
            f"McNemar requires paired observations; {missing} task x model cells are "
            "missing. Run on the unfiltered frame."
        )

    matrix = data.pass_matrix()
    results: List[McNemarResult] = []

    for model_a, model_b in combinations(sorted(matrix.columns), 2):
        a_only, b_only, both, neither, p_value = mcnemar_test(
            matrix[model_a].to_numpy(), matrix[model_b].to_numpy()
        )
        results.append(
            McNemarResult(model_a, model_b, a_only, b_only, both, neither, p_value)
        )

    adjusted, significant = holm_correction([item.p_value for item in results], alpha)
    for result, p_adjusted, is_significant in zip(results, adjusted, significant):
        result.p_value_holm = p_adjusted
        result.significant_holm = is_significant

    return results


def spearman_parameter_correlation(data: ExperimentData) -> Dict[str, float]:
    """Rank correlation between active parameter count and overall pass rate.

    n = 5. Reported as descriptive; the p-value is included only for
    completeness and should not be leaned on.
    """
    rates = data.overall_pass_rates()
    models = [model for model in rates.index if model in MODEL_PARAMS_B]

    params = [MODEL_PARAMS_B[model] for model in models]
    pass_rates = [float(rates[model]) for model in models]

    if len(models) < 3:
        return {"rho": float("nan"), "p_value": float("nan"), "n": len(models)}

    rho, p_value = stats.spearmanr(params, pass_rates)
    return {
        "rho": float(rho),
        "p_value": float(p_value),
        "n": len(models),
        "interpretation": "descriptive only; n=5 is far too small for inference",
    }


def build_stats_report(data: ExperimentData, alpha: float = 0.05) -> StatsReport:
    frame = data.runs
    notes: List[str] = []

    overall = chi_square_test(frame, "model_name", "Model x outcome (all tasks)")

    per_category: List[ChiSquareResult] = []
    for category in CATEGORIES:
        subset = frame[frame["category"] == category]
        if subset.empty:
            continue
        result = chi_square_test(subset, "model_name", f"Model x outcome ({category})")
        per_category.append(result)
        if not result.assumption_met:
            notes.append(
                f"{category}: min expected count {result.min_expected:.2f} < {MIN_EXPECTED_COUNT}; "
                "chi-squared approximation is unreliable for this subtable"
            )

    difficulty = chi_square_test(frame, "difficulty", "Difficulty x outcome (pooled)")

    if data.zero_test_rows and not data.excluded_zero_tests:
        notes.append(
            f"{data.zero_test_rows} of {len(frame)} runs reported tests_total=0 and were "
            "scored as failures; see the zero-test sensitivity check"
        )

    # McNemar needs complete pairing. On a filtered frame it is skipped rather
    # than computed on cells that would be silently treated as failures.
    if data.is_fully_paired():
        mcnemar_results = pairwise_mcnemar(data, alpha)
    else:
        mcnemar_results = []
        counts = data.attempt_counts().to_dict()
        notes.append(
            "McNemar skipped: the frame is not fully paired after filtering "
            f"(runs per model: {counts}). Pass rates on this frame have unequal "
            "denominators and are not directly comparable to the full-data rates."
        )

    return StatsReport(
        overall_chi2=overall,
        per_category_chi2=per_category,
        difficulty_chi2=difficulty,
        spearman_params=spearman_parameter_correlation(data),
        mcnemar=mcnemar_results,
        notes=notes,
    )
