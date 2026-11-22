"""REPRODUCTION ONLY: the ACRouter paper's metrics (arXiv 2606.22902; CLAIMS O10).

The paper's AvgPerf is the mean released score of the chosen model, in percent, on the zero-filled
paper view (missing cells scored 0) or the complete view. It is **not** acceptance: partial
credit counts, and the paper view scores missing cells as failures, which S0.11 forbids. Use it
only to reproduce published numbers (S2.6). Claim-facing numbers come from `router/sim/metrics.py`
(acceptance under S0.1). Only `router/experiments/s2_reproduce*.py` may import this module; a test
enforces that.

- AvgPerf: mean score of the chosen model, in percent;
- $Total: summed USD cost of the chosen model;
- Perf/$: AvgPerf / $Total (matches the published Always-Opus 1.29);
- CumReg: summed per-task gap to the reward oracle, reward = score - 0.1 * USD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PAPER_LAMBDA = 0.1


@dataclass
class Outcome:
    """Per-task result of a single-choice policy: the chosen model's score and cost."""
    score: pd.Series
    cost: pd.Series


def from_choice(score: pd.DataFrame, cost: pd.DataFrame, choice: pd.Series) -> Outcome:
    idx = score.index
    choice = choice.reindex(idx)
    rows = np.arange(len(idx))
    cols = score.columns.get_indexer(choice.values)
    if (cols < 0).any():
        raise ValueError("choice names a model not in the matrix")
    return Outcome(pd.Series(score.values[rows, cols], idx), pd.Series(cost.values[rows, cols], idx))


def paper_avgperf(out: Outcome) -> float:
    """The paper's AvgPerf: mean released score of the chosen model, percent (O10)."""
    return float(out.score.mean() * 100)


def paper_summary(out: Outcome, score: pd.DataFrame, cost: pd.DataFrame) -> dict:
    perf = paper_avgperf(out)
    total = float(out.cost.sum())
    oracle_reward = (score - PAPER_LAMBDA * cost).max(axis=1)
    reward = out.score - PAPER_LAMBDA * out.cost
    return {"PaperAvgPerf": perf, "$Total": total,
            "Perf/$": perf / total if total > 0 else np.inf,
            "CumReg": float((oracle_reward - reward).sum())}
