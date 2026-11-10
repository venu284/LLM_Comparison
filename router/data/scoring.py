"""Per-dimension score type and the acceptance rule.

The type tags are derived from the observed score distributions (Report 8 §3.3, R-b). They are
provisional until checked against each source benchmark's own scoring rule (FR4).
"""

from __future__ import annotations

import pandas as pd

ACCEPT_THRESHOLD = 0.5


def score_types(score: pd.DataFrame, dimension: pd.Series) -> pd.DataFrame:
    """Share of 0/1 scores and number of distinct values per dimension."""
    long = score.stack().rename("s").reset_index()
    long["dim"] = long.iloc[:, 0].map(dimension)
    g = long.groupby("dim")["s"]
    out = pd.DataFrame({"binary_share": g.apply(lambda s: s.isin([0.0, 1.0]).mean()),
                        "distinct": g.nunique()})
    out["type"] = pd.cut(out.binary_share, [-0.01, 0.5, 0.999, 1.0],
                         labels=["continuous", "mostly-binary", "binary"])
    return out


def accepted(score: pd.DataFrame | pd.Series, threshold: float = ACCEPT_THRESHOLD):
    """A task is accepted when its score reaches the threshold."""
    return score >= threshold
