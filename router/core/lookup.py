"""The shrunk per-group lookup table of router v1 (plan S4.1; `docs/research/S4_spec.md` §3.2).

Per group g and candidate j (a direct model or a fixed ladder), with n_g training tasks and s_gj of
them accepted, the shrunk acceptance is the beta-binomial posterior mean

    a_gj = (s_gj + kappa * p_j) / (n_g + kappa),

where p_j is j's pool-wide training acceptance and kappa is one prior strength shared by all
candidates. kappa comes from the ANOVA (method-of-moments) estimator of the intra-class correlation
rho, with numerator and denominator summed over candidates, and kappa = (1 - rho) / rho:

    N = sum n_g,  G = groups with rows,  n0 = (N - sum n_g^2 / N) / (G - 1)
    MSB_j = sum n_g (r_gj - p_j)^2 / (G - 1),   MSW_j = sum n_g r_gj (1 - r_gj) / (N - G)
    rho = sum_j (MSB_j - MSW_j) / sum_j (MSB_j + (n0 - 1) MSW_j)

G <= 1, N <= G, a zero denominator or rho <= 0 give kappa = inf (full pooling: every group uses
the pool row); rho >= 1 gives kappa = 0 (no pooling).

The decision rule is P0.6's Rule A on the shrunk values with a margin m: in each group, k is the
direct-dispatch candidate with the highest shrunk acceptance (ties to the cheaper, then the lower
column); choose the cheapest candidate whose shrunk acceptance is at least a_gk - m, ties to the
higher shrunk acceptance, then the lower column. With kappa = 0 and m = 0 this is
`baselines.lookup.rule_a_column` exactly (tested).

Everything here is a pure function of training-fold numbers; nothing reads a scored task.
"""

from __future__ import annotations

import math

import numpy as np

TOL = 1e-12          # as P0.6's rule tables (`baselines.common.TOL`)


def group_counts(groups: np.ndarray, A: np.ndarray, names) -> tuple[np.ndarray, np.ndarray]:
    """(S, n): per group (rows in `names` order) the column sums of A and the row count."""
    idx = {g: i for i, g in enumerate(names)}
    s = np.array([idx[g] for g in groups], dtype=int)
    n = np.bincount(s, minlength=len(names)).astype(float)
    S = np.zeros((len(names), A.shape[1]))
    np.add.at(S, s, A)
    return S, n


def anova_rho(S: np.ndarray, n: np.ndarray) -> float:
    """ANOVA estimator of the intra-class correlation, pooled over the columns of S; nan when
    undefined (one group, no within-group degrees of freedom, or a zero denominator)."""
    keep = n > 0
    S, n = S[keep], n[keep]
    G, N = len(n), float(n.sum())
    if G <= 1 or N <= G:
        return math.nan
    p = S.sum(0) / N
    r = S / n[:, None]
    n0 = (N - float((n ** 2).sum()) / N) / (G - 1)
    msb = (n[:, None] * (r - p) ** 2).sum(0) / (G - 1)
    msw = (n[:, None] * r * (1 - r)).sum(0) / (N - G)
    den = float((msb + (n0 - 1) * msw).sum())
    if den <= 0:
        return math.nan
    return float((msb - msw).sum()) / den


def kappa_from_rho(rho: float) -> float:
    if math.isnan(rho) or rho <= 0:
        return math.inf
    if rho >= 1:
        return 0.0
    return (1.0 - rho) / rho


def shrink(S: np.ndarray, n: np.ndarray, kappa: float) -> tuple[np.ndarray, np.ndarray]:
    """(shrunk acceptance per group and column, pool-wide acceptance per column). A group with no
    rows, and every group when kappa is inf, gets the pool row."""
    N = float(n.sum())
    p = S.sum(0) / N if N > 0 else np.zeros(S.shape[1])
    out = np.tile(p, (len(n), 1))
    if math.isinf(kappa):
        return out, p
    seen = n > 0
    out[seen] = (S[seen] + kappa * p) / (n[seen] + kappa)[:, None]
    return out, p


def best_direct(a: np.ndarray, c: np.ndarray, n_direct: int) -> int:
    """P0.6's CellBest column: highest acceptance among the direct columns, ties to the cheaper,
    then the lower column."""
    return int(np.lexsort((np.arange(n_direct), c[:n_direct], -a[:n_direct]))[0])


def choose(a: np.ndarray, c: np.ndarray, n_direct: int, margin: float) -> int:
    """Rule A with a margin: the cheapest column whose acceptance is at least the best direct
    column's minus `margin`; ties to the higher acceptance, then the lower column."""
    k = best_direct(a, c, n_direct)
    idx = np.arange(len(a))
    cand = idx[a >= a[k] - margin - TOL]
    return int(cand[np.lexsort((cand, -a[cand], c[cand]))[0]])
