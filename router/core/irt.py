"""The plain 1PL, P(model m passes task q) = sigmoid(theta_m - b_q): the special case of the S4.3
ability model without mu_g and gamma (plan S4.3: "`core/irt.py` kept as the 1PL case";
`docs/research/S4_spec.md` §6.1). Fitted on training tasks only, through the S4.6 guard.

The 2026-11-10 quick-check implementation (with a 2PL and the anchor code) is frozen in
`router/experiments/m3_quickcheck_irt.py`; this fit reproduces its 1PL (tested).
"""

from __future__ import annotations

from router.core import ability
from router.core.guards import check_non_transductive


def fit(Y, items, models, *, scored_ids, problem_of, maxiter: int = 500) -> ability.AbilityParams:
    """1PL on Y (models x training items; 1, 0, NaN for unavailable). theta is `delta`."""
    check_non_transductive(items, scored_ids, problem_of)
    return ability.fit(Y, items, models, None, scored_ids=scored_ids, problem_of=problem_of,
                       kind="1pl", maxiter=maxiter)


__all__ = ["fit"]
