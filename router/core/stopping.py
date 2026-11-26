"""The posterior a new model's per-group stopping rule reads, for router v2 without the S4.3 ability
model (plan S4.2 [M4.2]; CLAIMS S0.13; `docs/research/S4_spec.md` §4, §4.1, P8 (a)).

Per (new model, group): a Beta(alpha, beta) posterior on the new model's pass rate.
- Prior mean = the pool prior mu_g: the mean of the known models' shrunk acceptance in the group
  (R11: never theta = 0). Prior strength from the method-of-moments spread of those known-model
  rates: s = mu (1 - mu) / v - 1, floored at `MIN_STRENGTH`; v = 0 gives `MAX_STRENGTH`; mu is
  clipped to [`MU_CLIP`, 1 - `MU_CLIP`].
- PASS adds 1 to alpha, FAIL adds 1 to beta. UNKNOWN never updates (S0.13).
- The SD on the logit scale is exact for a Beta variable: Var[logit p] = psi1(alpha) + psi1(beta)
  (psi1 the trigamma function). The group's offset from the fixed pool mean has the same SD. Stop
  when it is <= `STOP_POSTERIOR_SD` (0.5).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import polygamma

from router.settings import STOP_POSTERIOR_SD
from router.sim.verifier import FAIL, PASS

MIN_STRENGTH = 1.0
MAX_STRENGTH = 1e6
MU_CLIP = 1e-3


def prior_from_known(rates) -> tuple[float, float]:
    """(alpha, beta) of the Beta prior from the known models' rates in one group."""
    r = np.asarray(rates, dtype=float)
    mu = float(np.clip(r.mean(), MU_CLIP, 1 - MU_CLIP))
    v = float(r.var())
    s = MAX_STRENGTH if v <= 0 else max(mu * (1 - mu) / v - 1.0, MIN_STRENGTH)
    s = min(s, MAX_STRENGTH)
    return mu * s, (1 - mu) * s


def logit_sd(alpha: float, beta: float) -> float:
    return math.sqrt(float(polygamma(1, alpha) + polygamma(1, beta)))


@dataclass
class GroupPosterior:
    alpha: float
    beta: float
    prior_mean: float = float("nan")
    prior_strength: float = float("nan")

    @classmethod
    def from_known(cls, rates) -> "GroupPosterior":
        a, b = prior_from_known(rates)
        return cls(a, b, a / (a + b), a + b)

    def update(self, verdict: str) -> bool:
        """PASS/FAIL update the posterior; UNKNOWN (or anything else) does not. Returns whether it
        updated."""
        if verdict == PASS:
            self.alpha += 1.0
            return True
        if verdict == FAIL:
            self.beta += 1.0
            return True
        return False

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def sd(self) -> float:
        return logit_sd(self.alpha, self.beta)

    def stop(self, tol: float = STOP_POSTERIOR_SD) -> bool:
        return self.sd <= tol
