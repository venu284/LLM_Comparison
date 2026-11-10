"""Simulated run-time verifier (Rule R1).

During replay a cascade must not escalate on the benchmark's ground truth. It sees only this
verifier's verdict, which agrees with the truth (score >= threshold) except with probability eps.
"""

from __future__ import annotations

import numpy as np

from router.data.scoring import ACCEPT_THRESHOLD


class Verifier:
    def __init__(self, eps: float = 0.0, seed: int = 0, threshold: float = ACCEPT_THRESHOLD):
        if not 0.0 <= eps <= 1.0:
            raise ValueError("eps must be in [0, 1]")
        self.eps = eps
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)

    def passes(self, true_score: float) -> bool:
        truth = true_score >= self.threshold
        flip = self.rng.random() < self.eps
        return truth != flip
