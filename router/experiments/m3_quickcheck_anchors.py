"""Anchor selection for onboarding a new model.

- `fisher`: adaptive, as in computerised adaptive testing. Each next anchor is the unused item
  with the most Fisher information a^2 p (1 - p) at the current ability estimate.
- `random`: k items drawn uniformly.
Both read outcomes through `probe(item_index) -> 0/1`, so the same code serves replay and live use.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.special import expit

from router.core.irt import IRTParams, ability_posterior


def onboard(params: IRTParams, k: int, probe: Callable[[int], float], method: str,
            seed: int, pool: np.ndarray | None = None) -> tuple[float, float, list[int]]:
    rng = np.random.default_rng(seed)
    pool = np.arange(len(params.b)) if pool is None else np.asarray(pool)
    if method == "random":
        idx = list(rng.choice(pool, size=k, replace=False))
        y = np.array([probe(i) for i in idx], dtype=float)
        th, sd = ability_posterior(params, np.array(idx), y)
        return th, sd, idx
    if method != "fisher":
        raise ValueError(method)
    used: list[int] = []
    ys: list[float] = []
    th, sd = 0.0, 1.0
    avail = np.ones(len(pool), dtype=bool)
    a, b = params.a[pool], params.b[pool]
    for _ in range(k):
        p = expit(a * (th - b))
        info = a ** 2 * p * (1 - p)
        info = np.where(avail, info + rng.random(len(pool)) * 1e-9, -np.inf)
        j = int(np.argmax(info))
        avail[j] = False
        used.append(int(pool[j]))
        ys.append(probe(int(pool[j])))
        th, sd = ability_posterior(params, np.array(used), np.array(ys, dtype=float))
    return th, sd, used
