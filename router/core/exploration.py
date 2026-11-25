"""Bounded onboarding of a new model for router v2 (plan S4.2 [M4.1]; CLAIMS S0.13;
`docs/research/S4_spec.md` §4, §4.1, P7 (a)).

- Calibration probes (`draw_probes`): charged calls on training tasks, drawn at fit time per
  training group, at most `MAX_PROBES_PER_GROUP` per group and `MAX_PROBES_PER_MODEL` (and the call
  cap) per model, without replacement, from stream `s4_probe_draw`. Groups go in sorted order; each
  draw is from the group's sorted task ids, so the draw is reproducible under the seed.
- Exploration (`is_exploration`): exactly 10% of each predicted group's requests: the j-th request
  of the group (counted from 1) is exploration iff j is a multiple of 10. No draw is involved.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.settings import EXPLORATION_SHARE, MAX_PROBES_PER_GROUP, MAX_PROBES_PER_MODEL
from router.sim.stats import rng

EVERY = round(1 / EXPLORATION_SHARE)          # 10
if abs(EVERY * EXPLORATION_SHARE - 1.0) > 1e-12:
    raise ValueError("EXPLORATION_SHARE must be 1/n for the deterministic schedule (P7 (a))")
CALL_CAP = 100                                # S4 spec §4.1: the largest DP2_K_GRID value


def is_exploration(j: int) -> bool:
    """Whether the j-th request (1-based) of a predicted group is exploration."""
    if j < 1:
        raise ValueError(j)
    return j % EVERY == 0


def draw_probes(groups: pd.Series, available: pd.Series, seed: int, *, cap: int = CALL_CAP,
                per_group: int = MAX_PROBES_PER_GROUP,
                per_model: int = MAX_PROBES_PER_MODEL) -> dict[str, list[str]]:
    """{group: [task ids]} of the calibration probes: per group (sorted), up to `per_group` of its
    training tasks where the new model is available, drawn without replacement; the total stops at
    min(per_model, cap)."""
    g = rng(seed, "s4_probe_draw")
    total = min(per_model, cap)
    out: dict[str, list[str]] = {}
    av = available.reindex(groups.index).fillna(False).astype(bool)
    for grp in sorted(groups.unique()):
        left = total - sum(len(v) for v in out.values())
        if left <= 0:
            break
        pool = np.array(sorted(groups.index[(groups == grp).to_numpy() & av.to_numpy()]))
        k = min(per_group, len(pool), left)
        if k == 0:
            continue
        out[grp] = list(pool[np.sort(g.choice(len(pool), size=k, replace=False))])
    return out
