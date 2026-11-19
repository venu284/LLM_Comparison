"""Sanity check for the anchor code: does Fisher selection beat random when item parameters are known?

Simulates a 2PL with 7,080 items (the probing-set size) and 300 respondents. For each respondent,
onboards with the TRUE item parameters (Fisher vs random, k in {10, 25, 50}) and also with item
parameters re-fitted from only 7 other respondents, as in the real-data experiment.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.m3_fisher_synthetic
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from router.core import irt
from router.core.anchors import onboard
from router.settings import REPO_ROOT

OUT = REPO_ROOT / "router" / "experiments" / "out"
Q, N_RESP, KS = 7080, 300, (10, 25, 50)


def main() -> None:
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1.5, Q)
    a = np.exp(rng.normal(0, 0.4, Q))
    theta = rng.normal(0, 1, N_RESP)
    P = 1 / (1 + np.exp(-a[None, :] * (theta[:, None] - b[None, :])))
    Y = (rng.random(P.shape) < P).astype(float)
    true = irt.IRTParams(list(range(Q)), list(range(N_RESP)), b, a, theta, "2pl")
    # Calibration from only 7 respondents, as in the leave-one-model-out experiment.
    seven = irt.fit(Y[:7], range(Q), range(7), "2pl")

    rows = []
    for label, params in (("true item parameters", true), ("fitted from 7 respondents", seven)):
        for r in range(7, 107):                                  # 100 respondents not used to fit
            for k in KS:
                for method in ("fisher", "random"):
                    th, _, _ = onboard(params, k, lambda i: Y[r, i], method, seed=r)
                    rows.append({"items": label, "k": k, "method": method,
                                 "abs_err": abs(th - theta[r])})
    df = pd.DataFrame(rows)
    tab = df.groupby(["items", "k", "method"]).abs_err.mean().unstack()
    tab.to_csv(OUT / "a6_fisher_synthetic.csv")
    print("mean |theta_hat - theta_true| over 100 simulated respondents")
    print(tab.round(3).to_string())


if __name__ == "__main__":
    main()
