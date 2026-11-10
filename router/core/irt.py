"""1PL / 2PL item response models fitted by penalised maximum likelihood (MAP).

P(model m solves task q) = sigmoid(a_q * (theta_m - b_q)), with a_q = 1 for the 1PL.
With only eight models per task, item parameters are weakly identified, so Gaussian priors
regularise them: theta ~ N(0, 1), b ~ N(0, 2^2), log a ~ N(0, 0.5^2).
Missing cells (NaN) are ignored.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

SD_THETA, SD_B, SD_LOGA = 1.0, 2.0, 0.5


@dataclass
class IRTParams:
    items: list[str]
    models: list[str]
    b: np.ndarray
    a: np.ndarray
    theta: np.ndarray
    kind: str

    def p(self, theta: np.ndarray | float, idx=None) -> np.ndarray:
        b = self.b if idx is None else self.b[idx]
        a = self.a if idx is None else self.a[idx]
        return expit(a * (np.asarray(theta)[..., None] - b))


def fit(Y: np.ndarray, items, models, kind: str = "1pl", maxiter: int = 500) -> IRTParams:
    """Y: models x items array of 0/1 with NaN for missing."""
    M, Q = Y.shape
    mask = ~np.isnan(Y)
    y = np.nan_to_num(Y)
    two = kind == "2pl"

    def unpack(z):
        th = z[:M]
        b = z[M:M + Q]
        la = z[M + Q:] if two else np.zeros(Q)
        return th, b, la

    def f(z):
        th, b, la = unpack(z)
        a = np.exp(la)
        eta = a[None, :] * (th[:, None] - b[None, :])
        p = expit(eta)
        ll = mask * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        r = mask * (y - p)                       # d ll / d eta
        g_th = (r * a[None, :]).sum(1) - th / SD_THETA ** 2
        g_b = -(r * a[None, :]).sum(0) - b / SD_B ** 2
        obj = ll.sum() - (th ** 2).sum() / (2 * SD_THETA ** 2) - (b ** 2).sum() / (2 * SD_B ** 2)
        grads = [g_th, g_b]
        if two:
            g_la = (r * eta).sum(0) - la / SD_LOGA ** 2
            obj -= (la ** 2).sum() / (2 * SD_LOGA ** 2)
            grads.append(g_la)
        return -obj, -np.concatenate(grads)

    z0 = np.zeros(M + Q + (Q if two else 0))
    res = minimize(f, z0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    th, b, la = unpack(res.x)
    return IRTParams(list(items), list(models), b, np.exp(la), th, kind)


def ability_posterior(params: IRTParams, idx: np.ndarray, y: np.ndarray,
                      prior_sd: float = SD_THETA) -> tuple[float, float]:
    """MAP ability and Laplace sd for a new model from its outcomes y on items idx."""
    if len(idx) == 0:
        return 0.0, prior_sd
    a, b = params.a[idx], params.b[idx]
    th = 0.0
    # Damped Newton on a concave objective. Undamped steps overshoot when the anchors are
    # highly discriminating items close to the current estimate (the Fisher case).
    for _ in range(200):
        p = expit(a * (th - b))
        g = (a * (y - p)).sum() - th / prior_sd ** 2
        h = -(a ** 2 * p * (1 - p)).sum() - 1 / prior_sd ** 2
        step = float(np.clip(g / h, -0.5, 0.5))
        th = float(np.clip(th - step, -6.0, 6.0))
        if abs(step) < 1e-8:
            break
    p = expit(a * (th - b))
    info = (a ** 2 * p * (1 - p)).sum() + 1 / prior_sd ** 2
    return float(th), float(1 / np.sqrt(info))


def heldout_loglik(params: IRTParams, Y: np.ndarray, test_mask: np.ndarray) -> float:
    """Mean Bernoulli log-likelihood per held-out cell."""
    p = expit(params.a[None, :] * (params.theta[:, None] - params.b[None, :]))
    y = Y[test_mask]
    q = np.clip(p[test_mask], 1e-12, 1 - 1e-12)
    return float((y * np.log(q) + (1 - y) * np.log(1 - q)).mean())
