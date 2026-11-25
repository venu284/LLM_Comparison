"""S4.3: the hierarchical ability model (plan §2 item 1, S4.3 [M3.1]; `docs/research/S4_spec.md`
§6.1; CLAIMS S0.13, R11).

    P(model m passes task q) = sigmoid(mu_g + delta_m + gamma_{m,g} - b_q)

g is the task's group (on training tasks, the true S1.1 family). Fitted on the known models and the
training tasks only (S4.6: `fit` calls `check_non_transductive` first).

Estimation (`kind="hier"`): MAP with empirical-Bayes variances.
- mu_g ~ N(0, 10^2), delta_m ~ N(0, 10^2): weak priors on fixed effects. After the fit delta is
  centred on the known models (delta -= mean, mu += mean; every logit unchanged), so delta = 0 is the
  average known model.
- gamma_{m,g} ~ N(0, sd_gamma^2), b_q ~ N(0, sd_b^2). The two variances come from an approximate EM:
  alternate a joint MAP fit (L-BFGS, exact gradient) with
  var <- mean(estimate^2 + 1 / (Fisher information + 1 / var)), from sd_gamma = 0.5, sd_b = 1, until
  both change by less than `EM_TOL` (relative) or `EM_ROUNDS` rounds; floored at `VAR_FLOOR`.
- sd_delta, the new model's prior SD on delta, is the sample SD (ddof 1) of the known models' delta.

`kind="1pl"`: the special case without mu and gamma, P = sigmoid(theta_m - b_q), with the fixed
priors theta ~ N(0, 1), b ~ N(0, 2^2) of the 2026-11-10 quick-check `irt.fit` (frozen in
`router/experiments/m3_quickcheck_irt.py`), which it reproduces (tested). theta is stored in
`delta` uncentred.

A new model (`NewModelPosterior`) starts at the pool prior: delta = mean known delta (0 after
centring), gamma = 0, so P = sigmoid(mu_g - b_q) for the average known model, never theta = 0
(R11). PASS / FAIL on a task with known b and group update a Laplace posterior over
(delta, gamma_1..G), found by a damped Newton iteration written out here; UNKNOWN never updates
(S0.13). Per group it reports the offset delta + gamma_g: its mean and SD (the SD S0.13's
`STOP_POSTERIOR_SD` is stated for).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from router.core.guards import check_non_transductive
from router.sim.verifier import FAIL, PASS

SD_FIXED = 10.0                    # mu_g and delta_m (fixed effects)
SD_GAMMA0, SD_B0 = 0.5, 1.0        # EM starting values
EM_ROUNDS, EM_TOL, VAR_FLOOR = 50, 1e-4, 1e-6
SD_THETA_1PL, SD_B_1PL = 1.0, 2.0  # the quick-check 1PL priors
KINDS = ("hier", "1pl")
_EPS = 1e-12


@dataclass
class AbilityParams:
    kind: str
    models: list
    items: list
    groups: list
    item_group: np.ndarray        # group index of each item
    mu: np.ndarray                # (G,)
    delta: np.ndarray             # (M,)
    gamma: np.ndarray             # (M, G)
    b: np.ndarray                 # (Q,)
    sd_gamma: float
    sd_b: float
    sd_delta: float
    n_group: np.ndarray           # training items per group
    em_rounds: int = 0

    @property
    def delta_pool(self) -> float:
        """A new model's prior mean on delta: the known models' mean (0 after centring)."""
        return float(self.delta.mean())

    def mu_of(self, group) -> float:
        """mu_g; for a group with no training row, the training-item-weighted mean of mu."""
        if group in self.groups:
            return float(self.mu[self.groups.index(group)])
        return float((self.mu * self.n_group).sum() / max(self.n_group.sum(), 1))

    def p(self) -> np.ndarray:
        """Fitted P(pass), models x items."""
        g = self.item_group
        return expit(self.mu[g][None, :] + self.delta[:, None] + self.gamma[:, g] - self.b[None, :])


def _onehot(idx: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(idx), n))
    out[np.arange(len(idx)), idx] = 1.0
    return out


def _fit_1pl(Y, mask, maxiter, guard):
    check_non_transductive(*guard)               # S4.6: every function that fits calls the guard
    M, Q = Y.shape
    y = np.nan_to_num(Y)

    def f(z):
        th, b = z[:M], z[M:]
        p = expit(th[:, None] - b[None, :])
        ll = mask * (y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS))
        r = mask * (y - p)
        obj = ll.sum() - (th ** 2).sum() / (2 * SD_THETA_1PL ** 2) - (b ** 2).sum() / (2 * SD_B_1PL ** 2)
        g = np.concatenate([r.sum(1) - th / SD_THETA_1PL ** 2, -r.sum(0) - b / SD_B_1PL ** 2])
        return -obj, -g

    res = minimize(f, np.zeros(M + Q), jac=True, method="L-BFGS-B", options={"maxiter": maxiter})
    return res.x[:M], res.x[M:]


def _fit_hier(Y, mask, gi, G, maxiter, guard):
    check_non_transductive(*guard)
    M, Q = Y.shape
    y = np.nan_to_num(Y)
    H = _onehot(gi, G)                                   # Q x G
    n_mu, n_d, n_g = G, M, M * G

    def unpack(z):
        return z[:n_mu], z[n_mu:n_mu + n_d], z[n_mu + n_d:n_mu + n_d + n_g].reshape(M, G), \
            z[n_mu + n_d + n_g:]

    def eta(mu, d, ga, b):
        return mu[gi][None, :] + d[:, None] + ga[:, gi] - b[None, :]

    var_g, var_b = SD_GAMMA0 ** 2, SD_B0 ** 2
    z = np.zeros(n_mu + n_d + n_g + Q)
    rounds = 0
    for rounds in range(1, EM_ROUNDS + 1):
        def f(z, var_g=var_g, var_b=var_b):
            mu, d, ga, b = unpack(z)
            p = expit(eta(mu, d, ga, b))
            ll = mask * (y * np.log(p + _EPS) + (1 - y) * np.log(1 - p + _EPS))
            r = mask * (y - p)
            obj = (ll.sum() - (mu ** 2).sum() / (2 * SD_FIXED ** 2) - (d ** 2).sum() / (2 * SD_FIXED ** 2)
                   - (ga ** 2).sum() / (2 * var_g) - (b ** 2).sum() / (2 * var_b))
            g = np.concatenate([r.sum(0) @ H - mu / SD_FIXED ** 2, r.sum(1) - d / SD_FIXED ** 2,
                                (r @ H - ga / var_g).ravel(), -r.sum(0) - b / var_b])
            return -obj, -g

        z = minimize(f, z, jac=True, method="L-BFGS-B", options={"maxiter": maxiter}).x
        mu, d, ga, b = unpack(z)
        pr = expit(eta(mu, d, ga, b))
        w = mask * pr * (1 - pr)
        new_g = max(float(np.mean(ga ** 2 + 1 / (w @ H + 1 / var_g))), VAR_FLOOR)
        new_b = max(float(np.mean(b ** 2 + 1 / (w.sum(0) + 1 / var_b))), VAR_FLOOR)
        done = abs(new_g - var_g) <= EM_TOL * var_g and abs(new_b - var_b) <= EM_TOL * var_b
        var_g, var_b = new_g, new_b
        if done:
            break
    mu, d, ga, b = unpack(z)
    return mu, d, ga, b, math.sqrt(var_g), math.sqrt(var_b), rounds


def fit(Y: np.ndarray, items, models, item_groups=None, *, scored_ids, problem_of,
        kind: str = "hier", maxiter: int = 500) -> AbilityParams:
    """Fit on Y (known models x training items; 1 pass, 0 fail, NaN unavailable). `item_groups`
    gives each item's group (ignored by the 1PL). The S4.6 guard runs first."""
    if kind not in KINDS:
        raise ValueError(kind)
    items, models = list(items), list(models)
    guard = (items, scored_ids, problem_of)
    check_non_transductive(*guard)
    Y = np.asarray(Y, dtype=float)
    if Y.shape != (len(models), len(items)):
        raise ValueError(f"Y is {Y.shape}, expected {(len(models), len(items))}")
    mask = ~np.isnan(Y)
    M, Q = Y.shape
    if kind == "1pl":
        th, b = _fit_1pl(Y, mask, maxiter, guard)
        return AbilityParams("1pl", models, items, ["all"], np.zeros(Q, dtype=int), np.zeros(1), th,
                             np.zeros((M, 1)), b, 0.0, SD_B_1PL,
                             float(np.std(th, ddof=1)) if M > 1 else SD_THETA_1PL,
                             np.array([float(Q)]))
    if item_groups is None:
        raise ValueError("the hierarchical model needs item_groups")
    groups = sorted(set(item_groups))
    gi = np.array([groups.index(g) for g in item_groups], dtype=int)
    mu, d, ga, b, sd_g, sd_b, rounds = _fit_hier(Y, mask, gi, len(groups), maxiter, guard)
    shift = float(d.mean())
    d, mu = d - shift, mu + shift
    sd_d = max(float(np.std(d, ddof=1)), math.sqrt(VAR_FLOOR)) if M > 1 else math.sqrt(VAR_FLOOR)
    return AbilityParams("hier", models, items, groups, gi, mu, d, ga, b, sd_g, sd_b, sd_d,
                         np.bincount(gi, minlength=len(groups)).astype(float), rounds)


@dataclass
class NewModelPosterior:
    """A new model's Laplace posterior over (delta, gamma_1..G) from PASS/FAIL evidence (S0.13).
    The 1PL has no gamma: the posterior is over delta (theta) only."""
    params: AbilityParams
    obs: list = field(default_factory=list)       # (group index or -1, b, y)
    mean: np.ndarray = None
    cov: np.ndarray = None

    def __post_init__(self):
        self._solve()

    @property
    def _G(self) -> int:
        return 0 if self.params.kind == "1pl" else len(self.params.groups)

    def _prior(self):
        G = self._G
        m0 = np.concatenate([[self.params.delta_pool], np.zeros(G)])
        prec = np.concatenate([[1 / max(self.params.sd_delta ** 2, VAR_FLOOR)],
                               np.full(G, 1 / max(self.params.sd_gamma ** 2, VAR_FLOOR))])
        return m0, prec

    def _design(self, gidx: int) -> np.ndarray:
        x = np.zeros(1 + self._G)
        x[0] = 1.0
        if self._G and gidx >= 0:
            x[1 + gidx] = 1.0
        return x

    def _solve(self):
        m0, prec = self._prior()
        x = m0.copy()
        if self.obs:
            X = np.stack([self._design(g) for g, _, _ in self.obs])
            off = np.array([self._mu(g) - b for g, b, _ in self.obs])
            y = np.array([v for _, _, v in self.obs])
            for _ in range(200):          # damped Newton on a concave objective
                p = expit(X @ x + off)
                grad = X.T @ (y - p) - prec * (x - m0)
                hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
                step = np.linalg.solve(hess, grad)
                step = step * min(1.0, 0.5 / max(float(np.abs(step).max()), 1e-300))
                x = x + step
                if float(np.abs(step).max()) < 1e-10:
                    break
            p = expit(X @ x + off)
            info = (X * (p * (1 - p))[:, None]).T @ X + np.diag(prec)
        else:
            info = np.diag(prec)
        self.mean, self.cov = x, np.linalg.inv(info)

    def _mu(self, gidx: int) -> float:
        if self.params.kind == "1pl":
            return 0.0
        return float(self.params.mu[gidx]) if gidx >= 0 else self.params.mu_of(None)

    def _gidx(self, group) -> int:
        return self.params.groups.index(group) if group in self.params.groups else -1

    def update(self, group, b: float, verdict: str) -> bool:
        """PASS / FAIL on a task of `group` with difficulty b update the posterior; UNKNOWN (or
        anything else) does not (S0.13). Returns whether it updated."""
        if verdict not in (PASS, FAIL):
            return False
        self.obs.append((self._gidx(group), float(b), 1.0 if verdict == PASS else 0.0))
        self._solve()
        return True

    def offset(self, group) -> tuple[float, float]:
        """(mean, SD) of the new model's offset delta + gamma_g for `group` (gamma = 0 with its
        prior variance for a group with no training row; the 1PL: theta)."""
        x = self._design(self._gidx(group))
        mean, var = float(x @ self.mean), float(x @ self.cov @ x)
        if self._G and self._gidx(group) < 0:
            var += self.params.sd_gamma ** 2
        return mean, math.sqrt(var)

    def p(self, group, b: float) -> float:
        """Plug-in P(pass) at the posterior mean."""
        return float(expit(self._mu(self._gidx(group)) + self.offset(group)[0] - b))


__all__ = ["AbilityParams", "NewModelPosterior", "fit", "KINDS"]
