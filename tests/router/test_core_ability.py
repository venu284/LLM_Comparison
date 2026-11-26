"""S4.3 ability model (docs/research/S4_spec.md §6.1): parameter recovery on simulated data, the 1PL
reduction against the frozen quick-check fit, the prior at k = 0 (the pool prior, never theta = 0),
posterior updates from PASS/FAIL with UNKNOWN ignored, and the S4.6 guard."""

import math

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from router.core import ability, irt
from router.core.ability import AbilityParams, NewModelPosterior
from router.core.guards import TransductiveFitError
from router.experiments import m3_quickcheck_irt as quick
from router.sim.stats import rng
from router.sim.verifier import FAIL, PASS, UNKNOWN


def _ids(n, prefix="q"):
    return [f"{prefix}{i:04d}" for i in range(n)]


def _problems(items, extra=()):
    return pd.Series({t: f"P_{t}" for t in list(items) + list(extra)})


def simulate(seed=0, M=30, G=4, per_group=300, sd_gamma=0.6, sd_b=1.2, missing=0.1):
    g = rng(seed, "s4_sim")
    mu = np.array([-1.0, -0.2, 0.5, 1.2])[:G]
    delta = g.normal(0, 1, M)
    gamma = g.normal(0, sd_gamma, (M, G))
    gi = np.repeat(np.arange(G), per_group)
    b = g.normal(0, sd_b, len(gi))
    p = expit(mu[gi][None, :] + delta[:, None] + gamma[:, gi] - b[None, :])
    Y = (g.random(p.shape) < p).astype(float)
    Y[g.random(p.shape) < missing] = np.nan
    items = _ids(len(gi))
    groups = [f"g{i}" for i in gi]
    return dict(Y=Y, items=items, models=[f"m{i}" for i in range(M)], groups=groups, mu=mu,
                delta=delta, gamma=gamma, b=b, p=p, sd_gamma=sd_gamma)


@pytest.fixture(scope="module")
def sim():
    s = simulate()
    s["fit"] = ability.fit(s["Y"], s["items"], s["models"], s["groups"], scored_ids=[],
                           problem_of=_problems(s["items"]))
    return s


# ----------------------------------------------------------------------------------- fitting
def test_parameter_recovery(sim):
    f = sim["fit"]
    assert f.kind == "hier" and f.groups == ["g0", "g1", "g2", "g3"]
    assert np.corrcoef(f.b, sim["b"])[0, 1] > 0.85
    assert np.corrcoef(f.delta, sim["delta"])[0, 1] > 0.95
    assert np.corrcoef(f.gamma.ravel(), sim["gamma"].ravel())[0, 1] > 0.7
    assert 0.6 * sim["sd_gamma"] < f.sd_gamma < 1.4 * sim["sd_gamma"]
    assert abs(f.delta.mean()) < 1e-9                       # centred on the known models
    assert f.sd_delta == pytest.approx(np.std(f.delta, ddof=1))
    assert np.abs(f.p() - sim["p"]).mean() < 0.08
    assert 1 <= f.em_rounds <= ability.EM_ROUNDS


def test_hierarchical_fit_beats_the_1pl_when_groups_matter(sim):
    one = ability.fit(sim["Y"], sim["items"], sim["models"], scored_ids=[],
                      problem_of=_problems(sim["items"]), kind="1pl")
    err = lambda f: np.abs(f.p() - sim["p"]).mean()          # noqa: E731
    assert err(sim["fit"]) < err(one)


def test_1pl_reduction_reproduces_the_frozen_quick_check_fit():
    g = rng(1, "s4_sim")
    M, Q = 10, 200
    th, b = g.normal(0, 1, M), g.normal(0, 1.5, Q)
    Y = (g.random((M, Q)) < expit(th[:, None] - b[None, :])).astype(float)
    Y[g.random(Y.shape) < 0.05] = np.nan
    items = _ids(Q)
    old = quick.fit(Y, items, range(M), "1pl")
    new = ability.fit(Y, items, range(M), scored_ids=[], problem_of=_problems(items), kind="1pl")
    via_irt = irt.fit(Y, items, range(M), scored_ids=[], problem_of=_problems(items))
    np.testing.assert_allclose(new.delta, old.theta, atol=1e-4)
    np.testing.assert_allclose(new.b, old.b, atol=1e-4)
    np.testing.assert_allclose(via_irt.b, new.b, atol=0)
    assert new.mu.tolist() == [0.0] and not new.gamma.any()


# ------------------------------------------------------------------------------- new model
def test_the_prior_at_k0_is_the_pool_prior_not_theta_zero(sim):
    f = sim["fit"]
    post = NewModelPosterior(f)
    for gname in f.groups:
        mean, sd = post.offset(gname)
        assert mean == pytest.approx(0.0, abs=1e-9)
        assert sd == pytest.approx(math.sqrt(f.sd_delta ** 2 + f.sd_gamma ** 2))
        assert post.p(gname, 0.3) == pytest.approx(float(expit(f.mu_of(gname) - 0.3)))
    far = max(f.groups, key=lambda x: abs(f.mu_of(x)))
    assert abs(f.mu_of(far)) > 0.5
    assert abs(post.p(far, 0.3) - float(expit(-0.3))) > 0.1   # theta = 0 would ignore mu_g
    # a group with no training row: the weighted mean of mu, gamma at its prior
    mean, sd = post.offset("unseen")
    assert mean == pytest.approx(0.0, abs=1e-9) and sd == pytest.approx(
        math.sqrt(f.sd_delta ** 2 + f.sd_gamma ** 2))
    assert f.mu_of("unseen") == pytest.approx(float((f.mu * f.n_group).sum() / f.n_group.sum()))


def test_posterior_updates_pass_fail_and_unknown_ignored(sim):
    f = sim["fit"]
    post = NewModelPosterior(f)
    m0, s0 = post.offset("g0")
    o0 = post.offset("g1")[0]
    assert not post.update("g0", 0.0, UNKNOWN) and post.offset("g0") == (m0, s0)
    assert post.update("g0", 0.0, PASS)
    m1, s1 = post.offset("g0")
    assert m1 > m0 and s1 < s0
    assert 0 < post.offset("g1")[0] - o0 < m1 - m0      # through delta, less than in g0 itself
    for _ in range(6):
        post.update("g0", 0.0, FAIL)
    assert post.offset("g0")[0] < m0
    sds = [post.offset("g0")[1]]
    for _ in range(20):
        post.update("g0", float(f.mu_of("g0")), PASS)
        post.update("g0", float(f.mu_of("g0")), FAIL)
        sds.append(post.offset("g0")[1])
    assert all(a > b for a, b in zip(sds, sds[1:]))
    assert len(post.obs) == 1 + 6 + 40                   # UNKNOWN never recorded


def test_1pl_posterior_matches_the_frozen_quick_check_posterior():
    """With the pool prior at 0 and the quick-check's prior SD, the 1PL posterior is the quick-check
    `ability_posterior` (MAP and Laplace SD)."""
    g = rng(2, "s4_sim")
    Q = 60
    b = g.normal(0, 1, Q)
    y = (g.random(Q) < expit(0.7 - b)).astype(float)
    p = AbilityParams("1pl", ["a", "b"], _ids(Q), ["all"], np.zeros(Q, dtype=int), np.zeros(1),
                      np.array([-0.8, 0.8]), np.zeros((2, 1)), b, 0.0, 2.0, 1.0, np.array([Q]))
    post = NewModelPosterior(p)
    for bq, yq in zip(b, y):
        post.update("all", bq, PASS if yq else FAIL)
    qc = quick.IRTParams(_ids(Q), ["a", "b"], b, np.ones(Q), np.array([-0.8, 0.8]), "1pl")
    th, sd = quick.ability_posterior(qc, np.arange(Q), y, prior_sd=1.0)
    mean, sd_new = post.offset("all")
    assert mean == pytest.approx(th, abs=1e-6) and sd_new == pytest.approx(sd, abs=1e-6)


# -------------------------------------------------------------------------------------- S4.6
def test_the_fit_refuses_scored_tasks_and_shared_problems():
    s = simulate(M=6, per_group=20)
    items = s["items"]
    with pytest.raises(TransductiveFitError, match="scored task"):
        ability.fit(s["Y"], items, s["models"], s["groups"], scored_ids=[items[0]],
                    problem_of=_problems(items))
    prob = _problems(items, extra=["scored"])
    prob["scored"] = prob[items[3]]
    with pytest.raises(TransductiveFitError, match="problem"):
        ability.fit(s["Y"], items, s["models"], s["groups"], scored_ids=["scored"], problem_of=prob)
    with pytest.raises(TransductiveFitError):
        irt.fit(s["Y"], items, s["models"], scored_ids=[items[0]], problem_of=_problems(items))
