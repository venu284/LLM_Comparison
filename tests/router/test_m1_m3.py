"""Tests for the M1-M3 router modules. Data-dependent tests skip when external/ is absent."""

import numpy as np
import pandas as pd
import pytest

from router.baselines import cascade, static
from router.core import irt
from router.core.anchors import onboard
from router.data import coderouterbench as crb
from router.data.schema import OutcomeMatrix
from router.sim.metrics import from_choice, summary
from router.sim.verifier import Verifier

HAVE_DATA = (crb.CRB_DIR / "id_test_results_long.csv").exists()


def toy() -> OutcomeMatrix:
    idx = pd.Index(["t1", "t2", "t3", "t4"])
    score = pd.DataFrame({"cheap": [1, 0, 1, 0], "strong": [1, 1, 0, 1]}, index=idx, dtype=float)
    cost = pd.DataFrame({"cheap": [0.1] * 4, "strong": [1.0] * 4}, index=idx)
    return OutcomeMatrix(score, cost, pd.Series("d", index=idx), pd.Series("s", index=idx))


def test_source_of():
    assert crb.source_of("dim1_mbpp_631") == "mbpp"
    assert crb.source_of("dim1_bcb_BigCodeBench/767") == "bcb"
    assert crb.source_of("dim2_lcb_abc388_d") == "lcb"
    assert crb.source_of("dim3_swebench_django__django-12856") == "swebench"


def test_metrics_on_toy():
    m = toy()
    s = summary(from_choice(m.score, m.cost, static.always(m, "strong")), m.score, m.cost)
    assert s["AvgPerf"] == pytest.approx(75.0)
    assert s["$Total"] == pytest.approx(4.0)
    assert s["Perf/$"] == pytest.approx(75.0 / 4.0)


def test_verifier_flip_rate():
    v = Verifier(eps=0.2, seed=1)
    flips = np.mean([not v.passes(1.0) for _ in range(20000)])
    assert flips == pytest.approx(0.2, abs=0.01)


def test_cascade_without_noise_escalates_only_failures():
    m = toy()
    out = cascade.run(m, ["cheap", "strong"], Verifier(0.0))
    assert out.score.tolist() == [1, 1, 1, 1]
    assert out.cost.tolist() == pytest.approx([0.1, 1.1, 0.1, 1.1])


def test_irt_recovers_difficulty():
    rng = np.random.default_rng(0)
    th = rng.normal(0, 1, 40)
    b = rng.normal(0, 1, 300)
    Y = (rng.random((40, 300)) < 1 / (1 + np.exp(-(th[:, None] - b[None, :])))).astype(float)
    p = irt.fit(Y, range(300), range(40), "1pl")
    assert np.corrcoef(p.b, b)[0, 1] > 0.9
    assert np.corrcoef(p.theta, th)[0, 1] > 0.9


def test_onboarding_sd_shrinks_and_is_deterministic():
    rng = np.random.default_rng(0)
    b = rng.normal(0, 1, 500)
    params = irt.IRTParams(list(range(500)), ["m"], b, np.ones(500), np.zeros(1), "1pl")
    y = (rng.random(500) < 1 / (1 + np.exp(-(0.5 - b)))).astype(float)
    sds = [onboard(params, k, lambda i: y[i], "random", seed=3)[1] for k in (5, 50)]
    assert sds[1] < sds[0]
    assert onboard(params, 10, lambda i: y[i], "fisher", seed=3)[2] == onboard(params, 10, lambda i: y[i], "fisher", seed=3)[2]


def test_fisher_beats_random_with_true_parameters():
    """Regression test: undamped Newton once made Fisher anchors diverge."""
    rng = np.random.default_rng(1)
    q = 2000
    b, a = rng.normal(0, 1.5, q), np.exp(rng.normal(0, 0.4, q))
    params = irt.IRTParams(list(range(q)), ["m"], b, a, np.zeros(1), "2pl")
    errs = {"fisher": [], "random": []}
    for r in range(30):
        th = rng.normal()
        y = (rng.random(q) < 1 / (1 + np.exp(-a * (th - b)))).astype(float)
        for method in errs:
            errs[method].append(abs(onboard(params, 25, lambda i: y[i], method, seed=r)[0] - th))
    assert np.mean(errs["fisher"]) < 0.8 * np.mean(errs["random"])


@pytest.mark.skipif(not HAVE_DATA, reason="CodeRouterBench not downloaded to external/")
def test_crb_counts_and_published_numbers():
    test = crb.load("id_test")
    assert test.score.shape == (2919, 8)
    assert not test.score.isna().any().any()
    opus = summary(from_choice(test.score, test.cost, static.always(test, "claude-opus-4-6")), test.score, test.cost)
    assert opus["AvgPerf"] == pytest.approx(43.83, abs=0.005)
    assert opus["Perf/$"] == pytest.approx(1.29, abs=0.005)
    paper = crb.load("id_test", view="paper")
    orc = summary(from_choice(paper.score, paper.cost, static.oracle(paper)), paper.score, paper.cost)
    assert orc["AvgPerf"] == pytest.approx(57.00, abs=0.005)
