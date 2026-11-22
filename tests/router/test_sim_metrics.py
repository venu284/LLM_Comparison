"""S2.3: claim-facing metrics (S0.1, S0.2, R3) and their separation from the paper metric (O10)."""

import numpy as np
import pandas as pd
import pytest

from router.data.schema import OutcomeMatrix
from router.settings import VERIFIER_RUN_USD
from router.sim import metrics, paper, toy
from router.sim.stream import GIVE_UP, Ladder, ReplayView, SingleChoice, replay
from router.sim.verifier import ReplayVerifier


def tiny(name="tiny/complete"):
    """Three tasks, two models, hand-checkable."""
    idx = pd.Index(["a", "b", "c"])
    score = pd.DataFrame({"cheap": [1.0, 0.8, 0.0], "strong": [1.0, 0.8, 1.0]}, index=idx)
    cost = pd.DataFrame({"cheap": [1.0, 1.0, 1.0], "strong": [10.0, 10.0, 10.0]}, index=idx)
    fam = pd.Series(["mbpp", "cruxeval", "mbpp"], index=idx)
    return OutcomeMatrix(score, cost, pd.Series("d", index=idx), fam, name)


def tiny_verdicts(m):
    return pd.DataFrame([{"task_id": t, "model": mdl, "observation": "PASS", "signal": True,
                          "unavailable": False} for t in m.tasks for mdl in m.models])


def run(m, router, **kw):
    res = replay(router, ReplayView.from_matrix(m), ReplayVerifier(tiny_verdicts(m)), seed=0)
    return res, metrics.task_outcomes(res, m, **kw)


def test_acceptance_is_s01_success_per_family():
    m = tiny()
    _, o = run(m, SingleChoice(lambda t, a: "strong"))
    assert o.accepted.to_dict() == {"a": True, "b": False, "c": True}
    assert metrics.acceptance(o) == pytest.approx(200 / 3)
    _, lenient = run(m, SingleChoice(lambda t, a: "strong"), cruxeval_lenient=True)
    assert lenient.accepted.to_dict() == {"a": True, "b": True, "c": True}
    fam = metrics.acceptance_by_family(o)
    assert fam.loc["cruxeval", "acceptance"] == 0 and fam.loc["mbpp", "acceptance"] == 100


def test_mbpp_partial_credit_never_counts_even_when_lenient():
    m = tiny()
    m.family.loc["b"] = "mbpp"
    _, o = run(m, SingleChoice(lambda t, a: "strong"), cruxeval_lenient=True)
    assert not o.accepted["b"]


def test_metric_separation_on_partial_credit():
    """The paper AvgPerf counts 0.8 as 80 points; acceptance counts it as a failure."""
    m = tiny()
    _, o = run(m, SingleChoice(lambda t, a: "strong"))
    pap = paper.paper_avgperf(paper.from_choice(m.score, m.cost, pd.Series("strong", index=m.tasks)))
    assert pap == pytest.approx(280 / 3)
    assert metrics.acceptance(o) == pytest.approx(200 / 3)
    assert not hasattr(metrics, "paper_avgperf") and not hasattr(metrics, "paper_summary")


def test_claim_metrics_refuse_the_paper_view():
    m = tiny("crb/id_test/paper")
    res = replay(SingleChoice(lambda t, a: "strong"), ReplayView.from_matrix(m),
                 ReplayVerifier(tiny_verdicts(m)), seed=0)
    with pytest.raises(ValueError, match="paper view"):
        metrics.task_outcomes(res, m)
    with pytest.raises(ValueError, match="paper view"):
        metrics.oracle_outcomes(m, m.tasks)


def test_cost_parts_give_up_and_probe_spread():
    m = tiny()
    res, o = run(m, Ladder(["cheap", "strong"]), probe_usd=3.0, n_probes=6)
    # every cell has a PASS verdict: the ladder stops at cheap
    assert (o.model_cost_usd == 1.0).all()
    assert np.allclose(o.verifier_usd, VERIFIER_RUN_USD)
    assert np.allclose(o.probe_usd, 1.0)
    b = metrics.cost_breakdown(o)
    assert b["n_probes"] == 6 and b["probe_usd"] == pytest.approx(3.0) and b["n_retries"] == 0
    assert b["total_usd"] == pytest.approx(3 * 1.0 + 3 * VERIFIER_RUN_USD + 3.0)
    _, g = run(m, SingleChoice(lambda t, a: GIVE_UP))
    assert not g.accepted.any() and (g.cost_usd == 0).all()


def test_price_sweep_scales_one_model():
    m = tiny()
    _, o = run(m, Ladder(["cheap", "strong"]))
    _, o2 = run(m, Ladder(["cheap", "strong"]), price_scale={"cheap": 2.0})
    assert np.allclose(o2.model_cost_usd, 2 * o.model_cost_usd)
    _, o3 = run(m, Ladder(["cheap", "strong"]), price_scale={"strong": 2.0})
    assert np.allclose(o3.model_cost_usd, o.model_cost_usd)
    with pytest.raises(ValueError):
        run(m, Ladder(["cheap"]), price_scale={"nope": 2.0})


def test_cpat_needs_a_floor():
    m = tiny()
    _, o = run(m, SingleChoice(lambda t, a: "strong"))
    with pytest.raises(TypeError):
        metrics.cpat_at_floor(o)
    hit = metrics.cpat_at_floor(o, 60.0)
    assert hit["meets_floor"] and hit["cpat"] == pytest.approx(o.cost_usd.sum() / 2)
    miss = metrics.cpat_at_floor(o, 70.0)
    assert not miss["meets_floor"] and np.isnan(miss["cpat"])


def test_cost_at_matched_acceptance():
    curve = pd.DataFrame({"acceptance": [50, 60, 70, 60], "cost": [1.0, 3.0, 5.0, 2.0]})
    assert metrics.cost_at_matched_acceptance(curve, 60) == pytest.approx(2.0)
    assert metrics.cost_at_matched_acceptance(curve, 65) == pytest.approx(3.5)
    assert np.isnan(metrics.cost_at_matched_acceptance(curve, 75))
    assert np.isnan(metrics.cost_at_matched_acceptance(curve, 40))


def test_regret_against_the_cost_aware_oracle():
    m = tiny()
    _, o = run(m, SingleChoice(lambda t, a: "strong"), probe_usd=1.5, n_probes=3)
    r = metrics.regret(o, m)
    # oracle: a -> cheap (1), b -> none (gives up, 0), c -> strong (10)
    assert r["oracle_cost_usd"] == pytest.approx(11.0)
    assert r["oracle_acceptance"] == pytest.approx(200 / 3)
    assert r["acceptance_regret_points"] == pytest.approx(0.0)
    assert r["cost_regret_usd"] == pytest.approx(30 + 3 * VERIFIER_RUN_USD + 1.5 - 11.0)
    assert r["n_probes"] == 3 and r["definition"] == metrics.OUR_DEFINITION


def test_unavailable_cells_are_not_failures_and_are_counted():
    m = toy.matrix()
    rv = ReplayVerifier(toy.verdict_table(m))
    res = replay(Ladder(["cheap", "mid", "strong"]), ReplayView.from_matrix(m), rv, seed=0)
    o = metrics.task_outcomes(res, m)
    cc, info = metrics.complete_case(o)
    assert "t1" not in cc.index and info["n_dropped"] == 1
    assert info["dropped_by_family"] == {"mbpp": 1}
    assert len(cc) + info["n_dropped"] == len(o)
    cp, info2 = metrics.cascade_population(o)
    assert "t3" not in cp.index and info2["n_dropped"] == 1


def test_probe_cost_is_spread_over_the_tasks_scored():
    m = toy.matrix()
    rv = ReplayVerifier(toy.verdict_table(m))
    res = replay(Ladder(["cheap", "strong"]), ReplayView.from_matrix(m), rv, seed=0)
    o = metrics.task_outcomes(res, m, probe_usd=2.3, n_probes=4)
    cc, _ = metrics.complete_case(o)
    assert cc.probe_usd.sum() == pytest.approx(2.3)
    assert np.allclose(cc.cost_usd, cc.model_cost_usd + cc.verifier_usd + 2.3 / len(cc))


def test_paired_frame_needs_the_same_tasks():
    m = tiny()
    _, a = run(m, SingleChoice(lambda t, a_: "strong"))
    _, b = run(m, SingleChoice(lambda t, a_: "cheap"))
    f = metrics.paired_frame(a, b, ("r", "base"))
    assert list(f.columns[:4]) == ["accepted_r", "accepted_base", "cost_r", "cost_base"]
    with pytest.raises(ValueError):
        metrics.paired_frame(a, b.iloc[:2])
