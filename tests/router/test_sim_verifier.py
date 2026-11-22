"""S2.2: the replay verifier (S0.11), its sensitivity modes, and where verifier cost is charged."""

import numpy as np
import pandas as pd
import pytest

from router.data import verdicts
from router.settings import VERIFIER_RUN_USD
from router.sim import metrics, toy
from router.sim.stream import GIVE_UP, Ladder, ReplayView, replay
from router.sim.verifier import (FAIL, NO_RECORD, NO_SIGNAL, PASS, SIGNAL, UNKNOWN,
                                 HeuristicVerifier, NoiseVerifier, ReplayVerifier, Unavailable)

HAVE_X13 = verdicts.X13_VERDICTS.exists()


@pytest.fixture
def m():
    return toy.matrix()


@pytest.fixture
def v(m):
    return toy.verdict_table(m)


def test_signal_cells_give_the_recorded_verdict_and_charge_a_run(m, v):
    rv = ReplayVerifier(v)
    sig = v[v.signal]
    assert len(sig) > 0
    for r in sig.itertuples():
        c = rv.check(r.task_id, r.model)
        assert (c.verdict, c.ran, c.state) == (r.observation, True, SIGNAL)
        assert c.verdict in (PASS, FAIL)


def test_no_signal_cells_give_unknown_and_no_run(m, v):
    rv = ReplayVerifier(v)
    ns = v[~v.signal & ~v.unavailable]
    assert len(ns) > 0
    for r in ns.itertuples():
        c = rv.check(r.task_id, r.model)
        assert (c.verdict, c.ran, c.state) == (UNKNOWN, False, NO_SIGNAL)


def test_empty_cells_are_unavailable(m, v):
    rv = ReplayVerifier(v)
    assert rv.empty_cells() == {("t1", "mid")}
    with pytest.raises(Unavailable):
        rv.check("t1", "mid")


def test_task_without_record_is_unknown_with_its_own_state(m, v):
    rv = ReplayVerifier(v)
    assert not rv.has_record("t3")
    c = rv.check("t3", "strong")
    assert (c.verdict, c.ran, c.state) == (UNKNOWN, False, NO_RECORD)


def test_heuristic_is_a_labelled_sensitivity_mode_and_charges_nothing(m, v):
    h = toy.heuristic_table(v)
    hv = HeuristicVerifier(v, h)
    assert hv.sensitivity and not ReplayVerifier.sensitivity
    acc = h.set_index(["task_id", "model"]).heuristic_accept
    for r in v[~v.signal & ~v.unavailable].itertuples():
        c = hv.check(r.task_id, r.model)
        assert c.verdict == (PASS if acc[(r.task_id, r.model)] else FAIL)
        assert not c.ran
    for r in v[v.signal].itertuples():
        assert hv.check(r.task_id, r.model) == ReplayVerifier(v).check(r.task_id, r.model)


def test_noise_rates_seed_and_call_order():
    idx = pd.Index([f"t{i}" for i in range(4000)])
    truth = pd.DataFrame({"a": np.arange(4000) % 2 == 0, "b": True}, index=idx)
    nv = NoiseVerifier(truth, false_pass=0.15, false_reject=0.27, seed=3)
    assert nv.sensitivity
    pa = np.array([nv.check(t, "a").verdict == PASS for t in idx])
    assert pa[1::2].mean() == pytest.approx(0.15, abs=0.02)        # true failures passed
    assert 1 - pa[0::2].mean() == pytest.approx(0.27, abs=0.025)   # true successes rejected
    again = NoiseVerifier(truth.iloc[::-1], false_pass=0.15, false_reject=0.27, seed=3)
    assert [again.check(t, "a").verdict for t in idx[:50]] == [nv.check(t, "a").verdict for t in idx[:50]]
    other = NoiseVerifier(truth, false_pass=0.15, false_reject=0.27, seed=4)
    assert any(other.check(t, "a").verdict != nv.check(t, "a").verdict for t in idx[:200])


def test_noise_verifier_keeps_no_signal_cells_unknown(m):
    ns = pd.DataFrame(False, index=m.tasks, columns=m.models)
    ns.loc["t2", "cheap"] = True
    nv = NoiseVerifier.from_matrix(m, false_pass=0.0, false_reject=0.0, seed=0, no_signal=ns)
    assert nv.check("t2", "cheap").verdict == UNKNOWN and not nv.check("t2", "cheap").ran
    # zero noise reproduces the S0.1 truth (score 1.0), not the 0.8 partial credit
    for t in m.tasks:
        assert (nv.check(t, "strong").verdict == PASS) == (m.score.at[t, "strong"] == 1.0)


def test_verifier_cost_is_charged_only_where_a_verifier_ran(m, v):
    rv = ReplayVerifier(v)
    res = replay(Ladder(["cheap", "mid", "strong"]), ReplayView.from_matrix(m), rv, seed=0)
    o = metrics.task_outcomes(res, m)
    ran = res.calls.groupby("task_id").verifier_ran.sum()
    assert (o.n_verifier_runs == ran.reindex(o.index).fillna(0)).all()
    assert np.allclose(o.verifier_usd, o.n_verifier_runs * VERIFIER_RUN_USD)
    no_signal_tasks = [t for t in m.tasks if not v[(v.task_id == t)].signal.any()]
    assert len(no_signal_tasks) > 0
    assert (o.loc[no_signal_tasks, "verifier_usd"] == 0).all()
    assert res.calls[res.calls.state != SIGNAL].verifier_ran.sum() == 0
    o10 = metrics.task_outcomes(res, m, verifier_scale=10)
    assert np.allclose(o10.verifier_usd, 10 * o.verifier_usd)


def test_x13_unavailable_cells_must_be_unavailable_in_the_matrix(m, v):
    bad = toy.with_scores(m, m.score)
    bad.available.loc["t1", "mid"] = True
    bad.unavailable_reason.loc["t1", "mid"] = None
    with pytest.raises(ValueError, match="empty X13 cell"):
        replay(Ladder(["cheap"]), ReplayView.from_matrix(bad), ReplayVerifier(v), seed=0)


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
def test_real_x13_counts_and_release_availability():
    from router.data.audit import load_release
    rv = ReplayVerifier.from_x13()
    assert len(rv.empty_cells()) == 1551
    m = load_release()
    for t, mdl in rv.empty_cells():
        assert not m.available.at[t, mdl]
