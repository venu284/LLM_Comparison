"""S3.3 fitted part (spec D8): the ladder traces match `policy_outcomes`, decision points and
targets are as defined, the perfect-information policy keeps the full ladder's acceptance, and the
threshold rule picks the largest t within the loss budget. Toy data, values checked by hand."""

import numpy as np
import pandas as pd
import pytest

from router.baselines.common import ESCALATE, Policy, policy_outcomes
from router.data.verdicts import X13_VERDICTS
from router.experiments import s3_classifier as C
from router.sim import toy
from router.sim.verifier import FAIL, PASS, UNKNOWN, ReplayVerifier


def _toy():
    m = toy.matrix(n=30)
    m = m.subset([t for t in m.tasks if t != "t3"])
    return m, ReplayVerifier(toy.verdict_table(m, no_record=()))


@pytest.mark.parametrize("rungs", [("cheap", "mid"), ("cheap", "strong"), ("cheap", "mid", "strong")])
def test_full_ladder_trace_matches_policy_outcomes(rungs):
    m, v = _toy()
    pol = Policy(rungs, True, ESCALATE)
    trs = C.traces(m, v, pol)
    A, Cst = policy_outcomes(m, v, [pol])
    for i, t in enumerate(m.tasks):
        a, c = C.full_outcome(trs[t])
        assert a == bool(A[i, 0]) and c == pytest.approx(Cst[i, 0]), t


def _trace(verdicts, ok, cost=(1.0, 2.0, 4.0)):
    n = len(verdicts)
    stop = next((i for i, x in enumerate(verdicts) if x == PASS), n - 1)
    return C.Trace(list("abc"[:stop + 1]), list(cost[:stop + 1]), list(verdicts[:stop + 1]),
                   list(ok[:stop + 1]), stop, list(ok))


def test_decision_points_and_targets_by_hand():
    trs = {"x": _trace([FAIL, UNKNOWN, FAIL], [False, False, True]),     # 2 decision points
           "y": _trace([PASS, FAIL, FAIL], [True, False, False]),         # none: rung 0 passes
           "z": _trace([UNKNOWN, PASS, FAIL], [False, True, True])}       # 1 point; c succeeds unseen
    r = C.decision_rows(trs, "post_failure")
    got = {(a, b): (h, y) for a, b, h, y in r.itertuples(index=False)}
    assert got == {("x", 0): ("r0:FAIL", 1), ("x", 1): ("r0:FAIL r1:UNKNOWN", 1),
                   ("z", 0): ("r0:UNKNOWN", 1)}
    pre = C.decision_rows(trs, "pre_attempt").set_index("task_id").y.to_dict()
    assert pre == {"x": 1, "y": 1, "z": 1}


def test_simulate_give_up_costs_by_hand():
    trs = {"x": _trace([FAIL, UNKNOWN, FAIL], [False, False, True])}
    assert C.simulate(trs, {}) == ({"x": True}, {"x": 7.0})
    assert C.simulate(trs, {("x", 0): True}) == ({"x": False}, {"x": 1.0})
    assert C.simulate(trs, {("x", 1): True}) == ({"x": False}, {"x": 3.0})
    assert C.simulate(trs, {("x", -1): True}) == ({"x": False}, {"x": 0.0})


def test_perfect_information_keeps_full_acceptance():
    m, v = _toy()
    pol = Policy(("cheap", "mid", "strong"), True, ESCALATE)
    trs = C.traces(m, v, pol)
    for kind in C.KINDS:
        rows = C.decision_rows(trs, kind)
        acc, cost = C.simulate(trs, {(t, j): y == 0 for t, j, y in zip(rows.task_id, rows.j, rows.y)})
        full = {t: C.full_outcome(tr) for t, tr in trs.items()}
        assert all(acc[t] == full[t][0] for t in trs), kind
        assert sum(cost.values()) <= sum(c for _, c in full.values()) + 1e-12


def test_choose_t_largest_within_budget():
    # four tasks; giving up on task a loses 25 points, on b loses nothing (y = 0)
    trs = {"a": _trace([FAIL, PASS], [False, True]), "b": _trace([FAIL, FAIL], [False, False]),
           "c": _trace([PASS, FAIL], [True, False]), "d": _trace([PASS, FAIL], [True, False])}
    rows = C.decision_rows(trs, "post_failure")
    p = np.array([0.9 if t == "a" else 0.1 for t in rows.task_id])
    assert C.choose_t(trs, rows, p) == pytest.approx(0.9)          # t = 0.9 stops b only
    assert C.choose_t(trs, rows, p, max_loss=30.0) > 0.9           # budget allows stopping a too


def test_d8_features_have_no_rung_position():
    """The primary classifier uses exactly D8's features (cell, verdict history, text); the rung
    position appears only in the labelled post-hoc variant (spec §7, 2026-11-24)."""
    rows = pd.DataFrame({"task_id": ["x", "x"], "j": [0, 1],
                         "history": ["r0:FAIL", "r0:FAIL r1:UNKNOWN"], "y": [1, 0]})
    cells = pd.Series({"x": "dim|fam"})
    d8 = C.GiveUpClassifier._meta(rows, cells)
    assert d8 == [{"cell=dim|fam": 1.0, "r0:FAIL": 1.0},
                  {"cell=dim|fam": 1.0, "r0:FAIL": 1.0, "r1:UNKNOWN": 1.0}]
    var = C.GiveUpClassifier._meta(rows, cells, rung_position=True)
    assert [sorted(k for k in d if k.startswith("pos=")) for d in var] == [["pos=0"], ["pos=1"]]
    assert C.FEATURES == {C.D8_FEATURES: False, C.RUNG_POSITION: True}
    texts = pd.Series({"x": "def f(): return 1"})
    assert C.GiveUpClassifier(rows, texts, cells).rung_position is False


@pytest.mark.skipif(not X13_VERDICTS.exists(), reason="external data not fetched")
def test_three_view_populations():
    """S3.3 fitted (X13, complete case) and S3.4 (complete case) on all three views' development
    splits. S3.3's sensitivity population equals DP1b's sensitivity column (279 tasks, 244
    problems); reviewer check of checkpoint 2, item 4."""
    from router.experiments import s3_kmatched as K
    pops, _, problems = C.populations()
    got = {k: (len(m), int(problems.reindex(m.tasks).nunique())) for k, m in pops.items()}
    assert got == {"filtered = headline": (195, 180), "all_eligible": (459, 355),
                   "sensitivity": (279, 244)}
    pk, problems = K.populations()
    got = {k: (len(m), int(problems.reindex(m.tasks).nunique())) for k, m in pk.items()}
    assert got == {"filtered = headline": (195, 180), "all_eligible": (966, 567),
                   "sensitivity": (294, 259)}
