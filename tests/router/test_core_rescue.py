"""S4.5 (docs/research/S4_spec.md §6.3): the rescue classifier (walks and decision points, Platt
calibration on simulated data, the threshold rule, an end-to-end fit on simulated walks) and the
policy rules in `policy.py` (break-even arithmetic, cheap-first ladder, budget cap)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

from router.baselines.common import ACCEPT, ESCALATE
from router.core import rescue as R
from router.core.policy import (break_even_ratio, cheap_first_ladder, cheap_first_pays,
                                stop_probability, within_budget)
from router.sim.stats import rng
from router.sim.verifier import FAIL, PASS, UNKNOWN


# ---------------------------------------------------------------------------- walks, decisions
def test_walk_and_decision_points():
    v = {"a": FAIL, "b": UNKNOWN, "c": PASS}
    cost = {"a": 1.0, "b": 2.0, "c": 5.0}
    ok = {"a": False, "b": False, "c": True}
    w = R.Walk.walk(["a", "b", "c"], v, cost, ok)
    assert w.stop_at == 2 and w.full() == (True, 8.0) and w.stopped_at(0) == (False, 1.0)
    rows = R.decision_rows({"t": w})
    assert rows.history.tolist() == ["a:FAIL", "a:FAIL b:UNKNOWN"] and rows.y.tolist() == [1, 1]
    wa = R.Walk.walk(["a", "b", "c"], v, cost, ok, on_unknown_accept=True)   # UNKNOWN ends it
    assert wa.stop_at == 1 and wa.full() == (False, 3.0)
    assert R.decision_rows({"t": wa}).history.tolist() == ["a:FAIL"]


def _hand_walks():
    walks = {}
    for t, ok2 in (("A", True), ("B", True), ("C", False), ("D", False)):
        walks[t] = R.Walk.walk(["s", "l"], {"s": FAIL, "l": PASS if ok2 else FAIL},
                               {"s": 1.0, "l": 10.0}, {"s": False, "l": ok2})
    rows = R.decision_rows(walks)
    p = rows.task_id.map({"A": 0.9, "B": 0.4, "C": 0.3, "D": 0.1}).to_numpy()
    return walks, rows, p


def test_the_threshold_rule_on_a_hand_example():
    walks, rows, p = _hand_walks()
    assert R.outcomes(walks, {}) == (50.0, 44.0)
    # t = 0.4 gives up on C and D (no later rung passes): no loss; t = 0.9 also gives up on B
    assert R.choose_t(walks, rows, p, max_loss=2.0) == 0.4
    assert R.outcomes(walks, {(t, 0): q < 0.4 for t, q in zip(rows.task_id, p)}) == (50.0, 24.0)
    assert R.choose_t(walks, rows, p, max_loss=30.0) == 0.9     # loses B: 25 points
    assert R.choose_t(walks, rows, p, max_loss=60.0) > 1.0      # gives up everywhere: 50 points


# --------------------------------------------------------------------------------- calibration
def _ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return sum(abs(p[idx == k].mean() - y[idx == k].mean()) * (idx == k).mean()
               for k in range(bins) if (idx == k).any())


def test_platt_calibrates_an_overconfident_score():
    g = rng(0, "s4_sim")
    p = g.uniform(0.05, 0.95, 20_000)
    y = (g.random(p.size) < p).astype(int)
    raw = expit(3.0 * logit(p))                     # over-confident
    half = p.size // 2
    pl = R.Platt().fit(raw[:half], y[:half])
    assert pl.a == pytest.approx(1 / 3, rel=0.1)
    cal = pl(raw[half:])
    assert _ece(cal, y[half:]) < 0.5 * _ece(raw[half:], y[half:])
    assert abs(cal.mean() - y[half:].mean()) < 0.02
    const = R.Platt().fit(raw[:10], np.ones(10, dtype=int))     # one class: its rate
    assert np.allclose(const(raw[:3]), 1 - R.P_CLIP)


def test_fit_rescue_on_simulated_walks():
    """The text says whether the big rung can rescue the task; the history and group carry no
    signal. Out-of-fold, calibrated, with t inside the loss budget."""
    g = rng(1, "s4_sim")
    walks, texts, groups, fold = {}, {}, {}, {}
    for i in range(600):
        t = f"t{i:03d}"
        rescuable = bool(g.random() < 0.5)
        words = ("solvable known pattern" if rescuable else "intractable unknown spec")
        texts[t] = f"{words} item {i % 37} variant {i % 11}"
        ok_l = bool(g.random() < (0.85 if rescuable else 0.05))
        walks[t] = R.Walk.walk(["s", "l"], {"s": FAIL, "l": PASS if ok_l else FAIL},
                               {"s": 1.0, "l": 10.0}, {"s": False, "l": ok_l})
        groups[t] = "g0"
        fold[t] = i % 4
    texts, groups, fold = pd.Series(texts), pd.Series(groups), pd.Series(fold)
    train = {t: w for t, w in walks.items() if int(t[1:]) < 480}
    test = {t: w for t, w in walks.items() if int(t[1:]) >= 480}
    res = R.fit_rescue(train, fold, texts, groups)
    rows = R.decision_rows(train)
    assert res.oof.notna().all() and len(res.oof) == len(rows)
    full, _ = R.outcomes(train, {})
    acc, _ = R.outcomes(train, {k: q < res.t for k, q in zip(zip(rows.task_id, rows.j),
                                                              res.platt(res.oof.to_numpy()))})
    assert full - acc <= R.MAX_LOSS + 1e-9 and res.t > 0
    te = R.decision_rows(test)
    p = res.predict(te, texts, groups)
    # knowing the flag exactly gives AUC 0.90 (P(y | flag) = 0.85 / 0.05); 120 test rows
    assert roc_auc_score(te.y, p) > 0.8
    assert abs(p.mean() - te.y.mean()) < 0.1


# ------------------------------------------------------------------------------- policy rules
def test_break_even_arithmetic_matches_the_expected_cost_comparison():
    for c in (0.1, 1.0, 3.0):
        for e in (1.0, 5.0, 40.0):
            if c >= e:
                continue
            assert break_even_ratio(c, e) == pytest.approx(c / e)
            for q in np.linspace(0, 1, 101):
                assert cheap_first_pays(q, c, e) == (c + (1 - q) * e < e)
    assert not cheap_first_pays(0.25, 1.0, 4.0)        # the tie is not cheaper
    assert cheap_first_pays(0.25 + 1e-9, 1.0, 4.0)


def test_stop_probability_is_not_the_success_rate():
    assert stop_probability(0.3, 0.6, ESCALATE) == 0.3
    assert stop_probability(0.3, 0.6, ACCEPT) == pytest.approx(0.9)
    q = stop_probability(0.0, 1.0, ESCALATE)            # a no-signal cell under escalate
    assert q == 0.0 and not cheap_first_pays(q, 0.01, 10.0)
    with pytest.raises(ValueError):
        stop_probability(0.7, 0.4, ESCALATE)


def test_cheap_first_ladder_on_hand_examples():
    cands = [("small", 1.0, 0.3), ("mid", 4.0, 0.5)]
    assert cheap_first_ladder(("big", 10.0), cands) == ("small", "mid", "big")
    assert cheap_first_ladder(("big", 10.0), [("small", 1.0, 0.2), ("mid", 4.0, 0.5)]) == ("mid", "big")
    # mid fails its ratio (0.3 < 0.4), so small is compared with big: 0.2 > 1/10
    assert cheap_first_ladder(("big", 10.0), [("small", 1.0, 0.2), ("mid", 4.0, 0.3)]) == ("small", "big")
    assert cheap_first_ladder(("big", 10.0), [("small", 1.0, 0.05), ("mid", 4.0, 0.3)]) == ("big",)
    # a candidate no cheaper than the current first rung is never put in front
    assert cheap_first_ladder(("big", 10.0), [("pricey", 12.0, 0.99)]) == ("big",)


def test_budget_cap():
    assert within_budget(100.0, 50.0, None)
    assert within_budget(3.0, 2.0, 5.0) and not within_budget(3.0, 2.5, 5.0)
    with pytest.raises(ValueError):
        within_budget(0.0, 1.0, -1.0)
