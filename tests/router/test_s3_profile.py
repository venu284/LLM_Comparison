"""S3 checkpoint 1 definitions (`router/sim/profile.py`; spec §5): toy cases checked by hand."""

import math

import numpy as np
import pandas as pd
import pytest

from router.sim import profile as P

T, F = True, False


# ------------------------------------------------------------------------------------- S3.6
def _verifier_toy():
    idx, cols = ["t1", "t2", "t3"], ["a", "b"]
    ok = pd.DataFrame([[T, F], [F, F], [T, F]], index=idx, columns=cols)
    verdict = pd.DataFrame([["PASS", "PASS"], ["FAIL", "UNKNOWN"], ["FAIL", None]],
                           index=idx, columns=cols, dtype=object)
    avail = pd.DataFrame(True, index=idx, columns=cols)
    avail.loc["t3", "b"] = False
    return ok, verdict, avail


def test_verifier_cells_keep_available_recorded_cells():
    cells = P.verifier_cells(*_verifier_toy())
    assert len(cells) == 5                                  # t3/b is unavailable
    assert cells.signal.sum() == 4                          # t2/b is UNKNOWN
    assert set(cells.loc[~cells.signal, "task_id"] + cells.loc[~cells.signal, "model"]) == {"t2b"}


def test_confusion_counts_and_rates():
    c = P.confusion(P.verifier_cells(*_verifier_toy()))
    # signal cells: t1a PASS/ok, t1b PASS/fail, t2a FAIL/fail, t3a FAIL/ok
    assert (c["pass_success"], c["pass_failure"], c["fail_success"], c["fail_failure"]) == (1, 1, 1, 1)
    assert c["false_pass_rate"] == 0.5 and c["false_reject_rate"] == 0.5
    assert (c["cells"], c["signal_cells"], c["no_signal_cells"]) == (5, 4, 1)


def test_confusion_zero_denominator_is_nan():
    cells = pd.DataFrame({"task_id": ["t"], "model": ["a"], "signal": [True], "passed": [True],
                          "success": [True]})
    c = P.confusion(cells)
    assert c["false_reject_rate"] == 0.0 and math.isnan(c["false_pass_rate"])


def test_confusion_frame_matches_confusion():
    cells = P.verifier_cells(*_verifier_toy())
    f = P.confusion_frame(cells)
    w = np.ones(len(f))
    assert P.ratio("fp_num", "fp_den")(f, w) == 50.0
    assert P.ratio("fr_num", "fr_den")(f, w) == 50.0


def test_no_signal_share():
    cells = P.verifier_cells(*_verifier_toy())
    s = P.no_signal_share(cells, ["t1", "t2", "t3", "t4"])     # t4 has no recorded cell
    assert (s["no_signal_cells"], s["cells"]) == (1, 5)
    assert (s["no_signal_tasks"], s["tasks"]) == (1, 4)


def test_clopper_pearson_known_values():
    assert P.clopper_pearson(0, 10) == (0.0, pytest.approx(1 - 0.025 ** 0.1))
    lo, hi = P.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187086, abs=1e-6) and hi == pytest.approx(0.812914, abs=1e-6)
    assert P.clopper_pearson(10, 10)[1] == 1.0
    assert all(math.isnan(x) for x in P.clopper_pearson(0, 0))


# ------------------------------------------------------------------------------------- S3.3
def test_unsolved_ignores_unavailable_success():
    ok, _, avail = _verifier_toy()
    ok = ok.copy()
    ok.loc["t2", "b"] = True
    avail = avail.copy()
    avail.loc["t2", "b"] = False                     # the only success is on an unavailable cell
    assert P.unsolved(ok, avail).tolist() == [False, True, False]


def _rescue_toy():
    idx = ["t1", "t2", "t3", "t4"]
    ok = pd.DataFrame({"a": [F, F, T, F], "b": [T, F, T, T]}, index=idx)
    verdict = pd.DataFrame({"a": ["FAIL", "UNKNOWN", "PASS", "PASS"],
                            "b": ["PASS", "FAIL", "PASS", "UNKNOWN"]}, index=idx, dtype=object)
    return ok, verdict


def test_rescue_pairs_hand_counts():
    ok, verdict = _rescue_toy()
    r = P.rescue_pairs(ok, verdict, order_price=["a", "b"], order_cost=["b", "a"]).set_index(
        ["lower", "higher"])
    ab = r.loc[("a", "b")]
    assert (ab.lower_fails, ab.true_rescues) == (3, 2)                 # t1, t2, t4; b solves t1, t4
    assert (ab.lower_verdict_fail, ab.observable_rescues) == (1, 1)    # only t1 shows a FAIL
    assert (ab.both_signal_after_fail, ab.verifier_only_rescues) == (1, 1)
    assert ab.lower_fails_shown == 1 and ab.observability == pytest.approx(1 / 3)
    assert ab.lower_cheaper_by_price and not ab.lower_cheaper_by_cost
    ba = r.loc[("b", "a")]
    assert (ba.lower_fails, ba.true_rescues, ba.true_rescue_rate) == (1, 0, 0.0)
    assert ba.both_signal_after_fail == 0 and math.isnan(ba.verifier_only_rescue_rate)
    assert len(r) == 2


def test_rescue_frame_matches_pairs():
    ok, verdict = _rescue_toy()
    f = P.rescue_frame(ok, verdict, "a", "b")
    w = np.ones(len(f))
    assert P.ratio("t_num", "t_den")(f, w) == pytest.approx(200 / 3)
    assert P.ratio("o_num", "o_den")(f, w) == 100.0


def _ladder_toy():
    idx, order = ["t1", "t2", "t3", "t4"], ["x", "y", "z"]
    cost = pd.DataFrame([[1.0, 2.0, 4.0]] * 4, index=idx, columns=order)
    ok = pd.DataFrame([[F, T, T], [F, F, F], [F, F, T], [F, F, F]], index=idx, columns=order)
    avail = pd.DataFrame(True, index=idx, columns=order)
    avail.loc[["t3", "t4"], "x"] = False
    return ok, cost, avail, order


def test_ladder_costs_hand_values():
    ok, cost, avail, order = _ladder_toy()
    lc = P.ladder_costs(ok, cost, avail, order)
    # t1 stops at y; t2 pays all; t3 skips unavailable x, stops at z; t4 pays y and z
    assert lc.ladder.tolist() == [3.0, 7.0, 6.0, 6.0]
    assert lc.perfect_stop.tolist() == [3.0, 1.0, 6.0, 2.0]          # unsolved pay the first available rung
    assert lc.rungs_reached.tolist() == [2, 3, 2, 2]
    assert lc.solved.tolist() == [True, False, True, False]


def test_ladder_costs_charge_verifier_per_rung_reached():
    ok, cost, avail, order = _ladder_toy()
    lc = P.ladder_costs(ok, cost, avail, order, verifier_usd=0.5)
    assert lc.ladder.tolist() == [4.0, 8.5, 7.0, 7.0]
    assert lc.perfect_stop.tolist() == [4.0, 1.5, 7.0, 2.5]


def test_ladder_summary():
    ok, cost, avail, order = _ladder_toy()
    s = P.ladder_summary(P.ladder_costs(ok, cost, avail, order), cost["z"])
    assert (s["ladder_usd"], s["always_usd"], s["perfect_stop_usd"]) == (22.0, 16.0, 12.0)
    assert s["unsolved"] == 2 and s["ladder_share_on_unsolved"] == pytest.approx(13 / 22)
    assert s["ladder_musd_per_task"] == pytest.approx(5500.0)


def test_mean_cost_order_skips_unavailable_and_breaks_ties_by_name():
    cost = pd.DataFrame({"b": [1.0, 1.0], "a": [1.0, 1.0], "c": [0.5, 100.0]})
    avail = pd.DataFrame({"b": [T, T], "a": [T, T], "c": [T, F]})
    assert P.mean_cost_order(cost, avail) == ["c", "a", "b"]
