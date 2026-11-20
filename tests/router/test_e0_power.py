"""Checks for the E0 power port (router/experiments/e0_power.py)."""

import json

import numpy as np
import pytest

from router import settings as S
from router.experiments import e0_power as e

HAVE_OUT = e.OUT_JSON.exists() and e.P0_6_JSON.exists()


def test_power_factor_is_the_original_scripts_1_43():
    assert e.POWER_FACTOR == pytest.approx((1.959964 + 0.841621) / 1.959964, abs=1e-5)
    assert round(e.POWER_FACTOR, 2) == 1.43


def test_power_at_margin_is_80pct_at_the_factor_margin():
    assert e.power_at_margin(e.POWER_FACTOR * 1.5, 1.5) == pytest.approx(0.80, abs=1e-9)
    assert e.power_at_margin(1.5, 1.5) == pytest.approx(0.5, abs=1e-12)


def test_oracle_takes_the_cheapest_best_model():
    V = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.2, 0.8, 0.8]])
    C = np.array([[5.0, 2.0, 1.0], [3.0, 1.0, 2.0], [1.0, 4.0, 3.0]])
    best, cost = e.oracle(V, C)
    assert best.tolist() == [1.0, 0.0, 0.8]
    assert cost.tolist() == [2.0, 1.0, 3.0]


def test_gate_view_uses_lower_bounds_and_strict_c_saving():
    m = {"vs_best_single": {"gap_points": [1.0, -2.0, 4.0], "saving_pct": [25.0, 20.0, 30.0]},
         "vs_cellbest_pred": {"gap_points": [0.0, -2.0, 2.0], "saving_pct": [1.0, 0.0, 2.0]}}
    v = e.gate_view(m)
    assert v["passes_ab"] is True                    # lower bounds exactly at 20 and -2 pass (a), (b)
    assert v["passes_c"] is False                    # (c) needs the saving lower bound above 0
    assert v["best_single"]["gap_half_width"] == 3.0


def test_aggregate_margin_is_factor_times_mean_half_width():
    views = [{"best_single": {"gap_points": [0, -h, h], "saving_pct": [0, 0, 0],
                              "gap_half_width": h, "saving_half_width": 0.0},
              "passes_ab": ok} for h, ok in ((1.0, True), (2.0, False))]
    a = e.aggregate(views)
    assert a["best_single"]["margin_80pct_power_true_gap_0"] == pytest.approx(1.5 * e.POWER_FACTOR)
    assert a["p_passes_ab"] == 0.5


@pytest.mark.skipif(not HAVE_OUT, reason="needs router/experiments/out/e0_power.json")
def test_committed_output_reproduces_p0_6_on_the_gate_seed():
    out = json.loads(e.OUT_JSON.read_text())
    assert out["p0_6_check"]["matches_p0_6_seed0"] is True
    ref = json.loads(e.P0_6_JSON.read_text())["seed0"]["metrics"]
    res = out["headline_complete_case"]["results"]
    # P0.6 section 8: gap half-widths against CellBest with predicted cells, bound policies.
    for key, hw in (("ruleA_bound_pred", 1.83), ("ruleB_bound_pred", 1.26)):
        g = res[key]["gate_seed"]["cellbest_pred"]
        assert round(g["gap_half_width"], 2) == hw
        assert g["gap_half_width"] == pytest.approx(ref[key]["ci_informational"]["gap_vs_cellbest_pred_half_width"], abs=1e-12)


@pytest.mark.skipif(not HAVE_OUT, reason="needs router/experiments/out/e0_power.json")
def test_committed_output_settings():
    s = json.loads(e.OUT_JSON.read_text())["settings"]
    assert s["boot"] == S.N_BOOT and s["folds"] == S.N_FOLDS and s["gate_seed"] == S.GATE_SEED
    assert s["draws"] >= S.N_SEEDS
