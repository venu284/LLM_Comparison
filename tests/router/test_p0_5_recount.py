"""Checks for the P0.5 recount script (router/experiments/p0_5_recount.py)."""

import numpy as np
import pandas as pd
import pytest

from router.experiments import p0_5_recount as rc

HAVE_DATA = rc.X13_VERDICTS.exists() and all((rc.CRB_DIR / f).exists() for f in rc.CRB_FILES)


def test_problem_id_groups_variants():
    assert rc.problem_id("dim1_mbpp_631") == rc.problem_id("dim4_mbpp_completion_631") == "mbpp631"
    assert rc.problem_id("dim1_humaneval_HumanEval_51") == "he51"


def test_lp_floor_meets_floor_on_fitting_data():
    # Two cells, three actions; action 0 is the expensive reference.
    pi = np.array([0.5, 0.5])
    v = np.array([[0.8, 0.8, 0.5], [0.9, 0.6, 0.85]])
    c = np.array([[10.0, 1.0, 0.5], [10.0, 1.0, 0.5]])
    g, _ = rc.ra_lp_floor(pi, v, c, 0.0, a_ref=0)
    vb = pi @ v[:, 0]
    if isinstance(g, tuple):
        ga, gb, w = g
        val = w * rc.ra_pol_vd(ga, pi, v, c)[0] + (1 - w) * rc.ra_pol_vd(gb, pi, v, c)[0]
    else:
        val = rc.ra_pol_vd(g, pi, v, c)[0]
    assert val >= vb - 1e-12
    # Cell 0: action 1 matches the reference's value at a tenth of the cost, so it is used.
    first = g[0] if isinstance(g, tuple) else g
    assert first[0] == 1


@pytest.mark.skipif(not HAVE_DATA, reason="external data not fetched")
def test_reproduction_and_unavailable_counts():
    d = rc.load()
    out = rc.mode_repro(d)                          # raises if any 2026-11-19 number fails
    assert out["reproduced"]
    full = rc.Pop(d, rc.headline_mask(d))
    counts = rc.unavailable_counts(d, full)         # asserts 1,551 and 148
    assert counts["headline_1971"]["empty_cells"] == 66
    assert counts["headline_1971"]["missing_run_cells"] == 0
    assert isinstance(full.B, pd.DataFrame) and len(full) == 1971
