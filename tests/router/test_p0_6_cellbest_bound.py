"""Checks for the P0.6 script (router/experiments/p0_6_cellbest_bound.py)."""

import numpy as np
import pytest

from router.experiments import p0_5_recount as rc
from router.experiments import p0_6_cellbest_bound as cb

HAVE_DATA = (rc.X13_VERDICTS.exists() and all((rc.CRB_DIR / f).exists() for f in rc.CRB_FILES)
             and cb.TASKS_JSONL.exists())


def test_cellbest_table_ties_to_cheaper():
    v = np.array([[0.8, 0.8, 0.5, 0.99]])          # column 3 is a ladder, not a direct model
    c = np.array([[3.0, 1.0, 0.1, 0.1]])
    assert cb.cellbest_table(v, c, n_direct=3).tolist() == [1]


def test_rule_tables_toy():
    # Cell 0: CellBest is column 0 (0.8 at cost 10). Columns 2 and 3 tie at 0.8 for 5.
    # Rule A takes the cheapest with acceptance >= 0.8: columns 2 and 3 tie, so the lower index.
    # Rule B takes the highest acceptance with cost <= 10: column 4 (0.9 at 9).
    # Cell 1: only CellBest's own column qualifies under either rule.
    v = np.array([[0.8, 0.6, 0.8, 0.8, 0.9],
                  [0.8, 0.7, 0.9, 0.1, 0.1]])
    c = np.array([[10.0, 1.0, 5.0, 5.0, 9.0],
                  [2.0, 1.0, 3.0, 4.0, 4.0]])
    cbt = np.array([0, 0])
    ra, rb = cb.rule_tables(v, c, cbt)
    assert ra.tolist() == [2, 0]
    assert rb.tolist() == [4, 0]


def test_rule_b_tie_goes_to_cheaper():
    v = np.array([[0.5, 0.9, 0.9]])
    c = np.array([[10.0, 8.0, 6.0]])
    _, rb = cb.rule_tables(v, c, np.array([0]))
    assert rb.tolist() == [2]


def test_cell_means():
    M = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    sidx = np.array([0, 0, 2])
    v, seen = cb.cell_means(np.array([0, 1, 2]), sidx, 3, M)
    assert np.allclose(v, [[0.5, 0.5], [0, 0], [1, 1]])
    assert seen.tolist() == [True, False, True]


@pytest.mark.skipif(not HAVE_DATA, reason="external data not fetched")
def test_reproduces_p0_5_and_joins_prompts():
    d = rc.load()
    cc = cb.complete_case(d)
    rep = cb.reproduce(d, cc, 1000)               # raises SystemExit if any P0.5 number differs
    assert rep["reproduced"]
    prompts, _ = cb.load_prompts(cc.ids)
    assert int(prompts.notna().sum()) == 1907


@pytest.mark.skipif(not HAVE_DATA, reason="external data not fetched")
def test_policy_matrices_match_p0_5_ladders():
    d = rc.load()
    cc = cb.complete_case(d)
    names, A, C = cb.policy_matrices(cc, "bound")
    assert len(names) == 8 + 84 and A.shape == C.shape == (1907, 92)
    # Direct-dispatch columns are the released success and cost; no verifier charge.
    assert np.array_equal(A[:, :8], cc.B.values) and np.array_equal(C[:, :8], cc.c.values)
    fa, fc, _ = rc.crossfit_baseline(cc, rc.folds(cc, 0))
    sav = 100 * (1 - C[:, 8:].sum(0) / fc.sum())
    gap = 100 * (A[:, 8:].mean(0) - fa.mean())
    assert int(((sav >= rc.GATE_SAVING) & (gap >= rc.GATE_GAP)).sum()) == 12
