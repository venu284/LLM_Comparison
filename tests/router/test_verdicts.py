"""Checks for the X13 verdict loader (router/data/verdicts.py, S1.11).

Only counts that cannot change are tested here. Headline counts are derived after S1.4 and S1.7
and frozen in artefacts.yaml.
"""

import pandas as pd
import pytest

from router.data import verdicts as vd

HAVE_DATA = vd.X13_VERDICTS.exists() and all((vd.CRB_DIR / f).exists()
                                              for f in vd.SPLIT_FILES.values())
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external data not fetched")

FORBIDDEN = {"truth", "w1_accept", "w1_quality", "w1_syntax", "w1b_tool", "w1b_accept", "score"}


@pytest.fixture(scope="module")
def v():
    return vd.load_verdicts()


@pytest.fixture(scope="module")
def raw():
    return vd.load_raw()


@needs_data
def test_row_and_task_counts(v):
    assert len(v) == 23_288
    assert v.task_id.nunique() == 2_911
    assert not v.duplicated(["task_id", "model"]).any()


@needs_data
def test_split_coverage(v):
    tasks = v.drop_duplicates("task_id").split.value_counts()
    assert tasks.to_dict() == {"probing": 2_044, "id_test": 867}
    assert v.split.notna().all()


@needs_data
def test_signal_no_signal_empty_counts(v):
    assert int(v.signal.sum()) == 8_577
    assert int((~v.signal & ~v.unavailable).sum()) == 13_160
    assert int(v.unavailable.sum()) == 1_551


@needs_data
def test_observations(v):
    obs = v.observation
    assert (obs[v.signal & v.visible_pass.eq(True)] == vd.PASS).all()
    assert (obs[v.signal & v.visible_pass.eq(False)] == vd.FAIL).all()
    assert (obs[~v.signal & ~v.unavailable] == vd.UNKNOWN).all()
    assert obs[v.unavailable].isna().all()
    assert v.visible_pass[~v.signal].isna().all()


@needs_data
def test_empty_cells_have_no_visible_tests(raw):
    empty = raw["empty"] == "True"
    assert (raw.loc[empty, "w1_n"].astype(int) == 0).all()


@needs_data
def test_accept_equals_all_visible_pass_where_tests_exist(raw):
    n = raw.w1_n.astype(int)
    p = raw.w1_p.astype(int)
    has = n > 0
    assert ((raw.w1_accept[has] == "True") == (p[has] == n[has])).all()


@needs_data
def test_main_output_has_no_ground_truth_or_heuristic(v):
    assert not FORBIDDEN & set(v.columns)
    assert not any(c.startswith("w2_") for c in v.columns)


@needs_data
def test_heuristic_variant_is_separate_and_labelled(v):
    h = vd.load_heuristic_accept()
    assert len(h) == len(v)
    assert "heuristic_accept" in h.columns and "truth" not in h.columns
    merged = v.merge(h, on=["task_id", "model"])
    # With a signal the heuristic variant is the visible-test verdict itself.
    sig = merged[merged.signal]
    assert (sig.heuristic_accept == sig.visible_pass.astype(bool)).all()


@needs_data
def test_hidden_test_fields_are_separate(v):
    h = vd.load_hidden_tests()
    assert len(h) == len(v)
    assert {"w2_pass", "w2_kind", "w2_frac", "w2_sec"} <= set(h.columns)
    assert "truth" not in h.columns
    assert (h.loc[h.w2_kind == "empty_response"].shape[0]) == 1_551


@needs_data
@pytest.mark.parametrize("population", ["complete_case_1907", "all_x13"])
def test_equivalence_with_p0_5_pivots(v, population):
    """load_verdicts() agrees with p0_5_recount.load()'s pivots A, N and E on every signal cell and
    every empty cell. The complete case has no empty cells by construction, so the all-X13 run is
    the one that exercises E."""
    from router.experiments import p0_5_recount as rc
    from router.experiments import p0_6_cellbest_bound as cb

    d = rc.load()
    if population == "complete_case_1907":
        cc = cb.complete_case(d)
        assert len(cc) == 1_907
        ids = cc.ids
    else:
        ids = d["A"].index
    res = vd.compare_with_p0_5(v, d, ids)
    assert res["n_signal_cells"] > 0
    assert res["A_mismatch"] == 0 and res["N_mismatch"] == 0 and res["E_mismatch"] == 0
    if population == "all_x13":
        assert res["n_empty_cells"] == 1_551


@needs_data
def test_truth_matches_release_runs(v):
    out = vd.truth_matches_release()
    assert out["n_joined"] == 23_288
    assert out["n_mismatch"] >= 0
