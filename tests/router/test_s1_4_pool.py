"""S1.4: the eligible-pool counts, pinned after they were reported (2026-11-20). Rules-only counts
need only the committed family table; the headline and X13 counts skip without external/."""

import pytest

from router.data.coderouterbench import CRB_DIR, SPLIT_FILES
from router.experiments import p0_5_recount as p05
from router.experiments import s1_4_pool as s14

HAVE_DATA = all((CRB_DIR / f).exists() for f in SPLIT_FILES.values()) and p05.X13_VERDICTS.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external/ data not downloaded")

POOL = {  # family: (probing, id_test)
    "bcb": (460, 215), "bcb_algo2": (78, 33), "bcb_completion": (58, 31), "bcb_ds": (79, 32),
    "bcb_v014": (81, 30), "cruxeval": (565, 235), "ds1000": (720, 280), "he_completion": (66, 32),
    "hex_cpp": (103, 61), "hex_go": (120, 44), "hex_java": (116, 48), "hex_js": (114, 50),
    "hex_python": (113, 51), "humaneval": (67, 33), "lcb": (721, 279), "mbpp": (156, 69),
    "mbpp_completion": (87, 37), "multiple_rs": (116, 40), "multiple_ts": (96, 39),
    "swebench": (52, 19),
}


@pytest.fixture(scope="module")
def t():
    return s14.id_table()


def test_eligible_pool_per_family_and_split(t):
    p = s14.pool(t)
    assert {f: (v["probing"], v["id_test"]) for f, v in p["families"].items()} == POOL
    assert p["total"] == {"probing": 3968, "id_test": 1658, "total": 5626}
    assert p["eligible_under_minimum"] == []


def test_diff_a_against_exec_dims_alone(t):
    d = s14.diff_a(t)
    assert d["provisional"] == {"probing": 3943, "id_test": 1612, "total": 5555}
    assert set(d["moved_in"]["families"]) == {"hex_cpp", "hex_go", "hex_java", "hex_js",
                                              "hex_python", "multiple_rs", "multiple_ts", "swebench"}
    assert d["moved_in"]["total"] == {"probing": 830, "id_test": 352, "total": 1182}
    assert set(d["moved_out"]["families"]) == {"he_testgen", "lcb_testgen", "mbpp_testgen"}
    assert d["moved_out"]["total"] == {"probing": 805, "id_test": 306, "total": 1111}


@needs_data
def test_diff_b_suspected_overlay(t):
    from router.data.verdicts import load_verdicts
    b = s14.diff_b(t, set(load_verdicts().task_id))
    assert set(b) == {"lcb", "lcb_testgen", "he_testgen", "mbpp_testgen"}
    assert all(r["all_family_tasks_suspected"] for r in b.values())
    assert [f for f, r in b.items() if r["eligible_by_rule"]] == ["lcb"]
    assert b["lcb"]["label"] == s14.PENDING_S17
    assert b["lcb"]["with_x13_verdicts"] == {"probing": 564, "id_test": 212, "total": 776}


@needs_data
def test_headline_sets_and_cruxeval(t):
    h = s14.headline(t)
    assert (h["headline_1971"]["tasks"], h["headline_1971"]["eligible"]) == (1971, 1971)
    assert (h["complete_case_1907"]["tasks"], h["complete_case_1907"]["eligible"]) == (1907, 1907)
    assert h["headline_1971"]["cruxeval_eligible"] == 800
    assert h["complete_case_1907"]["cruxeval_eligible"] == 792
    assert h["stop_rule_fires"] is False


@needs_data
def test_loader_views_carry_the_same_eligible_pool():
    """`annotate` through `coderouterbench.long_table`, in both views, agrees with the pool."""
    from router.data import coderouterbench as crb
    want = {"probing": 3968, "id_test": 1658}
    for split, n in want.items():
        for view in crb.VIEWS:
            long = crb.long_table(split, view)
            assert long.score_type.notna().all() and long.eligible.notna().all(), (split, view)
            assert long.loc[long.eligible.astype(bool), "task_id"].nunique() == n, (split, view)
