"""S2.6: the paper's published numbers reproduce on the paper view with the paper's AvgPerf
(REPRODUCTION ONLY; `router.sim.paper`), and the reference DimensionBest - ACRouter figures."""

import pytest

from router.data import coderouterbench as crb

HAVE_CRB = (crb.CRB_DIR / crb.SPLIT_FILES["id_test"]).exists()
pytestmark = pytest.mark.skipif(not HAVE_CRB, reason="CodeRouterBench not downloaded to external/")


@pytest.fixture(scope="module")
def res():
    from router.experiments import s2_reproduce
    return s2_reproduce.reproduce()


@pytest.mark.parametrize("router, target", [("oracle", 57.00), ("always_opus", 43.83),
                                            ("acrouter_released", 50.14)])
def test_paper_view_reproductions(res, router, target):
    r = res["views"]["paper"]["routers"][router]
    assert round(r["PaperAvgPerf"], 2) == target and r["matches_target"]


def test_acrouter_target_is_the_release_metrics_file(res):
    """50.14 is the release's summary.json; the paper's Table 3 says 49.98 (another run)."""
    assert res["release_summary_id"]["AvgPerf%"] == 50.14
    r = res["views"]["paper"]["routers"]["acrouter_released"]
    assert round(r["CumReg"], 1) == res["release_summary_id"]["CumReg"] == 202.0
    assert round(r["minus_paper_table3"], 2) == 0.16


def test_always_opus_cost_matches_table_12(res):
    r = res["views"]["paper"]["routers"]["always_opus"]
    assert round(r["$Total"], 2) == 34.02 and round(r["Perf/$"], 2) == 1.29


def test_dimension_best_vs_acrouter_reference(res):
    paper = res["views"]["paper"]["dimension_best_minus_acrouter"]["task"]
    complete = res["views"]["complete"]["dimension_best_minus_acrouter"]["task"]
    assert round(complete["point"], 2) == 1.54
    assert round(paper["point"], 2) in (-0.04, -0.05)            # -0.045
    assert paper["lo"] < 0 < paper["hi"] and complete["lo"] > 0
    assert paper["unit"] == "task" and paper["n_units"] == 2919
