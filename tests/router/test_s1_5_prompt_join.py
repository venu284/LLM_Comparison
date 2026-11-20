"""S1.5 report: P0.6 reproduction and content collisions on the joined task text (needs the
pinned data)."""

import pytest

from router.data import prompts as P

pytestmark = pytest.mark.skipif(
    not (P.TASKS_JSONL.is_file() and (P.SOURCES_DIR / "livecodebench.jsonl").is_file()),
    reason="external data not fetched (scripts/fetch_data.py)")


@pytest.fixture(scope="module")
def report():
    from router.experiments import s1_5_prompt_join as R

    return R


@pytest.fixture(scope="module")
def p0_6(report):
    try:
        return report.p0_6_reproduction(P.prompt_table())
    except FileNotFoundError as e:  # CodeRouterBench matrices not present
        pytest.skip(str(e))


def test_p0_6_complete_case_set_joins(p0_6):
    assert (p0_6["tasks"], p0_6["joined"]) == (1907, 1907)


@pytest.mark.parametrize("field", ["content", "raw", "wrapped"])
def test_p0_6_identical_pairs_and_small_bcb_cells_reproduce(p0_6, field):
    q = p0_6[field]
    assert len(q["pairs_identical"]) == 3 and all(q["pairs_identical"].values())
    assert (q["small_bcb_identical_in_other_cell"], q["small_bcb_tasks"]) == (63, 67)


def test_eligible_families_share_content_only_where_allowed(report):
    """Addendum A stop rule: across eligible families, content is shared only by the BigCodeBench
    cross-dimension prompts and HumanEval variants of the same problem."""
    from router.data.scoring import eligible_families

    t = P.prompt_table()
    el = t[(t.split_kind == "id") & t.family.isin(eligible_families())]
    assert report.unexpected_content_collisions(el) == []
    h = report.shared_hashes(el, "content_sha256")
    assert (h["tasks_hashed"], h["distinct_hashes"]) == (5555, 4988)
    assert (h["tasks_in_within_family_groups"], h["tasks_in_across_family_groups"]) == (10, 830)
