"""S1.5 report: P0.6 reproduction on the joined task text (needs the pinned data)."""

import pytest

from router.data import prompts as P

pytestmark = pytest.mark.skipif(
    not (P.TASKS_JSONL.is_file() and (P.SOURCES_DIR / "livecodebench.jsonl").is_file()),
    reason="external data not fetched (scripts/fetch_data.py)")


@pytest.fixture(scope="module")
def p0_6():
    from router.experiments import s1_5_prompt_join as R

    try:
        return R.p0_6_reproduction(P.prompt_table())
    except FileNotFoundError as e:  # CodeRouterBench matrices not present
        pytest.skip(str(e))


def test_p0_6_complete_case_set_joins(p0_6):
    assert (p0_6["tasks"], p0_6["joined"]) == (1907, 1907)


@pytest.mark.parametrize("field", ["wrapped", "raw"])
def test_p0_6_identical_pairs_and_small_bcb_cells_reproduce(p0_6, field):
    q = p0_6[field]
    assert len(q["pairs_identical"]) == 3 and all(q["pairs_identical"].values())
    assert (q["small_bcb_identical_in_other_cell"], q["small_bcb_tasks"]) == (63, 67)
