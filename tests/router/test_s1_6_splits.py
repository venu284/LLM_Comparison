"""S1.6 report: problem ids re-derived from the pinned data (needs `external/`)."""

import json

import pytest

from router.data import prompts as P
from router.data import splits as S

pytestmark = pytest.mark.skipif(
    not (P.TASKS_JSONL.is_file() and (P.SOURCES_DIR / f"{S.DS1000_ORIGIN_EXTRACT}.jsonl").is_file()),
    reason="external data not fetched (scripts/fetch_data.py)")


@pytest.fixture(scope="module")
def kt():
    return S.key_table()


@pytest.fixture(scope="module")
def derived(kt):
    return S.derive(kt)


def test_problems_csv_reproduces_from_the_pinned_data(derived, tmp_path):
    out = tmp_path / "problems.csv"
    S.write_problems(derived, out)
    assert out.read_bytes() == S.PROBLEMS_CSV.read_bytes()


def test_splits_csv_reproduces_from_the_pinned_data(kt, derived, tmp_path):
    out = tmp_path / "splits.csv"
    S.write_splits(S.split_table(derived, kt[["task_id", "family", "split_kind"]]), out)
    assert out.read_bytes() == S.SPLITS_CSV.read_bytes()


def test_only_s1_5_pairs_share_content_beyond_the_source_keys(kt):
    from router.experiments import s1_6_splits as R

    assert R.unimplied_links(kt, "content_sha256") == sorted(sorted(p) for p in R.S1_5_PAIRS)


def test_report_reproduces_its_committed_json(kt, derived):
    from router.experiments import s1_6_splits as R

    sp = S.split_table(derived, kt[["task_id", "family", "split_kind"]])
    text = json.dumps(R.build(kt, derived, sp), indent=1, sort_keys=True, default=int) + "\n"
    assert text == R.OUT_JSON.read_text()
