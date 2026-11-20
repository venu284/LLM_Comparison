"""S1.1: the task-id-to-family table. The frozen-table tests read the committed
`router/data/task_families.csv`; the drift test needs external/ and skips without it."""

import re

import pandas as pd
import pytest

from router.data import sources
from router.data.coderouterbench import CRB_DIR
from router.settings import MIN_FAMILY_TASKS

HAVE_DATA = all((CRB_DIR / f).exists()
                for f in (*sources.ID_SPLIT_FILES.values(), sources.OOD_FILE))

# The one family the S1 gate's "zero singleton families" check fails on, reported to the user
# as a gate fail (2026-11-20); OOD families are exempt only from the 30-task minimum.
KNOWN_SINGLETON_GATE_FAIL = {"swe_ci"}


@pytest.fixture(scope="module")
def table():
    return sources.load_table()


def test_table_covers_every_release_id(table):
    n = table.groupby("split").size().to_dict()
    assert n == {"probing": 7080, "id_test": 2919, "ood": 176}
    assert not table.task_id.duplicated().any()


def test_every_id_maps_to_a_frozen_family(table):
    assert set(table.family) == set(sources.FAMILIES)
    assert (table.family == table.task_id.map(sources.parse_family)).all()


def test_counts_match_frozen_table(table):
    assert sources.family_counts(table) == sources.FAMILIES
    assert sum(f.n_tasks for f in sources.FAMILIES.values()) == 10_175


def test_each_family_has_one_split_kind_and_dimension(table):
    per = table.groupby("family").agg(kinds=("split_kind", "nunique"), dims=("dimension", "nunique"))
    assert (per.kinds == 1).all() and (per.dims == 1).all()


def _task_tokens(task_id: str) -> set[str]:
    """The parts of an id that identify one task: its numbers and, for MultiPL-E, its function."""
    tokens = set(re.findall(r"\d+", task_id.split("::")[-1].split("_", 1)[-1]))
    m = re.search(r"HumanEval_\d+_(\w+)$", task_id)
    if m:
        tokens.add(m.group(1))
    return tokens


def test_no_family_name_carries_a_task_number_or_function(table):
    for task_id, fam in zip(table.task_id, table.family):
        parts = set(fam.split("_"))
        assert not parts & _task_tokens(task_id), (task_id, fam)
    assert not [f for f in sources.FAMILIES if any(p.isdigit() for p in f.split("_"))]


def test_id_families_meet_the_minimum_or_await_the_user():
    under = sources.families_under(MIN_FAMILY_TASKS, "id")
    assert set(under) <= sources.PENDING_USER_DECISION, sorted(set(under) - sources.PENDING_USER_DECISION)
    # A pending entry must still be under the minimum, so a stale entry cannot hide a family.
    assert sources.PENDING_USER_DECISION <= set(under)


def test_ood_families_are_listed_and_exempt():
    ood = {k: f.n_tasks for k, f in sources.FAMILIES.items() if f.split_kind == "ood"}
    assert ood == {"featurebench": 49, "longcli": 14, "old112": 112, "swe_ci": 1}


def test_singleton_families_are_only_the_reported_gate_fail():
    singletons = {k for k, f in sources.FAMILIES.items() if f.n_tasks == 1}
    assert singletons == KNOWN_SINGLETON_GATE_FAIL
    assert all(sources.FAMILIES[k].split_kind == "ood" for k in singletons)


@pytest.mark.parametrize("task_id, family", [
    ("dim1_mbpp_631", "mbpp"),
    ("dim3_mbpp_completion_12", "mbpp_completion"),
    ("dim5_mbpp_testgen_12", "mbpp_testgen"),
    ("dim2_lcb_abc388_d", "lcb"),
    ("dim5_lcb_testgen_abc388_d", "lcb_testgen"),
    ("dim1_bcb_BigCodeBench/767", "bcb"),
    ("dim1_bcb_v014_BigCodeBench/14", "bcb_v014"),
    ("dim7_hex_js_JavaScript/3", "hex_js"),
    ("dim7_multiple_rs_HumanEval_12_longest", "multiple_rs"),
    ("dim4_bugs2fix_sup_7", "bugs2fix_sup"),
    ("dim4_bugs2fix_7", "bugs2fix"),
    ("dim3_swebench_django__django-12856", "swebench"),
    ("new64::swe_ci::some.task", "swe_ci"),
    ("old112::django__django-11099", "old112"),
])
def test_parse_rules(task_id, family):
    assert sources.parse_family(task_id) == family


@pytest.mark.parametrize("task_id", ["mbpp_631", "dim1_newbench_3", "dim1_mbpp_x", "new64::other::t"])
def test_unparsed_id_raises(task_id):
    with pytest.raises(sources.UnparsedTaskId):
        sources.parse_family(task_id)


@pytest.mark.skipif(not HAVE_DATA, reason="CodeRouterBench not downloaded to external/")
def test_committed_table_matches_pinned_release(table):
    built = sources.build_table(CRB_DIR).astype(str)
    pd.testing.assert_frame_equal(built.reset_index(drop=True), table.reset_index(drop=True))


@pytest.mark.skipif(not HAVE_DATA, reason="CodeRouterBench not downloaded to external/")
def test_ood_families_agree_with_bench_column():
    ood = pd.read_csv(CRB_DIR / sources.OOD_FILE, usecols=["task_id", "bench"]).drop_duplicates("task_id")
    assert (ood.task_id.map(sources.parse_family) == ood.bench).all()
