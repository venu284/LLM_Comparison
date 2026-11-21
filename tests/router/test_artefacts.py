"""S1.7 checkpoint 3: the frozen labels, `artefacts.yaml` and the view builder (plan S1.7, S1 gate).

Synthetic tests check the label and view rules. The data tests check that the committed yaml
reproduces from the code, that each built view's counts equal the yaml, that the views nest and
that no view holds an ineligible task. They skip without external/.
"""

import pandas as pd
import pytest

from router.data import audit
from router.data.scoring import eligible_families

HAVE_DATA = audit.RESPONSE_MATRIX.exists() and audit.ARTEFACTS_YAML.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external/ data not downloaded")


def test_labels_cover_every_eligible_family_once():
    labelled = {u["family"] for u in audit.LABELS.values()}
    eligible = set(eligible_families())
    assert eligible - labelled == set(audit.NOT_SUSPECTED)
    assert labelled - eligible == {"lcb_testgen", "he_testgen", "mbpp_testgen"}
    assert all(u["label"] in (audit.CONFIRMED, audit.CLEARED, audit.UNDECIDED)
               for u in audit.LABELS.values())
    assert all(u.get("eligible", True) is False for k, u in audit.LABELS.items() if "testgen" in k)


def test_lcb_is_split_by_the_subset_rule():
    assert audit.label_unit("t1", "lcb", {"t1"}) == "lcb"
    assert audit.label_unit("t2", "lcb", {"t1"}) == "lcb_kept"
    assert audit.LABELS["lcb"]["label"] == audit.CONFIRMED
    assert audit.LABELS["lcb_kept"]["label"] == audit.UNDECIDED
    assert audit.label_unit("t3", "he_completion", set()) is None


def test_views_nest_and_follow_the_labels():
    fam = pd.Series({"a": "mbpp", "b": "bcb", "c": "humaneval", "d": "he_completion",
                     "e": "lcb", "f": "lcb", "g": "he_testgen", "h": "cruxeval"})
    v = audit.view_task_ids(fam, leetcode={"e"}, x13={"a", "d", "e", "f"})
    assert v["all_eligible"] == ["a", "b", "c", "d", "e", "f", "h"]     # no ineligible task
    assert v["filtered"] == ["a", "d", "h"]                             # cleared + not suspected
    assert v["sensitivity"] == ["a", "c", "d", "f", "h"]                # + undecided (lcb kept)
    assert v["headline"] == ["a", "d"]                                  # filtered with X13 verdicts
    assert set(v["headline"]) <= set(v["filtered"]) <= set(v["sensitivity"]) <= set(v["all_eligible"])


@pytest.fixture(scope="module")
def built():
    art = audit.load_artefacts()
    m = audit.load_release()
    return art, m, audit.build_views(m, art)


@needs_data
def test_yaml_reproduces_from_the_code():
    assert audit.build_artefacts() == audit.load_artefacts()


@needs_data
def test_view_counts_equal_the_yaml(built):
    art, _, views = built
    for name, m in views.items():
        y = art["views"][name]
        assert len(m) == y["tasks"]
        assert m.long.drop_duplicates("task_id").problem_id.nunique() == y["problems"]
        assert m.family.nunique() == y["families"]
        assert m.family.value_counts().to_dict() == y["tasks_per_family"]
        assert int(m.available.all(axis=1).sum()) == y["complete_case_tasks"]
        reasons = m.unavailable_reason.stack().value_counts().to_dict()
        assert {k: v for k, v in y["unavailable_cells"].items() if v} == reasons
        assert audit._sha(m.tasks) == y["task_ids_sha256"]


@needs_data
def test_views_nest_and_hold_no_ineligible_task(built):
    art, _, views = built
    t = {k: set(v.tasks) for k, v in views.items()}
    assert t["headline"] <= t["filtered"] <= t["sensitivity"] <= t["all_eligible"]
    eligible = set(eligible_families())
    assert all(set(v.family) <= eligible for v in views.values())
    assert art["label_task_counts_eligible"]["confirmed defect"] + \
        art["label_task_counts_eligible"]["undecided"] == len(t["all_eligible"] - t["filtered"])


@needs_data
def test_counts_reviewed_at_checkpoint_2(built):
    """The view sizes given in the plan log after the checkpoint 2 review (main a3a6627)."""
    art = built[0]
    assert [art["views"][v]["tasks"] for v in ("all_eligible", "filtered", "sensitivity")] == \
        [5626, 1287, 2160]
    assert art["subset_rules"]["lcb"]["excluded_tasks"] == 422
    assert art["unavailable"]["x13_empty_cells"] == 1551
    assert art["unavailable"]["x13_empty_cells_released_score_1"] == 877
    assert art["unavailable"]["missing_token_record_cells"] == 148
    assert art["response_matrix"]["empty_cells_8_models"] == 3330
