"""S1.6 problem ids: unit tests and checks on the committed files (no external data needed)."""

import inspect
import json
from pathlib import Path

import pandas as pd
import pytest

from router.data import splits as S
from router.data.sources import load_table
from router.settings import DEV_SEED, N_FOLDS

REPORT = Path(S.__file__).resolve().parents[1] / "experiments/out/s1_6_splits.json"


# --------------------------------------------------------------------------- source keys
@pytest.mark.parametrize("task_id, family, key", [
    ("dim1_humaneval_HumanEval_12", "humaneval", "he/12"),
    ("dim4_he_completion_HumanEval_0", "he_completion", "he/0"),
    ("dim9_he_testgen_HumanEval_99", "he_testgen", "he/99"),
    ("dim7_hex_go_Go/47", "hex_go", "he/47"),
    ("dim7_hex_cpp_CPP/0", "hex_cpp", "he/0"),
    ("dim7_hex_js_JavaScript/163", "hex_js", "he/163"),
    ("dim7_multiple_rs_HumanEval_0_has_close_elements", "multiple_rs", "he/0"),
    ("dim7_multiple_ts_HumanEval_9_rolling_max", "multiple_ts", "he/9"),
    ("dim1_mbpp_138", "mbpp", "mbpp/138"),
    ("dim4_mbpp_completion_94", "mbpp_completion", "mbpp/94"),
    ("dim9_mbpp_testgen_100", "mbpp_testgen", "mbpp/100"),
    ("dim6_ds1000_138", "ds1000", "ds1000/138"),
    ("dim2_bcb_algo2_BigCodeBench/10", "bcb_algo2", "bcb/10"),
    ("dim1_bcb_v014_BigCodeBench/0", "bcb_v014", "bcb/0"),
    ("dim2_lcb_1873_A", "lcb", "lcb/1873_A"),
    ("dim9_lcb_testgen_abc301_b", "lcb_testgen", "lcb/abc301_b"),
    ("dim4_cruxeval_7", "cruxeval", "crux/7"),
    ("dim3_swebench_astropy__astropy-12907", "swebench", "swe/astropy__astropy-12907"),
    ("dim5_bugs2fix_sup_7000", "bugs2fix_sup", "bugs2fix_sup/7000"),
    ("dim3_debugbench_1004", "debugbench", "debugbench/1004"),
])
def test_source_key(task_id, family, key):
    assert S.source_key(task_id, family) == key


def test_source_keys_are_namespaced_so_bare_ids_do_not_collide():
    assert S.source_key("dim6_ds1000_138", "ds1000") != S.source_key("dim1_mbpp_138", "mbpp")


def test_ood_tasks_get_their_own_namespace():
    assert S.source_key("old112::astropy__astropy-13033", "old112", "ood") == "ood/old112/astropy__astropy-13033"


def test_a_task_id_without_its_family_is_an_error():
    with pytest.raises(ValueError):
        S.source_key("dim1_mbpp_12", "bugs2fix")


def test_key_order_is_by_namespace_then_natural():
    keys = ["mbpp/71", "he/10", "he/9", "mbpp/141", "ds1000/2", "bugs2fix/1", "crux/3"]
    assert sorted(keys, key=S.key_order) == ["he/9", "he/10", "mbpp/71", "mbpp/141", "ds1000/2", "crux/3",
                                             "bugs2fix/1"]


# --------------------------------------------------------------------------- DS-1000 origins
def _ds(pid, lib, lpid, kind, origin):
    return {"metadata.problem_id": pid, "metadata.library": lib, "metadata.library_problem_id": lpid,
            "metadata.perturbation_type": kind, "metadata.perturbation_origin_id": origin}


def test_origin_ids_resolve_within_the_library():
    rows = [_ds(0, "Pandas", 0, "Origin", 0), _ds(1, "Pandas", 1, "Surface", 0),
            _ds(2, "Numpy", 0, "Origin", 0), _ds(3, "Numpy", 1, "Semantic", 0),
            _ds(4, "Numpy", 2, "Surface", 1)]          # a perturbation of a perturbation
    assert S.ds1000_origin_pairs(rows) == [(1, 0), (3, 2), (4, 3)]


def test_an_origin_row_that_does_not_point_to_itself_is_an_error():
    with pytest.raises(ValueError, match="point to itself"):
        S.ds1000_origin_pairs([_ds(0, "Pandas", 0, "Origin", 1), _ds(1, "Pandas", 1, "Origin", 1)])


def test_an_unresolved_origin_is_an_error():
    with pytest.raises(ValueError, match="not found"):
        S.ds1000_origin_pairs([_ds(0, "Pandas", 0, "Origin", 0), _ds(1, "Pandas", 1, "Surface", 7)])


# --------------------------------------------------------------------------- derivation on a toy table
def _toy():
    rows = [  # task_id, family, split_kind, source_key, content
        ("a1", "fa", "id", "he/1", "h1"), ("a2", "fb", "id", "he/1", "h2"),
        ("b1", "fa", "id", "mbpp/7", "h3"), ("b2", "fa", "id", "mbpp/71", "h3"),
        ("c1", "ds1000", "id", "ds1000/0", "h4"), ("c2", "ds1000", "id", "ds1000/1", "h5"),
        ("d1", "fx", "id", "fx/1", None), ("d2", "fx", "id", "fx/2", None),
        ("o1", "old", "ood", "ood/old/1", None),
    ]
    return pd.DataFrame(rows, columns=["task_id", "family", "split_kind", "source_key", "content_sha256"])


DS_TOY = [_ds(0, "Pandas", 0, "Origin", 0), _ds(1, "Pandas", 1, "Surface", 0)]


def test_derive_links_keys_origins_and_content_but_not_null_hashes():
    pr = S.derive(_toy(), DS_TOY).set_index("task_id")
    assert pr.problem_id.to_dict() == {"a1": "he/1", "a2": "he/1", "b1": "mbpp/7", "b2": "mbpp/7",
                                       "c1": "ds1000/0", "c2": "ds1000/0", "d1": "fx/1", "d2": "fx/2",
                                       "o1": "ood/old/1"}
    assert pr.link.to_dict() == {"a1": "source_key", "a2": "source_key", "b1": "content", "b2": "content",
                                 "c1": "ds1000_origin", "c2": "ds1000_origin", "d1": "none", "d2": "none",
                                 "o1": "none"}
    assert list(S.derive(_toy(), DS_TOY).columns) == S.PROBLEM_COLUMNS


def test_content_already_implied_is_not_marked_as_a_content_link():
    kt = _toy()
    kt.loc[kt.task_id == "c2", "content_sha256"] = "h4"      # same content as its own origin
    assert S.derive(kt, DS_TOY).set_index("task_id").link["c2"] == "ds1000_origin"


def test_derive_refuses_a_problem_that_mixes_id_and_ood_tasks():
    kt = _toy()
    kt.loc[kt.task_id == "o1", "content_sha256"] = "h1"
    with pytest.raises(ValueError, match="ID and OOD"):
        S.derive(kt, DS_TOY)


def test_problem_names_do_not_depend_on_row_order():
    kt = _toy()
    a = S.derive(kt, DS_TOY).set_index("task_id").problem_id
    b = S.derive(kt.iloc[::-1].reset_index(drop=True), DS_TOY).set_index("task_id").problem_id
    assert a.sort_index().equals(b.sort_index())


# --------------------------------------------------------------------------- committed problems.csv
@pytest.fixture(scope="module")
def problems():
    return S.load_problems()


def test_every_task_has_exactly_one_problem_id(problems):
    table = load_table()
    assert list(problems.columns) == S.PROBLEM_COLUMNS
    assert problems.task_id.is_unique and set(problems.task_id) == set(table.task_id)
    assert problems.problem_id.notna().all() and (problems.problem_id != "").all()


def test_problem_counts_per_family_match_the_report(problems):
    fam = load_table().set_index("task_id").family
    got = problems.assign(family=problems.task_id.map(fam)).groupby("family").problem_id.nunique()
    rep = json.loads(REPORT.read_text())["problems"]["families"]
    assert {f: int(n) for f, n in got.items()} == {f: v["problems"] for f, v in rep.items()}
    assert problems.problem_id.nunique() == json.loads(REPORT.read_text())["problems"]["problems"]


def test_link_values_are_known_and_match_the_report(problems):
    allowed = {"+".join(c) for n in range(1, 4) for c in __import__("itertools").combinations(S.LINKS, n)}
    assert set(problems.link) <= allowed | {S.NO_LINK}
    assert problems.link.value_counts().to_dict() == json.loads(REPORT.read_text())["problems"]["links"]


def test_no_problem_mixes_id_and_ood_tasks(problems):
    kind = load_table().set_index("task_id").split_kind
    mixed = problems.assign(kind=problems.task_id.map(kind)).groupby("problem_id").kind.nunique()
    assert (mixed == 1).all()


def test_annotate_fills_problem_id_from_the_committed_file(problems):
    long = pd.DataFrame({"task_id": ["dim7_hex_go_Go/12", "dim7_multiple_rs_HumanEval_12_longest",
                                     "dim1_humaneval_HumanEval_12"], "model": "m"})
    out = S.annotate(long)
    assert out.problem_id.tolist() == ["he/12"] * 3 and "problem_id" not in long


def test_annotate_refuses_an_unknown_task():
    with pytest.raises(KeyError):
        S.annotate(pd.DataFrame({"task_id": ["dim1_nope_1"]}))


# --------------------------------------------------------------------------- development split and folds
@pytest.fixture(scope="module")
def splits():
    return S.load_splits()


def test_splits_csv_reproduces_from_problems_csv_and_the_constants(problems, tmp_path):
    """Pin: if problem ids change, this fails and the redraw becomes an explicit decision."""
    out = tmp_path / "splits.csv"
    S.write_splits(S.split_table(problems, load_table()), out)
    assert out.read_bytes() == S.SPLITS_CSV.read_bytes()


def test_r13_no_problem_content_or_source_key_in_two_of_dev_and_the_folds(problems, splits):
    assert S.r13_violations(problems, splits) == []


def test_r13_check_catches_a_task_moved_to_another_side(problems, splits):
    """Move one task of a dev problem into an eval problem: its source key now sits in two groups,
    which the check that does not use problem ids must catch."""
    dev = set(splits.problem_id[splits.split == S.DEV])
    first_eval = sorted(splits.problem_id[splits.split == S.EVAL])[0]
    leaked = problems.copy()
    i = leaked.index[leaked.problem_id.isin(dev) & leaked.duplicated("source_key", keep=False)][0]
    leaked.loc[i, "problem_id"] = first_eval
    found = S.r13_violations(leaked, splits)
    assert {v["field"] for v in found} >= {"source_key"}
    assert {v["seed"] for v in found} == set(S.FOLD_SEEDS)


def test_labels_partition_the_problems_and_every_eligible_task_is_in_dev_or_eval(problems, splits):
    from router.data.scoring import eligible_families

    assert list(splits.columns) == S.SPLIT_COLUMNS
    assert splits.problem_id.is_unique and set(splits.problem_id) == set(problems.problem_id)
    assert set(splits.split) <= set(S.SPLIT_LABELS)
    t = S.task_splits(problems, splits).merge(load_table()[["task_id", "family", "split_kind"]], on="task_id")
    eligible = (t.split_kind == "id") & t.family.isin(eligible_families())
    assert t.split.notna().all()
    assert t.split[eligible].isin([S.DEV, S.EVAL]).all()
    assert (t.split[t.split_kind == "ood"] == S.OOD).all() and (t.split[t.split_kind == "id"] != S.OOD).all()
    assert set(splits.problem_id[splits.split.isin([S.DEV, S.EVAL])]) == set(t.problem_id[eligible])


def test_folds_exist_only_for_eval_problems_and_differ_in_size_by_at_most_one(splits):
    ev = splits.split == S.EVAL
    for s in S.FOLD_SEEDS:
        col = splits[S.fold_column(s)]
        assert col[ev].notna().all() and col[~ev].isna().all()
        assert set(col[ev]) == set(range(N_FOLDS))
        sizes = col[ev].value_counts()
        assert sizes.max() - sizes.min() <= 1


def test_the_development_split_is_the_same_for_every_fold_seed(splits):
    pool = sorted(splits.problem_id[splits.split.isin([S.DEV, S.EVAL])])
    committed = set(splits.problem_id[splits.split == S.DEV])
    for s in S.FOLD_SEEDS:
        d = S.draw(pool, DEV_SEED, [s])
        assert set(d.problem_id[d.split == S.DEV]) == committed
    assert len(committed) == S.n_dev(len(pool))


def test_fold_seeds_give_different_assignments(splits):
    ev = splits[splits.split == S.EVAL]
    cols = [tuple(ev[S.fold_column(s)]) for s in S.FOLD_SEEDS]
    assert len(set(cols)) == len(cols)


def test_draw_takes_only_problem_ids_and_seeds():
    assert list(inspect.signature(S.draw).parameters) == ["problem_ids", "dev_seed", "fold_seeds"]


def test_draw_does_not_depend_on_input_order():
    ids = [f"p/{i}" for i in range(50)]
    assert S.draw(ids, 0, [0, 1]).equals(S.draw(ids[::-1], 0, [0, 1]))


@pytest.mark.parametrize("n, k", [(0, 0), (5, 1), (12, 2), (13, 3), (3487, 697)])
def test_n_dev_rounds_half_up_exactly(n, k):
    assert S.n_dev(n) == k


def test_splits_module_imports_no_score_source():
    src = Path(S.__file__).read_text()
    for name in ("coderouterbench", "verdicts", "phase5", "prices", "audit", "artefacts"):
        assert f"data.{name}" not in src and f"import {name}" not in src
