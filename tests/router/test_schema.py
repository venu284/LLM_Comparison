"""S1.2: long-format storage behind OutcomeMatrix. No external data needed."""

import pandas as pd
import pytest

from router.data.schema import (EMPTY_RESPONSE, LONG_COLUMNS, MISSING_TOKEN_RECORD, OutcomeMatrix,
                                long_frame)


def toy_long() -> pd.DataFrame:
    rows = []
    for i, t in enumerate(["t1", "t2", "t3"]):
        for j, m in enumerate(["a", "b", "c"]):
            rows.append({"task_id": t, "model": m, "run": 0, "score": float((i + j) % 2),
                         "cost_usd": 0.1 * (j + 1), "input_tokens": 10, "output_tokens": 20,
                         "family": "fam_x" if t != "t3" else "fam_y", "dimension": "d",
                         "split": "probing"})
    df = pd.DataFrame(rows)
    hit = (df.task_id == "t2") & (df.model == "b")
    df["available"] = ~hit
    df["unavailable_reason"] = None
    df.loc[hit, "unavailable_reason"] = MISSING_TOKEN_RECORD
    return df


@pytest.fixture
def m() -> OutcomeMatrix:
    return OutcomeMatrix.from_long(toy_long(), name="toy")


def test_long_frame_has_every_column_and_leaves_deferred_ones_empty():
    df = long_frame(toy_long())
    assert list(df.columns) == LONG_COLUMNS
    assert df.problem_id.isna().all() and df.score_type.isna().all() and df.eligible.isna().all()


def test_from_long_derives_wide_views(m):
    assert m.score.shape == (3, 3) and m.models == ["a", "b", "c"]
    assert m.cost.loc["t1", "c"] == pytest.approx(0.3)
    assert m.available.sum().sum() == 8
    assert not m.available.loc["t2", "b"]
    assert m.unavailable_reason.loc["t2", "b"] == MISSING_TOKEN_RECORD
    assert m.unavailable_reason.loc["t1", "a"] is None
    assert m.family.to_dict() == {"t1": "fam_x", "t2": "fam_x", "t3": "fam_y"}
    assert m.score.loc["t2", "b"] == toy_long().set_index(["task_id", "model"]).score["t2", "b"]   # kept, not imputed


def _assert_consistent(sub: OutcomeMatrix, full: OutcomeMatrix, tasks, models):
    tasks = pd.Index(tasks)
    for attr in ("score", "cost", "available", "unavailable_reason"):
        pd.testing.assert_frame_equal(getattr(sub, attr), getattr(full, attr).loc[tasks, models],
                                      check_names=False)
    pd.testing.assert_series_equal(sub.dimension, full.dimension.loc[tasks], check_names=False)
    pd.testing.assert_series_equal(sub.source, full.source.loc[tasks], check_names=False)
    assert sub.name == full.name
    assert set(sub.long.task_id) == set(tasks) and set(sub.long.model) == set(models)
    assert len(sub.long) == len(tasks) * len(models)
    assert list(sub.long.columns) == LONG_COLUMNS
    back = OutcomeMatrix.from_long(sub.long)
    for attr in ("score", "cost", "available", "unavailable_reason"):
        pd.testing.assert_frame_equal(getattr(back, attr), getattr(sub, attr), check_names=False)


def test_subset_carries_every_field(m):
    sub = m.subset(["t2", "t3"])
    _assert_consistent(sub, m, ["t2", "t3"], ["a", "b", "c"])
    assert sub.unavailable_reason.loc["t2", "b"] == MISSING_TOKEN_RECORD


def test_without_carries_every_field(m):
    sub = m.without("a")
    _assert_consistent(sub, m, ["t1", "t2", "t3"], ["b", "c"])
    assert not sub.available.loc["t2", "b"]
    assert m.without("b").available.all().all()


def test_mark_unavailable_returns_a_new_matrix(m):
    before = m.available.copy()
    out = m.mark_unavailable([("t1", "a"), ("t2", "b"), ("zz", "a"), ("t1", "zz")], EMPTY_RESPONSE)
    pd.testing.assert_frame_equal(m.available, before)           # original untouched
    assert not out.available.loc["t1", "a"]
    assert out.unavailable_reason.loc["t1", "a"] == EMPTY_RESPONSE
    assert out.unavailable_reason.loc["t2", "b"] == MISSING_TOKEN_RECORD   # first reason kept
    assert out.available.sum().sum() == 7
    row = out.long[(out.long.task_id == "t1") & (out.long.model == "a")].iloc[0]
    assert not row.available and row.unavailable_reason == EMPTY_RESPONSE
    pd.testing.assert_frame_equal(out.score, m.score)
    with pytest.raises(ValueError):
        m.mark_unavailable([("t1", "a")], "bogus")


def test_wide_constructor_still_works_and_defaults_to_available():
    idx = pd.Index(["t1", "t2"])
    score = pd.DataFrame({"x": [1.0, 0.0]}, index=idx)
    w = OutcomeMatrix(score, score * 0.1, pd.Series("d", index=idx), pd.Series("s", index=idx), "w")
    assert w.available.all().all() and w.long is None
    assert w.unavailable_reason.isna().all().all()
    assert len(w.subset(["t2"]).without("x").models) == 0
    marked = w.mark_unavailable([("t1", "x")], EMPTY_RESPONSE)
    assert not marked.available.loc["t1", "x"] and marked.long is None


def test_reason_and_availability_must_agree():
    df = toy_long()
    df.loc[0, "unavailable_reason"] = EMPTY_RESPONSE      # reason on an available cell
    with pytest.raises(ValueError):
        OutcomeMatrix.from_long(df)
    df = toy_long()
    df.loc[df.unavailable_reason.notna(), "unavailable_reason"] = "bogus"
    with pytest.raises(ValueError):
        OutcomeMatrix.from_long(df)


def test_several_runs_need_a_run_choice():
    df = pd.concat([toy_long(), toy_long().assign(run=1)], ignore_index=True)
    with pytest.raises(ValueError):
        OutcomeMatrix.from_long(df)
    assert OutcomeMatrix.from_long(df, run=1).score.shape == (3, 3)
