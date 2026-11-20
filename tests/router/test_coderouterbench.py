"""S1.3: the CodeRouterBench loader on the S1.2 schema. Skips when external/ is absent."""

import pandas as pd
import pytest

from router.data import coderouterbench as crb
from router.data import sources
from router.data.schema import MISSING_TOKEN_RECORD
from router.experiments import p0_5_recount as rc

HAVE_DATA = all((crb.CRB_DIR / f).exists() for f in crb.SPLIT_FILES.values()) \
    and (crb.CRB_DIR / crb.OBS_MATRIX).exists()
HAVE_X13 = HAVE_DATA and rc.X13_VERDICTS.exists()
pytestmark = pytest.mark.skipif(not HAVE_DATA, reason="CodeRouterBench not downloaded to external/")

CELLS = {"probing": 7080 * 8, "id_test": 2919 * 8}
UNAVAILABLE = {"probing": 106, "id_test": 42}     # the 148 missing-token-record rows, per split


@pytest.fixture(scope="module")
def mats():
    return {(s, v): crb.load(s, v) for s in CELLS for v in crb.VIEWS}


@pytest.mark.parametrize("split", CELLS)
@pytest.mark.parametrize("view", crb.VIEWS)
def test_cell_counts_and_no_nan(mats, split, view):
    m = mats[split, view]
    assert m.score.size == CELLS[split] and m.score.shape[1] == 8
    assert len(m.long) == CELLS[split]
    assert not m.score.isna().any().any()
    assert not m.cost.isna().any().any()


def test_paper_view_missing_cells():
    assert crb.missing_cells("id_test") == 881
    assert crb.missing_cells("probing") == 2417


@pytest.mark.parametrize("split", CELLS)
@pytest.mark.parametrize("view", crb.VIEWS)
def test_missing_token_record_cells_are_unavailable_not_rescored(mats, split, view):
    m = mats[split, view]
    off = ~m.available
    assert int(off.sum().sum()) == UNAVAILABLE[split]
    assert set(off.columns[off.any()]) == {"Qwen3-Max"}
    assert set(m.unavailable_reason.to_numpy()[off.to_numpy()]) == {MISSING_TOKEN_RECORD}
    raw = crb.load_long(split)
    hit = raw[raw.cost_source == crb.MISSING_TOKEN_COST_SOURCE]
    flagged = off.stack()
    assert set(map(tuple, hit[["task_id", "model"]].to_numpy())) == set(flagged[flagged].index)
    # Score stays exactly as released (0 at cost 0); nothing is set to NaN.
    for t, mdl, s, c in hit[["task_id", "model", "score", "cost_usd"]].itertuples(index=False):
        assert m.score.loc[t, mdl] == s and m.cost.loc[t, mdl] == c


@pytest.mark.parametrize("split, n_missing", [("probing", 2417), ("id_test", 881)])
def test_paper_view_zero_fills_the_cells_obs_lacks(mats, split, n_missing):
    obs = pd.read_json(crb.CRB_DIR / crb.OBS_MATRIX, typ="series")
    m = mats[split, "paper"]
    absent = pd.DataFrame([[mdl not in obs.get(t, {}) for mdl in m.models] for t in m.tasks],
                          index=m.tasks, columns=m.models)
    assert int(absent.sum().sum()) == n_missing
    assert (m.score.to_numpy()[absent.to_numpy(dtype=bool)] == 0.0).all()


def test_unavailable_cells_are_inside_the_paper_missing_cells(mats):
    obs = pd.read_json(crb.CRB_DIR / crb.OBS_MATRIX, typ="series")
    for split in CELLS:
        off = ~mats[split, "complete"].available
        off = off.stack()
        for t, mdl in off[off].index:
            assert mdl not in obs.get(t, {})


def test_every_task_carries_its_family(mats):
    for (split, _), m in mats.items():
        assert m.family.notna().all()
        assert set(m.family) <= set(sources.FAMILIES)
        assert (m.family == m.tasks.map(sources.family_of)).all()
        assert (m.long.family == m.long.task_id.map(sources.family_of)).all()
        assert set(m.long.split) == {split} and set(m.long.run) == {0}


def test_complete_and_paper_views_share_costs_and_availability(mats):
    for split in CELLS:
        c, p = mats[split, "complete"], mats[split, "paper"]
        pd.testing.assert_frame_equal(c.cost, p.cost)
        pd.testing.assert_frame_equal(c.available, p.available)


@pytest.mark.skipif(not HAVE_X13, reason="X13 verdicts not downloaded to external/")
def test_agrees_with_p0_5_on_missing_token_cells(mats):
    d = rc.load()
    ours = pd.concat([~mats[s, "complete"].available for s in CELLS]).sort_index()
    theirs = d["missing_run"].loc[ours.index, ours.columns]
    pd.testing.assert_frame_equal(ours, theirs, check_names=False)
