"""S3.7 resampling (spec D10 (c)) and S3.4 k-matched bookkeeping (spec D9), on toy data."""

import numpy as np
import pandas as pd
import pytest

from router.baselines.common import Context
from router.baselines.lookup import KMatchedDimensionBest
from router.experiments import s3_kmatched as K
from router.experiments import s3_power as P
from router.sim import toy
from router.sim.verifier import ReplayVerifier


def _toy():
    m = toy.matrix(n=30)
    m = m.subset([t for t in m.tasks if t not in ("t1", "t3")])        # complete case, X13 records
    problems = pd.Series({t: f"p{int(t[1:]) // 2}" for t in m.tasks})  # pairs of tasks share a problem
    return m, problems, ReplayVerifier(toy.verdict_table(m, no_record=()))


def test_resample_sizes_copies_and_shared_folds(monkeypatch):
    monkeypatch.setattr(P, "N_EVAL_PROBLEMS", 40)
    m, problems, _ = _toy()
    mm, folds, orig, cluster = P.resample(m, problems, 0, 0)
    assert cluster.nunique() == 40                                        # one cluster per draw
    assert len(mm) == sum(int((problems == p).sum()) for p in cluster.str.split("#").str[0]
                          .groupby(cluster).first())
    # every copy keeps its original's row
    for c in list(mm.tasks)[:10]:
        assert mm.score.loc[c].equals(m.score.loc[orig[c]].rename(c))
    # all draws of one original problem share a fold
    prob = pd.Series({c: problems[o] for c, o in orig.items()})
    assert (folds.groupby(prob).nunique() == 1).all()
    assert sorted(folds.unique()) == [0, 1, 2, 3, 4]


def test_resample_is_seeded():
    m, problems, _ = _toy()
    a = P.resample(m, problems, 1, 3)
    b = P.resample(m, problems, 1, 3)
    c = P.resample(m, problems, 1, 4)
    assert list(a[0].tasks) == list(b[0].tasks) and a[1].equals(b[1])
    assert list(a[0].tasks) != list(c[0].tasks)


def test_copy_verifier_maps_to_originals():
    m, problems, v = _toy()
    mm, folds, orig, cluster = P.resample(m, problems, 0, 0)
    cv = P.CopyVerifier(v, orig)
    for c in list(mm.tasks)[:20]:
        assert cv.check(c, "mid") == v.check(orig[c], "mid")
    assert cv.empty_cells() == set()


def test_probe_labels_match_the_router():
    m, problems, v = _toy()
    for k in (3, 8):
        for lab in K.LABELS:
            r = KMatchedDimensionBest("strong", k, lab).fit(m, Context(v, seed=2))
            p = K.probe_labels(m, "strong", k, 2, v)
            assert p["n_probes"] == r.n_probes == k
            assert p["probe_usd"] == pytest.approx(r.probe_usd)
            assert 0 <= p["n_verifier_labels"] <= k


def test_unreachable_k_is_refused():
    m, problems, v = _toy()
    folds = pd.Series([i % 5 for i in range(len(m))], index=m.tasks)
    n = K.min_training_availability(m, folds, "mid")
    K.require_reachable(m, folds, "mid", n)
    with pytest.raises(K.Unreachable):
        K.require_reachable(m, folds, "mid", n + 1)


def test_e0_power_by_hand():
    from scipy.stats import norm
    z = norm.ppf(0.975)
    # the CI lower bound sits exactly at -2 when gap - h = -2: power 1/2
    assert P.e0_power(-1.0, 1.0) == pytest.approx(0.5)
    assert P.e0_power(0.0, 1.0) == pytest.approx(norm.cdf(z))
    assert P.e0_power(0.0, 0.5) > P.e0_power(0.0, 1.0) > P.e0_power(-1.0, 1.0)


def test_posthoc_shift_power_rows_and_width_check_by_hand():
    # four draws; resample-world mean gap -1.5; development gap -1.0
    d = pd.DataFrame({"seed": [0, 0, 1, 1], "gap": [-1.0, -2.0, -1.0, -2.0],
                      "lo": [-2.5, -3.6, -1.9, -3.2], "hi": [0.5, -0.5, 0.1, -0.8]})
    d["pass_at_development_gap"] = d.lo >= P.GAP_MIN
    d["pass_at_gap_0"] = (d.lo - (-1.0)) >= P.GAP_MIN
    d["pass_at_gap_0_resample_mean_posthoc"] = (d.lo - d.gap.mean()) >= P.GAP_MIN
    r = P.power_row(d)
    assert r["n"] == 4
    assert r["power_at_development_gap"] == pytest.approx(0.25)                 # -1.9 only
    assert r["power_at_gap_0"] == pytest.approx(0.5)                            # -1.5, -0.9
    assert r["power_at_gap_0_resample_mean_posthoc"] == pytest.approx(0.75)     # -1.0, -0.4, -1.7
    assert r["median_half_width"] == pytest.approx(1.35)                        # of 1.5, 1.55, 1.0, 1.2
    proj = {"development_gap": -1.0, "projected_half_width": 0.5,
            "power_at_development_gap": P.e0_power(-1.0, 0.5), "power_at_gap_0": P.e0_power(0.0, 0.5)}
    s = P.side_by_side(proj, d, float(d.gap.mean()))
    assert s["resampling_c_seed_0"]["n"] == 2 and s["resample_mean_gap"] == pytest.approx(-1.5)
    w = s["width_check_e0_at_resampled_half_width"]
    assert w["half_width"] == pytest.approx(1.35)
    assert w["power_at_development_gap"] == pytest.approx(P.e0_power(-1.0, 1.35))
    assert w["power_at_gap_0"] == pytest.approx(P.e0_power(0.0, 1.35))
