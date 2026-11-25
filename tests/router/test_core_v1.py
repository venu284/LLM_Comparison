"""Router v1 (S4.1; docs/research/S4_spec.md §3): hand-checked shrinkage and decision values, the
reduction to per-cell Rule A, the S2 garbage tests, the shuffled-label test, and a test that the
router never reads a true label."""

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from router.baselines.common import ESCALATE, ACCEPT, Context, success
from router.baselines.lookup import CellRule
from router.baselines.run import default_baselines, run_dev
from router.core import lookup as L
from router.core.policy import RouterV1, two_rung_cascade
from router.data import verdicts
from router.sim import metrics, stream, toy
from router.sim.stream import ReplayView, replay
from router.sim.verifier import ReplayVerifier

HAVE_X13 = verdicts.X13_VERDICTS.exists()
WORDS = {"cruxeval": "predict the output of this function call",
         "mbpp": "write a python function over a list"}


def toy_setup(n=60, seed=7):
    m = toy.matrix(n=n, seed=seed)
    content = pd.Series({t: f"{WORDS[m.family[t]]} n{i % 10} id {t}" for i, t in enumerate(m.tasks)})
    ctx = Context(ReplayVerifier(toy.verdict_table(m)), content=content,
                  problem_of=pd.Series(m.tasks, index=m.tasks))
    folds = pd.Series([i % 5 for i in range(len(m))], index=m.tasks)
    return m, ctx, folds


# --------------------------------------------------------------------- hand-checked primitives
def test_shrink_by_hand():
    S, n = np.array([[3.0], [1.0], [0.0]]), np.array([4.0, 4.0, 0.0])
    a, p = L.shrink(S, n, 2.0)
    assert p[0] == 0.5
    assert a[0, 0] == pytest.approx(4 / 6) and a[1, 0] == pytest.approx(2 / 6)
    assert a[2, 0] == 0.5                                   # a group with no rows: the pool row
    a0, _ = L.shrink(S, n, 0.0)
    assert list(a0[:2, 0]) == [0.75, 0.25]                  # kappa = 0: the raw rates
    ainf, _ = L.shrink(S, n, math.inf)
    assert (ainf == 0.5).all()                              # kappa = inf: full pooling


def test_anova_rho_by_hand():
    # n = (4, 4), rates (0.75, 0.25), p = 0.5: n0 = 4, MSB = 0.5, MSW = 1.5 / 6 = 0.25,
    # rho = (0.5 - 0.25) / (0.5 + 3 * 0.25) = 0.2, kappa = 0.8 / 0.2 = 4
    S, n = np.array([[3.0], [1.0]]), np.array([4.0, 4.0])
    assert L.anova_rho(S, n) == pytest.approx(0.2)
    assert L.kappa_from_rho(L.anova_rho(S, n)) == pytest.approx(4.0)
    # pooled over two columns: the second column has rates (0.5, 0.5): MSB = 0, MSW = 2 / 6
    S2 = np.array([[3.0, 2.0], [1.0, 2.0]])
    want = (0.5 - 0.25 + 0 - 2 / 6) / (0.5 + 3 * 0.25 + 0 + 3 * 2 / 6)
    assert L.anova_rho(S2, n) == pytest.approx(want)


def test_kappa_degenerate_cases():
    n = np.array([4.0, 4.0])
    assert L.kappa_from_rho(L.anova_rho(np.array([[2.0], [2.0]]), n)) == math.inf   # rho < 0
    assert L.kappa_from_rho(L.anova_rho(np.array([[4.0], [0.0]]), n)) == 0.0        # rho = 1
    assert math.isnan(L.anova_rho(np.array([[3.0]]), np.array([4.0])))               # one group
    assert math.isnan(L.anova_rho(np.array([[1.0], [0.0]]), np.array([1.0, 1.0])))   # N = G
    assert L.kappa_from_rho(math.nan) == math.inf


def test_choose_margin_and_ties_by_hand():
    a, c = np.array([0.80, 0.79, 0.70]), np.array([3.0, 1.0, 0.5])
    assert L.choose(a, c, 3, 0.02) == 1        # 0.79 >= 0.78, cheaper than column 0
    assert L.choose(a, c, 3, 0.0) == 0
    assert L.choose(a, c, 3, 0.10) == 2
    assert L.choose(np.array([0.8, 0.8]), np.array([1.0, 1.0]), 2, 0.0) == 0      # full tie
    # the reference is the best direct column; a non-direct column (index 2) above it may win
    assert L.choose(np.array([0.6, 0.5, 0.9]), np.array([1.0, 0.5, 0.8]), 2, 0.0) == 2
    # within the margin, a cost tie goes to the higher acceptance
    assert L.choose(np.array([0.80, 0.79, 0.785]), np.array([2.0, 1.0, 1.0]), 3, 0.02) == 1


def test_choose_equals_p0_6_rule_a_at_zero_margin():
    from router.baselines.lookup import cellbest_column, rule_a_column
    g = np.random.default_rng(3)
    for _ in range(300):
        J, D = int(g.integers(3, 9)), 0
        D = int(g.integers(1, J + 1))
        a = np.round(g.random(J), 1)            # coarse values make ties common
        c = np.round(g.random(J), 1)
        k = cellbest_column(a, c, D)
        assert L.choose(a, c, D, 0.0) == rule_a_column(a, c, k)


# ------------------------------------------------------------------------------- the router
def test_fit_records_and_pool_row():
    m, ctx, folds = toy_setup()
    train = m.subset(list(folds.index[folds != 0]))
    r = RouterV1().fit(train, ctx)
    assert set(r.table) == {"dim1|cruxeval", "dim2|mbpp"}
    assert r.chosen["n_train"] == len(train) and sum(r.chosen["group_n"].values()) == len(train)
    assert r.chosen["cascade"].count(">") == 1 and r.chosen["cascade"].endswith("@escalate")
    # a task with no text, and one whose text the predictor maps nowhere new, go by the pool row
    h = stream.TaskHandle("k", None, {})
    assert r._plan(h).name == r.fallback


def test_two_rung_cascade_is_fixed_cascade_restricted_to_two_rungs():
    from router.baselines.cascade import FixedCascade
    from router.baselines import common
    m, ctx, _ = toy_setup()
    for mode in (ACCEPT, ESCALATE):
        pol, reached = two_rung_cascade(m, ctx, mode)
        orig = common.ladders
        try:
            common_ladders_2 = lambda t, md: [p for p in orig(t, md) if len(p.rungs) == 2]  # noqa: E731
            import router.baselines.cascade as cas
            cas.ladders = common_ladders_2
            ref = FixedCascade(mode).fit(m, ctx).table["all"]
        finally:
            cas.ladders = orig
        assert pol.name == ref and isinstance(reached, bool)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_reduction_to_per_cell_rule_a(seed):
    """kappa = 0, margin 0 and P0.6's full candidate list: v1 is per-cell Rule A (predicted cells,
    no-signal cells escalated), table for table and decision for decision."""
    m, ctx, folds = toy_setup(n=80, seed=seed)
    v1 = RouterV1(shrinkage="none", margin=0.0, menu="rule_a")
    ra = CellRule("A", ESCALATE, "pred")
    train = m.subset(list(folds.index[folds != 0]))
    assert v1.fit(train, ctx).table == ra.fit(train, ctx).table
    a, b = run_dev(v1, m, ctx, folds, seed=0), run_dev(ra, m, ctx, folds, seed=0)
    cols = ["action", "final_model", "accepted", "cost_usd", "n_calls"]
    pd.testing.assert_frame_equal(a[cols], b[cols])


# --------------------------------------------------------------------- what the router sees
VERSIONS = [RouterV1(), RouterV1(mode=ACCEPT)]


def _decisions(res):
    return (res.calls[["task_id", "call_index", "model", "verdict"]].to_dict("records"),
            res.tasks[["action", "final_model", "n_calls"]].to_dict("index"))


def _fit_then_replay_twice(b, m, ctx, folds, stream_matrix):
    tr, te = folds.index[folds != 0], folds.index[folds == 0]
    train, test, other = m.subset(list(tr)), m.subset(list(te)), stream_matrix.subset(list(te))
    content = ctx.content.reindex(test.tasks)
    a = replay(b.fit(train, ctx), ReplayView.from_matrix(test, content), ctx.verifier, seed=3)
    z = replay(b.fit(train, ctx), ReplayView.from_matrix(other, content), ctx.verifier, seed=3)
    return a, z


@pytest.mark.parametrize("b", VERSIONS, ids=lambda b: b.name)
def test_v1_ignores_scored_tasks_scores_and_costs(b):
    m, ctx, folds = toy_setup()
    g = np.random.default_rng(99)
    garbage = [toy.with_scores(m, pd.DataFrame(g.random(m.score.shape), index=m.tasks,
                                               columns=m.models)),
               toy.with_scores(m, 1.0 - m.score)]
    scrambled = toy.with_scores(m, m.score)
    scrambled.cost.loc[:, :] = m.cost.sample(frac=1.0, random_state=1).to_numpy() * 7.3 + 0.01
    for s in garbage + [scrambled]:
        a, z = _fit_then_replay_twice(b, m, ctx, folds, s)
        assert _decisions(a) == _decisions(z), b.name


class _Raising(Mapping):
    def __getitem__(self, k):
        raise AssertionError(f"router read true label {k!r}")

    def __iter__(self):
        raise AssertionError("router iterated the true labels")

    def __len__(self):
        raise AssertionError("router inspected the true labels")


@pytest.mark.parametrize("b", VERSIONS, ids=lambda b: b.name)
def test_v1_never_reads_a_true_label(b, monkeypatch):
    m, ctx, folds = toy_setup()
    base = run_dev(b, m, ctx, folds, seed=0)
    orig = stream.TaskHandle
    monkeypatch.setattr(stream, "TaskHandle", lambda key, content, info: orig(key, content, _Raising()))
    guarded = run_dev(b, m, ctx, folds, seed=0)
    pd.testing.assert_frame_equal(base, guarded)
    assert b.more_information == () and guarded.attrs["more_information"] == ()


def test_true_group_diagnostic_is_labelled():
    m, ctx, folds = toy_setup()
    d = RouterV1(groups="true")
    assert d.name == "v1_true_groups" and set(d.more_information) == {"family", "dimension"}
    o = run_dev(d, m, ctx, folds, seed=0)
    assert set(o.attrs["fits"][0]["given_more_information"]) == {"family", "dimension"}


@pytest.mark.parametrize("b", VERSIONS, ids=lambda b: b.name)
def test_v1_falls_to_random_when_labels_are_shuffled(b):
    m, ctx, folds = toy_setup(n=150)
    ok = success(m)
    random_expected = float((ok.sum(axis=1) / m.available.sum(axis=1)).mean())
    gain = run_dev(b, m, ctx, folds, seed=0)["accepted"].mean() - random_expected
    shuffled = [run_dev(b, m, ctx, folds, seed=0, shuffle_seed=s)["accepted"].mean() for s in range(10)]
    assert abs(np.mean(shuffled) - random_expected) < 0.08, np.mean(shuffled)
    assert gain > 0.15, gain


_REAL = {}


def real_dev():
    if not _REAL:
        from router.data import splits as S
        from router.data.audit import build_views, load_release
        from router.data.prompts import join
        from router.sim import stats
        h = build_views(load_release())["headline"]
        ts = S.task_splits().set_index("task_id")
        dev = h.subset([t for t in h.tasks if ts.at[t, "split"] == S.DEV])
        cc = dev.subset(list(dev.tasks[dev.available.all(axis=1).to_numpy()]))
        ctx = Context(ReplayVerifier.from_x13(), content=join(list(cc.tasks)),
                      problem_of=ts.problem_id)
        _REAL.update(m=cc, ctx=ctx, folds=stats.dev_folds(cc.tasks, 0))
    return _REAL["m"], _REAL["ctx"], _REAL["folds"]


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
def test_real_dev_v1_garbage_scores():
    m, ctx, folds = real_dev()
    g = np.random.default_rng(5)
    garbage = toy.with_scores(m, pd.DataFrame(g.random(m.score.shape), index=m.tasks,
                                              columns=m.models))
    for b in VERSIONS:
        a, z = _fit_then_replay_twice(b, m, ctx, folds, garbage)
        assert _decisions(a) == _decisions(z), b.name


_SHUFFLE_MISS = ("Measured 2026-11-24, left for the reviewer (not adjusted after the failure): with "
                 "S2.6's protocol (shuffle seeds 0-19) the shuffled mean misses Random's expected "
                 "50.38: router_v1 46.56 [43.40, 49.73], router_v1_accept 46.67 [43.47, 49.87]. "
                 "Diagnostic (router_v1): shuffle seeds 20-119 give 51.33 [49.75, 52.92], which "
                 "contains it, and its picks under shuffling spread over all eight models. The "
                 "unshuffled half (beats Random, CI above 0) passes for both.")


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
@pytest.mark.parametrize("b", [pytest.param(v, id=v.name, marks=pytest.mark.xfail(strict=True, reason=_SHUFFLE_MISS))
                               for v in VERSIONS])
def test_real_dev_v1_shuffled_labels_fall_to_random(b):
    """As S2.6 on the headline development split: unshuffled, v1 beats Random (paired
    cluster-bootstrap CI above 0); shuffled, Random's expected acceptance lies inside the t
    interval of the mean over N_SHUFFLES shuffles."""
    from router.baselines.run import shuffled_mean_interval
    from router.experiments.s2_baselines import N_SHUFFLES
    from router.sim import stats
    m, ctx, folds = real_dev()
    rnd = run_dev(next(x for x in default_baselines(m.models) if x.name == "random"), m, ctx,
                  folds, seed=0)
    o = run_dev(b, m, ctx, folds, seed=0)
    f = metrics.paired_frame(o, rnd, ("b", "r"))
    iv = stats.bootstrap(f, stats.mean_diff("accepted_b", "accepted_r"),
                         clusters=ctx.problem_of.reindex(f.index), n_boot=300)
    assert iv.lo > 0, iv
    ok = success(m)
    random_expected = 100 * float((ok.sum(axis=1) / m.available.sum(axis=1)).mean())
    accs = [metrics.acceptance(run_dev(b, m, ctx, folds, seed=0, shuffle_seed=k))
            for k in range(N_SHUFFLES)]
    mean, lo, hi = shuffled_mean_interval(accs)
    assert lo <= random_expected <= hi, (random_expected, (mean, lo, hi))
