"""Router v2 (S4.2; docs/research/S4_spec.md §4, §4.1): the posterior and stopping rule, the
exploration schedule and probe draw, the k = 0 equivalence, exactly 10% exploration per predicted
group, the probe limits, UNKNOWN never a label, probe cost charged, the held-out model's outcomes
read only through probes, and the S2 garbage, true-label and shuffled-label tests."""

import math
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from router.baselines.common import Context, success, true_cells
from router.baselines.run import default_baselines, run_dev, shuffled_check_fails, shuffled_mean_interval
from router.core import exploration as X
from router.core.policy import EXPLORING, OUT_OF_BUDGET, STOPPED, RouterV1, RouterV2
from router.core.stopping import GroupPosterior, logit_sd, prior_from_known
from router.data import verdicts
from router.settings import MAX_PROBES_PER_GROUP, STOP_POSTERIOR_SD
from router.sim import metrics, stream, toy
from router.sim.stream import ReplayView, replay, task_key
from router.sim.verifier import FAIL, PASS, UNKNOWN, ReplayVerifier

HAVE_X13 = verdicts.X13_VERDICTS.exists()
WORDS = {"cruxeval": "predict the output of this function call",
         "mbpp": "write a python function over a list"}


def toy_setup(n=200, seed=7):
    m = toy.matrix(n=n, seed=seed)
    content = pd.Series({t: f"{WORDS[m.family[t]]} n{i % 10} id {t}" for i, t in enumerate(m.tasks)})
    ctx = Context(ReplayVerifier(toy.verdict_table(m)), content=content,
                  problem_of=pd.Series(m.tasks, index=m.tasks))
    folds = pd.Series([i % 5 for i in range(len(m))], index=m.tasks)
    return m, ctx, folds


# -------------------------------------------------------------------------- posterior, stopping
def test_prior_by_hand():
    # rates (0.2, 0.4, 0.6): mean 0.4, variance 0.08 / 3; s = 0.24 / (0.08 / 3) - 1 = 8
    a, b = prior_from_known([0.2, 0.4, 0.6])
    assert (a, b) == pytest.approx((3.2, 4.8))
    a, b = prior_from_known([0.0, 1.0])          # variance 0.25 = mu(1 - mu): s floored at 1
    assert a + b == pytest.approx(1.0)
    a, b = prior_from_known([0.5, 0.5])          # no spread: the strength cap
    assert a + b == pytest.approx(1e6)


def test_logit_sd_by_hand_and_by_simulation():
    assert logit_sd(1, 1) == pytest.approx(math.sqrt(2 * math.pi ** 2 / 6))     # psi1(1) = pi^2/6
    g = np.random.default_rng(0)
    p = g.beta(6.0, 9.0, size=400_000)
    assert logit_sd(6.0, 9.0) == pytest.approx(np.std(np.log(p / (1 - p))), rel=0.01)


def test_updates_ignore_unknown_and_stop_at_the_tolerance():
    post = GroupPosterior(2.0, 2.0, 0.5)
    assert not post.update(UNKNOWN) and (post.alpha, post.beta) == (2.0, 2.0)
    assert post.update(PASS) and post.update(FAIL) and (post.alpha, post.beta) == (3.0, 3.0)
    while not post.stop():
        post.update(PASS)
        post.update(FAIL)
    assert post.sd <= STOP_POSTERIOR_SD < logit_sd(post.alpha - 1, post.beta - 1)


# ---------------------------------------------------------------------- exploration, probes
def test_exploration_is_exactly_every_tenth_request():
    assert X.EVERY == 10
    for n in (0, 9, 10, 19, 20, 101):
        assert sum(X.is_exploration(j) for j in range(1, n + 1)) == n // 10
    with pytest.raises(ValueError):
        X.is_exploration(0)


def test_probe_draw_limits_and_reproducibility():
    groups = pd.Series(["a"] * 12 + ["b"] * 3 + ["c"] * 40, index=[f"t{i:02d}" for i in range(55)])
    avail = pd.Series(True, index=groups.index)
    avail.iloc[0:10] = False                                 # group a: 2 available tasks
    p = X.draw_probes(groups, avail, seed=3)
    assert {g: len(v) for g, v in p.items()} == {"a": 2, "b": 3, "c": MAX_PROBES_PER_GROUP}
    assert all(avail[t] for v in p.values() for t in v)
    assert p == X.draw_probes(groups, avail, seed=3) and p != X.draw_probes(groups, avail, seed=4)
    assert sum(len(v) for v in X.draw_probes(groups, avail, seed=3, cap=4).values()) == 4
    big = pd.Series([f"g{i % 30}" for i in range(600)], index=[f"u{i}" for i in range(600)])
    assert sum(len(v) for v in X.draw_probes(big, pd.Series(True, index=big.index), 0).values()) == 100


# --------------------------------------------------------------------------------- the router
def _fit_fold(b, m, ctx, folds, f=0):
    tr, te = folds.index[folds != f], folds.index[folds == f]
    train, test = m.subset(list(tr)), m.subset(list(te))
    r = b.fit(train, ctx)
    res = replay(r, ReplayView.from_matrix(test, ctx.content.reindex(test.tasks)), ctx.verifier, seed=3)
    return r, res, train, test


def test_cap_zero_is_v1_without_the_new_model():
    m, ctx, folds = toy_setup()
    for M in m.models:
        a = run_dev(RouterV2(M, cap=0), m, ctx, folds, seed=0)
        b = run_dev(RouterV1(), m.without(M), ctx, folds, seed=0)
        cols = ["action", "final_model", "accepted", "cost_usd", "n_calls"]
        pd.testing.assert_frame_equal(a[cols], b[cols])
        assert all(f["chosen"]["n_probes"] == 0 for f in a.attrs["fits"])


@pytest.mark.parametrize("labels", ["ground_truth", "verifier"])
def test_exploration_is_ten_percent_per_predicted_group_while_exploring(labels):
    m, ctx, folds = toy_setup(n=400)
    r, res, _, _ = _fit_fold(RouterV2("mid", labels, cap=10_000), m, ctx, folds)   # keeps exploring
    order = res.tasks.sort_values("position").index
    seen, n_explored = Counter(), 0
    for t in order:
        g, explored, _ = r.tasks[task_key(t)]
        seen[g] += 1
        if explored:
            n_explored += 1
            assert seen[g] % 10 == 0
            first = res.calls[(res.calls.task_id == t) & (res.calls.call_index == 0)]
            assert first.model.iloc[0] == "mid" and bool(first.verifier_ran.iloc[0]) == \
                (first.state.iloc[0] == "signal")
    assert n_explored > 0
    states = {d["state"] for d in r.chosen["groups"].values()}
    assert EXPLORING in states
    for g, d in r.chosen["groups"].items():
        assert d["requests"] == seen[g]
        if d["state"] == EXPLORING:            # never stopped: exactly every tenth request
            assert d["k_explore"] == d["requests"] // 10


@pytest.mark.parametrize("model", ["mid", "strong"])
def test_unknown_never_updates_and_probe_limits_hold(model):
    m, ctx, folds = toy_setup(n=400)
    r, res, train, _ = _fit_fold(RouterV2(model, "verifier", cap=10_000), m, ctx, folds)
    probes = X.draw_probes(true_cells(train), train.available[model], ctx.seed)
    assert all(len(v) <= MAX_PROBES_PER_GROUP for v in probes.values())
    first = res.calls[(res.calls.model == model) & (res.calls.call_index == 0)]
    first = first[np.array([r.tasks[task_key(t)][1] for t in first.task_id], dtype=bool)]
    n_unknown = 0
    for g, d in r.chosen["groups"].items():
        vs = [ctx.verifier.check(t, model).verdict for t in probes.get(g, [])]
        vs += [v for t, v in zip(first.task_id, first.verdict) if r.tasks[task_key(t)][0] == g]
        n_unknown += sum(v == UNKNOWN for v in vs)
        i = r.v.names.index(g)
        a0, b0 = prior_from_known(r.v.a_hat[i, :r.v.n_direct])
        assert d["alpha"] == pytest.approx(a0 + sum(v == PASS for v in vs), abs=1e-5)
        assert d["beta"] == pytest.approx(b0 + sum(v == FAIL for v in vs), abs=1e-5)
        assert d["k_probe"] + d["k_explore"] == len(vs)
    assert n_unknown > 0                 # the toy verifier is mostly no-signal: UNKNOWNs occur


def test_probe_cost_is_charged_and_spread_over_the_scored_tasks():
    """Per fold, the probe cost spread over the scored tasks equals the new model's released cost
    on its probe tasks plus one verifier run per probe that ran (verifier-only pairing)."""
    m, ctx, folds = toy_setup()
    o = run_dev(RouterV2("strong", "verifier"), m, ctx, folds, seed=0)
    for f in sorted(folds.unique()):
        train = m.subset(list(folds.index[folds != f]))
        probes = [t for v in X.draw_probes(true_cells(train), train.available["strong"], ctx.seed).values()
                  for t in v]
        want = sum(float(train.cost.at[t, "strong"]) + ctx.verifier_usd * ctx.verifier.check(t, "strong").ran
                   for t in probes)
        assert len(probes) > 0
        assert o[o.fold == f].probe_usd.sum() == pytest.approx(want, rel=1e-9)
        assert o.attrs["fits"][int(f)]["chosen"]["n_probes"] == len(probes)
    assert (o.cost_usd - (o.model_cost_usd + o.verifier_usd + o.probe_usd)).abs().max() < 1e-12


def test_the_call_cap_stops_exploration_and_sets_out_of_budget():
    m, ctx, folds = toy_setup(n=400)
    r, res, _, _ = _fit_fold(RouterV2("strong", "ground_truth", cap=3), m, ctx, folds)
    assert r.total_calls == 3 and r.chosen["n_probes"] == 3
    assert not (res.calls.model == "strong").any() or \
        all(not r.tasks[task_key(t)][1] for t in res.calls.task_id[res.calls.model == "strong"])
    assert {d["state"] for d in r.chosen["groups"].values()} <= {STOPPED, OUT_OF_BUDGET}


def test_held_out_model_outcomes_are_read_only_through_probes():
    """Garbage in the new model's training outcomes outside its probe tasks leaves v2's decisions
    unchanged (S0.5)."""
    m, ctx, folds = toy_setup()
    tr = folds.index[folds != 0]
    train = m.subset(list(tr))
    probes = {t for v in X.draw_probes(true_cells(train), train.available["strong"], ctx.seed).values()
              for t in v}
    sc = m.score.copy()
    rest = [t for t in tr if t not in probes]
    sc.loc[rest, "strong"] = 1.0 - sc.loc[rest, "strong"]
    cost = m.cost.copy()
    cost.loc[rest, "strong"] = cost.loc[rest, "strong"] * 9.1
    g = toy.with_scores(m, sc)
    g.cost.loc[:, :] = cost.to_numpy()
    a = _fit_fold(RouterV2("strong", "ground_truth"), m, ctx, folds)[1]
    z = _fit_fold(RouterV2("strong", "ground_truth"), g, ctx, folds)[1]
    assert a.calls[["task_id", "model", "verdict"]].equals(z.calls[["task_id", "model", "verdict"]])


VERSIONS = [RouterV2("strong", "ground_truth"), RouterV2("strong", "verifier")]


def _decisions(res):
    return (res.calls[["task_id", "call_index", "model", "verdict"]].to_dict("records"),
            res.tasks[["action", "final_model", "n_calls"]].to_dict("index"))


@pytest.mark.parametrize("b", VERSIONS, ids=lambda b: b.name)
def test_v2_ignores_scored_tasks_scores_and_costs(b):
    m, ctx, folds = toy_setup()
    g = np.random.default_rng(99)
    te = folds.index[folds == 0]
    garbage = [toy.with_scores(m, pd.DataFrame(g.random(m.score.shape), index=m.tasks, columns=m.models)),
               toy.with_scores(m, 1.0 - m.score)]
    scrambled = toy.with_scores(m, m.score)
    scrambled.cost.loc[:, :] = m.cost.sample(frac=1.0, random_state=1).to_numpy() * 7.3 + 0.01
    for s in garbage + [scrambled]:
        stream_m = m.subset(list(m.tasks))
        sc, co = m.score.copy(), m.cost.copy()
        sc.loc[te], co.loc[te] = s.score.loc[te], s.cost.loc[te]    # only the scored fold changes
        z_m = toy.with_scores(m, sc)
        z_m.cost.loc[:, :] = co.to_numpy()
        tr = folds.index[folds != 0]
        train = stream_m.subset(list(tr))
        view_a = ReplayView.from_matrix(m.subset(list(te)), ctx.content.reindex(te))
        view_z = ReplayView.from_matrix(z_m.subset(list(te)), ctx.content.reindex(te))
        a = replay(b.fit(train, ctx), view_a, ctx.verifier, seed=3)
        z = replay(b.fit(train, ctx), view_z, ctx.verifier, seed=3)
        assert _decisions(a) == _decisions(z), b.name


class _Raising(Mapping):
    def __getitem__(self, k):
        raise AssertionError(f"router read true label {k!r}")

    def __iter__(self):
        raise AssertionError("router iterated the true labels")

    def __len__(self):
        raise AssertionError("router inspected the true labels")


@pytest.mark.parametrize("b", VERSIONS, ids=lambda b: b.name)
def test_v2_never_reads_a_true_label(b, monkeypatch):
    m, ctx, folds = toy_setup()
    base = run_dev(b, m, ctx, folds, seed=0)
    orig = stream.TaskHandle
    monkeypatch.setattr(stream, "TaskHandle", lambda key, content, info: orig(key, content, _Raising()))
    guarded = run_dev(b, m, ctx, folds, seed=0)
    pd.testing.assert_frame_equal(base, guarded)


@pytest.mark.parametrize("labels", ["ground_truth", "verifier"])
def test_v2_shuffled_labels_one_sided_toy(labels):
    """New model `mid`: with `strong` known, there is something to learn, so unshuffled v2 beats
    Random clearly; shuffled, it does not rise above Random (one-sided). Holding out `strong`
    instead leaves only the two weaker models known, and no router of them can beat Random over all
    three, so that case does not test learning."""
    b = RouterV2("mid", labels)
    m, ctx, folds = toy_setup(n=150)
    ok = success(m)
    random_expected = float((ok.sum(axis=1) / m.available.sum(axis=1)).mean())
    gain = run_dev(b, m, ctx, folds, seed=0)["accepted"].mean() - random_expected
    shuffled = [run_dev(b, m, ctx, folds, seed=0, shuffle_seed=s)["accepted"].mean() for s in range(10)]
    assert np.mean(shuffled) - random_expected < 0.08, np.mean(shuffled)
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
        ctx = Context(ReplayVerifier.from_x13(), content=join(list(cc.tasks)), problem_of=ts.problem_id)
        _REAL.update(m=cc, ctx=ctx, folds=stats.dev_folds(cc.tasks, 0))
    return _REAL["m"], _REAL["ctx"], _REAL["folds"]


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
@pytest.mark.parametrize("labels", ["ground_truth", "verifier"])
def test_real_dev_v2_garbage_scores_and_shuffled_labels(labels):
    """Headline development split, new model Opus (the best single model): garbage scores on the
    scored fold leave the decisions unchanged; unshuffled v2 beats Random (CI above 0); shuffled,
    the one-sided check passes (reviewer ruling 2026-11-25)."""
    from router.experiments.s2_baselines import N_SHUFFLES
    from router.sim import stats
    m, ctx, folds = real_dev()
    b = RouterV2("claude-opus-4-6", labels)
    g = np.random.default_rng(5)
    te = folds.index[folds == 0]
    sc = m.score.copy()
    sc.loc[te] = g.random((len(te), len(m.models)))
    gm = toy.with_scores(m, sc)
    tr = folds.index[folds != 0]
    a = replay(b.fit(m.subset(list(tr)), ctx), ReplayView.from_matrix(m.subset(list(te)), ctx.content), ctx.verifier, seed=3)
    z = replay(b.fit(m.subset(list(tr)), ctx), ReplayView.from_matrix(gm.subset(list(te)), ctx.content), ctx.verifier, seed=3)
    assert _decisions(a) == _decisions(z)
    rnd = run_dev(next(x for x in default_baselines(m.models) if x.name == "random"), m, ctx, folds, seed=0)
    o = run_dev(b, m, ctx, folds, seed=0)
    f = metrics.paired_frame(o, rnd, ("b", "r"))
    iv = stats.bootstrap(f, stats.mean_diff("accepted_b", "accepted_r"),
                         clusters=ctx.problem_of.reindex(f.index), n_boot=300)
    assert iv.lo > 0, iv
    ok = success(m)
    random_expected = 100 * float((ok.sum(axis=1) / m.available.sum(axis=1)).mean())
    accs = [metrics.acceptance(run_dev(b, m, ctx, folds, seed=0, shuffle_seed=k)) for k in range(N_SHUFFLES)]
    mean, lo, hi = shuffled_mean_interval(accs)
    assert not shuffled_check_fails(lo, random_expected), (random_expected, (mean, lo, hi))
