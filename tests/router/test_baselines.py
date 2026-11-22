"""S2.5 baselines on the replay simulator: no ground truth at routing time, P0.5/P0.6 definitions,
labels, and the shuffled-label check (S2.6)."""

import numpy as np
import pandas as pd
import pytest

from router.baselines import common, lookup
from router.baselines.acrouter import ReleasedACRouter, covered
from router.baselines.cells import CellPredictor, out_of_fold_accuracy, training_pool
from router.baselines.common import ACCEPT, ESCALATE, Context, policy_outcomes, success
from router.baselines.lookup import KMatchedDimensionBest
from router.baselines.run import default_baselines, run_dev, shuffle_models
from router.data import verdicts
from router.sim import metrics, toy
from router.sim.stream import ReplayView, replay, task_key
from router.sim.verifier import ReplayVerifier

HAVE_X13 = verdicts.X13_VERDICTS.exists()
WORDS = {"cruxeval": "predict the output of this function call",
         "mbpp": "write a python function over a list"}


def stub_embed(texts):
    """A deterministic bag-of-words embedding for tests (the real one is `embeddings.embed`)."""
    vocab = sorted({w for t in WORDS.values() for w in t.split()} | {f"n{i}" for i in range(10)})
    out = []
    for t in texts:
        v = np.array([t.split().count(w) for w in vocab], float) + 1e-3
        out.append(v / np.linalg.norm(v))
    return np.array(out)


def toy_setup(n=60):
    m = toy.matrix(n=n)
    content = pd.Series({t: f"{WORDS[m.family[t]]} n{i % 10} id {t}" for i, t in enumerate(m.tasks)})
    rv = ReplayVerifier(toy.verdict_table(m))
    ctx = Context(rv, content=content, problem_of=pd.Series(m.tasks, index=m.tasks),
                  embed=stub_embed)
    folds = pd.Series([i % 5 for i in range(len(m))], index=m.tasks)
    return m, ctx, folds


def toy_baselines(m):
    decisions = pd.Series({t: m.models[i % 3] for i, t in enumerate(m.tasks) if i % 4})
    return default_baselines(m.models) + [
        ReleasedACRouter(decisions),
        KMatchedDimensionBest("strong", k=8),
        KMatchedDimensionBest("strong", k=8, labels="verifier")]


def _decisions(res):
    return (res.calls[["task_id", "call_index", "model", "verdict"]].to_dict("records"),
            res.tasks[["action", "final_model", "n_calls"]].to_dict("index"))


# ------------------------------------------------------------------------------ no ground truth
def _fit_then_replay_twice(b, m, ctx, folds, stream_matrix):
    tr, te = folds.index[folds != 0], folds.index[folds == 0]
    train, test = m.subset(list(tr)), m.subset(list(te))
    other = stream_matrix.subset(list(te))
    content = ctx.content.reindex(test.tasks)
    a = replay(b.fit(train, ctx), ReplayView.from_matrix(test, content), ctx.verifier, seed=3)
    z = replay(b.fit(train, ctx), ReplayView.from_matrix(other, content), ctx.verifier, seed=3)
    return a, z


def test_every_baseline_ignores_scored_tasks_scores():
    """Fit on the training folds with true outcomes, then replay the scored fold with its released
    scores replaced by garbage: every baseline's decisions are identical (S0.13, R16)."""
    m, ctx, folds = toy_setup()
    g = np.random.default_rng(99)
    garbage = [pd.DataFrame(g.random(m.score.shape), index=m.tasks, columns=m.models),
               1.0 - m.score]
    names = []
    for b in toy_baselines(m):
        names.append(b.name)
        for gs in garbage:
            a, z = _fit_then_replay_twice(b, m, ctx, folds, toy.with_scores(m, gs))
            assert _decisions(a) == _decisions(z), b.name
    assert len(names) == len(set(names)) == len(m.models) + 24


def test_every_baseline_ignores_scored_tasks_costs():
    m, ctx, folds = toy_setup()
    scrambled = toy.with_scores(m, m.score)
    scrambled.cost.loc[:, :] = (m.cost.sample(frac=1.0, random_state=1).to_numpy() * 7.3 + 0.01)
    for b in toy_baselines(m):
        a, z = _fit_then_replay_twice(b, m, ctx, folds, scrambled)
        assert _decisions(a) == _decisions(z), b.name


def test_fit_sees_only_training_tasks():
    m, ctx, folds = toy_setup()
    seen = []

    class Spy:
        name, learned, more_information, definition = "spy", True, (), None

        def fit(self, train, c):
            seen.append((set(train.tasks), set(c.exclude_problems)))
            return default_baselines(m.models)[0].fit(train, c)

    run_dev(Spy(), m, ctx, folds, seed=0)
    for f, (tr, excl) in enumerate(seen):
        held = set(folds.index[folds == f])
        assert not (tr & held) and excl == held


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
def test_real_dev_split_garbage_scores_every_baseline():
    """The same test on the development-split headline tasks, real X13 verdicts, real texts."""
    m, ctx, folds = real_dev()
    g = np.random.default_rng(5)
    garbage = toy.with_scores(m, pd.DataFrame(g.random(m.score.shape), index=m.tasks,
                                              columns=m.models))
    for b in default_baselines(m.models) + [ReleasedACRouter()]:
        a, z = _fit_then_replay_twice(b, m, ctx, folds, garbage)
        assert _decisions(a) == _decisions(z), b.name


_REAL = {}


def real_dev():
    if not _REAL:
        from router.baselines import embeddings
        from router.data import splits as S
        from router.data.audit import build_views, load_release
        from router.data.prompts import join
        from router.sim import stats
        h = build_views(load_release())["headline"]
        ts = S.task_splits().set_index("task_id")
        dev = h.subset([t for t in h.tasks if ts.at[t, "split"] == S.DEV])
        cc = dev.subset(list(dev.tasks[dev.available.all(axis=1).to_numpy()]))
        try:
            embeddings.verify()
            embed = embeddings.embed
        except embeddings.EmbeddingPinError:
            embed = stub_embed
        ctx = Context(ReplayVerifier.from_x13(), content=join(list(cc.tasks)),
                      problem_of=ts.problem_id, embed=embed)
        _REAL.update(m=cc, ctx=ctx, folds=stats.dev_folds(cc.tasks, 0))
    return _REAL["m"], _REAL["ctx"], _REAL["folds"]


# ------------------------------------------------------------------------- policy outcomes
def _replayed(train, ctx, pol):
    r = common.PlanRouter(lambda h: pol)
    res = replay(r, ReplayView.from_matrix(train), ctx.verifier, seed=0)
    o = metrics.task_outcomes(res, train)
    return o["accepted"].astype(float).loc[train.tasks].to_numpy(), o.cost_usd.loc[train.tasks].to_numpy()


def test_policy_outcomes_match_replay_toy():
    m, ctx, _ = toy_setup()
    for mode in (ACCEPT, ESCALATE):
        pols = common.candidates(m, mode)
        A, C = policy_outcomes(m, ctx.verifier, pols)
        for j, pol in enumerate(pols):
            a, c = _replayed(m, ctx, pol)
            assert np.array_equal(A[:, j], a), pol.name
            assert np.allclose(C[:, j], c, rtol=0, atol=1e-15), pol.name


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
def test_policy_outcomes_match_replay_real_dev():
    m, ctx, _ = real_dev()
    assert len(common.ladders(m, ESCALATE)) == 84
    for mode in (ACCEPT, ESCALATE):
        pols = common.candidates(m, mode)
        A, C = policy_outcomes(m, ctx.verifier, pols)
        for j in list(range(8)) + list(range(8, len(pols), 7)):
            a, c = _replayed(m, ctx, pols[j])
            assert np.array_equal(A[:, j], a) and np.allclose(C[:, j], c, rtol=1e-12, atol=0)


def test_direct_dispatch_pays_no_verifier_and_ladders_do():
    m, ctx, folds = toy_setup()
    o = run_dev(lookup.CellBest("true"), m, ctx, folds, seed=0)
    assert o.n_verifier_runs.sum() == 0 and (o.verifier_usd == 0).all()
    from router.baselines.cascade import FixedCascade
    o = run_dev(FixedCascade(ESCALATE), m, ctx, folds, seed=0)
    assert o.n_verifier_runs.sum() > 0


# ------------------------------------------------------------------- P0.5 / P0.6 definitions
def _p06():
    pytest.importorskip("sklearn")
    from router.experiments import p0_6_cellbest_bound as p06
    return p06


@pytest.mark.parametrize("seed", range(20))
def test_rules_match_p0_6_on_random_tables_with_ties(seed):
    p06 = _p06()
    g = np.random.default_rng(seed)
    X, K, M = 4, 12, 5
    v = np.round(g.random((X, K)) * 4) / 4          # coarse values: many ties
    c = np.round(g.random((X, K)) * 3 + 1)
    cb = p06.cellbest_table(v, c, M)
    ra, rb = p06.rule_tables(v, c, cb)
    for x in range(X):
        k = lookup.cellbest_column(v[x], c[x], M)
        assert k == cb[x]
        assert lookup.rule_a_column(v[x], c[x], k) == ra[x]
        assert lookup.rule_b_column(v[x], c[x], k) == rb[x]


def test_rule_a_escalated_toy_case():
    """A hand-checked cell: CellBest is direct:strong (acceptance 0.8, cost 4). Rule A takes the
    cheapest candidate with acceptance >= 0.8 (a ladder at 0.8 for 2.5); Rule B the best acceptance
    at cost <= 4 (the 0.9 ladder at 3.5)."""
    v = np.array([0.5, 0.8, 0.8, 0.9, 0.7])
    c = np.array([1.0, 4.0, 2.5, 3.5, 0.5])
    k = lookup.cellbest_column(v, c, 2)
    assert k == 1
    assert lookup.rule_a_column(v, c, k) == 2
    assert lookup.rule_b_column(v, c, k) == 3
    p06 = _p06()
    ra, rb = p06.rule_tables(v[None], c[None], np.array([k]))
    assert (ra[0], rb[0]) == (2, 3)


@pytest.mark.parametrize("seed", range(15))
def test_zero_loss_matches_p0_5(seed):
    from router.experiments import p0_5_recount as p05
    g = np.random.default_rng(seed)
    X, U = 5, 6
    pi = g.dirichlet(np.ones(X))
    v = g.random((X, U))
    c = g.random((X, U)) * 5 + 0.1
    a_ref = int(np.argmax(pi @ v))
    ours = lookup.lp_floor(pi, v, c, 0.0, a_ref)
    theirs, _ = p05.ra_lp_floor(pi, v, c, 0.0, a_ref)
    if isinstance(theirs, tuple):
        assert isinstance(ours, tuple)
        assert np.array_equal(ours[0], theirs[0]) and np.array_equal(ours[1], theirs[1])
        assert ours[2] == pytest.approx(theirs[2])
    else:
        assert np.array_equal(ours, theirs)


def test_ladders_are_p0_5s():
    m, _, _ = toy_setup()
    names = [p.name for p in common.ladders(m, ESCALATE)]
    assert names == ["cheap>mid@escalate", "cheap>strong@escalate", "mid>strong@escalate",
                     "cheap>mid>strong@escalate"]


# ------------------------------------------------------------------------------------ labels
def test_given_more_information_labels_match_what_the_router_reads():
    m, ctx, folds = toy_setup()
    for b in toy_baselines(m):
        o = run_dev(b, m, ctx, folds, seed=0)
        used = {x for f in o.attrs["fits"] for x in f["given_more_information"]}
        assert used == set(b.more_information), b.name
    gmi = {b.name for b in default_baselines(m.models) if b.more_information}
    assert gmi == {"dimension_best", "dimension_best_cost_aware", "source_best", "cellbest_true",
                   "ruleA_accept_true", "ruleB_accept_true", "ruleA_escalate_true",
                   "ruleB_escalate_true", "zero_loss_true"}
    ours = {b.name for b in default_baselines(m.models)
            if b.definition and b.definition.startswith(common.OUR_DEFINITION)}
    assert {"fixed_cascade_accept", "fixed_cascade_escalate", "linucb", "knn",
            "dimension_best_cost_aware", "zero_loss_pred"} <= ours


# ------------------------------------------------------------------------ shuffled labels
def test_shuffle_permutes_whole_cell_records():
    m, ctx, _ = toy_setup()
    s, pv = shuffle_models(m, ctx.verifier, seed=4)
    for t in m.tasks[:12]:
        assert sorted(s.score.loc[t]) == sorted(m.score.loc[t])
        for mdl in m.models:
            src = pv.perm[t][mdl]
            assert s.score.at[t, mdl] == m.score.at[t, src]
            assert s.cost.at[t, mdl] == m.cost.at[t, src]
            assert s.available.at[t, mdl] == m.available.at[t, src]
            if m.available.at[t, src]:
                assert pv.check(t, mdl) == ctx.verifier.check(t, src)


def test_learned_baselines_fall_to_random_when_labels_are_shuffled():
    """Unshuffled, every learned baseline beats Random's expected acceptance clearly; with the
    models' records permuted inside each training task (10 shuffles), each falls to it."""
    m, ctx, folds = toy_setup(n=150)
    ok = success(m)
    random_expected = float((ok.sum(axis=1) / m.available.sum(axis=1)).mean())
    weak_unshuffled = {"fixed_cascade_accept"}   # accepts the cheap rung on UNKNOWN: no gain
    for b in default_baselines(m.models):
        if not b.learned:
            continue
        true_gain = run_dev(b, m, ctx, folds, seed=0)["accepted"].mean() - random_expected
        shuffled = [run_dev(b, m, ctx, folds, seed=0, shuffle_seed=s)["accepted"].mean()
                    for s in range(10)]
        assert abs(np.mean(shuffled) - random_expected) < 0.08, (b.name, np.mean(shuffled))
        if b.name not in weak_unshuffled:
            assert true_gain > 0.15, (b.name, true_gain)


# ------------------------------------------------------------------------- other baselines
def test_k_matched_dimension_best_uses_only_k_probes_of_the_new_model():
    m, ctx, folds = toy_setup()
    b = KMatchedDimensionBest("strong", k=6)
    train = m.subset(list(folds.index[folds != 0]))
    r = b.fit(train, ctx)
    assert r.n_probes == 6 and r.probe_usd > 0
    # garbage in the new model's non-probe training outcomes leaves the table unchanged
    from router.sim.stats import rng
    g = rng(ctx.seed, "probe_draw")
    avail = train.tasks[train.available["strong"].astype(bool).to_numpy()]
    probes = set(pd.Index(sorted(avail))[np.sort(g.choice(len(avail), size=6, replace=False))])
    sc = train.score.copy()
    rest = [t for t in train.tasks if t not in probes]
    sc.loc[rest, "strong"] = 1.0 - sc.loc[rest, "strong"]
    assert b.fit(toy.with_scores(train, sc), ctx).table == r.table


def test_released_acrouter_gives_up_on_uncovered_tasks():
    m, ctx, folds = toy_setup()
    dec = pd.Series({t: "mid" for t in m.tasks[::2]})
    assert list(covered(m.tasks, dec)) == list(m.tasks[::2])
    o = run_dev(ReleasedACRouter(dec), m, ctx, folds, seed=0)
    assert (o.loc[m.tasks[1::2], "action"] == "give_up").all()
    cov = o.loc[m.tasks[::2]]
    assert (cov.final_model.dropna() == "mid").all()


def test_cell_predictor_pool_excludes_scored_problems():
    m, ctx, folds = toy_setup()
    cells = common.true_cells(m)
    held = set(folds.index[folds == 2])
    pool = training_pool(cells, ctx.for_fold(held))
    assert not (set(pool.index) & held) and len(pool) == len(m) - len(held)
    p = CellPredictor(ctx.content.loc[pool.index], pool)
    assert set(p.predict(ctx.content.loc[list(held)])) <= set(cells)
    acc = out_of_fold_accuracy(ctx.content, cells, folds, problem_of=ctx.problem_of)
    assert acc["n"] == len(m) and 0.9 <= acc["accuracy"] <= 1.0


def test_embedding_pin_refuses_a_changed_file(tmp_path):
    from router.baselines import embeddings
    for rel in embeddings.MINILM_SHA256:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"x")
    with pytest.raises(embeddings.EmbeddingPinError, match="sha256"):
        embeddings.verify(tmp_path)
    try:
        embeddings.verify()
    except embeddings.EmbeddingPinError:
        pytest.skip("pinned MiniLM revision not in the local cache")


def test_baselines_draw_from_their_streams():
    m, ctx, folds = toy_setup()
    from router.baselines.static import RandomChoice
    a = run_dev(RandomChoice(), m, ctx, folds, seed=0)
    b = run_dev(RandomChoice(), m, ctx, folds, seed=0)
    c = run_dev(RandomChoice(), m, ctx, folds, seed=1)
    assert a.final_model.equals(b.final_model) and not a.final_model.equals(c.final_model)
    assert task_key("x") != "x"


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
@pytest.mark.parametrize("name", ["best_single", "cellbest_pred", "ruleA_escalate_pred",
                                  "fixed_cascade_escalate"])
def test_real_dev_shuffled_labels_fall_to_random(name):
    """On the development-split headline tasks: unshuffled, the baseline beats Random (paired
    cluster-bootstrap CI above 0); with shuffled labels, the paired CI against Random holds 0."""
    from router.sim import stats
    m, ctx, folds = real_dev()
    b = next(x for x in default_baselines(m.models) if x.name == name)
    rnd = run_dev(next(x for x in default_baselines(m.models) if x.name == "random"), m, ctx,
                  folds, seed=0)
    pid = ctx.problem_of
    for shuffle, expect_gain in ((None, True), (0, False), (1, False)):
        o = run_dev(b, m, ctx, folds, seed=0, shuffle_seed=shuffle)
        f = metrics.paired_frame(o, rnd, ("b", "r"))
        iv = stats.bootstrap(f, stats.mean_diff("accepted_b", "accepted_r"),
                             clusters=pid.reindex(f.index), n_boot=300)
        assert (iv.lo > 0) if expect_gain else (iv.lo <= 0 <= iv.hi), (name, shuffle, iv)
