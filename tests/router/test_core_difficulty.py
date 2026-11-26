"""S4.4 difficulty from the prompt (docs/research/S4_spec.md §6.2): ridge recovers a linear b from
an injected embedding, the grouped CV never splits a problem, predictions depend on the text
only, the S4.6 guard, the optional family classifier, and one test on the pinned MiniLM."""

import inspect

import numpy as np
import pandas as pd
import pytest

from router.baselines import embeddings
from router.core import difficulty as D
from router.core.guards import TransductiveFitError
from router.sim.stats import rng

DIM = 16


def setup(seed=0, n_problems=100, per_problem=3, noise=0.2):
    g = rng(seed, "s4_sim")
    ids = [f"t{i:04d}" for i in range(n_problems * per_problem)]
    problem_of = pd.Series({t: f"P{i // per_problem:03d}" for i, t in enumerate(ids)})
    texts = pd.Series({t: f"task text {t}" for t in ids})
    vec = {texts[t]: g.normal(0, 1, DIM) for t in ids}
    w = g.normal(0, 1, DIM)
    b_true = pd.Series({t: float(vec[texts[t]] @ w) for t in ids})
    b = b_true + g.normal(0, noise, len(ids))
    calls = []

    def embed(xs):
        calls.append(list(xs))
        return np.stack([vec[x] for x in xs]) if len(xs) else np.zeros((0, DIM))
    return dict(ids=ids, problem_of=problem_of, texts=texts, b=b, b_true=b_true, embed=embed,
                calls=calls, vec=vec)


def split(s, n_scored_problems=20):
    probs = sorted(set(s["problem_of"]))
    scored_p = set(probs[-n_scored_problems:])
    train = [t for t in s["ids"] if s["problem_of"][t] not in scored_p]
    scored = [t for t in s["ids"] if s["problem_of"][t] in scored_p]
    return train, scored


def test_ridge_recovers_a_linear_difficulty_on_unseen_problems():
    s = setup()
    train, scored = split(s)
    m = D.fit(train, s["b"], s["texts"], scored_ids=scored, problem_of=s["problem_of"], embed=s["embed"])
    pred = m.predict([s["texts"][t] for t in scored])
    assert m.alpha in D.ALPHAS and m.n_train == len(train)
    assert np.corrcoef(pred, s["b_true"][scored])[0, 1] > 0.95


def test_grouped_cv_never_splits_a_problem():
    s = setup()
    train, scored = split(s)
    probs = s["problem_of"].reindex(train).tolist()
    folds = D.grouped_folds(probs)
    assert len(set(folds)) == D.N_CV
    for p in set(probs):
        assert len({f for f, q in zip(folds, probs) if q == p}) == 1
    cv = D.grouped_cv(train, s["b"], s["texts"], scored_ids=scored, problem_of=s["problem_of"],
                      embed=s["embed"])
    assert cv["pred"].notna().all() and cv["r2"] > 0.8 and cv["spearman"] > 0.9
    fp = pd.DataFrame({"fold": cv["folds"], "problem": s["problem_of"].reindex(cv["folds"].index)})
    assert (fp.groupby("problem").fold.nunique() == 1).all()


def test_features_are_the_embedding_of_the_text_only():
    s = setup()
    train, scored = split(s)
    params = set(inspect.signature(D.fit).parameters)
    assert params == {"train_ids", "b", "texts", "scored_ids", "problem_of", "embed"}  # no labels
    m = D.fit(train, s["b"], s["texts"], scored_ids=scored, problem_of=s["problem_of"], embed=s["embed"])
    assert all(x.startswith("task text ") for call in s["calls"] for x in call)
    xs = [s["texts"][t] for t in scored[:5]]
    np.testing.assert_allclose(m.predict(xs), s["embed"](xs) @ m.coef + m.intercept)
    # the same text gives the same b, whatever task, family or problem it belongs to
    assert m.predict([xs[0], xs[0]])[0] == m.predict([xs[0], xs[0]])[1]
    out = m.predict([xs[0], None, ""])
    assert np.isfinite(out[0]) and np.isnan(out[1:]).all()


def test_tasks_without_text_are_left_out_of_the_fit():
    s = setup()
    train, scored = split(s)
    texts = s["texts"].copy()
    texts[train[0]] = None
    m = D.fit(train, s["b"], texts, scored_ids=scored, problem_of=s["problem_of"], embed=s["embed"])
    assert m.n_train == len(train) - 1


def test_the_s46_guard_refuses_scored_tasks_and_shared_problems():
    s = setup()
    train, scored = split(s)
    kw = dict(problem_of=s["problem_of"], embed=s["embed"])
    with pytest.raises(TransductiveFitError, match="scored task"):
        D.fit(train + scored[:1], s["b"], s["texts"], scored_ids=scored, **kw)
    pof = s["problem_of"].copy()
    pof[train[0]] = pof[scored[0]]
    with pytest.raises(TransductiveFitError, match="problem"):
        D.fit(train, s["b"], s["texts"], scored_ids=scored, problem_of=pof, embed=s["embed"])
    with pytest.raises(TransductiveFitError):
        D.grouped_cv(train + scored[:1], s["b"], s["texts"], scored_ids=scored, **kw)


def test_family_classifier_from_the_prompt():
    g = rng(3, "s4_sim")
    centres = {"cruxeval": g.normal(0, 1, DIM), "mbpp": g.normal(0, 1, DIM)}
    texts, labels, vec = [], [], {}
    for i in range(200):
        fam = "cruxeval" if i % 2 else "mbpp"
        x = f"prompt {i}"
        vec[x] = centres[fam] + g.normal(0, 0.3, DIM)
        texts.append(x)
        labels.append(fam)
    clf = D.FamilyClassifier(embed=lambda xs: np.stack([vec[x] for x in xs])).fit(texts[:150], labels[:150])
    assert (clf.predict(texts[150:]) == np.array(labels[150:])).mean() > 0.95


@pytest.mark.skipif(not embeddings.MINILM_DIR.is_dir(), reason="pinned MiniLM not downloaded")
def test_the_pinned_minilm_embeds_content_text():
    texts = pd.Series({f"t{i}": f"def f{i}(xs):\n    return sorted(xs)[{i % 3}]" for i in range(12)})
    E = D._default_embed(list(texts))
    assert E.shape == (12, 384)
    np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-5)
    b = pd.Series(np.arange(12, dtype=float), index=texts.index)
    problem_of = pd.Series({t: f"P{i // 2}" for i, t in enumerate(list(texts.index) + ["s"])})
    m = D.fit(list(texts.index), b, texts, scored_ids=["s"], problem_of=problem_of)
    assert np.isfinite(m.predict(["def g(xs):\n    return xs"])).all()
