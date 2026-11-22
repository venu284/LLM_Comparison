"""S2.4: stream ids, the paired and cluster bootstrap, seeds, the S0.8 label, and the nested
cross-fitting driver (S0.6, O7, R13, R14)."""

import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.data import splits as S
from router.settings import GATE_SEED, N_FOLDS, N_INNER_FOLDS
from router.sim import stats

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------------------- stream ids
def test_stream_ids_are_in_s2_range_unique_and_documented():
    ids = list(stats.STREAM_IDS.values())
    assert len(set(ids)) == len(ids)
    assert all(2 <= i <= 99 for i in ids)
    table = dict(re.findall(r"^\| (\d+) \| `(\w+)` \|", inspect.getdoc(stats), flags=re.M))
    assert {name: int(i) for i, name in table.items()} == stats.STREAM_IDS
    for bad in (0, 1, 100, 250):
        with pytest.raises(ValueError):
            stats.rng(0, bad)
    a = stats.rng(4, "probe_draw").random(3)
    b = np.random.default_rng(np.random.SeedSequence([4, 3])).random(3)
    assert np.array_equal(a, b)


def test_sim_draws_only_through_stats_rng():
    for f in (REPO / "router" / "sim").glob("*.py"):
        if f.name in ("stats.py", "toy.py"):
            continue
        src = f.read_text()
        assert "default_rng(" not in src and "np.random.seed" not in src, f.name


# -------------------------------------------------------------------------------- bootstrap
def _frame(n_problems=40, per=3, seed=0):
    g = np.random.default_rng(seed)
    rows = []
    for p in range(n_problems):
        for k in range(1 + p % per):
            rows.append({"task": f"p{p}_{k}", "problem_id": f"p{p}", "a": g.random() < 0.6,
                         "b": g.random() < 0.5, "ca": g.random(), "cb": 1 + g.random()})
    return pd.DataFrame(rows).set_index("task")


def test_bootstrap_reproduces_under_a_seed():
    f = _frame()
    st = stats.mean_diff("a", "b")
    x = stats.bootstrap(f, st, n_boot=300, seed=1)
    y = stats.bootstrap(f, st, n_boot=300, seed=1)
    z = stats.bootstrap(f, st, n_boot=300, seed=2)
    assert np.array_equal(x.draws, y.draws) and (x.lo, x.hi) == (y.lo, y.hi)
    assert not np.array_equal(x.draws, z.draws)
    c1 = stats.bootstrap(f, st, clusters=f.problem_id, n_boot=300, seed=1)
    c2 = stats.bootstrap(f, st, clusters=f.problem_id, n_boot=300, seed=1)
    assert np.array_equal(c1.draws, c2.draws)
    assert x.unit == "task" and c1.unit == "problem" and c1.n_units == 40


def test_cluster_bootstrap_resamples_whole_problems():
    f = _frame()
    seen = []

    def spy(fr, w):
        seen.append(w.copy())
        return float(np.sum(w))

    stats.bootstrap(f, spy, clusters=f.problem_id, n_boot=200, seed=0)
    for w in seen[:-1]:               # the last call is the point estimate (all ones)
        per_problem = pd.Series(w, index=f.index).groupby(f.problem_id).nunique()
        assert (per_problem == 1).all()          # every task of a problem has the same weight
        assert np.all(w == np.round(w))
        counts = pd.Series(w, index=f.index).groupby(f.problem_id).first()
        assert counts.sum() == 40                # 40 problems drawn with replacement
    seen.clear()
    stats.bootstrap(f, spy, n_boot=50, seed=0)
    assert all(np.sum(w) == len(f) for w in seen)


def test_gate_interval_is_the_cluster_bootstrap_at_gate_settings():
    from router.settings import N_BOOT_GATE
    f = _frame()
    iv = stats.gate_interval(f, stats.mean_diff("a", "b"), f.problem_id)
    assert iv.unit == "problem" and iv.n_boot == N_BOOT_GATE == 2000 and iv.seed == GATE_SEED
    with pytest.raises(ValueError):
        stats.gate_interval(f, stats.mean_diff("a", "b"), None)


def test_bootstrap_is_paired():
    f = _frame()
    f["a2"] = f["a"]
    iv = stats.bootstrap(f, stats.mean_diff("a", "a2"), clusters=f.problem_id, n_boot=200)
    assert np.all(iv.draws == 0) and iv.point == 0


def test_saving_is_on_resampled_totals():
    f = pd.DataFrame({"c": [1.0, 9.0], "base": [2.0, 10.0]})
    assert stats.saving_pct("c", "base")(f, np.ones(2)) == pytest.approx(100 * (1 - 10 / 12))
    assert stats.rate("c", 1)(f, np.array([3.0, 1.0])) == pytest.approx((3 + 9) / 4)


def test_missing_cluster_is_an_error():
    f = _frame()
    with pytest.raises(ValueError):
        stats.bootstrap(f, stats.rate("a"), clusters=f.problem_id.iloc[:-1], n_boot=10)


# ---------------------------------------------------------------------------- seeds, label
def test_across_seeds_marks_the_gate_seed():
    f = _frame()
    tab = stats.across_seeds(lambda s: stats.bootstrap(f, stats.rate("a"), n_boot=50, seed=s))
    assert list(tab.index) == [0, 1, 2, 3, 4]
    assert tab.gate.tolist() == [True, False, False, False, False] and GATE_SEED == 0


@pytest.mark.parametrize("points, lower, strict, label", [
    ({0: 25.0, 1: 24.0, 2: 23.0}, 21.0, False, stats.CLEARS),
    ({0: 25.0, 1: 19.5, 2: 23.0}, 21.0, False, stats.AT_THRESHOLD),     # another seed falls below
    ({0: 25.0, 1: 24.0, 2: 23.0}, 20.0, False, stats.AT_THRESHOLD),     # lower bound at threshold
    ({0: 20.05, 1: 24.0, 2: 23.0}, 20.01, False, stats.AT_THRESHOLD),   # margin under 0.1
    ({0: 19.9, 1: 24.0, 2: 23.0}, 19.0, False, stats.DOES_NOT_CLEAR),
    ({0: 0.0, 1: 1.0}, -1.0, True, stats.DOES_NOT_CLEAR),               # strict (C_SAVING)
])
def test_threshold_label(points, lower, strict, label):
    assert stats.threshold_label(points, lower, 20.0 if not strict else 0.0, strict=strict) == label


# ------------------------------------------------------------------------ cross-fitting driver
def toy_tables(n_problems=50, tasks_per=2, leak=False):
    """Toy problems.csv and splits.csv (problem ids "toy/<i>", not in the committed files)."""
    prob, spl = [], []
    for i in range(n_problems):
        pid = f"toy/{i}"
        folds = {S.fold_column(s): (i + s) % N_FOLDS for s in S.FOLD_SEEDS}
        spl.append({"problem_id": pid, "split": S.EVAL, **folds})
        for k in range(tasks_per):
            sha = f"h{i}_{k}"
            if leak and i == 1 and k == 0:
                sha = "h0_0"          # planted: problem 1 shares a task's content with problem 0
            prob.append({"task_id": f"toy_t{i}_{k}", "problem_id": pid, "link": S.NO_LINK,
                         "source_key": f"sk{i}", "content_sha256": sha})
    splits = pd.DataFrame(spl)
    for s in S.FOLD_SEEDS:
        splits[S.fold_column(s)] = splits[S.fold_column(s)].astype("Int64")
    return pd.DataFrame(prob), splits


class Spy:
    """A candidate that records the problems it was fitted on and asked to score."""
    log = []

    def __init__(self, name, bias):
        self.name, self.bias = name, bias

    def fit(self, train):
        trained = frozenset(train)
        spy = self

        class F:
            def score(self, tasks):
                Spy.log.append((spy.name, trained, frozenset(tasks)))
                return pd.DataFrame({"value": spy.bias, "saw_task": [t in trained for t in tasks]},
                                    index=pd.Index(tasks))
        return F()


def test_cross_fit_never_scores_a_task_with_a_configuration_that_saw_its_problem():
    problems, splits = toy_tables()
    task_ids = problems.task_id
    Spy.log = []
    picked = []

    def select(inner):
        seen = set().union(*(set(f.index) for f in inner.values()))
        picked.append(seen)
        return max(inner, key=lambda n: inner[n].value.mean())

    res = stats.cross_fit(task_ids, [Spy("a", 0.1), Spy("b", 0.2)], select, seed=0,
                          problems=problems, splits=splits)
    pid = problems.set_index("task_id").problem_id
    fold = S.task_splits(problems, splits).set_index("task_id")[S.fold_column(0)]
    assert sorted(res.scores.index) == sorted(task_ids)         # each task scored exactly once
    assert not res.scores.index.has_duplicates
    assert not res.scores.saw_task.any()
    assert set(res.selected.values()) == {"b"}
    for k in range(N_FOLDS):
        held = set(pid[fold == k])
        for f in res.fits:
            assert not (f["train_problems"] & f["scored_problems"])
            if f["outer_fold"] == k:
                assert not (f["train_problems"] & held)            # fold k never fitted on
                if f["stage"] == "inner":
                    assert not (f["scored_problems"] & held)       # nor used to select
        assert not (picked[k] & set(pid.index[pid.isin(held)]))    # the rule never saw fold k
    inner = [f for f in res.fits if f["stage"] == "inner"]
    assert len(inner) == N_FOLDS * N_INNER_FOLDS * 2
    for name, trained, scored in Spy.log:
        assert not ({pid[t] for t in trained} & {pid[t] for t in scored})


def test_cross_fit_stops_on_a_planted_content_leak():
    problems, splits = toy_tables(leak=True)
    with pytest.raises(ValueError, match="R13"):
        stats.cross_fit(problems.task_id, [Spy("a", 0)], lambda inner: "a",
                        problems=problems, splits=splits)


def test_cross_fit_refuses_the_committed_evaluation_population():
    t = S.task_splits()
    ev = t[t.split == S.EVAL].task_id.iloc[:20]
    with pytest.raises(stats.EvaluationPopulationRefused):
        stats.cross_fit(ev, [Spy("a", 0)], lambda inner: "a")
    with pytest.raises(stats.EvaluationPopulationRefused):
        stats.cross_fit(ev, [Spy("a", 0)], lambda inner: "a", frozen=True)
    with pytest.raises(stats.EvaluationPopulationRefused, match="does not exist"):
        stats.cross_fit(ev, [Spy("a", 0)], lambda inner: "a", frozen=True,
                        tag="no-such-tag-s2-test")
    # a substitute table that relabels real evaluation tasks does not get around the refusal
    problems, splits = toy_tables(n_problems=len(ev), tasks_per=1)
    problems["task_id"] = list(ev)
    with pytest.raises(stats.EvaluationPopulationRefused):
        stats.cross_fit(problems.task_id, [Spy("a", 0)], lambda inner: "a",
                        problems=problems, splits=splits)


def test_cross_fit_rejects_development_tasks_and_bad_rules():
    problems, splits = toy_tables()
    splits.loc[0, "split"] = S.DEV
    for s in S.FOLD_SEEDS:
        splits.loc[0, S.fold_column(s)] = pd.NA
    with pytest.raises(ValueError, match="outer fold"):
        stats.cross_fit(problems.task_id, [Spy("a", 0)], lambda i: "a",
                        problems=problems, splits=splits)
    problems, splits = toy_tables()
    with pytest.raises(ValueError, match="unknown candidate"):
        stats.cross_fit(problems.task_id, [Spy("a", 0)], lambda i: "zzz",
                        problems=problems, splits=splits)
    with pytest.raises(ValueError, match="unique"):
        stats.cross_fit(problems.task_id, [Spy("a", 0), Spy("a", 1)], lambda i: "a",
                        problems=problems, splits=splits)
