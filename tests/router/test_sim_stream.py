"""S2.1: seeded replay with bandit feedback, and the no-ground-truth test (S0.13, R16)."""

import dataclasses
import hashlib

import numpy as np
import pandas as pd
import pytest

from router.data import verdicts
from router.sim import toy
from router.sim.stats import STREAM_IDS
from router.sim.stream import (GIVE_UP, Accept, Ladder, Observation, ReplayView, SingleChoice,
                               TaskHandle, replay, task_key, task_order)
from router.sim.verifier import PASS, UNKNOWN, ReplayVerifier

HAVE_X13 = verdicts.X13_VERDICTS.exists()


@pytest.fixture
def m():
    return toy.matrix()


@pytest.fixture
def rv(m):
    return ReplayVerifier(toy.verdict_table(m))


# ------------------------------------------------------------------------------------ routers
class Spy:
    """Records everything it is given and derives every decision from a hash of it, so any
    change in what replay passes (a leaked score, say) changes its decisions."""

    def __init__(self, more_information=()):
        self.more_information = tuple(more_information)
        self.seen = []
        self.state = hashlib.sha256()

    def _mix(self, *objs):
        for o in objs:
            self.seen.append(o)
            self.state.update(repr(o).encode())
        return int(self.state.hexdigest(), 16)

    def decide(self, task, available, history):
        h = self._mix(task, available, history)
        left = [mdl for mdl in available if mdl not in {o.model for o in history}]
        if history and (h % 3 == 0 or not left):
            return Accept(history[h % len(history)].model)
        if not left:
            return GIVE_UP
        if h % 11 == 0 and not history:
            return GIVE_UP
        return left[h % len(left)]

    def feedback(self, obs):
        self._mix(obs)


class GreedyLearner:
    """An online learner: estimates each model's PASS share from its own observations and calls
    the best-looking model first, escalating once on FAIL."""
    more_information = ()

    def __init__(self):
        self.n, self.k = {}, {}

    def decide(self, task, available, history):
        if len(history) >= 2 or (history and history[-1].verdict != "FAIL"):
            return Accept()
        left = [mdl for mdl in available if mdl not in {o.model for o in history}]
        if not left:
            return Accept() if history else GIVE_UP
        return max(left, key=lambda x: ((self.k.get(x, 0) + 1) / (self.n.get(x, 0) + 2), x))

    def feedback(self, obs):
        if obs.verdict != UNKNOWN:
            self.n[obs.model] = self.n.get(obs.model, 0) + 1
            self.k[obs.model] = self.k.get(obs.model, 0) + (obs.verdict == PASS)


def routers():
    return {
        "spy": lambda: Spy(),
        "spy_given_family": lambda: Spy(("family", "dimension")),
        "ladder_escalate": lambda: Ladder(["cheap", "mid", "strong"], "escalate"),
        "ladder_accept": lambda: Ladder(["cheap", "mid", "strong"], "accept"),
        "greedy_learner": lambda: GreedyLearner(),
        "single_by_content": lambda: SingleChoice(lambda t, a: a[len(t.key) % len(a)]),
    }


def _decisions(res):
    return (res.calls[["task_id", "call_index", "model", "verdict"]].to_dict("records"),
            res.tasks[["position", "action", "final_model", "n_calls"]].to_dict("index"))


# --------------------------------------------------------------------------- no ground truth
@pytest.mark.parametrize("name", sorted(routers()))
def test_garbage_scores_leave_every_decision_unchanged(name, m, rv):
    """Keep the verifier observations fixed, replace every released score with garbage: every
    router must make identical decisions (S0.13, R16)."""
    make = routers()[name]
    g = np.random.default_rng(99)
    for garbage in (pd.DataFrame(g.random(m.score.shape), index=m.tasks, columns=m.models),
                    pd.DataFrame(-1e9, index=m.tasks, columns=m.models),
                    1.0 - m.score):
        assert not np.allclose(garbage.to_numpy(), m.score.to_numpy())
        for seed in (0, 3):
            a = replay(make(), ReplayView.from_matrix(m), rv, seed=seed,
                       arrivals={"strong": 5})
            b = replay(make(), ReplayView.from_matrix(toy.with_scores(m, garbage)), rv, seed=seed,
                       arrivals={"strong": 5})
            assert _decisions(a) == _decisions(b)


@pytest.mark.parametrize("name", sorted(routers()))
def test_garbage_costs_leave_every_decision_unchanged(name, m, rv):
    """Realised cell costs are not router-visible either (Qwen3-Max's cost is a per-dimension
    constant, so it would reveal the dimension). Keep availability and verdicts, scramble costs."""
    make = routers()[name]
    garbage = m.cost.sample(frac=1.0, random_state=1).set_axis(m.tasks) * 7.3 + 0.01
    scrambled = toy.with_scores(m, m.score)
    scrambled.cost.loc[:, :] = garbage.to_numpy()
    a = replay(make(), ReplayView.from_matrix(m), rv, seed=0, arrivals={"strong": 5})
    b = replay(make(), ReplayView.from_matrix(scrambled), rv, seed=0, arrivals={"strong": 5})
    assert _decisions(a) == _decisions(b)


class NoScore:
    """A matrix stand-in whose score raises on access."""

    def __init__(self, m):
        self._m = m

    def __getattr__(self, name):
        if name == "score":
            raise AssertionError("replay read the score")
        return getattr(self._m, name)


def test_replay_never_reads_the_score(m, rv):
    view = ReplayView.from_matrix(NoScore(m))
    res = replay(Spy(), view, rv, seed=0)
    assert len(res.tasks) == len(m)
    assert not any(f.name == "score" for f in dataclasses.fields(ReplayView))


def _walk(obj, seen=None):
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return
    seen.add(id(obj))
    yield obj
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            yield from _walk(getattr(obj, f.name), seen)
    elif isinstance(obj, dict) or hasattr(obj, "items") and hasattr(obj, "keys"):
        for k, val in obj.items():
            yield from _walk(k, seen)
            yield from _walk(val, seen)
    elif isinstance(obj, (tuple, list, set, frozenset)):
        for x in obj:
            yield from _walk(x, seen)


def test_no_router_facing_object_carries_a_score(m, rv):
    for cls in (TaskHandle, Observation):
        assert {f.name for f in dataclasses.fields(cls)}.isdisjoint({"score", "truth", "released"})
    # S0.13: an observation is the verdict on the router's own call, nothing more
    assert [f.name for f in dataclasses.fields(Observation)] == ["task", "model", "verdict",
                                                                 "call_index"]
    spy = Spy(("family", "dimension"))
    replay(spy, ReplayView.from_matrix(m), rv, seed=1)
    kinds = set()
    for obj in _walk(spy.seen):
        kinds.add(type(obj).__name__)
        assert not isinstance(obj, (pd.DataFrame, pd.Series, np.ndarray)), type(obj)
        if dataclasses.is_dataclass(obj):
            assert "score" not in {f.name for f in dataclasses.fields(obj)}
        if isinstance(obj, dict):
            assert "score" not in obj
    assert {"TaskHandle", "Observation", "str"} <= kinds


# ----------------------------------------------------------------------------- stream rules
def test_task_order_uses_its_own_stream_and_reproduces(m):
    o0 = task_order(m.tasks, 0)
    assert list(o0) == list(task_order(m.tasks[::-1], 0))
    assert list(o0) != list(task_order(m.tasks, 1))
    expect = pd.Index(sorted(m.tasks))[np.random.default_rng(
        np.random.SeedSequence([0, STREAM_IDS["stream_order"]])).permutation(len(m))]
    assert list(o0) == list(expect)


def test_model_arrival(m, rv):
    seen = []

    class Rec(Spy):
        def decide(self, task, available, history):
            if not history:
                seen.append(available)
            return super().decide(task, available, history)

    res = replay(Rec(), ReplayView.from_matrix(m), rv, seed=0, arrivals={"strong": 10})
    assert all("strong" not in a for a in seen[:10])
    assert sum("strong" in a for a in seen[10:]) >= len(seen[10:]) - 1
    assert res.events == [{"position": 10, "model": "strong"}]


def test_bandit_feedback_only_and_opaque_handles(m, rv):
    got = []

    class Rec(Spy):
        def decide(self, task, available, history):
            got.append((task, history))
            assert all(o.task == task.key for o in history)
            return super().decide(task, available, history)

    res = replay(Rec(), ReplayView.from_matrix(m), rv, seed=2)
    for task, history in got:
        assert task.info == {}
        assert task.key not in set(m.tasks) and not task.key.startswith("t")
        assert all(o.verdict in ("PASS", "FAIL", "UNKNOWN") for o in history)
    assert res.given_more_information == ()
    keys = {task_key(t): t for t in m.tasks}
    for task, history in got:
        t = keys[task.key]
        called = list(res.calls[res.calls.task_id == t].model)
        assert [o.model for o in history] == called[:len(history)]


def test_given_more_information_is_recorded(m, rv):
    spy = Spy(("family",))
    res = replay(spy, ReplayView.from_matrix(m), rv, seed=0)
    assert res.given_more_information == ("family",)
    handles = [o for o in spy.seen if isinstance(o, TaskHandle)]
    assert all(set(h.info) == {"family"} for h in handles)


def test_unavailable_rung_is_skipped_and_counted(m, rv):
    res = replay(Ladder(["cheap", "mid", "strong"]), ReplayView.from_matrix(m), rv, seed=0)
    assert "mid" not in set(res.calls[res.calls.task_id == "t1"].model)
    assert res.tasks.at["t1", "n_skipped_unavailable"] == 1
    assert res.tasks.at["t1", "skipped_models"] == ("mid",)
    assert res.tasks.drop("t1").n_skipped_unavailable.sum() == 0


def test_protocol_errors(m, rv):
    view = ReplayView.from_matrix(m)
    with pytest.raises(ValueError, match="already called"):
        replay(SingleChoice_twice(), view, rv, seed=0)
    with pytest.raises(ValueError, match="before any call"):
        replay(SingleChoice(lambda t, a: Accept()), view, rv, seed=0)
    res = replay(SingleChoice(lambda t, a: GIVE_UP), view, rv, seed=0)
    assert (res.tasks.action == "give_up").all() and res.tasks.final_model.isna().all()
    assert res.calls.empty and res.n_retries == 0


class SingleChoice_twice:
    more_information = ()

    def decide(self, task, available, history):
        return available[0]

    def feedback(self, obs):
        pass


@pytest.mark.skipif(not HAVE_X13, reason="route-anything-data not downloaded to external/")
def test_garbage_scores_on_the_development_split_headline_tasks():
    """The no-ground-truth test on real data: development-split headline tasks, real X13."""
    from router.data import splits as S
    from router.data.audit import build_views, load_release
    m = load_release()
    head = build_views(m)["headline"]
    dev = set(S.task_splits().query("split == 'dev'").task_id)
    sub = head.subset([t for t in head.tasks if t in dev])
    rv = ReplayVerifier.from_x13()
    garbage = pd.DataFrame(np.random.default_rng(5).random(sub.score.shape), index=sub.tasks,
                           columns=sub.models)
    for make in (lambda: Spy(), lambda: Ladder(["Qwen3-Max", "gpt-5.4", "claude-opus-4-6"]),
                 lambda: GreedyLearner()):
        a = replay(make(), ReplayView.from_matrix(sub), rv, seed=0)
        b = replay(make(), ReplayView.from_matrix(toy.with_scores(sub, garbage)), rv, seed=0)
        assert _decisions(a) == _decisions(b)


def test_direct_dispatch_runs_no_verifier(m, rv):
    """`Call(model, verify=False)`: no verifier run is charged and the router learns nothing."""
    from router.sim.stream import NOT_RUN, Call
    seen = []

    class Direct:
        more_information = ()

        def decide(self, task, available, history):
            return Accept() if history else Call(available[-1], verify=False)

        def feedback(self, obs):
            seen.append(obs.verdict)

    res = replay(Direct(), ReplayView.from_matrix(m), rv, seed=0)
    assert set(seen) == {UNKNOWN} and set(res.calls.state) == {NOT_RUN}
    assert res.tasks.n_verifier_runs.sum() == 0 and not res.calls.verifier_ran.any()
    verified = replay(SingleChoice(lambda t, a: a[-1], verify=True), ReplayView.from_matrix(m), rv,
                      seed=0)
    assert verified.tasks.n_verifier_runs.sum() > 0
    assert list(verified.calls.model) == list(res.calls.model)
