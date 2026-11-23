"""S3 checkpoint 2, built before any evaluation-population number (spec D3, D4; reviewer's
checkpoint 1b go-ahead, item 1):

- the DP1a evaluation path `stats.fit_free_headroom` is closed: no argument takes a router, a
  candidate, a view or a matrix; it runs only the default BestSingle and DimensionBest (checked by
  class identity, not by name); it reads the filtered view only; no scored task's problem is in its
  training set; its result carries the path record; and `cross_fit` still refuses router
  candidates on evaluation tasks without a tag;
- the oracle-on-no-signal bound (`router.baselines.oracle_bound`) can be imported only by the S3.2
  module, reads the truth only on no-signal cells, charges no verifier run there, and matches
  `policy_outcomes` when replayed.
"""

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.baselines import common, lookup, static
from router.baselines.common import Context, PlanRouter, policy_outcomes, success
from router.baselines.oracle_bound import (ORACLE, VERIFIER_MODE, BoundRule, OracleOnNoSignal,
                                           oracle_candidates)
from router.baselines.run import default_baselines
from router.data import splits as S
from router.data import verdicts
from router.sim import metrics, stats, toy
from router.sim.stream import ReplayView, replay
from router.sim.verifier import FAIL, NO_SIGNAL, PASS, ReplayVerifier

REPO = Path(__file__).resolve().parents[2]
HAVE_DATA = verdicts.X13_VERDICTS.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external data not fetched")
ORACLE_MODULE = "router.baselines.oracle_bound"
ORACLE_IMPORTERS = {"router.experiments.s3_dp1b"}


# ===================================================================== D3: the DP1a evaluation path
def test_fit_free_headroom_takes_no_object_argument():
    sig = inspect.signature(stats.fit_free_headroom)
    assert list(sig.parameters) == ["seed", "cruxeval_lenient", "empty_unavailable", "all_tasks"]
    for name in ("cruxeval_lenient", "empty_unavailable", "all_tasks"):
        p = sig.parameters[name]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False


def test_default_routes_pass_the_identity_check():
    stats.check_fit_free_routes(stats._fit_free_routes())


class _SubBestSingle(static.BestSingle):
    pass


class _Impostor:
    """A router that calls itself DimensionBest."""
    name = "dimension_best"
    key = "dimension"


@pytest.mark.parametrize("routes", [
    {"best_single": _SubBestSingle(), "dimension_best": lookup.GroupBest("dimension")},
    {"best_single": static.BestSingle(), "dimension_best": _Impostor()},
    {"best_single": static.BestSingle(), "dimension_best": lookup.GroupBest("dimension", cost_aware=True)},
    {"best_single": static.BestSingle(), "dimension_best": lookup.GroupBest("family")},
    {"best_single": static.BestSingle(name="dimension_best"), "dimension_best": lookup.GroupBest("dimension")},
    {"best_single": static.BestSingle(), "dimension_best": lookup.CellBest("pred")},
    {"best_single": static.BestSingle(), "dimension_best": lookup.GroupBest("dimension"),
     "extra": lookup.CellRule("A", "escalate")},
])
def test_identity_check_refuses_subclasses_renames_and_other_routes(routes):
    with pytest.raises(stats.RouteRefused):
        stats.check_fit_free_routes(routes)


def test_fit_free_headroom_refuses_an_injected_route(monkeypatch):
    monkeypatch.setattr(stats, "_fit_free_routes", lambda: {
        "best_single": _SubBestSingle(), "dimension_best": lookup.GroupBest("dimension")})
    with pytest.raises(stats.RouteRefused):
        stats.fit_free_headroom(0)


class _Candidate:
    name = "router"

    def fit(self, train):
        raise AssertionError("must be refused before fitting")


def test_cross_fit_still_refuses_router_candidates_on_evaluation_tasks():
    ts = S.task_splits().set_index("task_id")
    ev = list(ts.index[ts.split == S.EVAL][:200])
    with pytest.raises(stats.EvaluationPopulationRefused):
        stats.cross_fit(ev, [_Candidate()], lambda s: "router")
    with pytest.raises(stats.EvaluationPopulationRefused):
        stats.cross_fit(ev, [_Candidate()], lambda s: "router", frozen=True)
    with pytest.raises(stats.EvaluationPopulationRefused, match="does not exist"):
        stats.cross_fit(ev, [_Candidate()], lambda s: "router", frozen=True, tag="no-such-tag-s3")


@needs_data
def test_fit_free_headroom_refuses_another_view(monkeypatch):
    from router.data import audit
    real = audit.build_views

    def swapped(m):
        v = real(m)
        return {**v, "filtered": v["all_eligible"]}

    monkeypatch.setattr(audit, "build_views", swapped)
    with pytest.raises(stats.RouteRefused, match="filtered view only"):
        stats.fit_free_headroom(0)


@needs_data
def test_fit_free_headroom_structure_on_the_filtered_view():
    """Structure only: groups, problem separation and the path record. No headroom is asserted."""
    r = stats.fit_free_headroom(0)
    rec = r.record
    assert rec["evaluation_population_path"] == stats.EVALUATION_PATH == "S3 spec D3 whitelist"
    assert rec["view"] == "filtered" and rec["groups"] == ["dev"] + [f"fold{k}" for k in range(5)]
    assert rec["routes"]["best_single"] == "router.baselines.static.BestSingle"
    assert rec["routes"]["dimension_best"] == "router.baselines.lookup.GroupBest"
    assert (rec["n_tasks"], rec["n_development_tasks"], rec["n_evaluation_tasks_read"]) == (1046, 195, 851)
    assert len(r.tasks) == 1046 and r.tasks.index.is_unique
    for f in r.fits:
        assert not (f["train_problems"] & f["scored_problems"]), f["group"]
        assert f["n_train"] + f["n_scored"] == 1046
    assert r.tasks.groupby("group").size().sum() == 1046
    # every scored task's problem is absent from the training problems of its own group's fit
    fit = {f["group"]: f for f in r.fits}
    for g, rows in r.tasks.groupby("group"):
        assert not set(rows.problem_id) & fit[g]["train_problems"]


# ===================================================================== D4: the oracle import fence
def _imports(path: Path) -> set[str]:
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
            mods |= {f"{node.module}.{a.name}" for a in node.names}
    return mods


def _module(path: Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    parts = rel.parts[:-1] if rel.name == "__init__" else rel.parts
    return ".".join(parts)


def _files():
    out = {_module(p): p for p in (REPO / "router").rglob("*.py")}
    out.update({_module(p): p for p in (REPO / "scripts").rglob("*.py")})
    return out


def _closure(start, files):
    seen, todo = set(), [start]
    while todo:
        m = todo.pop()
        if m in seen or m not in files:
            continue
        seen.add(m)
        todo += [x for x in _imports(files[m]) if x.startswith("router")]
    return seen


def test_only_the_s3_2_module_imports_the_oracle():
    files = _files()
    assert ORACLE_MODULE in files
    for mod, path in files.items():
        if mod == ORACLE_MODULE or mod in ORACLE_IMPORTERS:
            continue
        direct = {x for x in _imports(path) if x == ORACLE_MODULE or x.startswith(ORACLE_MODULE + ".")}
        assert not direct, mod
        assert ORACLE_MODULE not in _closure(mod, files), mod


def test_no_other_module_names_the_oracle_mode_or_wrapper():
    pat = re.compile(r"OracleOnNoSignal|oracle_on_no_signal|on_unknown\s*=\s*['\"]oracle['\"]")
    for mod, path in _files().items():
        if mod == ORACLE_MODULE or mod in ORACLE_IMPORTERS:
            continue
        assert not pat.search(path.read_text()), mod


def test_bound_is_not_a_default_baseline_and_ladders_refuse_the_mode():
    assert not any(n.startswith("bound") for n in (b.name for b in default_baselines(toy.MODELS)))
    with pytest.raises(ValueError):
        common.ladders(toy.matrix(), ORACLE)


# ===================================================================== D4: the wrapper and the bound
def _toy():
    m = toy.matrix(n=30)
    rec = [t for t in m.tasks if t != "t3"]          # t3 has no X13 record
    m = m.subset(rec)
    base = ReplayVerifier(toy.verdict_table(m, no_record=()))
    return m, base


def test_wrapper_reads_truth_only_on_no_signal_cells():
    m, base = _toy()
    w = OracleOnNoSignal.from_matrix(base, m)
    ok = success(m)
    n_oracle = 0
    for t in m.tasks:
        for mod in m.models:
            if not m.available.at[t, mod]:
                continue
            b, c = base.check(t, mod), w.check(t, mod)
            if b.state == NO_SIGNAL:
                n_oracle += 1
                assert (c.verdict, c.ran, c.state) == ((PASS if ok.at[t, mod] else FAIL), False, "oracle")
            else:
                assert c == b
    assert n_oracle > 20


def test_wrapper_lenient_truth_and_no_record():
    m, base = _toy()
    lenient = OracleOnNoSignal.from_matrix(base, m, cruxeval_lenient=True)
    strict = OracleOnNoSignal.from_matrix(base, m)
    hit = [(t, mod) for t in m.tasks for mod in m.models
           if m.available.at[t, mod] and m.family[t] == "cruxeval" and m.score.at[t, mod] == 0.8
           and base.check(t, mod).state == NO_SIGNAL]
    assert hit
    for t, mod in hit:
        assert lenient.check(t, mod).verdict == PASS and strict.check(t, mod).verdict == FAIL
    full = toy.matrix(n=30)
    w = OracleOnNoSignal.from_matrix(ReplayVerifier(toy.verdict_table(full)), full)
    with pytest.raises(ValueError, match="no X13 record"):
        w.check("t3", "cheap")


def test_bound_outcomes_match_replay_for_every_column():
    m, base = _toy()
    w = OracleOnNoSignal.from_matrix(base, m)
    pols = oracle_candidates(m)
    assert len(pols) == 3 + sum(1 for p in pols if p.verify) and all(
        p.on_unknown == ORACLE for p in pols if p.verify)
    A, C = policy_outcomes(m, w, pols)
    for j, pol in enumerate(pols):
        res = replay(PlanRouter(lambda h, p=pol: p), ReplayView.from_matrix(m), w, seed=0)
        o = metrics.task_outcomes(res, m).loc[m.tasks]
        assert o.attrs["verifier_mode"] == VERIFIER_MODE
        assert np.array_equal(o.accepted.to_numpy(float), A[:, j]), pol.name
        assert np.allclose(o.cost_usd.to_numpy(), C[:, j]), pol.name


def test_bound_hand_case():
    """A no-signal task where cheap fails and strong succeeds: the bound escalates past cheap
    (truth), accepts strong, pays both calls and no verifier run."""
    m, base = _toy()
    ok = success(m)
    w = OracleOnNoSignal.from_matrix(base, m)
    cand = [t for t in m.tasks if base.check(t, "cheap").state == NO_SIGNAL
            and base.check(t, "strong").state == NO_SIGNAL and m.available.loc[t].all()
            and not ok.at[t, "cheap"] and ok.at[t, "strong"]]
    assert cand
    t = cand[0]
    pol = common.Policy(("cheap", "strong"), True, ORACLE)
    A, C = policy_outcomes(m.subset([t]), w, [pol])
    assert A[0, 0] == 1.0
    assert C[0, 0] == pytest.approx(m.cost.at[t, "cheap"] + m.cost.at[t, "strong"])


def test_bound_rule_reads_the_truth_and_needs_the_wrapper():
    """The D4 exemption made explicit: unlike every default baseline, the bound's decisions change
    when the scored tasks' truth changes; and it refuses the plain replay verifier."""
    m, base = _toy()
    folds = pd.Series([i % 5 for i in range(len(m))], index=m.tasks)
    tr, te = folds.index[folds != 0], folds.index[folds == 0]
    ctx = Context(OracleOnNoSignal.from_matrix(base, m), problem_of=pd.Series(m.tasks, index=m.tasks))
    router = BoundRule("A", cells="true").fit(m.subset(list(tr)), ctx)
    garbage = toy.with_scores(m, 1.0 - m.score)
    gw = OracleOnNoSignal.from_matrix(base, garbage)
    a = replay(router, ReplayView.from_matrix(m.subset(list(te))), ctx.verifier, seed=0)
    z = replay(router, ReplayView.from_matrix(garbage.subset(list(te))), gw, seed=0)
    # the calls made differ on at least one task
    assert a.calls[["task_id", "model"]].values.tolist() != z.calls[["task_id", "model"]].values.tolist()
    with pytest.raises(TypeError):
        BoundRule("A", cells="true").fit(m.subset(list(tr)), Context(base))
