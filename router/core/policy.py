"""Router v1 (plan S4.1; `docs/research/S4_spec.md` §3): a shrunk per-group lookup table plus a
fixed two-rung cascade. Router v2 (S4.2; spec §4, §4.1, §4.2) at the end of this module: v1 plus
bounded onboarding of one new model (calibration probes, every-tenth-request exploration per
predicted group, a Beta posterior per group with the logit-SD stopping rule, the call cap and the
pool prior), with the new model in each group's menu at its posterior mean from the probes on.
The S4.5 policy rules (spec §6.3: break-even ratio, cheap-first ladder, budget cap) close the
module; no router uses them before S5.

Fitting (`RouterV1.fit(train, ctx)`) reads the training folds only (O1, S0.6):
1. The fixed cascade (§3.4): among P0.5's two-rung ladders (`common.ladders`, non-decreasing
   training mean cost) in mode `mode`, the cheapest whose training acceptance is at least the
   training best single model's; ties to the higher acceptance, then the lower index; if none
   reaches it, the highest-acceptance two-rung ladder (ties to the cheaper).
2. The menu (§3.2, P6 (a)): direct dispatch to each model, then that cascade. `menu="rule_a"`
   replaces it by P0.6's full candidate list (8 direct + every 2- and 3-rung ladder), used only for
   the reduction test against per-cell Rule A.
3. Per group (the training tasks' true S1.1 family, which equals the cell in every development
   population; P3 (a)): training acceptance and mean cost of each candidate
   (`common.policy_outcomes`, which matches replay), the acceptance shrunk toward the pool by
   `core.lookup` with one shared kappa (P4 (a)), and Rule A on the shrunk values with margin
   `margin` (P5 (a)).
4. The pool row: the pool-wide acceptance and mean cost, for a task whose predicted group has no
   training row or that has no text (§3.1, R11).

At decision time the router reads the scored task's group from its `content_text` through the S2.5
predictor refit on the training folds (`groups="pred"`, P2 (a)). `groups="true"` reads the true
family from the handle: a diagnostic, labelled given more information, never a router version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from router.baselines.cells import fit_predictor
from router.baselines.common import (ACCEPT, ESCALATE, MODES, OUR_DEFINITION, Context, PlanRouter,
                                     Policy, best_column, candidates, direct, ladders,
                                     policy_outcomes, success, true_cells)
from router.baselines.lookup import TRUE_CELL_INFO
from router.baselines.static import best_single_model
from router.core import lookup as L
from router.core.exploration import CALL_CAP, draw_probes, is_exploration
from router.core.stopping import GroupPosterior
from router.data.schema import OutcomeMatrix
from router.sim.stream import GIVE_UP, Accept, Call
from router.sim.verifier import FAIL, PASS, UNKNOWN

DEFAULT_MARGIN = 0.02      # S4 spec §3.2, fixed before any run (P0.5's cost-aware DimensionBest)


def two_rung_cascade(train: OutcomeMatrix, ctx: Context, mode: str) -> tuple[Policy, bool]:
    """The fixed cascade of S4 spec §3.4, and whether it reached the best single model."""
    pols = [p for p in ladders(train, mode) if len(p.rungs) == 2]
    A, C = policy_outcomes(train, ctx.verifier, pols, verifier_usd=ctx.verifier_usd)
    v, c = A.mean(0), C.mean(0)
    ref = float(success(train)[best_single_model(train)].mean())
    idx = np.arange(len(pols))
    ok = idx[v >= ref - L.TOL]
    if len(ok):
        return pols[int(ok[np.lexsort((ok, -v[ok], c[ok]))[0]])], True
    return pols[best_column(v, c)], False


@dataclass
class RouterV1:
    mode: str = ESCALATE
    margin: float = DEFAULT_MARGIN
    shrinkage: str = "anova"       # "anova" (P4 (a)) or "none" (kappa = 0)
    groups: str = "pred"           # "pred" (router) or "true" (diagnostic, given more information)
    menu: str = "v1"               # "v1" (models + one two-rung cascade) or "rule_a" (P0.6's list)
    learned: bool = True

    def __post_init__(self):
        if self.mode not in MODES or self.shrinkage not in ("anova", "none") \
                or self.groups not in ("pred", "true") or self.menu not in ("v1", "rule_a"):
            raise ValueError(self)

    @property
    def name(self) -> str:
        base = "router_v1" if self.groups == "pred" else "v1_true_groups"
        if self.mode == ACCEPT:
            base += "_accept"
        if self.shrinkage != "anova":
            base += "_noshrink"
        if self.margin != DEFAULT_MARGIN:
            base += f"_m{self.margin:g}"
        if self.menu != "v1":
            base += f"_menu_{self.menu}"
        return base

    @property
    def more_information(self) -> tuple:
        return TRUE_CELL_INFO if self.groups == "true" else ()

    @property
    def definition(self) -> str:
        return (f"{OUR_DEFINITION}: S4 spec §3; groups = {self.groups} S1.1 family, shrinkage = "
                f"{self.shrinkage}, margin = {self.margin:g}, no-signal cells {self.mode}d")

    def fit_parts(self, train: OutcomeMatrix, ctx: Context) -> "V1Parts":
        """Everything v1 fits on the training folds (router v2 extends it)."""
        cascade, reached = two_rung_cascade(train, ctx, self.mode)
        if self.menu == "v1":
            pols = [direct(x) for x in train.models] + [cascade]
        else:
            pols = candidates(train, self.mode)
        A, C = policy_outcomes(train, ctx.verifier, pols, verifier_usd=ctx.verifier_usd)
        groups = true_cells(train).to_numpy()
        names = sorted(set(groups))
        S, n = L.group_counts(groups, A, names)
        Csum, _ = L.group_counts(groups, C, names)
        kappa = 0.0 if self.shrinkage == "none" else L.kappa_from_rho(L.anova_rho(S, n))
        a_hat, p = L.shrink(S, n, kappa)
        if self.groups == "true":
            read = lambda h: f"{h.info['dimension']}|{h.info['family']}"   # noqa: E731
        else:
            pred = fit_predictor(true_cells(train), ctx)
            read = lambda h: pred.predict_one(h.content_text)               # noqa: E731
        return V1Parts(pols=pols, n_direct=len(train.models), names=names, n=n, a_hat=a_hat,
                       cost=Csum / np.maximum(n, 1)[:, None], p=p, pool_cost=C.mean(0),
                       kappa=kappa, cascade=cascade, reached=reached, read=read,
                       margin=self.margin)

    def fit(self, train: OutcomeMatrix, ctx: Context):
        v = self.fit_parts(train, ctx)
        table = {g: v.pols[L.choose(v.a_hat[i], v.cost[i], v.n_direct, v.margin)]
                 for i, g in enumerate(v.names)}
        pool = v.pool_policy()
        read = v.read
        router = PlanRouter(lambda h: table.get(read(h), pool), self.more_information)
        router.table = {g: pol.name for g, pol in table.items()}
        router.fallback = pool.name
        router.chosen = v.record()
        return router


@dataclass
class V1Parts:
    pols: list
    n_direct: int
    names: list
    n: np.ndarray
    a_hat: np.ndarray
    cost: np.ndarray
    p: np.ndarray
    pool_cost: np.ndarray
    kappa: float
    cascade: Policy
    reached: bool
    read: object
    margin: float

    def pool_policy(self) -> Policy:
        return self.pols[L.choose(self.p, self.pool_cost, self.n_direct, self.margin)]

    def record(self) -> dict:
        return {"kappa": None if math.isinf(self.kappa) else round(float(self.kappa), 6),
                "kappa_inf": math.isinf(self.kappa), "cascade": self.cascade.name,
                "cascade_reached_best_single": self.reached,
                "n_train": int(self.n.sum()),
                "group_n": {g: int(x) for g, x in zip(self.names, self.n)}}


# ------------------------------------------------------------------------------------ router v2
EXPLORING, STOPPED, OUT_OF_BUDGET = "exploring", "stopped", "out_of_budget"


@dataclass
class RouterV2:
    """Router v1 plus bounded onboarding of one new model (S4.2; spec §4, §4.1, §4.2).

    `fit(train, ctx)` fits v1 on the training folds without `new_model` (its outcomes are used only
    through the calibration probes, S0.5), draws the probes and returns a router for a stream in
    which the new model is available from position 0. `labels` is the pairing: "gt_probes" (probes
    labelled by the S0.1 ground truth, live calls by the verifier) or "verifier" (everything by the
    verifier; UNKNOWN never a label). Live calls are always verifier-labelled (S0.13). `cap` = 0
    makes no call to the new model: v1 without it (the k = 0 comparator, tested).
    """
    new_model: str
    labels: str = "gt_probes"
    cap: int = CALL_CAP
    learned: bool = True
    more_information: tuple = ()

    def __post_init__(self):
        if self.labels not in ("gt_probes", "verifier"):
            raise ValueError(self.labels)

    @property
    def name(self) -> str:
        return f"router_v2_{self.labels}" + ("" if self.cap == CALL_CAP else f"_cap{self.cap}")

    @property
    def definition(self) -> str:
        return (f"{OUR_DEFINITION}: S4 spec §4.2; new model {self.new_model}, pairing "
                f"{self.labels}, call cap {self.cap}")

    def fit(self, train: OutcomeMatrix, ctx: Context):
        M = self.new_model
        if M not in train.models:
            raise ValueError(M)
        known = train.without(M)
        v = RouterV1().fit_parts(known, ctx)
        groups = true_cells(train)
        probes = draw_probes(groups, train.available[M], ctx.seed, cap=self.cap) if self.cap else {}
        ok = success(train)[M] if self.labels == "gt_probes" else None
        post, probe_cost, k_probe = {}, {}, {}
        probe_usd, n_probes = 0.0, 0
        for i, g in enumerate(v.names):
            post[g] = GroupPosterior.from_known(v.a_hat[i, :v.n_direct])
        for g, ts in probes.items():
            costs = []
            for t in ts:
                if self.labels == "gt_probes":
                    verdict = PASS if bool(ok.at[t]) else FAIL
                    ran = False
                else:
                    c = ctx.verifier.check(t, M)
                    verdict, ran = c.verdict, c.ran
                post[g].update(verdict)
                cst = float(train.cost.at[t, M])
                costs.append(cst)
                probe_usd += cst + ctx.verifier_usd * ran
                n_probes += 1
            probe_cost[g] = float(np.mean(costs))
            k_probe[g] = len(ts)
        all_costs = [train.cost.at[t, M] for ts in probes.values() for t in ts]
        router = _V2Router(v, M, post, probe_cost,
                           float(np.mean(all_costs)) if all_costs else None, k_probe, n_probes,
                           self.cap)
        router.probe_usd, router.n_probes = probe_usd, n_probes
        return router


class _V2Router:
    """The fitted v2 router (spec §4.2).

    Counters per group: `probe`, `explore` (exploration calls), `plan_verified` and
    `plan_unverified` (plan calls to M). k = all four; the cap counter `total_calls` = probes +
    exploration calls. `tasks` maps each task key to [predicted group, explored, the group's state
    at request time ("pool" for a pool-routed task), M a rung of the plan the task was routed by
    (the exploration call excluded)].
    """

    def __init__(self, v: V1Parts, M: str, post: dict, probe_cost: dict, mean_probe_cost,
                 k_probe: dict, n_probes: int, cap: int):
        self.v, self.M, self.post, self.cap = v, M, post, cap
        self.probe_cost, self.mean_probe_cost = probe_cost, mean_probe_cost
        self.more_information = ()
        self.total_calls = n_probes
        self.k = {g: {"probe": k_probe.get(g, 0), "explore": 0, "plan_verified": 0,
                      "plan_unverified": 0} for g in v.names}
        self.state = {g: EXPLORING for g in v.names}
        self.k_at_end = {}
        self.requests = {g: 0 for g in v.names}
        self.base = {g: v.pols[L.choose(v.a_hat[i], v.cost[i], v.n_direct, v.margin)]
                     for i, g in enumerate(v.names)}
        self.plan = dict(self.base)
        self.entered = {g: False for g in v.names}
        self.pool = v.pool_policy()
        self.tasks: dict[str, list] = {}
        self._cur = None
        self._pol = None                       # the policy the last decision followed
        for g in v.names:
            self._check(g)
            self._refresh(g)
        self._check_cap()
        self.table = {g: p.name for g, p in self.base.items()}
        self.fallback = self.pool.name
        self.chosen = {**v.record(), "new_model": M, "cap": cap, "n_probes": n_probes,
                       "groups": self._report_groups(), "tasks": self.tasks}

    # ------------------------------------------------------------------ onboarding state
    def _m_cost(self, g):
        return self.probe_cost.get(g, self.mean_probe_cost)

    def _k(self, g) -> int:
        return sum(self.k[g].values())

    def _enter(self, g, acc):
        """Recompute group g's choice with M as a direct candidate at acceptance `acc`."""
        cost = self._m_cost(g)
        if cost is None:                      # no probe, no cost information: M not entered
            return
        i = self.v.names.index(g)
        nd = self.v.n_direct
        a = np.concatenate([self.v.a_hat[i, :nd], [acc], self.v.a_hat[i, nd:]])
        c = np.concatenate([self.v.cost[i, :nd], [cost], self.v.cost[i, nd:]])
        pols = self.v.pols[:nd] + [direct(self.M)] + self.v.pols[nd:]
        self.plan[g] = pols[L.choose(a, c, nd + 1, self.v.margin)]
        self.entered[g] = True

    def _refresh(self, g):
        """P9 (b): M at the posterior mean, exploring or stopped. Out of budget: fixed at the pool
        prior, entered once by `_check_cap`."""
        if self.state[g] != OUT_OF_BUDGET:
            self._enter(g, self.post[g].mean)

    def _check(self, g):
        if self.state[g] == EXPLORING and self.post[g].stop():
            self.state[g] = STOPPED
            self.k_at_end[g] = self._k(g)

    def _check_cap(self):
        if self.total_calls >= self.cap:
            for g in self.v.names:
                if self.state[g] == EXPLORING:
                    self.state[g] = OUT_OF_BUDGET
                    self.k_at_end[g] = self._k(g)
                    self._enter(g, self.post[g].prior_mean)

    def _report_groups(self):
        return {g: {"state": self.state[g], "k_probe": self.k[g]["probe"],
                    "k_explore": self.k[g]["explore"],
                    "k_plan_verified": self.k[g]["plan_verified"],
                    "k_plan_unverified": self.k[g]["plan_unverified"], "k": self._k(g),
                    "k_at_end": self.k_at_end.get(g),
                    "alpha": round(self.post[g].alpha, 6), "beta": round(self.post[g].beta, 6),
                    "prior_mean": round(self.post[g].prior_mean, 6),
                    "prior_strength": round(self.post[g].prior_strength, 6),
                    "logit_sd": round(self.post[g].sd, 6), "m_entered": self.entered[g],
                    "plan": self.plan[g].name, "requests": self.requests[g]}
                for g in self.v.names}

    # ------------------------------------------------------------------------ the router
    def _start(self, task, available):
        g = self.v.read(task)
        known = g in self.state
        explore = False
        if known:
            self.requests[g] += 1
            explore = (self.state[g] == EXPLORING and self.total_calls < self.cap
                       and is_exploration(self.requests[g]) and self.M in available)
        pol = self.plan[g] if known else self.pool
        if explore and self.M in pol.rungs:   # §4.2: after M's exploration call, v1 without M
            pol = self.base[g]
        self.tasks[task.key] = [g, explore, self.state[g] if known else "pool",
                                bool(self.M in pol.rungs)]
        self._cur = (task.key, g, explore, pol)
        self.chosen["groups"] = self._report_groups()

    def decide(self, task, available, history):
        if self._cur is None or self._cur[0] != task.key:
            self._start(task, available)
        _, g, explore, pol = self._cur
        self._pol = pol
        off = 0
        if explore:
            if not history:
                return Call(self.M, verify=True)
            if history[0].verdict == PASS:
                return Accept(model=self.M)
            off = 1
        rungs = [x for x in pol.rungs if x in available]
        h = history[off:]
        if not rungs:                 # never accept a failed exploration call (§4.2)
            return Accept() if h else GIVE_UP
        if not pol.verify:
            return Accept() if h else Call(rungs[0], verify=False)
        if h:
            last = h[-1]
            stop = last.verdict == PASS or (last.verdict == UNKNOWN and pol.on_unknown == ACCEPT)
            if stop or len(h) == len(rungs):
                return Accept()
        return rungs[len(h)]

    def feedback(self, obs) -> None:
        if self._cur is None or obs.task != self._cur[0] or obs.model != self.M:
            return
        _, g, explore, _ = self._cur
        if g not in self.state:
            return
        if explore and obs.call_index == 0:
            self.total_calls += 1
            self.k[g]["explore"] += 1
            updated = self.post[g].update(obs.verdict)      # UNKNOWN does not update (S0.13)
        elif self._pol.verify:
            self.k[g]["plan_verified"] += 1
            updated = self.post[g].update(obs.verdict)
        else:                                               # unverified: no verdict
            self.k[g]["plan_unverified"] += 1
            updated = False
        if updated:
            self._check(g)
            self._refresh(g)
        self._check_cap()
        self.chosen["groups"] = self._report_groups()


# --------------------------------------------------------- S4.5 policy rules (spec §6.3; plan §2)
def break_even_ratio(c_cheap: float, c_next: float) -> float:
    """c / e: the stop probability a cheap rung needs before trying it first pays (spec §6.3).
    Costs include each rung's verifier run when it is verified."""
    if c_next <= 0:
        raise ValueError(c_next)
    return c_cheap / c_next


def stop_probability(p_pass: float, p_unknown: float, mode: str) -> float:
    """q = P(the ladder stops at this rung): PASS, plus UNKNOWN only in accept mode. Not P(success):
    under escalate on a no-signal cell q = 0."""
    if mode not in MODES:
        raise ValueError(mode)
    if min(p_pass, p_unknown) < 0 or p_pass + p_unknown > 1 + 1e-12:
        raise ValueError((p_pass, p_unknown))
    return p_pass + (p_unknown if mode == ACCEPT else 0.0)


def cheap_first_pays(q: float, c_cheap: float, c_next: float) -> bool:
    """Cheap-first expected cost c + (1 - q) e is below e iff q > c / e (equality: not cheaper)."""
    return q > break_even_ratio(c_cheap, c_next)


def cheap_first_ladder(target: tuple, candidates) -> tuple:
    """Build a cheap-first ladder ending at `target` = (model, cost). `candidates` are
    (model, cost, q). Walking the candidates cheaper than the current first rung in descending
    cost, each goes in front iff q > its cost / the first rung's cost."""
    ladder, c_first = [target[0]], float(target[1])
    for m, c, q in sorted(candidates, key=lambda x: (-x[1], x[0])):
        if c < c_first and cheap_first_pays(q, c, c_first):
            ladder.insert(0, m)
            c_first = float(c)
    return tuple(ladder)


def within_budget(spent: float, cost: float, cap: float | None) -> bool:
    """Whether a rung may be called under the per-task cap B (P12: no default; None = no cap)."""
    if cap is None:
        return True
    if cap < 0:
        raise ValueError(cap)
    return spent + cost <= cap + 1e-12


__all__ = ["RouterV1", "RouterV2", "two_rung_cascade", "DEFAULT_MARGIN", "break_even_ratio",
           "cheap_first_ladder", "cheap_first_pays", "stop_probability", "within_budget"]
