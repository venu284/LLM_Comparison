"""S4.2: router v2 (v1 plus bounded onboarding) on leave-one-model-out streams of the development
split (docs/research/S4_spec.md §4, §4.1, §4.2; CLAIMS S0.5, S0.13, R8, R11).

Populations: as `s4_v1` (spec §2, P1 (b)), complete case, seeds 0-4, `stats.dev_folds`. For each
population, seed and model M (the new model, never pooled over models, R8) and each pairing
(`gt_probes`: probes by the ground truth, live calls by the verifier; `verifier`: everything by the
verifier): v1 is fitted on the training folds without M, M's calibration probes are drawn from the
training folds (stream 21), and the scored fold is replayed with M available from position 0, in
each group's menu at its posterior mean (P9 (b)), explored on every tenth request of each predicted
group until the group stops (logit SD <= 0.5), the call cap (100, probes + exploration) is reached,
or the stream ends. M arrives at position 0, so `replay(arrivals=...)` is not needed (ruled spec
§4).

Comparators, paired on the same tasks and folds:
- v1 without M (k = 0, the pool prior; equal to v2 with no call, tested);
- k-matched DimensionBest at v2's k (per fold: k = v2's calls to M there, plan calls included;
  in the `gt_probes` pairing the first n_gt draws, n_gt = v2's probes there, by the ground truth and
  the rest by the verifier, §4.2; probes from the training folds; given more information: it reads
  the dimension);
- v1 with all eight models (k = inf), as a reference.

Per row: acceptance and cost (S0.2: probe cost spread over the scored tasks, exploration calls
charged in the stream), k per fold (probes, exploration, plan calls verified and unverified), each
group's end state and k, probe and exploration cost, acceptance on tasks routed while the group
explored against after, and with M in the plan against without (k = 0 on the same tasks), and
paired cluster-bootstrap CIs on the gate seed. Gate seed only: every S2 metric for v2 (CPAT at the best single model's acceptance,
regret including probe cost) with paired CIs against the comparators.

Development figures, not results.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s4_v2
Writes router/experiments/out/s4_v2_seeds.csv, s4_v2_seed0.csv, s4_v2_groups.csv and s4_v2.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from router.baselines.common import Context
from router.baselines.lookup import KMatchedDimensionBest
from router.baselines.run import run_dev
from router.baselines.static import BestSingle
from router.core.exploration import CALL_CAP
from router.core.policy import RouterV1, RouterV2
from router.experiments.s2_baselines import compare, row
from router.core.stopping import logit_sd
from router.experiments.s4_v1 import POPULATIONS, inputs
from router.settings import N_BOOT, REPO_ROOT, STOP_POSTERIOR_SD
from router.sim import metrics, stats
from router.sim.stream import task_key

OUT = REPO_ROOT / "router" / "experiments" / "out"
LABELS = ("gt_probes", "verifier")
SPLITS = ("exploring", "settled", "m_in_plan", "m_not_in_plan", "pool")


@dataclass
class KMatchedAtV2K:
    """k-matched DimensionBest with k set per fold to v2's calls to the new model there and, in
    the `gt_probes` pairing, v2's probe count there labelled by the ground truth (spec §4.2)."""
    new_model: str
    labels: str
    k_by_fold: dict          # frozenset(scored problems) -> k
    n_gt_by_fold: dict       # frozenset(scored problems) -> v2's probes
    learned: bool = True
    more_information: tuple = ("dimension",)
    definition: str = "k-matched DimensionBest (S0.5) at router v2's k in each fold"

    @property
    def name(self) -> str:
        return f"dimension_best_kmatched_v2_{self.labels}"

    def fit(self, train, ctx):
        key = frozenset(ctx.exclude_problems)
        k = self.k_by_fold[key]
        if self.labels == "gt_probes":
            b = KMatchedDimensionBest(self.new_model, k, "gt_probes",
                                      n_ground_truth=self.n_gt_by_fold[key])
        else:
            b = KMatchedDimensionBest(self.new_model, k, self.labels)
        r = b.fit(train, ctx)
        r.chosen = {"k": k, "labels_used": r.labels_used}
        return r


def fold_problems(folds: pd.Series, problems: pd.Series) -> dict:
    return {int(f): frozenset(problems.reindex(folds.index[folds == f])) for f in sorted(folds.unique())}


def v2_record(o: pd.DataFrame, m, ctx, M: str) -> tuple[dict, list]:
    """Per-fold onboarding record of one v2 run, and per-group rows. `split` holds the scored task
    ids of each split of spec §4.2 (state at request time; M a rung of the plan)."""
    folds, groups = [], []
    key_of = {task_key(t): t for t in o.index}
    expl_usd, n_expl = 0.0, 0
    split = {s: [] for s in SPLITS}
    for f in o.attrs["fits"]:
        ch = f["chosen"]
        gs = ch["groups"].values()
        k_expl = sum(g["k_explore"] for g in gs)
        k_pv = sum(g["k_plan_verified"] for g in gs)
        k_pu = sum(g["k_plan_unverified"] for g in gs)
        folds.append({"fold": f["fold"], "k_total": ch["n_probes"] + k_expl + k_pv + k_pu,
                      "cap_counter": ch["n_probes"] + k_expl, "n_probes": ch["n_probes"],
                      "k_explore": k_expl, "k_plan_verified": k_pv, "k_plan_unverified": k_pu})
        for g, d in sorted(ch["groups"].items()):
            groups.append({"fold": f["fold"], "group": g, **{k: d[k] for k in (
                "state", "k_probe", "k_explore", "k_plan_verified", "k_plan_unverified", "k",
                "k_at_end", "requests", "prior_mean", "prior_strength", "alpha", "beta",
                "logit_sd", "m_entered", "plan")}})
        for key, (g, explored, state, m_plan) in ch["tasks"].items():
            t = key_of[key]
            if explored:
                n_expl += 1
                expl_usd += float(m.cost.at[t, M]) + ctx.verifier_usd * ctx.verifier.check(t, M).ran
            if state == "pool":
                split["pool"].append(t)
                continue
            split["exploring" if state == "exploring" else "settled"].append(t)
            split["m_in_plan" if m_plan else "m_not_in_plan"].append(t)
    rec = {"k_total_by_fold": [x["k_total"] for x in folds],
           "n_probes_by_fold": [x["n_probes"] for x in folds],
           "k_total_mean": float(np.mean([x["k_total"] for x in folds])),
           "cap_counter_max": int(max(x["cap_counter"] for x in folds)),
           "n_probes": int(sum(x["n_probes"] for x in folds)), "n_exploration_calls": n_expl,
           "n_plan_calls_verified": int(sum(x["k_plan_verified"] for x in folds)),
           "n_plan_calls_unverified": int(sum(x["k_plan_unverified"] for x in folds)),
           "probe_usd": float(o.probe_usd.sum()), "exploration_usd": expl_usd}
    states = pd.Series([g["state"] for g in groups])
    for s in ("stopped", "out_of_budget", "exploring"):
        rec[f"groups_{s}"] = int((states == s).sum())
    for g in groups:      # the SD the prior alone gives: <= the tolerance means a prior-driven stop
        a0 = g["prior_mean"] * g["prior_strength"]
        g["prior_logit_sd"] = logit_sd(a0, g["prior_strength"] - a0)
    stopped = [g for g in groups if g["state"] == "stopped"]
    rec["k_at_stop_median"] = float(np.median([g["k_at_end"] for g in stopped])) if stopped else float("nan")
    rec["n_stopped_groups_prior_alone_meets_tolerance"] = int(
        sum(g["prior_logit_sd"] <= STOP_POSTERIOR_SD for g in stopped))
    rec["split"] = split
    return rec, groups


def break_even(v2: pd.DataFrame, k0: pd.DataFrame, rec: dict) -> dict:
    """S0.2: the onboarding cost (probes plus exploration calls) and the stream length at which
    v2's per-task saving over k = 0, onboarding cost excluded, pays it back. No positive per-task
    saving means no break-even ("none"). The per-task saving credits explored tasks where the new
    model passed, since the rest of the plan was skipped there."""
    n = len(v2)
    onboard = rec["probe_usd"] + rec["exploration_usd"]
    per_task = (float(k0.cost_usd.sum()) - (float(v2.cost_usd.sum()) - onboard)) / n
    return {"onboarding_usd": onboard,
            "saving_per_task_excl_onboarding_musd": 1e3 * per_task,
            "break_even_tasks": onboard / per_task if per_task > 0 else "none"}


def run():
    d = inputs()
    problems = d["problems"]
    res = {"spec": "docs/research/S4_spec.md §4, §4.1, §4.2", "call_cap": CALL_CAP, "n_boot": N_BOOT,
           "seeds": list(stats.SEEDS), "gate_seed": stats.GATE_SEED, "labels": list(LABELS),
           "populations": {p: d["counts"][p] for p in POPULATIONS}}
    rows, seed0, grows = [], [], []
    for pop in POPULATIONS:
        m = d["mats"][pop]
        for seed in stats.SEEDS:
            ctx = Context(d["verifier"], content=d["content"], problem_of=problems, seed=seed)
            folds = stats.dev_folds(m.tasks, seed)
            fp = fold_problems(folds, problems)
            gate = seed == stats.GATE_SEED
            kinf = run_dev(RouterV1(), m, ctx, folds, seed=seed)
            best = run_dev(BestSingle(), m, ctx, folds, seed=seed) if gate else None
            for M in m.models:
                k0 = run_dev(RouterV1(), m.without(M), ctx, folds, seed=seed)
                for lab in LABELS:
                    v2 = run_dev(RouterV2(M, lab), m, ctx, folds, seed=seed)
                    rec, groups = v2_record(v2, m, ctx, M)
                    fits = v2.attrs["fits"]
                    kmap = {fp[x["fold"]]: k for x, k in zip(fits, rec["k_total_by_fold"])}
                    gmap = {fp[x["fold"]]: k for x, k in zip(fits, rec["n_probes_by_fold"])}
                    km = run_dev(KMatchedAtV2K(M, lab, kmap, gmap), m, ctx, folds, seed=seed)
                    used = [f["chosen"]["labels_used"] for f in km.attrs["fits"]]
                    r = {"population": pop, "seed": seed, "gate_seed": gate, "new_model": M,
                         "labels": lab, "acceptance": metrics.acceptance(v2),
                         "cost_musd_per_task": 1e3 * float(v2.cost_usd.mean()),
                         "k0_acceptance": metrics.acceptance(k0),
                         "k0_cost_musd_per_task": 1e3 * float(k0.cost_usd.mean()),
                         "kmatched_acceptance": metrics.acceptance(km),
                         "kmatched_cost_musd_per_task": 1e3 * float(km.cost_usd.mean()),
                         **break_even(v2, k0, rec),
                         "kinf_acceptance": metrics.acceptance(kinf),
                         "kinf_cost_musd_per_task": 1e3 * float(kinf.cost_usd.mean()),
                         "kmatched_labels_ground_truth": sum(u["ground_truth"] for u in used),
                         "kmatched_labels_verifier": sum(u["verifier"] for u in used),
                         **{k: v for k, v in rec.items()
                            if k not in ("k_total_by_fold", "n_probes_by_fold", "split")},
                         "k_total_by_fold": " ".join(map(str, rec["k_total_by_fold"]))}
                    for tag, ref in (("k0", k0), ("kmatched", km), ("kinf", kinf)):
                        f = metrics.paired_frame(v2, ref, ("x", "r"))
                        one = np.ones(len(f))
                        r[f"gap_vs_{tag}"] = stats.mean_diff("accepted_x", "accepted_r")(f, one)
                        r[f"saving_vs_{tag}"] = stats.saving_pct("cost_x", "cost_r")(f, one)
                        if gate:
                            c = compare(v2, ref, problems)
                            r[f"gap_vs_{tag}_lo"], r[f"gap_vs_{tag}_hi"] = c["gap_points"][1:]
                            r[f"saving_vs_{tag}_lo"], r[f"saving_vs_{tag}_hi"] = c["saving_pct"][1:]
                    for s, ts in rec["split"].items():   # spec §4.2, paired with k = 0
                        r[f"n_{s}"] = len(ts)
                        if not ts:
                            continue
                        f = metrics.paired_frame(v2.loc[ts], k0.loc[ts], ("x", "r"))
                        one = np.ones(len(f))
                        r[f"{s}_acceptance"] = metrics.acceptance(v2.loc[ts])
                        r[f"{s}_k0_acceptance"] = metrics.acceptance(k0.loc[ts])
                        r[f"{s}_cost_diff_musd_vs_k0"] = 1e3 * float((f.cost_x - f.cost_r).mean())
                        if gate and s != "pool":
                            c = compare(v2.loc[ts], k0.loc[ts], problems)
                            r[f"{s}_gap_vs_k0_lo"], r[f"{s}_gap_vs_k0_hi"] = c["gap_points"][1:]
                    rows.append(r)
                    grows += [{"population": pop, "seed": seed, "new_model": M, "labels": lab, **g}
                              for g in groups]
                    if gate:
                        floor = metrics.acceptance(best)
                        refs = {"k0": k0, "kmatched": km, "kinf": kinf, "best_single": best}
                        g0 = row(f"router_v2_{lab}", v2, refs, problems, floor, m)
                        seed0.append({"population": pop, "new_model": M, "labels": lab, **g0,
                                      "floor": floor})
    return res, pd.DataFrame(rows), pd.DataFrame(seed0), pd.DataFrame(grows)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, rows, seed0, groups = run()
    rows.round(10).to_csv(OUT / "s4_v2_seeds.csv", index=False, lineterminator="\n")
    seed0.round(10).to_csv(OUT / "s4_v2_seed0.csv", index=False, lineterminator="\n")
    groups.round(10).to_csv(OUT / "s4_v2_groups.csv", index=False, lineterminator="\n")
    (OUT / "s4_v2.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    g = rows[rows.gate_seed]
    print(g[["population", "new_model", "labels", "acceptance", "cost_musd_per_task", "k_total_mean",
             "groups_stopped", "groups_out_of_budget", "groups_exploring", "gap_vs_k0", "saving_vs_k0",
             "gap_vs_kmatched", "saving_vs_kmatched"]].round(2).to_string(index=False))
