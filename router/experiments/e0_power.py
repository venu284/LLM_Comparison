"""E0 power: CI half-widths and gate pass rates for S0.3 and S0.6 (P0.4).

Port of the four original scripts in `router/experiments/e0_power/` onto the current definitions:
success = score 1.0 (S0.1); unavailable cells (empty X13 responses, missing Qwen3-Max runs) treated
as missing, not as failures, by taking the complete case (S0.11); the real X13 visible-test
verdicts instead of a random false-pass rate (R1); the best single model chosen inside the
training folds (S0.3); 5-fold cross-fitting by problem (S0.6); cluster bootstrap over problems
(S0.8).

It reuses the P0.5 and P0.6 functions and their seed scheme (fold seed s, bootstrap seed 1000 + s),
so the seed-0 numbers for the shared policies equal P0.6's exactly; the script checks this.

Three arms:
- `headline_complete_case`: the 1,907-task complete-case headline set of P0.5/P0.6;
- `headline_evaluation`: the same set with a provisional 20% development split of problems removed
  (S0.6), i.e. the population S0.3 is scored on. The development-split seed is not fixed by the
  plan; the one used here is provisional;
- `pool`: the full artefact-filtered pool (exec dimensions, suspected families removed), complete
  case, for DP1a. Only policies that need no verdicts or prompts: the oracle and the lookup routes.

Stand-in policies are not routers. Each draw re-randomises folds and bootstrap on fixed data, so
the pass rates show how stable a gate verdict is to the draw; they are not sampling power. The
margin for 80% power at a true gap of 0 is the original scripts' formula,
(z_0.975 + z_0.80) / z_0.975 x CI half-width, about 1.43 x half-width.

Every count is provisional (S0): S1.4 and S1.7 fix the final population, and S3.7 reruns this.

Run from the repo root:
    pipeline/venv/bin/python router/experiments/e0_power.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router import settings as S  # noqa: E402
from router.experiments import p0_5_recount as rc  # noqa: E402
from router.experiments import p0_6_cellbest_bound as p6  # noqa: E402

OUT_JSON = REPO_ROOT / "router/experiments/out/e0_power.json"
P0_6_JSON = REPO_ROOT / "router/experiments/out/p0_6_cellbest_bound.json"

DEV_SEED_PROVISIONAL = 0   # S0.6 "drawn once with a recorded seed": not fixed yet (open item)
N_DRAWS = 20               # fold draws for pass rates; power4.py used 20 repetitions
POWER_FACTOR = float((norm.ppf(0.975) + norm.ppf(0.80)) / norm.ppf(0.975))   # 1.4294

# (key in P0.6 run_seed metrics, label). Rules use predicted cells: the router cannot see the source.
HEADLINE_POLICIES = [
    ("cellbest_pred", "CellBest, predicted cells (the S0.3(c) comparator)"),
    ("ruleA_accept_pred", "Per-cell Rule A, no-signal cells accepted (no oracle)"),
    ("ruleA_escalate_pred", "Per-cell Rule A, no-signal cells escalated (no oracle)"),
    ("ruleB_accept_pred", "Per-cell Rule B, no-signal cells accepted (no oracle)"),
    ("ruleB_escalate_pred", "Per-cell Rule B, no-signal cells escalated (no oracle)"),
    ("ruleA_bound_pred", "Per-cell Rule A, oracle on no-signal cells (upper bound)"),
    ("ruleB_bound_pred", "Per-cell Rule B, oracle on no-signal cells (upper bound)"),
    ("ladder_escalate", "Best fixed ladder, escalate mode (in-sample pick: optimistic)"),
]
POOL_ROUTES = ["oracle", "dimensionbest", "cost_aware_dimensionbest", "cellbest_true"]


# ----------------------------------------------------------------------------- populations

def pool_pop(d) -> rc.Pop:
    """Artefact-filtered pool: exec dimensions, suspected families removed (no X13 requirement)."""
    susp = np.array([bool(rc.SUSPECTED.search(t)) for t in d["S"].index])
    return rc.Pop(d, (d["dim"].isin(rc.EXEC_DIMS) & ~susp).values)


def dev_problems(pool: rc.Pop, seed: int) -> set:
    """Provisional development split: DEV_FRACTION of the pool's problems, drawn once (S0.6)."""
    probs = np.array(sorted(pool.problem.unique()))
    perm = np.random.default_rng(seed).permutation(probs)
    return set(perm[: int(round(S.DEV_FRACTION * len(probs)))])


# ----------------------------------------------------------------------------- summaries

def half_width(ci3) -> float:
    return float((ci3[2] - ci3[1]) / 2)


def power_at_margin(margin: float, hw: float) -> float:
    """P(CI lower bound >= -margin) for a router whose true gap is 0, with a normal CI of half-width
    hw: Phi(z (margin / hw) - z), z = z_0.975. At margin = POWER_FACTOR x hw this is 0.80."""
    z = norm.ppf(0.975)
    return float(norm.cdf(z * margin / hw - z))


def gate_view(m: dict) -> dict:
    """Gap and saving CIs against both comparators, half-widths and lower-bound pass flags."""
    out = {}
    for ref in ("best_single", "cellbest_pred"):
        v = m.get(f"vs_{ref}")
        if v is None:
            continue
        out[ref] = {"gap_points": v["gap_points"], "saving_pct": v["saving_pct"],
                    "gap_half_width": half_width(v["gap_points"]),
                    "saving_half_width": half_width(v["saving_pct"])}
    if "best_single" in out:
        b = out["best_single"]
        out["passes_ab"] = bool(b["saving_pct"][1] >= S.GATE_SAVING and b["gap_points"][1] >= S.GATE_GAP)
    if "cellbest_pred" in out:
        c = out["cellbest_pred"]
        out["passes_c"] = bool(c["saving_pct"][1] > S.C_SAVING and c["gap_points"][1] >= S.C_GAP)
    if "passes_ab" in out and "passes_c" in out:
        out["passes_abc"] = out["passes_ab"] and out["passes_c"]
    return out


def aggregate(views: list[dict]) -> dict:
    """Mean half-widths, pass rates over draws, and the margin for 80% power at a true gap of 0."""
    agg = {"n_draws": len(views)}
    for ref in ("best_single", "cellbest_pred"):
        if ref not in views[0]:
            continue
        g = [v[ref]["gap_half_width"] for v in views]
        s = [v[ref]["saving_half_width"] for v in views]
        agg[ref] = {"gap_half_width_mean": float(np.mean(g)), "gap_half_width_range": [float(min(g)), float(max(g))],
                    "saving_half_width_mean": float(np.mean(s)),
                    "gap_point_mean": float(np.mean([v[ref]["gap_points"][0] for v in views])),
                    "saving_point_mean": float(np.mean([v[ref]["saving_pct"][0] for v in views])),
                    "gap_lower_mean": float(np.mean([v[ref]["gap_points"][1] for v in views])),
                    "saving_lower_mean": float(np.mean([v[ref]["saving_pct"][1] for v in views])),
                    "margin_80pct_power_true_gap_0": float(POWER_FACTOR * np.mean(g)),
                    "power_at_gate_gap_true_gap_0": power_at_margin(-S.GATE_GAP if ref == "best_single" else -S.C_GAP,
                                                                    float(np.mean(g)))}
    for k in ("passes_ab", "passes_c", "passes_abc"):
        if k in views[0]:
            agg[f"p_{k}"] = float(np.mean([v[k] for v in views]))
    return agg


def summarise(per_draw: list[dict], names) -> dict:
    """Per policy: the gate seed in full, seeds 0..N_SEEDS-1 (S0.8) and all draws (power)."""
    out = {}
    for key in names:
        views = [d[key] for d in per_draw]
        out[key] = {"gate_seed": views[S.GATE_SEED],
                    "s0_8_seeds": aggregate(views[: S.N_SEEDS]),
                    "all_draws": aggregate(views)}
    return out


# ----------------------------------------------------------------------------- headline arms

def headline_draw(cc: rc.Pop, s: int, prompts, cells, cell_names, mats, n_boot: int) -> dict:
    r = p6.run_seed(cc, s, prompts, cells, cell_names, mats, n_boot)
    m = dict(r["metrics"])
    m["ladder_escalate"] = r["fixed_ladders"]["escalate"]["best_ladder_metrics"]
    views = {k: gate_view(m[k]) for k, _ in HEADLINE_POLICIES}
    views["_ladder_escalate_name"] = r["fixed_ladders"]["escalate"]["best_ladder"]
    views["_predictor_accuracy"] = r["predictor"]["accuracy"]
    return views


def headline_arm(d, cc: rc.Pop, n_draws: int, n_boot: int, label: str) -> dict:
    prompts, _ = p6.load_prompts(cc.ids)
    if int(prompts.notna().sum()) != len(cc):
        raise SystemExit(f"{label}: prompt join incomplete; stop and report")
    cells = rc.ra_cells(cc.ids, d["dim"]).values
    cell_names = sorted(set(cells))
    mats = {m: p6.policy_matrices(cc, m) for m in p6.MODES}
    per_draw = []
    for s in range(n_draws):
        per_draw.append(headline_draw(cc, s, prompts.values, cells, cell_names, mats, n_boot))
        print(f"  {label} draw {s}: done", flush=True)
    return {"n": len(cc), "n_problems": int(cc.problem.nunique()),
            "policies": {k: lbl for k, lbl in HEADLINE_POLICIES},
            "ladder_escalate_pick_by_draw": [p["_ladder_escalate_name"] for p in per_draw],
            "predictor_accuracy_by_draw": [p["_predictor_accuracy"] for p in per_draw],
            "results": summarise(per_draw, [k for k, _ in HEADLINE_POLICIES])}


def dev_seed_sensitivity(d, cc: rc.Pop, pool: rc.Pop, dev_seeds, n_boot: int) -> dict:
    """The evaluation population depends on the development-split draw, which the plan has not
    fixed. For each development seed, the gate-seed draw only: CellBest's own result against the
    best single model, and each stand-in's gap half-width and pass flags."""
    out = {}
    for ds in dev_seeds:
        dev = dev_problems(pool, ds)
        ev = cc.subset(~cc.problem.isin(dev).values)
        prompts, _ = p6.load_prompts(ev.ids)
        cells = rc.ra_cells(ev.ids, d["dim"]).values
        cell_names = sorted(set(cells))
        mats = {m: p6.policy_matrices(ev, m) for m in p6.MODES}
        v = headline_draw(ev, S.GATE_SEED, prompts.values, cells, cell_names, mats, n_boot)
        out[str(ds)] = {"n": len(ev),
                        **{k: {r: {"gap_points": v[k][r]["gap_points"], "saving_pct": v[k][r]["saving_pct"],
                                   "gap_half_width": v[k][r]["gap_half_width"]}
                               for r in ("best_single", "cellbest_pred") if r in v[k]}
                           | {f: v[k][f] for f in ("passes_ab", "passes_c") if f in v[k]}
                           for k, _ in HEADLINE_POLICIES}}
        print(f"  dev seed {ds}: done", flush=True)
    return out


# ----------------------------------------------------------------------------- pool arm (DP1a)

def oracle(V: np.ndarray, Cm: np.ndarray):
    """Per task: the best value any model reaches, at the cost of the cheapest model reaching it."""
    best = V.max(axis=1)
    cost = np.where(V >= best[:, None] - p6.TOL, Cm, np.inf).min(axis=1)
    return best, cost


def pool_draw(p: rc.Pop, s: int, V: np.ndarray, Cm: np.ndarray, sidx, X, names, n_boot: int) -> dict:
    """One fold draw on the pool: lookup routes from P0.5 fitted on training folds, the oracle,
    and the best single model, all against the training-fold best single model."""
    fold = rc.folds(p, s)
    acc = {r: np.zeros(len(p)) for r in POOL_ROUTES + ["best_single"]}
    cost = {r: np.zeros(len(p)) for r in acc}
    acc["oracle"], cost["oracle"] = oracle(V, Cm)
    for f in range(S.N_FOLDS):
        te, tr = np.where(fold == f)[0], np.where(fold != f)[0]
        res, _ = rc.ra_routes_fold(p, tr, te, sidx, X, V, Cm, names)
        for r, key in (("dimensionbest", "dimensionbest"), ("cost_aware_dimensionbest", "cost_aware_dimensionbest"),
                       ("cellbest_true", "cellbest"), ("best_single", "best_single")):
            acc[r][te], cost[r][te] = res[key]
    cols = {f"a_{r}": acc[r] for r in acc} | {f"c_{r}": cost[r] for r in cost} | {"n": np.ones(len(p))}
    Bt = rc.cluster_boot(p.problem.values, cols, n_boot, seed=1000 + s)
    views = {}
    for r in POOL_ROUTES:
        m = {"vs_best_single": {"gap_points": p6.ci(100 * (Bt[f"a_{r}"] - Bt["a_best_single"]) / Bt["n"]),
                                "saving_pct": p6.ci(100 * (1 - Bt[f"c_{r}"] / Bt["c_best_single"]))}}
        v = gate_view(m)
        v["passes_dp1a"] = bool(v["best_single"]["gap_points"][1] >= S.DP1A_MIN_LOWER)
        views[r] = v
    return views


def pool_arm(d, n_draws: int, n_boot: int) -> dict:
    full = pool_pop(d)
    p = full.subset(~full.unavail.any(axis=1).values)
    cells = rc.ra_cells(p.ids, d["dim"])
    cell_names = sorted(cells.unique())
    sidx = cells.map({x: i for i, x in enumerate(cell_names)}).values
    names = [f"direct:{m}" for m in p.models]
    # Two readings of DP1a's "AvgPerf points": acceptance under S0.1 (score = 1.0, the primary
    # reading) and the mean released score (sensitivity; partial credit counts).
    score = d["S"].loc[p.ids][p.models].values
    if np.isnan(score).any():
        raise SystemExit("pool: released score missing on a complete-case cell; stop and report")
    readings = {"acceptance_score_1": p.B.values, "mean_released_score": score}
    out = {"n_before_complete_case": len(full), "n": len(p), "n_problems": int(p.problem.nunique()),
           "dropped_unavailable": {"tasks": int(full.unavail.any(axis=1).sum()),
                                   "missing_runs": int(full.M.any(axis=1).sum()),
                                   "empty_x13_responses": int(full.E.any(axis=1).sum())},
           "routes": POOL_ROUTES, "readings": {}}
    for name, V in readings.items():
        per_draw = [pool_draw(p, s, V, p.c.values, sidx, len(cell_names), names, n_boot) for s in range(n_draws)]
        res = summarise(per_draw, POOL_ROUTES)
        for r in POOL_ROUTES:
            for k, sl in (("s0_8_seeds", slice(0, S.N_SEEDS)), ("all_draws", slice(None))):
                res[r][k]["p_passes_dp1a"] = float(np.mean([pd_[r]["passes_dp1a"] for pd_ in per_draw[sl]]))
        out["readings"][name] = res
        print(f"  pool ({name}): done", flush=True)
    return out


# ----------------------------------------------------------------------------- checks

def check_against_p0_6(arm: dict) -> dict:
    """The gate-seed CIs of the shared policies must equal P0.6's committed output exactly."""
    ref = json.loads(P0_6_JSON.read_text())["seed0"]["metrics"]
    res = arm["results"]
    mism = []
    for key, _ in HEADLINE_POLICIES:
        if key == "ladder_escalate":
            continue
        g = res[key]["gate_seed"]
        for r in ("best_single", "cellbest_pred"):
            if r not in g:
                continue
            for q in ("gap_points", "saving_pct"):
                if not np.allclose(g[r][q], ref[key][f"vs_{r}"][q], rtol=0, atol=1e-9):
                    mism.append(f"{key} vs {r} {q}")
    if mism:
        raise SystemExit(f"Port does not reproduce P0.6 on seed 0, stop and report: {mism}")
    return {"matches_p0_6_seed0": True, "policies_checked": [k for k, _ in HEADLINE_POLICIES if k != "ladder_escalate"]}


# ----------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=N_DRAWS, help="fold draws (seeds 0..draws-1)")
    ap.add_argument("--boot", type=int, default=S.N_BOOT, help="cluster-bootstrap resamples")
    ap.add_argument("--dev-seed", type=int, default=DEV_SEED_PROVISIONAL, help="provisional development-split seed")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--no-p0-6-check", action="store_true", help="skip the P0.6 seed-0 check (for a new population)")
    a = ap.parse_args(argv)
    if a.draws < S.N_SEEDS:
        raise SystemExit(f"--draws must be at least {S.N_SEEDS} (the S0.8 seeds)")
    t0 = time.time()
    d = rc.load()

    cc = p6.complete_case(d)
    print(f"headline complete case: {len(cc)} tasks")
    arm_cc = headline_arm(d, cc, a.draws, a.boot, "complete case")
    check = None if a.no_p0_6_check else check_against_p0_6(arm_cc)

    pool = pool_pop(d)
    dev = dev_problems(pool, a.dev_seed)
    ev = cc.subset(~cc.problem.isin(dev).values)
    print(f"headline evaluation population: {len(ev)} tasks ({len(cc) - len(ev)} in the development split)")
    arm_ev = headline_arm(d, ev, a.draws, a.boot, "evaluation")
    arm_ev["development_split"] = {"seed": a.dev_seed, "pool_problems": int(pool.problem.nunique()),
                                   "dev_problems": len(dev), "headline_tasks_removed": len(cc) - len(ev)}
    print("development-split sensitivity")
    arm_ev["dev_seed_sensitivity"] = dev_seed_sensitivity(d, cc, pool, range(S.N_SEEDS), a.boot)

    print("pool (DP1a)")
    arm_pool = pool_arm(d, a.draws, a.boot)

    res = {
        "settings": {"draws": a.draws, "s0_8_seeds": S.N_SEEDS, "gate_seed": S.GATE_SEED, "boot": a.boot,
                     "folds": S.N_FOLDS, "dev_fraction": S.DEV_FRACTION, "dev_seed_provisional": a.dev_seed,
                     "power_factor": POWER_FACTOR, "verifier_run_usd": rc.VERIFIER_RUN_USD,
                     "thresholds": {"gate_saving": S.GATE_SAVING, "gate_gap": S.GATE_GAP, "c_saving": S.C_SAVING,
                                    "c_gap": S.C_GAP, "dp1a_min_lower": S.DP1A_MIN_LOWER}},
        "p0_6_check": check,
        "headline_complete_case": arm_cc,
        "headline_evaluation": arm_ev,
        "pool": arm_pool,
        "runtime_s": round(time.time() - t0, 1),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(f"wrote {a.out.relative_to(REPO_ROOT) if a.out.is_relative_to(REPO_ROOT) else a.out} in {res['runtime_s']} s")


if __name__ == "__main__":
    main()
