"""S3.7: power of the S0.3 gap test at -2 points (S3 spec D10, ruled (c) with (b) as a cross-check).
Development split only; no evaluation outcome is read.

Stand-ins (baselines, not routers): the best single model, CellBest-pred and Rule A escalated with
predicted cells. The bound stand-in is left out: the D4 ruling lets only the S3.2 module import the
oracle (spec §7, 2026-11-23). Comparisons: CellBest-pred and Rule A escalated against the best
single model; the best single model and Rule A escalated against CellBest-pred. A draw "passes"
when the gate interval's lower bound of the gap (`stats.gate_interval`) is >= -2 points.

(c) Resampling. Population: the headline development split, complete case (195 tasks, 180
problems). Each resample draws 774 problems with replacement from the 180 (the size of the
headline evaluation population, complete case), stream 13 (`s3_power_resample`, under
`SeedSequence([1000 * seed + r, 13])` for resample r of seed `seed`); every copy of a
drawn problem keeps all its tasks. All draws of one original problem share a fold (five folds,
round-robin over the drawn problems in a seeded order from the same stream), so no task text is on
both sides of a fit. Each stand-in is cross-fitted with `run_dev` on those folds; the gap CI
clusters on the draws. Two points per comparison:
- power at the development gap: the share of resamples whose lower bound is >= -2;
- power at gap 0: the same with the paired differences shifted by minus the stand-in's
  development gap (seed 0, the real development run), which moves the interval by that constant
  and leaves its width unchanged.
True-gap assumption: the resampling treats the development split as the population, so each
stand-in's true gap is taken to be its development gap. It is an assumption, not an estimate of the
evaluation gap. 100 resamples per seed, seeds 0 to 4 (seed 0 decides nothing here; power is
information); the Monte Carlo standard error is reported.

(b) Projection: the gate-seed development CI half-width h_dev (`gate_interval`), scaled to
h = h_dev * sqrt(180 / 774); power at true gap g = Phi(z (g + 2) / h - z), z = z_0.975 (the E0
formula), at g = the development gap and g = 0.

Added after checkpoint 2 (reviewer check of checkpoint 2, item 3; spec §7, 2026-11-24):
- `summary`: (c) and (b) side by side for each comparison, with the half-widths; (c) pooled over
  all draws of seeds 0 to 4 and on seed 0 alone.
- A labelled post-hoc variant of "power at gap 0": shift each draw by minus the resample-world mean
  gap (the mean of the resampled gaps over all draws of seeds 0 to 4, per comparison) instead of the
  development gap. Post hoc and data-dependent: chosen after the resampled gaps were seen to differ
  from the development gaps. It uses no new draw.
- A width check: the E0 formula evaluated with the median resampled half-width in place of the
  projected one, to separate the effect of the wider interval from that of the shifted mean gap.

Nesting: the stand-ins select nothing, so nested cross-fitting reduces to the five folds.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.s3_power
Writes router/experiments/out/s3_7_power.json and s3_7_power_draws.csv.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

from router.baselines.common import ESCALATE, Context
from router.baselines.lookup import CellBest, CellRule
from router.baselines.run import run_dev
from router.baselines.static import BestSingle
from router.data.prompts import join
from router.data.schema import OutcomeMatrix
from router.experiments.s2_baselines import population
from router.settings import CI_LEVEL, REPO_ROOT
from router.sim import metrics, stats
from router.sim.verifier import ReplayVerifier

OUT = REPO_ROOT / "router" / "experiments" / "out"
N_EVAL_PROBLEMS = 774            # headline evaluation population, complete case (spec D10)
N_RESAMPLES = 100                # per seed (spec §7, 2026-11-23)
GAP_MIN = -2.0
BEST, CBP, RA = "best_single", "cellbest_pred", "ruleA_escalate_pred"
COMPARISONS = ((CBP, BEST), (RA, BEST), (BEST, CBP), (RA, CBP))


def stand_ins():
    return [BestSingle(), CellBest("pred"), CellRule("A", ESCALATE, "pred")]


class CopyVerifier:
    """The replay verifier seen through renamed task copies (copy id -> original id)."""

    def __init__(self, base, original: dict):
        self.base, self.original = base, original
        self.mode, self.sensitivity = base.mode, base.sensitivity
        by_task = {}
        for t, m in base.empty_cells():
            by_task.setdefault(t, set()).add(m)
        self._empty = {(c, m) for c, o in original.items() for m in by_task.get(o, ())}

    def check(self, task_id, model):
        return self.base.check(self.original[task_id], model)

    def has_record(self, task_id):
        return self.base.has_record(self.original[task_id])

    def empty_cells(self):
        return set(self._empty)


def resample(m: OutcomeMatrix, problems: pd.Series, seed: int, r: int):
    """One resample: (matrix of copies, folds, copy -> original, copy -> draw cluster)."""
    g = stats.rng(seed * 1000 + r, "s3_power_resample")
    probs = np.array(sorted(problems.reindex(m.tasks).unique()))
    drawn = probs[g.integers(0, len(probs), N_EVAL_PROBLEMS)]
    uniq = np.array(sorted(set(drawn)))
    order = uniq[g.permutation(len(uniq))]
    fold_of = {p: i % 5 for i, p in enumerate(order)}
    by_prob = pd.Series(m.tasks, index=problems.reindex(m.tasks).to_numpy())
    ids, orig, cluster, fold = [], {}, {}, {}
    for k, p in enumerate(drawn):
        for t in np.atleast_1d(by_prob.loc[p]):
            c = f"{t}#r{k}"
            ids.append(c)
            orig[c], cluster[c], fold[c] = t, f"{p}#r{k}", fold_of[p]
    src = [orig[c] for c in ids]

    def take(frame):
        x = frame.loc[src].copy()
        x.index = pd.Index(ids)
        return x

    mm = OutcomeMatrix(take(m.score), take(m.cost), take(m.dimension), take(m.family),
                       m.name + "/resample", available=take(m.available).astype(bool),
                       unavailable_reason=take(m.unavailable_reason))
    return mm, pd.Series(fold, name="dev_fold").loc[ids], orig, pd.Series(cluster).loc[ids]


def gap_ci(o, ref, clusters) -> tuple[float, float, float]:
    f = metrics.paired_frame(o, ref, ("x", "r"))
    iv = stats.gate_interval(f, stats.mean_diff("accepted_x", "accepted_r"), clusters.reindex(f.index))
    return iv.point, iv.lo, iv.hi


def projection(dev: dict, problems: pd.Series, n_dev: int) -> dict:
    out = {}
    for a, b in COMPARISONS:
        p, lo, hi = gap_ci(dev[a], dev[b], problems)
        h = (hi - lo) / 2 * np.sqrt(n_dev / N_EVAL_PROBLEMS)
        out[f"{a}_vs_{b}"] = {"development_gap": p, "development_ci": [lo, hi],
                              "projected_half_width": h,
                              "power_at_development_gap": e0_power(p, h),
                              "power_at_gap_0": e0_power(0.0, h)}
    return out


def run(n_resamples: int = N_RESAMPLES, seeds=stats.SEEDS):
    pop, hd, ed, problems, comp = population()
    base = ReplayVerifier.from_x13()
    content = join(list(pop.tasks))
    ctx = Context(base, content=content, problem_of=problems)
    folds0 = stats.dev_folds(pop.tasks, stats.GATE_SEED)
    dev = {b.name: run_dev(b, pop, ctx, folds0, seed=stats.GATE_SEED) for b in stand_ins()}
    n_dev = int(problems.reindex(pop.tasks).nunique())
    proj = projection(dev, problems, n_dev)
    dev_gap = {k: v["development_gap"] for k, v in proj.items()}

    draws = []
    for seed in seeds:
        for r in range(n_resamples):
            mm, folds, orig, cluster = resample(pop, problems, seed, r)
            prob_of = pd.Series({c: problems[o] for c, o in orig.items()})
            cctx = Context(CopyVerifier(base, orig), content=pd.Series({c: content[o] for c, o in orig.items()}),
                           problem_of=prob_of)
            outs = {b.name: run_dev(b, mm, cctx, folds, seed=seed) for b in stand_ins()}
            for a, b in COMPARISONS:
                p, lo, hi = gap_ci(outs[a], outs[b], cluster)
                key = f"{a}_vs_{b}"
                draws.append({"seed": seed, "resample": r, "comparison": key, "gap": p, "lo": lo,
                              "hi": hi, "n_tasks": len(mm), "n_draws": N_EVAL_PROBLEMS,
                              "pass_at_development_gap": bool(lo >= GAP_MIN),
                              "pass_at_gap_0": bool(lo - dev_gap[key] >= GAP_MIN)})
            print(f"seed {seed} resample {r}", file=sys.stderr, flush=True)
    d = pd.DataFrame(draws)
    mean_gap = d.groupby("comparison").gap.mean()
    d["pass_at_gap_0_resample_mean_posthoc"] = (d.lo - d.comparison.map(mean_gap)) >= GAP_MIN
    res = {"spec": "docs/research/S3_analysis_spec.md D10", "population": comp,
           "n_development_problems": n_dev, "n_evaluation_problems": N_EVAL_PROBLEMS,
           "n_resamples_per_seed": n_resamples, "gap_threshold": GAP_MIN,
           "stand_ins": [b.name for b in stand_ins()],
           "bound_stand_in": "left out: the D4 ruling restricts the oracle to the S3.2 module (spec §7)",
           "true_gap_assumption": "each stand-in's true gap is taken to be its development gap "
                                  "(seed 0); not an estimate of the evaluation gap",
           "nesting": "the stand-ins select nothing, so nested cross-fitting reduces to five folds",
           "projection_b": proj, "resampling_c": {},
           "posthoc_gap_0_shift": "post hoc, data-dependent: each draw shifted by minus the "
                                  "resample-world mean gap (all draws of seeds 0 to 4, per "
                                  "comparison); spec §7, 2026-11-24",
           "summary": {}}
    for key, g in d.groupby("comparison"):
        rows = {}
        for s, gs in g.groupby("seed"):
            rows[int(s)] = power_row(gs)
        res["resampling_c"][key] = {"development_gap": dev_gap[key],
                                    "resample_mean_gap": float(mean_gap[key]), "by_seed": rows}
        res["summary"][key] = side_by_side(proj[key], g, float(mean_gap[key]))
    return res, d


def power_row(gs: pd.DataFrame) -> dict:
    """Power, its Monte Carlo standard error and the median half-width over a set of draws."""
    n = len(gs)
    out = {"n": n, "median_half_width": float(((gs.hi - gs.lo) / 2).median())}
    for tag, col in (("development_gap", "pass_at_development_gap"), ("gap_0", "pass_at_gap_0"),
                     ("gap_0_resample_mean_posthoc", "pass_at_gap_0_resample_mean_posthoc")):
        p = float(gs[col].mean())
        out[f"power_at_{tag}"] = p
        out[f"mc_se_{tag}"] = float(np.sqrt(p * (1 - p) / n))
    return out


def e0_power(gap: float, h: float) -> float:
    """The E0 formula: P(CI lower bound >= -2) at true gap `gap` with half-width `h`."""
    z = norm.ppf(0.5 + CI_LEVEL / 2)
    return float(norm.cdf(z * (gap - GAP_MIN) / h - z))


def side_by_side(proj: dict, g: pd.DataFrame, mean_gap: float) -> dict:
    """(c) and (b) for one comparison, at the development gap and at gap 0, with half-widths."""
    h_res = float(((g.hi - g.lo) / 2).median())
    return {"development_gap": proj["development_gap"], "resample_mean_gap": mean_gap,
            "resampling_c_all_seeds": power_row(g),
            "resampling_c_seed_0": power_row(g[g.seed == stats.GATE_SEED]),
            "projection_b": {"half_width": proj["projected_half_width"],
                             "power_at_development_gap": proj["power_at_development_gap"],
                             "power_at_gap_0": proj["power_at_gap_0"]},
            "width_check_e0_at_resampled_half_width": {
                "half_width": h_res,
                "power_at_development_gap": e0_power(proj["development_gap"], h_res),
                "power_at_gap_0": e0_power(0.0, h_res)}}


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    res, d = run()
    d.to_csv(OUT / "s3_7_power_draws.csv", index=False, lineterminator="\n")
    (OUT / "s3_7_power.json").write_text(json.dumps(_clean(res), indent=1, sort_keys=True) + "\n")
    print(json.dumps(_clean({k: res[k] for k in ("projection_b", "resampling_c")}), indent=1))
