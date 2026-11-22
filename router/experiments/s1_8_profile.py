"""S1.8 step 4: a descriptive profile of LLMRouterBench's code subsets.

Descriptive only: no oracle, DimensionBest or routing gain is computed (those are S3's, once a
population is fixed). Flags for a future audit are counts; no response is read for labelling.
Definitions are in docs/research/S1.8_llmrouterbench.md. The JSON is deterministic (sorted keys,
fixed rounding, no timestamps), so it reproduces byte for byte.

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.s1_8_profile
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import llmrouterbench as L  # noqa: E402
from router.settings import SUCCESS_SCORE  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out/s1_8_profile.json"
NEAR_ZERO_PASS = 0.02          # a model x subset pass rate at or below this is flagged
FAR_BELOW_MEDIAN = 0.5         # flagged when below this fraction of the subset's median pass rate
CAP_MIN_CELLS = 2              # a model's max output tokens hit by at least this many cells
ROUND = 6

# The 2026-08-18 audit (docs/history/handover/2026-08-18-router-goal-review-findings.md §6, §7).
AUDIT_2026_08_18 = {
    "code_rows": 70678, "null_scores": 1, "models": 39, "presence_density": 0.609,
    "models_in_all_four": 0, "binary_subsets": ["humaneval", "livecodebench", "mbpp"],
    "zero_cost_models": {"humaneval": [9, 22], "mbpp": [9, 20], "livecodebench": [9, 38],
                         "swe-bench/verified": [0, 15]},
}


def _r(x):
    if isinstance(x, dict):
        return {str(k): _r(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if np.isnan(x) else round(float(x), ROUND)
    return x


def subset_counts(d: L.LrbData) -> dict:
    out = {}
    for fam, g in d.long.groupby("family"):
        subset = L.CODE_SUBSETS[list(L.FAMILY.values()).index(fam)]
        tasks, models = g.task_id.nunique(), g.model.nunique()
        avail = g[g.available]
        flags = d.cost_flags[d.cost_flags.family == fam]
        out[fam] = {
            "subset": subset, "tasks": tasks, "models": models, "cells": len(g),
            "density": len(g) / (tasks * models),
            "null_scores": int((~g.available).sum()),
            "distinct_score_values": sorted(float(v) for v in avail.score.unique()),
            "zero_cost_models": sorted(flags.model[flags.zero_cost]),
            "partial_zero_cost_models": {m: [int(z), int(c)] for m, z, c in
                                         zip(flags.model[flags.partial_zero_cost],
                                             flags.zero_cost_cells[flags.partial_zero_cost],
                                             flags.cells[flags.partial_zero_cost])},
            "zero_file_cost_models": sorted(d.files.model[(d.files.family == fam) & (d.files.file_cost == 0)
                                                          & ~d.files.router_baseline]),
        }
    return out


def raw_counts(d_all: L.LrbData) -> dict:
    """Counts with the router baseline included, comparable with the 2026-08-18 audit."""
    files = d_all.files
    models = sorted(set(files.model))
    pres = files.groupby("model").subset.nunique()
    return {"cells": len(d_all.cells), "null_scores": int(d_all.cells.score.isna().sum()),
            "models": len(models), "presence_density": len(files) / (len(models) * len(L.CODE_SUBSETS)),
            "models_in_all_four": int((pres == len(L.CODE_SUBSETS)).sum()),
            "files": len(files)}


def presence(d: L.LrbData) -> dict:
    fams = sorted(d.long.family.unique())
    pres = {m: sorted(g.family.unique()) for m, g in d.long.groupby("model")}
    models = sorted(pres)
    pairs = {f"{a}&{b}": sorted(m for m in models if a in pres[m] and b in pres[m])
             for i, a in enumerate(fams) for b in fams[i + 1:]}
    cell_density = len(d.long) / (d.long.task_id.nunique() * len(models))
    return {"models": len(models), "matrix": {m: {f: f in pres[m] for f in fams} for m in models},
            "presence_density": sum(len(v) for v in pres.values()) / (len(models) * len(fams)),
            "cell_density": cell_density,
            "models_in_all_subsets": sorted(m for m in models if len(pres[m]) == len(fams)),
            "shared_models_per_pair": pairs}


def per_model(d: L.LrbData) -> dict:
    out = {}
    c = d.cells.assign(passed=d.cells.score >= SUCCESS_SCORE)
    for (fam, model), g in c.groupby(["family", "model"]):
        a = g[g.score.notna()]
        cost = g.recorded_cost[~g.cost_excluded]
        out.setdefault(fam, {})[model] = {
            "pass_rate": float(a.passed.mean()), "available_cells": len(a),
            "mean_cost_usd": float(cost.mean()) if len(cost) else float("nan"),
            "cost_excluded": bool(g.cost_excluded.all()),
            "mean_input_tokens": float(g.input_tokens.mean()),
            "mean_output_tokens": float(g.output_tokens.mean()),
        }
    return out


def audit_flags(d: L.LrbData, pm: dict) -> dict:
    out = {}
    c = d.cells
    for fam, models in pm.items():
        rates = pd.Series({m: v["pass_rate"] for m, v in models.items()})
        med = float(rates.median())
        g = c[c.family == fam]
        caps = {}
        for m, gm in g.groupby("model"):
            top = gm.output_tokens.max()
            n = int((gm.output_tokens == top).sum())
            if n >= CAP_MIN_CELLS and top > 0:
                caps[m] = {"max_output_tokens": int(top), "cells_at_max": n}
        out[fam] = {
            "median_pass_rate": med,
            "near_zero_models": sorted(rates.index[rates <= NEAR_ZERO_PASS]),
            "all_zero_models": sorted(rates.index[rates == 0]),
            "far_below_median_models": sorted(rates.index[rates < FAR_BELOW_MEDIAN * med]),
            "empty_raw_output_cells": int(g.raw_output_empty.sum()),
            "empty_raw_output_by_model": {m: int(n) for m, n in
                                          g[g.raw_output_empty].model.value_counts().sort_index().items()},
            "zero_cost_cells_of_priced_models": int((~g.cost_excluded & (g.recorded_cost == 0)).sum()),
            "zero_cost_cells_of_priced_models_with_zero_tokens": int(
                (~g.cost_excluded & (g.recorded_cost == 0) & (g.input_tokens == 0) & (g.output_tokens == 0)).sum()),
            "zero_cost_cells_of_priced_models_with_empty_output": int(
                (~g.cost_excluded & (g.recorded_cost == 0) & g.raw_output_empty).sum()),
            "empty_raw_output_scored_0": int((g.raw_output_empty & (g.score == 0)).sum()),
            "processing_failed_cells": int(g.processing_failed.sum()),
            "zero_output_token_cells": int((g.output_tokens == 0).sum()),
            "output_token_caps": caps,
            "tests_shown_in_prompt": int(
                d.tasks[(d.tasks.family == fam)].prompt.str.contains("Your code should pass these tests:",
                                                                     regex=False).sum()),
        }
    return out


def crb_overlap(d: L.LrbData) -> dict:
    """Shared source keys with CodeRouterBench, per subset: any CRB task, its eligible pool, and its
    filtered (headline) view; and whether the task text matches CRB's content_text."""
    from router.data import audit, prompts
    from router.data.sources import load_table
    from router.data.splits import load_problems

    table = load_table()
    probs = load_problems().merge(table[["task_id", "family", "split_kind"]], on="task_id")
    fam = table[table.split_kind == "id"].set_index("task_id").family
    art = audit.load_artefacts()
    views = audit.view_task_ids(fam, set(art["subset_rules"]["lcb"]["excluded_task_ids"]),
                                audit.x13_tasks())
    for v in ("all_eligible", "filtered"):      # the frozen views, rebuilt from the labels
        if audit._sha(views[v]) != art["views"][v]["task_ids_sha256"]:
            raise ValueError(f"view {v} does not match artefacts.yaml")
    eligible, filtered = set(views["all_eligible"]), set(views["filtered"])
    content = prompts.join(list(probs.task_id))
    out = {}
    for f, t in d.tasks.groupby("family"):
        keys = set(t.source_key.dropna())
        p = probs[probs.source_key.isin(keys)]
        text = t.set_index("source_key").origin_query
        exact = ws = 0
        for r in p.itertuples():
            c = content.get(r.task_id)
            if c is None:
                continue
            exact += L.sha256_text(c) == L.sha256_text(text[r.source_key])
            ws += L._ws(c) == L._ws(text[r.source_key])
        out[f] = {
            "lrb_tasks": len(t),
            "shared_keys_any_crb_task": int(p.source_key.nunique()),
            "shared_keys_eligible_crb": int(p[p.task_id.isin(eligible)].source_key.nunique()),
            "shared_keys_filtered_view": int(p[p.task_id.isin(filtered)].source_key.nunique()),
            "crb_tasks_sharing_a_key": len(p),
            "crb_problem_ids_sharing_a_key": int(p.problem_id.nunique()),
            "crb_families": {k: int(v) for k, v in p.family.value_counts().sort_index().items()},
            "crb_filtered_tasks_sharing_a_key": int(p.task_id.isin(filtered).sum()),
            "text_equal_exact_crb_tasks": int(exact),
            "text_equal_whitespace_crb_tasks": int(ws),
        }
    return out


def model_names_vs_crb(d: L.LrbData) -> dict:
    from router.data.prices import CRB_PRICES
    lrb = sorted(d.long.model.unique())
    return {"crb_models": sorted(CRB_PRICES), "exact_name_matches": sorted(set(lrb) & set(CRB_PRICES)),
            "lrb_models": lrb}


def implied_prices(d: L.LrbData) -> dict:
    """Least-squares $/M input and output price per model from its non-excluded cells with cost
    > 0: recorded cost ~ a * input_tokens + b * output_tokens. Descriptive; nothing is priced."""
    out = {}
    c = d.cells[~d.cells.cost_excluded & (d.cells.recorded_cost > 0)]
    for m, g in c.groupby("model"):
        X = g[["input_tokens", "output_tokens"]].to_numpy(float) / 1e6
        y = g.recorded_cost.to_numpy(float)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ coef
        out[m] = {"input_per_m": float(coef[0]), "output_per_m": float(coef[1]), "cells": len(g),
                  "max_abs_rel_residual": float(np.max(np.abs(resid) / y))}
    return out


def build() -> dict:
    d = L.load_cached()
    d_all = L.load(include_router_baselines=True)
    pm = per_model(d)
    return _r({
        "pin": {"archive": "NPULH/LLMRouterBench@b95870f9e60c589e82f7972c91ffa8de9ca89f85",
                "tarball_sha256": "b79f8cde1a6f029c2efa663a3a3b6f7748defb22341fe59f328cebef6648c8f1"},
        "router_baselines_excluded": sorted(L.ROUTER_BASELINES),
        "raw_counts_with_router_baseline": raw_counts(d_all),
        "audit_2026_08_18": AUDIT_2026_08_18,
        "subsets": subset_counts(d),
        "presence": presence(d),
        "per_model": pm,
        "audit_flags": audit_flags(d, pm),
        "source_keys": {f: {"mapped": int(t.source_key.notna().sum()), "distinct": int(t.source_key.nunique()),
                            "tasks": len(t), "method": L.MAPPING[t.subset.iloc[0]]}
                        for f, t in d.tasks.groupby("family")},
        "humaneval_index_is_number_plus_1": int(
            (d.tasks[d.tasks.family == "lrb_humaneval"].apply(
                lambda r: int(r.source_key.split("/")[1]) + 1 == r["index"], axis=1)).sum()),
        "crb_overlap": crb_overlap(d),
        "model_names_vs_crb": model_names_vs_crb(d),
        "implied_prices": implied_prices(d),
        "thresholds": {"near_zero_pass": NEAR_ZERO_PASS, "far_below_median": FAR_BELOW_MEDIAN,
                       "cap_min_cells": CAP_MIN_CELLS, "success_score": SUCCESS_SCORE},
    })


def dumps(profile: dict) -> str:
    return json.dumps(profile, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dumps(build()), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
