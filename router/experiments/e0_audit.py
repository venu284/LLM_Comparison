"""S1.7 checkpoint 1: what can be known without reading a response.

Rules: docs/research/S1.7_audit_criteria.md (§5), committed before this script was first run.
Inputs are the released scores and the response matrix reduced to presence, length and SHA-256
(`router.data.audit.response_stats`); no response text is read or written. No oracle or
DimensionBest gain is computed (S1.7: gain shares come only after the labels are committed).

    PYTHONPATH=.:pipeline pipeline/venv/bin/python -m router.experiments.e0_audit

Outputs (byte-for-byte reproducible on the pinned data), under router/experiments/out/:
  e0_audit.json                    flagship, mapping, flags, truncation screen, identical prompts,
                                   coverage totals;
  e0_audit_family_model.csv        family × model table (§5.2);
  e0_audit_coverage.csv            family × response key coverage (§5.4), keys mapped by §5.5;
  e0_audit_coverage_name_match.csv the same with keys mapped by name (NOT the §5.5 rule; for the
                                   empty-response decision while §5.5's outcome is reviewed);
  e0_audit_truncation_cells.csv    cells flagged by the truncation screen (§5.3).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import audit  # noqa: E402

OUT = REPO_ROOT / "router/experiments/out"
DIGITS = 6


def _round(df: pd.DataFrame) -> pd.DataFrame:
    return df.round(DIGITS)


def _csv(df: pd.DataFrame, path: Path) -> None:
    _round(df).to_csv(path, index=False, lineterminator="\n")


def _clean(x):
    if isinstance(x, float):
        return round(x, DIGITS)
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item"):
        return _clean(x.item())
    return x


OUTPUTS = ("e0_audit.json", "e0_audit_family_model.csv", "e0_audit_coverage.csv",
           "e0_audit_coverage_name_match.csv", "e0_audit_truncation_cells.csv")


def main(out_dir: Path = OUT) -> dict:
    long = audit.released_long()
    flagship = audit.flagship_models()           # §5.1, before the table
    stats = audit.response_stats()

    aliases = audit.alias_shares(stats)
    corr = audit.length_correlation(stats, long)
    mapping = audit.map_keys(aliases, corr)
    cpt = audit.chars_per_token(stats, long)
    cpt_control = audit.over_ratio_matrix(stats, long)
    const = audit.constant_token_records(long)
    key_tasks_by_family = (stats[["task_id", "key"]].assign(family=lambda d: d.task_id.map(
        long.drop_duplicates("task_id").set_index("task_id")["family"]))
        .groupby(["key", "family"]).size())

    table = audit.family_model_table(long)
    flags = audit.family_flags(table, flagship)
    screen, screened = audit.truncation_screen(long)
    cov = audit.coverage(stats, long, mapping["key_to_model"])
    ident = audit.identical_prompt_bcb(long, audit.content_hashes(), across="family")
    ident_plan = audit.identical_prompt_bcb(long, audit.wrapped_prompt_hashes(), across="dimension")
    name_map = audit.name_match_keys(sorted(stats.key.unique()), aliases)
    cov_name = audit.coverage(stats, long, name_map)

    key_tasks = stats.groupby("key").agg(tasks=("task_id", "size"), nonempty=("is_empty", lambda e: int((~e).sum())))
    graded = {m: k for m, k in mapping["graded_key"].items()}
    graded_keys = sorted({k for v in graded.values() if v for k in ([v] if isinstance(v, str) else v)})
    cov_graded = cov[cov.key.isin(graded_keys)]

    res = {
        "flagship": flagship,
        "released_models": list(audit.RELEASED_MODELS),
        "cells": {"total": int(len(long)), "unavailable": int((~long.available.astype(bool)).sum())},
        "response_keys": {k: {"tasks": int(r.tasks), "nonempty": int(r.nonempty)}
                          for k, r in key_tasks.iterrows()},
        "mapping": {**mapping,
                    "alias_shares": aliases.to_dict("records"),
                    "length_correlation": {k: corr.loc[k].to_dict() for k in corr.index}},
        "flags": flags.to_dict("records"),
        "flagged_eligible": sorted(flags[flags.flagged & flags.eligible].family),
        "flagged_ineligible": sorted(flags[flags.flagged & ~flags.eligible].family),
        "truncation_screen": screen.to_dict("records"),
        "identical_prompt_bcb": ident,
        "identical_prompt_bcb_plan_wording": ident_plan,
        "name_match_mapping_not_the_rule": name_map,
        "chars_per_token_diagnostic_not_the_rule": cpt.to_dict("records"),
        "chars_per_token_control_key_x_model": {k: cpt_control.loc[k].to_dict() for k in cpt_control.index},
        "constant_token_records": const.to_dict("records"),
        "key_tasks_by_family": {f"{k}|{f}": int(n) for (k, f), n in key_tasks_by_family.items()
                                if k not in audit.RELEASED_MODELS},
        "coverage_name_match_total_not_the_rule": {
            c: int(cov_name[cov_name.key.isin([k for k, v in name_map.items() if v])][c].sum())
            for c in ["cells", "nonempty", "empty", "empty_tokens_pos", "empty_tokens_zero",
                      "empty_score_one", "empty_unavailable"]},
        "coverage_graded_keys_total": {c: int(cov_graded[c].sum()) for c in
                                       ["cells", "nonempty", "empty", "empty_tokens_pos",
                                        "empty_tokens_zero", "empty_score_one", "empty_unavailable"]},
    }
    res = _clean(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "e0_audit.json").write_text(json.dumps(res, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    _csv(table, out_dir / "e0_audit_family_model.csv")
    _csv(cov, out_dir / "e0_audit_coverage.csv")
    _csv(cov_name, out_dir / "e0_audit_coverage_name_match.csv")
    _csv(screened, out_dir / "e0_audit_truncation_cells.csv")
    print(json.dumps({k: res[k] for k in ("flagship", "flagged_eligible", "flagged_ineligible",
                                          "coverage_graded_keys_total", "coverage_name_match_total_not_the_rule")}, indent=1, ensure_ascii=False))
    print(json.dumps(res["mapping"]["graded_key"], ensure_ascii=False), res["mapping"]["notes"])
    return res


if __name__ == "__main__":
    main()
