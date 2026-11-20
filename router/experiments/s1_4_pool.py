"""S1.4: the eligible pool under the documented scoring rules, against the provisional S0 rule.

Every number in the S1.4 Outcome comes from this script, run from the repo root on the pinned
data (scripts/fetch_data.py --verify-only must print `fetched: 0, verified: 187`):

    pipeline/venv/bin/python router/experiments/s1_4_pool.py

Sections, in the order they are computed:
  pool        eligible tasks per family and split (rules only: no score is read);
  diff_a      documented eligibility against the provisional EXEC_DIMS rule alone, with no
              suspected-family removal (the S1.4 effect);
  diff_b      the four suspected families with their documented eligibility, pending S1.7;
  headline    CRUXEval and the eligible share against the provisional 1,971-task headline set and
              its 1,907-task complete case (stop-rule check);
  crosscheck  report-only, after the rules were frozen (commit 993388d): released scores strictly
              between 0 and 1, scores in [1 - 1e-9, 1.0), distinct values, cells at 1.0, families
              whose every score is 0, and Route Anything's runnable=false share.

No router or baseline result is computed here (S1.7, R10). Output: router/experiments/out/s1_4_pool.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import coderouterbench as crb  # noqa: E402
from router.data import sources  # noqa: E402
from router.data.scoring import RULES  # noqa: E402
from router.data.verdicts import load_verdicts  # noqa: E402
from router.experiments import p0_5_recount as p05  # noqa: E402
from router.settings import EXTERNAL_DIR, MIN_FAMILY_TASKS  # noqa: E402

OUT_JSON = REPO_ROOT / "router/experiments/out/s1_4_pool.json"
RA_TASKS = EXTERNAL_DIR / "route-anything-data/experiments_x/common/tasks.jsonl"
SPLITS = ("probing", "id_test")
NEAR_ONE = 1e-9
PENDING_S17 = "eligible by rule, artefact-suspected, pending S1.7"


def id_table() -> pd.DataFrame:
    t = sources.load_table()
    t = t[t.split_kind == "id"].copy()
    t["eligible"] = t.family.map(lambda f: RULES[f].eligible)
    t["provisional"] = t.dimension.isin(p05.EXEC_DIMS)
    t["suspected"] = t.task_id.map(lambda s: bool(p05.SUSPECTED.search(s)))
    return t


def per_split(t: pd.DataFrame, col: str) -> dict:
    g = t[t[col]].groupby(["family", "split"]).size().unstack(fill_value=0)
    g = g.reindex(columns=list(SPLITS), fill_value=0)
    out = {f: {s: int(g.loc[f, s]) for s in SPLITS} | {"total": int(g.loc[f].sum())} for f in g.index}
    tot = {s: int(g[s].sum()) for s in SPLITS}
    return {"families": out, "total": tot | {"total": sum(tot.values())}}


def pool(t: pd.DataFrame) -> dict:
    p = per_split(t, "eligible")
    p["n_eligible_families"] = len(p["families"])
    p["eligible_under_minimum"] = sorted(f for f, v in p["families"].items()
                                         if v["total"] < MIN_FAMILY_TASKS)
    p["ineligible"] = {f: {"score_type": RULES[f].score_type,
                           "test_checkable": RULES[f].test_checkable,
                           "tasks": sources.FAMILIES[f].n_tasks}
                       for f in sorted(RULES) if not RULES[f].eligible}
    return p


def diff_a(t: pd.DataFrame) -> dict:
    fam = t.groupby("family").agg(dimension=("dimension", "first"), eligible=("eligible", "first"),
                                  provisional=("provisional", "first"))
    counts = t.groupby(["family", "split"]).size().unstack(fill_value=0).reindex(columns=list(SPLITS))

    def block(sel):
        rows = {f: {"dimension": fam.loc[f, "dimension"]}
                | {s: int(counts.loc[f, s]) for s in SPLITS}
                | {"total": int(counts.loc[f].sum())} for f in fam.index[sel]}
        return {"families": rows,
                "total": {s: sum(r[s] for r in rows.values()) for s in (*SPLITS, "total")}}

    return {"provisional_rule": f"dimension in {p05.EXEC_DIMS} (no suspected removal)",
            "provisional": per_split(t, "provisional")["total"],
            "documented": per_split(t, "eligible")["total"],
            "moved_in": block(fam.eligible & ~fam.provisional),
            "moved_out": block(~fam.eligible & fam.provisional)}


def diff_b(t: pd.DataFrame, verdict_ids: set) -> dict:
    out = {}
    for f, g in t[t.suspected].groupby("family"):
        r = RULES[f]
        row = {"tasks": len(g), "score_type": r.score_type, "test_checkable": r.test_checkable,
               "eligible_by_rule": r.eligible,
               "all_family_tasks_suspected": len(g) == sources.FAMILIES[f].n_tasks,
               "label": PENDING_S17 if r.eligible else "ineligible by rule"}
        if r.eligible:
            has = g.task_id.isin(verdict_ids)
            row["with_x13_verdicts"] = {s: int(has[g.split == s].sum()) for s in SPLITS} \
                | {"total": int(has.sum())}
        out[f] = row
    return out


def headline(t: pd.DataFrame) -> dict:
    d = p05.load()
    mask = p05.headline_mask(d)
    full = p05.Pop(d, mask)
    cc = full.subset(~full.unavail.any(axis=1).values)
    fam = t.set_index("task_id").family
    elig = t.set_index("task_id").eligible

    def count(ids, name):
        ids = pd.Index(ids)
        e = elig.reindex(ids)
        assert e.notna().all()
        crux = fam.reindex(ids) == "cruxeval"
        inel = ~e.astype(bool)
        return {"tasks": len(ids), "eligible": int(e.sum()), "ineligible": int(inel.sum()),
                "ineligible_share": round(float(inel.mean()), 4),
                "ineligible_by_family": fam.reindex(ids)[inel].value_counts().to_dict(),
                "cruxeval": int(crux.sum()), "cruxeval_eligible": int((crux & e.astype(bool)).sum()),
                "set": name}

    out = {"headline_1971": count(full.ids, "provisional headline (S0.11)"),
           "complete_case_1907": count(cc.ids, "headline tasks with no unavailable cell")}
    out["stop_rule_fires"] = out["complete_case_1907"]["ineligible_share"] > 0.5
    return out


def crosscheck(t: pd.DataFrame) -> dict:
    L = pd.concat([crb.load_long(s) for s in SPLITS], ignore_index=True)
    L["family"] = L.task_id.map(sources.family_of)
    L["frac"] = (L.score > 0) & (L.score < 1)
    L["near_one"] = (L.score >= 1 - NEAR_ONE) & (L.score < 1)
    by = L.groupby("family")
    frac_tasks = L[L.frac].groupby("family").task_id.nunique()
    ra = pd.read_json(RA_TASKS, lines=True)[["task_id", "runnable"]]
    ra["family"] = ra.task_id.map(sources.family_of)
    rag = ra.groupby("family").runnable
    out = {}
    for f in sorted(RULES):
        r = RULES[f]
        n_frac = int(frac_tasks.get(f, 0))
        if r.partial_credit == "no" and n_frac:
            flag = "rule says binary, data has fractional scores"
        elif r.partial_credit == "yes" and not n_frac:
            flag = "rule allows partial credit, data are binary"
        elif r.partial_credit == "unstated" and n_frac:
            flag = "rule unstated, data has fractional scores"
        else:
            flag = ""
        sc = L.score[L.family == f]
        if (sc == 0).all():
            flag = (flag + "; " if flag else "") + "every released score is 0"
        row = {"partial_credit_rule": r.partial_credit, "eligible": r.eligible,
               "tasks": sources.FAMILIES[f].n_tasks, "cells": int(len(sc)),
               "distinct_scores": int(sc.nunique()),
               "score_values": sorted(map(float, sc.unique())) if sc.nunique() <= 10 else None,
               "cells_at_one": int((sc == 1).sum()),
               "tasks_with_fractional_score": n_frac,
               "cells_with_fractional_score": int(by.frac.sum().get(f, 0)),
               "cells_in_near_one": int(by.near_one.sum().get(f, 0)),
               "flag": flag}
        if f in rag.groups:
            n_ra = int(rag.size()[f])
            n_false = int((~rag.get_group(f).astype(bool)).sum())
            row["ra_tasks"], row["ra_runnable_false"] = n_ra, n_false
            row["ra_runnable_false_share"] = round(n_false / n_ra, 4)
        else:
            row["ra_tasks"] = 0
        out[f] = row
    return out


def main() -> dict:
    t = id_table()
    verdict_ids = set(load_verdicts().task_id)
    res = {"pool": pool(t), "diff_a": diff_a(t), "diff_b": diff_b(t, verdict_ids),
           "headline": headline(t)}
    if res["headline"]["stop_rule_fires"]:
        print("STOP: more than half of the 1,907 complete-case tasks are ineligible; report, "
              "do not pin counts.")
    res["crosscheck"] = crosscheck(t)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: res[k] for k in ("diff_a", "diff_b", "headline")}, indent=1))
    print(json.dumps(res["pool"]["total"]))
    return res


if __name__ == "__main__":
    main()
