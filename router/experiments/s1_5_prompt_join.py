#!/usr/bin/env python
"""S1.5 report: task-text join rate per family and split, source checks, shared hashes.

Every number in the S1.5 Outcome comes from this script, run on the pinned data
(`scripts/fetch_data.py`). It reports:

  1. join rate per family, per split and overall against the S1 gate's 95%, on S1.4's eligible
     pool (the gate, addendum A) and on all ID families (family table `split_kind == "id"`),
     with the reason for every family below it;
  2. OOD families, listed separately (not joined here);
  3. tasks.jsonl split against the family table split;
  4. the builders against all tasks.jsonl prompts, and each adapter against tasks.jsonl on the
     overlap, for the content, the raw text and the wrapped prompt;
  5. tasks sharing a hash, within and across families, for all three fields, on the eligible
     pool and on all ID tasks, with the largest groups; and any cross-family content collision
     among eligible families outside the allowed sets (exit status 2 if there is one);
  6. P0.6 reproduction on its complete-case headline set (1,907 tasks): all join, the three
     byte-identical pairs, and the identical-prompt counts per cell, for all three fields.

Output: router/experiments/out/s1_5_prompt_join.json.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import prompts as P  # noqa: E402
from router.data.scoring import eligible_families  # noqa: E402
from router.data.sources import family_of  # noqa: E402

OUT_JSON = REPO_ROOT / "router/experiments/out/s1_5_prompt_join.json"
GATE = 0.95
P0_6_PAIRS = [("dim6_ds1000_138", "dim6_ds1000_180"), ("dim6_ds1000_137", "dim6_ds1000_179"),
              ("dim1_mbpp_141", "dim1_mbpp_71")]
SMALL_BCB_CELLS = ("algorithm|bcb", "code_completion|bcb", "data_science|bcb")
FIELDS = (("content", "content_sha256"), ("raw", "raw_sha256"), ("wrapped", "prompt_sha256"))


def join_rates(idt: pd.DataFrame) -> dict:
    """Join rate per family and split over the given ID tasks."""
    fams = {}
    for fam, g in idt.groupby("family"):
        n, j = len(g), int(g.joined.sum())
        rate = j / n
        entry = {"joined": j, "total": n, "rate": round(rate, 4), "meets_gate": rate >= GATE,
                 "by_split": {s: [int(h.joined.sum()), len(h)] for s, h in g.groupby("split")},
                 "sources": {str(k): int(v) for k, v in g.prompt_source.value_counts(dropna=False).items()},
                 "templates": {str(k): int(v) for k, v in g.template.value_counts(dropna=False).items()}}
        if rate < GATE:
            entry["reason"] = P.GAPS.get(fam, "partial join: see unmatched upstream ids")
            entry["unjoined_examples"] = g.task_id[~g.joined].head(5).tolist()
        fams[fam] = entry
    by_split = {s: {"joined": int(g.joined.sum()), "total": len(g), "rate": round(g.joined.mean(), 4)}
                for s, g in idt.groupby("split")}
    return {"families": fams, "by_split": by_split,
            "overall": {"joined": int(idt.joined.sum()), "total": len(idt), "rate": round(idt.joined.mean(), 4)},
            "families_below_gate": sorted(f for f, e in fams.items() if not e["meets_gate"])}


def ood(table: pd.DataFrame) -> dict:
    o = table[table.split_kind == "ood"]
    return {"tasks": len(o), "families": o.family.value_counts().to_dict(),
            "joined": int(o.joined.sum()),
            "note": "OOD prompts are published by CodeRouterBench (raw_matrices/phase2_ood/unified/tasks.jsonl); "
                    "out of S1.5 scope"}


def split_agreement(table: pd.DataFrame) -> dict:
    t = table.set_index("task_id").split
    rows = P._tasks_jsonl_rows()
    mism = [r["task_id"] for r in rows if t.get(r["task_id"]) != r["split"]]
    return {"rows": len(rows), "mismatches": len(mism), "examples": mism[:10]}


def source_checks() -> dict:
    rows = P._tasks_jsonl_rows()
    builders = Counter()
    for r in rows:
        builders[(family_of(r["task_id"]), P.rebuild_tasks_jsonl_prompt(r) == r["prompt"])] += 1
    jl = P.from_tasks_jsonl()
    ad = P.adapter_texts()
    overlap = {}
    for tid, a in ad.items():
        if tid not in jl:
            continue
        e = overlap.setdefault(a.family, {"overlap": 0, "content_equal": 0, "raw_equal": 0, "wrapped_equal": 0,
                                          "raw_differs": []})
        e["overlap"] += 1
        e["content_equal"] += a.content_text == jl[tid].content_text
        e["raw_equal"] += a.raw_text == jl[tid].raw_text
        e["wrapped_equal"] += a.wrapped_prompt == jl[tid].wrapped_prompt
        if a.raw_text != jl[tid].raw_text:
            e["raw_differs"].append(tid)
    fam_builders = {}
    for (fam, ok), n in builders.items():
        fam_builders.setdefault(fam, {"rows": 0, "rebuilt": 0})
        fam_builders[fam]["rows"] += n
        fam_builders[fam]["rebuilt"] += n if ok else 0
    adapter_cover = Counter(a.family for a in ad.values())
    return {"builders": {"rows": len(rows), "rebuilt": sum(n for (_, ok), n in builders.items() if ok),
                         "by_family": fam_builders},
            "adapter_overlap": overlap, "adapter_tasks_by_family": dict(adapter_cover)}


def shared_hashes(idt: pd.DataFrame, col: str, top: int = 8) -> dict:
    """Tasks sharing a hash of one field, within one family and across families, over the given
    ID tasks; with the largest groups."""
    j = idt[idt.joined & idt[col].notna()]
    groups = j.groupby(col)
    multi = [g for _, g in groups if len(g) > 1]
    within = Counter(); across = Counter(); across_pairs = Counter()
    tasks_within = tasks_across = 0
    for g in multi:
        fams = sorted(set(g.family))
        if len(fams) == 1:
            within[fams[0]] += 1
            tasks_within += len(g)
        else:
            across[" + ".join(fams)] += 1
            tasks_across += len(g)
            for i, a in enumerate(fams):
                for b in fams[i + 1:]:
                    across_pairs[f"{a} + {b}"] += 1
    # The largest group for each distinct family set, largest first.
    by_set: dict = {}
    for g in sorted(multi, key=lambda g: (-len(g), g.task_id.min())):
        by_set.setdefault(tuple(sorted(set(g.family))), g)
    biggest = list(by_set.values())[:top]
    return {"tasks_hashed": len(j), "distinct_hashes": j[col].nunique(),
            "hashes_shared_within_one_family": dict(within), "tasks_in_within_family_groups": tasks_within,
            "hashes_shared_across_families": dict(across.most_common()), "tasks_in_across_family_groups": tasks_across,
            "family_pairs_sharing_a_hash": dict(across_pairs.most_common()),
            "top_groups": [{"tasks": len(g), "families": sorted(set(g.family)), "examples": sorted(g.task_id)[:4]}
                           for g in biggest]}


# Families whose tasks may share content across families (addendum A): BigCodeBench prompts reused
# across dimensions, and HumanEval variants of the same problem. Any other cross-family content
# collision among eligible families is a stop-and-report.
ALLOWED_CONTENT_SHARING = (
    frozenset({"bcb", "bcb_algo2", "bcb_completion", "bcb_ds", "bcb_v014"}),
    frozenset({"humaneval", "he_completion", "hex_python"}),
)


def _problem_number(upstream: str | None) -> str | None:
    """Trailing problem number of an upstream id (BigCodeBench/10, HumanEval/10, Python/10)."""
    m = re.search(r"(\d+)$", upstream or "")
    return m.group(1) if m else None


def unexpected_content_collisions(eligible_idt: pd.DataFrame) -> list[dict]:
    """Cross-family content groups that are not the same problem within an allowed family set."""
    j = eligible_idt[eligible_idt.joined]
    out = []
    for _, g in j.groupby("content_sha256"):
        fams = set(g.family)
        if len(fams) < 2:
            continue
        allowed = any(fams <= ok for ok in ALLOWED_CONTENT_SHARING)
        problems = {_problem_number(u) for u in g.upstream_id}
        if not allowed or len(problems) != 1 or None in problems:
            out.append({"families": sorted(fams), "tasks": sorted(g.task_id)})
    return out


def p0_6_reproduction(table: pd.DataFrame) -> dict:
    from router.experiments import p0_5_recount as rc
    from router.experiments import p0_6_cellbest_bound as p6

    d = rc.load()
    cc = p6.complete_case(d)
    texts = P.task_texts()
    ids = list(cc.ids)
    joined = sum(i in texts for i in ids)
    cells = rc.ra_cells(cc.ids, d["dim"]).values
    names = sorted(set(cells))
    out = {"tasks": len(ids), "joined": joined}
    by = table.set_index("task_id")
    for field, col in FIELDS:
        hashes = by[col].reindex(ids).values
        ov = p6.prompt_overlap(cc, cells, hashes, names)
        small = {c: ov[c] for c in SMALL_BCB_CELLS}
        out[field] = {
            "pairs_identical": {f"{a}/{b}": bool(by.at[a, col] == by.at[b, col]) for a, b in P0_6_PAIRS},
            "small_bcb_identical_in_other_cell": sum(v["identical_prompt_in_other_cell"] for v in small.values()),
            "small_bcb_tasks": sum(v["tasks"] for v in small.values()),
            "overlap_by_cell": ov,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    a = ap.parse_args(argv)
    table = P.prompt_table()
    idt = table[table.split_kind == "id"]
    eligible = eligible_families()
    elt = idt[idt.family.isin(eligible)]
    rep = {
        "eligible_families": eligible,
        "join_eligible": join_rates(elt),
        "join": join_rates(idt),
        "ood": ood(table),
        "split_agreement": split_agreement(table),
        "sources": source_checks(),
        "shared_hashes_eligible": {f: shared_hashes(elt, col) for f, col in FIELDS},
        "shared_hashes": {f: shared_hashes(idt, col) for f, col in FIELDS},
        "unexpected_content_collisions_eligible": unexpected_content_collisions(elt),
        "upstream_ids": {f: int(g.upstream_id.notna().sum()) for f, g in idt.groupby("family")},
        "p0_6": p0_6_reproduction(table),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=int) + "\n")

    for label, key in (("eligible pool", "join_eligible"), ("all ID tasks", "join")):
        j = rep[key]
        print(f"-- {label}\n{'family':18s} {'joined':>7s} {'total':>6s} {'rate':>7s}  probing   id_test")
        for fam, e in sorted(j["families"].items()):
            bs = e["by_split"]
            print(f"{fam:18s} {e['joined']:7d} {e['total']:6d} {e['rate']:7.1%}  "
                  f"{'%d/%d' % tuple(bs.get('probing', (0, 0))):9s} {'%d/%d' % tuple(bs.get('id_test', (0, 0)))}"
                  f"{'' if e['meets_gate'] else '  < 95%'}")
        print(f"{label} overall: {j['overall']['joined']}/{j['overall']['total']} = {j['overall']['rate']:.1%}; "
              + "; ".join(f"{s} {v['joined']}/{v['total']} = {v['rate']:.1%}" for s, v in j["by_split"].items()))
    s = rep["sources"]
    print(f"builders: {s['builders']['rebuilt']}/{s['builders']['rows']} tasks.jsonl prompts rebuilt")
    for fam, e in sorted(s["adapter_overlap"].items()):
        print(f"adapter {fam}: overlap {e['overlap']}, content equal {e['content_equal']}, "
              f"raw equal {e['raw_equal']}, wrapped equal {e['wrapped_equal']}")
    print(f"split agreement: {rep['split_agreement']['mismatches']} mismatches in {rep['split_agreement']['rows']} rows")
    p = rep["p0_6"]
    print(f"P0.6 complete-case set: {p['joined']}/{p['tasks']} joined")
    for field, _ in FIELDS:
        q = p[field]
        print(f"  {field}: pairs {q['pairs_identical']}; small bcb identical in other cell "
              f"{q['small_bcb_identical_in_other_cell']}/{q['small_bcb_tasks']}")
    for scope, key in (("eligible", "shared_hashes_eligible"), ("all ID", "shared_hashes")):
        for field, _ in FIELDS:
            h = rep[key][field]
            print(f"shared {field} hashes ({scope}): {h['tasks_hashed']} tasks, {h['distinct_hashes']} distinct; "
                  f"within-family groups hold {h['tasks_in_within_family_groups']} tasks, "
                  f"across-family groups {h['tasks_in_across_family_groups']}")
    bad = rep["unexpected_content_collisions_eligible"]
    print(f"unexpected cross-family content collisions among eligible families: {len(bad)}")
    print(f"wrote {a.out}")
    return 2 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
