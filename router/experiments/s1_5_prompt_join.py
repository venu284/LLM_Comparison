#!/usr/bin/env python
"""S1.5 report: task-text join rate per family and split, source checks, shared hashes.

Every number in the S1.5 Outcome comes from this script, run on the pinned data
(`scripts/fetch_data.py`). It reports:

  1. join rate for ID families (family table `split_kind == "id"`): per family, per split,
     overall, against the S1 gate's 95%, with the reason for every family below it;
  2. OOD families, listed separately (not joined here);
  3. tasks.jsonl split against the family table split;
  4. the builders against all tasks.jsonl prompts, and each adapter against tasks.jsonl on the
     overlap, for the raw text and the wrapped prompt;
  5. tasks sharing a hash, within and across families, for both fields;
  6. P0.6 reproduction on its complete-case headline set (1,907 tasks): all join, the three
     byte-identical pairs, and the identical-prompt counts per cell, for both fields.

Output: router/experiments/out/s1_5_prompt_join.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from router.data import prompts as P  # noqa: E402
from router.data.sources import family_of  # noqa: E402

OUT_JSON = REPO_ROOT / "router/experiments/out/s1_5_prompt_join.json"
GATE = 0.95
P0_6_PAIRS = [("dim6_ds1000_138", "dim6_ds1000_180"), ("dim6_ds1000_137", "dim6_ds1000_179"),
              ("dim1_mbpp_141", "dim1_mbpp_71")]
SMALL_BCB_CELLS = ("algorithm|bcb", "code_completion|bcb", "data_science|bcb")


def join_rates(table: pd.DataFrame) -> dict:
    idt = table[table.split_kind == "id"]
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
        e = overlap.setdefault(a.family, {"overlap": 0, "raw_equal": 0, "wrapped_equal": 0, "raw_differs": []})
        e["overlap"] += 1
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


def shared_hashes(table: pd.DataFrame, col: str) -> dict:
    j = table[table.joined & table[col].notna() & (table.split_kind == "id")]
    groups = j.groupby(col)
    multi = {h: g for h, g in groups if len(g) > 1}
    within = Counter(); across = Counter(); across_pairs = Counter()
    tasks_within = tasks_across = 0
    for g in multi.values():
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
    return {"tasks_hashed": len(j), "distinct_hashes": j[col].nunique(),
            "hashes_shared_within_one_family": dict(within), "tasks_in_within_family_groups": tasks_within,
            "hashes_shared_across_families": dict(across.most_common()), "tasks_in_across_family_groups": tasks_across,
            "family_pairs_sharing_a_hash": dict(across_pairs.most_common())}


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
    for field, col in (("wrapped", "prompt_sha256"), ("raw", "raw_sha256")):
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
    rep = {
        "join": join_rates(table),
        "ood": ood(table),
        "split_agreement": split_agreement(table),
        "sources": source_checks(),
        "shared_hashes": {"raw": shared_hashes(table, "raw_sha256"), "wrapped": shared_hashes(table, "prompt_sha256")},
        "upstream_ids": {f: int(g.upstream_id.notna().sum()) for f, g in table[table.split_kind == "id"].groupby("family")},
        "p0_6": p0_6_reproduction(table),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, default=int) + "\n")

    j = rep["join"]
    print(f"{'family':18s} {'joined':>7s} {'total':>6s} {'rate':>7s}  probing   id_test")
    for fam, e in sorted(j["families"].items()):
        bs = e["by_split"]
        print(f"{fam:18s} {e['joined']:7d} {e['total']:6d} {e['rate']:7.1%}  "
              f"{'%d/%d' % tuple(bs.get('probing', (0, 0))):9s} {'%d/%d' % tuple(bs.get('id_test', (0, 0)))}"
              f"{'' if e['meets_gate'] else '  < 95%'}")
    print(f"ID overall: {j['overall']['joined']}/{j['overall']['total']} = {j['overall']['rate']:.1%}; "
          + "; ".join(f"{s} {v['joined']}/{v['total']} = {v['rate']:.1%}" for s, v in j["by_split"].items()))
    s = rep["sources"]
    print(f"builders: {s['builders']['rebuilt']}/{s['builders']['rows']} tasks.jsonl prompts rebuilt")
    for fam, e in sorted(s["adapter_overlap"].items()):
        print(f"adapter {fam}: overlap {e['overlap']}, raw equal {e['raw_equal']}, wrapped equal {e['wrapped_equal']}")
    print(f"split agreement: {rep['split_agreement']['mismatches']} mismatches in {rep['split_agreement']['rows']} rows")
    p = rep["p0_6"]
    print(f"P0.6 complete-case set: {p['joined']}/{p['tasks']} joined")
    for field in ("wrapped", "raw"):
        q = p[field]
        print(f"  {field}: pairs {q['pairs_identical']}; small bcb identical in other cell "
              f"{q['small_bcb_identical_in_other_cell']}/{q['small_bcb_tasks']}")
    for field in ("raw", "wrapped"):
        h = rep["shared_hashes"][field]
        print(f"shared {field} hashes: {h['tasks_hashed']} tasks, {h['distinct_hashes']} distinct; "
              f"within-family groups hold {h['tasks_in_within_family_groups']} tasks, "
              f"across-family groups {h['tasks_in_across_family_groups']}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
