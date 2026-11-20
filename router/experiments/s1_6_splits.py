#!/usr/bin/env python
"""S1.6 report, checkpoint 1: problem ids.

Every number in the S1.6 checkpoint-1 Outcome comes from this script, run on the pinned data
(`scripts/fetch_data.py`). It derives the problem ids (`router.data.splits.derive`), writes
`router/data/problems.csv` with `--write-problems`, and reports:

  1. problems per family and in the eligible pool, problem counts after each kind of link, the
     distribution of tasks per problem, and the largest problems with their family make-up;
  2. the sharp check: content links (and, for information, raw-text links) that the source keys
     alone do not imply;
  3. DS-1000 origins: the release's perturbation counts, chains, problems before and after, and
     which of S1.5's within-family pairs the origin field explains;
  4. the difference from the provisional ids (`p0_5_recount.problem_id`), as changes in which
     tasks share a problem, by cause;
  5. the second review's recount ("643 of 1,430 joined test tasks share prompt text with the
     probing split") on today's denominators;
  6. cross-checks: MultiPL-E function names, MBPP texts, LiveCodeBench partners, `_sup`
     families, and OOD instance ids.

Exit status 2 if the sharp check finds a content link outside S1.5's five within-family pairs.
Output: router/experiments/out/s1_6_splits.json.
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
from router.data import splits as S  # noqa: E402
from router.data.scoring import eligible_families  # noqa: E402

OUT_JSON = REPO_ROOT / "router/experiments/out/s1_6_splits.json"
CRB_TASKS = P.EXTERNAL_DIR / "CodeRouterBench" / "id_tasks.jsonl"
S1_5_PAIRS = [("dim6_ds1000_138", "dim6_ds1000_180"), ("dim6_ds1000_137", "dim6_ds1000_179"),
              ("dim1_mbpp_141", "dim1_mbpp_71"), ("dim6_ds1000_515", "dim6_ds1000_516"),
              ("dim6_ds1000_621", "dim6_ds1000_622")]
SUP = {"bugs2fix_sup": "bugs2fix", "codexglue_sup": "codexglue", "debugbench_sup": "debugbench"}
MBPP_FAMILIES = ("mbpp", "mbpp_completion", "mbpp_testgen")
# DS-1000 paper (arXiv:2211.11501v1) Table 3, "Detailed statistics of DS-1000", Total column.
DS1000_PAPER_COUNTS = {"Origin": 451, "Surface": 152, "Semantic": 235, "Difficult-Rewrite": 162}
TOP = 15


def _counts(s: pd.Series) -> dict:
    return {str(k): int(v) for k, v in sorted(s.value_counts().items(), key=lambda kv: str(kv[0]))}


# --------------------------------------------------------------------------- 1. problems
def problem_summary(kt: pd.DataFrame, pr: pd.DataFrame) -> dict:
    d = kt.assign(problem_id=pr.problem_id.to_numpy(), link=pr.link.to_numpy())
    eligible = set(eligible_families())
    d["eligible"] = (d.split_kind == "id") & d.family.isin(eligible)
    ids = list(d.task_id)
    key_e = S.group_edges(d, "source_key")
    orig_e = S.origin_edges(d, S.read_ds1000_origin())
    cont_e = S.group_edges(d, "content_sha256")
    idk = d.split_kind == "id"
    stages = {}
    for name, edges in (("none", []), ("source_key", [key_e]), ("source_key+ds1000_origin", [key_e, orig_e]),
                        ("source_key+ds1000_origin+content", [key_e, orig_e, cont_e])):
        comp = pd.Series(S.components(ids, *edges))
        stages[name] = {"id_problems": int(comp[d.task_id[idk]].nunique()),
                        "ood_problems": int(comp[d.task_id[~idk]].nunique())}
    fams = {}
    for fam, g in d.groupby("family"):
        fams[fam] = {"tasks": len(g), "problems": int(g.problem_id.nunique()),
                     "split_kind": g.split_kind.iloc[0], "eligible": bool(g.eligible.any()),
                     "links": _counts(g.link)}
    pool = d[d.eligible]
    pool_problems = set(pool.problem_id)
    size = d.groupby("problem_id").size()
    pool_size = pool.groupby("problem_id").size()
    largest = []
    for pid in sorted(size.index, key=lambda p: (-size[p], S.key_order(p)))[:TOP]:
        g = d[d.problem_id == pid]
        largest.append({"problem_id": pid, "tasks": len(g), "eligible_tasks": int(g.eligible.sum()),
                        "families": _counts(g.family)})
    return {
        "tasks": len(d), "problems": int(d.problem_id.nunique()),
        "id": {"tasks": int(idk.sum()), "problems": int(d.problem_id[idk].nunique())},
        "ood": {"tasks": int((~idk).sum()), "problems": int(d.problem_id[~idk].nunique())},
        "eligible_pool": {
            "families": len(eligible), "tasks": len(pool), "problems": len(pool_problems),
            "tasks_in_those_problems_incl_ineligible": int(d.problem_id.isin(pool_problems).sum()),
            "eligible_tasks_per_problem": {str(k): int(v) for k, v in sorted(Counter(pool_size).items())},
        },
        "problems_after_each_link": stages,
        "links": _counts(d.link),
        "tasks_per_problem": {str(k): int(v) for k, v in sorted(Counter(size).items())},
        "largest_problems": largest,
        "families": fams,
    }


# --------------------------------------------------------------------------- 2. sharp check
def unimplied_links(kt: pd.DataFrame, column: str, *extra_edges) -> list[list[str]]:
    """Groups of tasks sharing `column` that span more than one component of the source-key
    links (plus `extra_edges`): each group is listed as its tasks, sorted."""
    comp = S.components(list(kt.task_id), S.group_edges(kt, "source_key"), *extra_edges)
    out = []
    for _, g in kt[kt[column].notna()].groupby(column):
        if len({comp[t] for t in g.task_id}) > 1:
            out.append(sorted(g.task_id))
    return sorted(out)


def sharp_check(kt: pd.DataFrame) -> dict:
    content = unimplied_links(kt, "content_sha256")
    expected = sorted(sorted(p) for p in S1_5_PAIRS)
    raw = unimplied_links(kt, "raw_sha256")
    fam = dict(zip(kt.task_id, kt.family))
    raw_sets = Counter(" | ".join(sorted({fam[t] for t in g})) for g in raw)
    return {
        "content_unimplied_groups": content,
        "content_matches_s1_5_pairs": content == expected,
        "raw_unimplied_groups": len(raw),
        "raw_unimplied_tasks": sum(len(g) for g in raw),
        "raw_unimplied_by_family_set": dict(sorted(raw_sets.items())),
        "raw_unimplied_examples": raw[:TOP],
    }


# --------------------------------------------------------------------------- 3. DS-1000
def ds1000_report(kt: pd.DataFrame, pr: pd.DataFrame) -> dict:
    rows = S.read_ds1000_origin()
    pairs = S.ds1000_origin_pairs(rows)
    by_pid = {r["metadata.problem_id"]: r for r in rows}
    origin_of = dict(pairs)
    chains = [{"problem_id": a, "origin_id": b, "origin_type": by_pid[b]["metadata.perturbation_type"],
               "origin_of_origin": origin_of[b]}
              for a, b in pairs if b in origin_of]
    ds = kt.family == "ds1000"
    d = kt[ds]
    ids = list(d.task_id)
    orig_e = S.origin_edges(d, rows)
    key_only = pd.Series(S.components(ids))
    with_orig = pd.Series(S.components(ids, orig_e))
    with_all = pd.Series(S.components(ids, orig_e, S.group_edges(d, "content_sha256")))
    comp_orig = dict(with_orig)
    explained = {f"{a}/{b}": comp_orig[a] == comp_orig[b] for a, b in S1_5_PAIRS if a.startswith("dim6_ds1000")}
    pair_types = {f"{a}/{b}": [by_pid[int(t.rsplit('_', 1)[1])]["metadata.perturbation_type"] for t in (a, b)]
                  for a, b in S1_5_PAIRS if a.startswith("dim6_ds1000")}
    release = Counter(r["metadata.perturbation_type"] for r in rows)
    return {
        "extract": "xlangai/DS-1000@4416080ac5cb80bdf7576aefb8f9a0b4d5426a44 test.jsonl, metadata fields",
        "fields": ["metadata.problem_id", "metadata.library", "metadata.library_problem_id",
                   "metadata.perturbation_type", "metadata.perturbation_origin_id"],
        "origin_id_space": {
            "origin_rows_pointing_to_self_by_library_problem_id": sum(
                r["metadata.perturbation_origin_id"] == r["metadata.library_problem_id"]
                for r in rows if r["metadata.perturbation_type"] == "Origin"),
            "origin_rows_pointing_to_self_by_global_problem_id": sum(
                r["metadata.perturbation_origin_id"] == r["metadata.problem_id"]
                for r in rows if r["metadata.perturbation_type"] == "Origin"),
            "origin_rows": release["Origin"],
        },
        "perturbation_types_release": dict(sorted(release.items())),
        "perturbation_types_paper_table3": DS1000_PAPER_COUNTS,
        "perturbed_problems": len(pairs),
        "perturbations_of_a_perturbation": chains,
        "ds1000_problems": {"source_keys_only": int(key_only.nunique()),
                            "with_origin": int(with_orig.nunique()),
                            "with_origin_and_content": int(with_all.nunique())},
        "s1_5_pairs_explained_by_origin": explained,
        "s1_5_pair_perturbation_types": pair_types,
        "ds1000_problem_sizes": {str(k): int(v) for k, v in
                                 sorted(Counter(pr[kt.family.to_numpy() == "ds1000"].groupby("problem_id").size()).items())},
    }


# --------------------------------------------------------------------------- 4. provisional diff
def provisional_diff(kt: pd.DataFrame, pr: pd.DataFrame) -> dict:
    """Compare partitions, not labels: a task changed if the set of tasks sharing its problem
    differs between the provisional ids and the new ones.

    The cause is read from each pair of tasks the change joins or separates: `hex` or `multiple`
    if either task is in such a family, `ds1000_origin` if the DS-1000 origin links alone join the
    pair, `content` if only the content links do, else `other`. A task lists every cause of its
    pairs."""
    from router.experiments.p0_5_recount import problem_id as provisional

    old = pd.Series([provisional(t) for t in kt.task_id], index=kt.task_id)
    new = pd.Series(pr.problem_id.to_numpy(), index=kt.task_id)
    fam = dict(zip(kt.task_id, kt.family))
    by_origin = S.components(list(kt.task_id), S.origin_edges(kt, S.read_ds1000_origin()))
    old_m = {k: frozenset(v) for k, v in old.groupby(old).groups.items()}
    new_m = {k: frozenset(v) for k, v in new.groupby(new).groups.items()}

    def pair_cause(x: str, y: str) -> str:
        if fam[x].startswith("hex_") or fam[y].startswith("hex_"):
            return "hex"
        if fam[x].startswith("multiple_") or fam[y].startswith("multiple_"):
            return "multiple"
        if by_origin[x] == by_origin[y]:
            return "ds1000_origin"
        return "content" if new[x] == new[y] else "other"

    causes, changed = Counter(), 0
    examples: dict[str, list] = {}
    for t in kt.task_id:
        a, b = old_m[old[t]], new_m[new[t]]
        if a == b:
            continue
        changed += 1
        cause = "+".join(sorted({pair_cause(t, y) for y in a ^ b}))
        causes[cause] += 1
        ex = examples.setdefault(cause, [])
        if len(ex) < 5:
            ex.append({"task_id": t, "provisional": old[t], "provisional_size": len(a),
                       "problem_id": new[t], "size": len(b)})
    old_sets, new_sets = set(old_m.values()), set(new_m.values())
    return {
        "provisional_problems": len(old_m), "new_problems": len(new_m),
        "tasks_changed": changed, "tasks_changed_by_cause": dict(sorted(causes.items())),
        "new_problems_not_a_provisional_problem": len(new_sets - old_sets),
        "provisional_problems_not_a_new_problem": len(old_sets - new_sets),
        "examples": dict(sorted(examples.items())),
    }


# --------------------------------------------------------------------------- 5. recount
def recount(kt: pd.DataFrame, pr: pd.DataFrame) -> dict:
    d = kt.assign(problem_id=pr.problem_id.to_numpy())
    d = d[d.split_kind == "id"]
    eligible = d.family.isin(eligible_families())
    out = {}
    for pool_name, mask in (("all_id", pd.Series(True, index=d.index)), ("eligible_pool", eligible)):
        g = d[mask]
        probe, test = g[g.split == "probing"], g[g.split == "id_test"]
        entry = {"id_test_tasks": len(test), "id_test_joined": int(test.joined.sum())}
        for col in ("content_sha256", "raw_sha256", "prompt_sha256", "problem_id"):
            seen = set(probe[col].dropna())
            hit = test[col].isin(seen) & test[col].notna()
            entry[col] = {"joined": [int(hit[test.joined].sum()), int(test.joined.sum())],
                          "all": [int(hit.sum()), len(test)]}
        out[pool_name] = entry
    out["note"] = ("[shared, denominator] over ID-test tasks; a task shares a field when a probing task "
                   "in the same pool has the same non-null value. Hash fields are null for unjoined tasks.")
    return out


# --------------------------------------------------------------------------- 6. cross-checks
def multiple_names(kt: pd.DataFrame) -> dict:
    he = {r["task_id"]: r["prompt"] for r in P.read_extract("humanevalplus")}
    rows = []
    for t, f, up in zip(kt.task_id, kt.family, kt.upstream_id):
        if not f.startswith("multiple_"):
            continue
        m = re.fullmatch(r"HumanEval_(\d+)_(\w+)", up)
        n, fn = int(m.group(1)), m.group(2)
        text = he.get(f"HumanEval/{n}")
        ok = text is not None and re.search(rf"\bdef {re.escape(fn)}\(", text) is not None
        if not ok:
            entry = re.findall(r"\bdef (\w+)\(", text or "")
            rows.append({"task_id": t, "n": n, "fn": fn, "humaneval_defs": entry})
    return {"source": "humanevalplus extract (HumanEval/0-163)", "checked": int(kt.family.str.startswith("multiple_").sum()),
            "humaneval_problems_in_extract": len(he), "mismatches": rows}


def mbpp_texts(kt: pd.DataFrame) -> dict:
    d = kt[kt.family.isin(MBPP_FAMILIES)]
    out, multi = [], 0
    for key, g in d.groupby("source_key"):
        if g.family.nunique() < 2:
            continue
        multi += 1
        j = g[g.joined]
        if j.raw_sha256.nunique() > 1:
            out.append({"source_key": key, "raw_sha256": dict(zip(j.family, j.raw_sha256))})
    no_text = d[~d.joined]
    return {"field": "raw_sha256 (mbpp_completion content adds a def header; raw text does not)",
            "keys_in_two_or_more_families": multi,
            "keys_with_two_or_more_joined_families": int(sum(g[g.joined].family.nunique() >= 2
                                                             for _, g in d.groupby("source_key"))),
            "families_without_text": _counts(no_text.family),
            "disagreements": sorted(out, key=lambda r: S.key_order(r["source_key"]))}


def lcb_partners(kt: pd.DataFrame) -> dict:
    lcb_ext = {r["question_id"] for r in P.read_extract("livecodebench")}
    tj = {r["task_id"]: r for r in P._tasks_jsonl_rows()}
    d = kt[kt.family.isin(["lcb", "lcb_testgen"])]
    q = {f: set(g.upstream_id) for f, g in d.groupby("family")}
    raw = {(f, u): h for f, u, h in zip(d.family, d.upstream_id, d.raw_sha256)}
    both = sorted(q["lcb"] & q["lcb_testgen"])
    disagree = [u for u in both if raw[("lcb", u)] is not None and raw[("lcb_testgen", u)] is not None
                and raw[("lcb", u)] != raw[("lcb_testgen", u)]]
    in_tj = {f: int(sum(t in tj for t in g.task_id)) for f, g in d.groupby("family")}
    return {
        "lcb_qids": len(q["lcb"]), "lcb_testgen_qids": len(q["lcb_testgen"]),
        "lcb_not_in_livecodebench_extract": sorted(q["lcb"] - lcb_ext),
        "lcb_testgen_not_in_livecodebench_extract": sorted(q["lcb_testgen"] - lcb_ext),
        "lcb_testgen_without_lcb_partner": sorted(q["lcb_testgen"] - q["lcb"]),
        "lcb_without_lcb_testgen_partner": len(q["lcb"] - q["lcb_testgen"]),
        "partners": len(both), "partners_raw_text_disagree": disagree,
        "tasks_in_tasks_jsonl": in_tj,
    }


def sup_check(kt: pd.DataFrame) -> dict:
    crb = {}
    if CRB_TASKS.is_file():
        with open(CRB_TASKS, encoding="utf-8") as f:
            crb = {r["task_id"]: r for r in map(json.loads, f) if r}
    out = {}
    for sup, base in SUP.items():
        s, b = kt[kt.family == sup], kt[kt.family == base]
        sn = {int(t.rsplit("_", 1)[1]): t for t in s.task_id}
        bn = {int(t.rsplit("_", 1)[1]): t for t in b.task_id}
        ov = sorted(set(sn) & set(bn))
        same_split = sum(crb.get(sn[n], {}).get("source_split") == crb.get(bn[n], {}).get("source_split") for n in ov)
        out[sup] = {
            "base": base, "tasks": len(s), "base_tasks": len(b),
            "joined": int(s.joined.sum()), "base_joined": int(b.joined.sum()),
            "shared_content_or_raw_hash": 0 if not (s.joined.any() and b.joined.any()) else int(
                s.content_sha256.isin(set(b.content_sha256.dropna())).sum()
                + s.raw_sha256.isin(set(b.raw_sha256.dropna())).sum()),
            "id_range": [min(sn), max(sn)], "base_id_range": [min(bn), max(bn)],
            "ids_also_used_by_base": ov,
            "of_those_same_crb_source_split": int(same_split),
            "linked": 0,
        }
    out["note"] = ("No task text exists for these six families (S1.5 GAPS), so no hash can link them. "
                   "Equal id numbers are not evidence of one problem: no field maps a _sup id to a base "
                   "id, so none is linked.")
    return out


def ood_check(kt: pd.DataFrame) -> dict:
    swe = {k.split("/", 1)[1] for k in kt.source_key if k.startswith("swe/")}
    ood = kt[kt.split_kind == "ood"]
    inst = {f: {k.rsplit("/", 1)[1] for k in g.source_key} for f, g in ood.groupby("family")}
    return {"swebench_instances": len(swe),
            "ood_instances_also_in_swebench": {f: sorted(v & swe) for f, v in sorted(inst.items())},
            "ood_tasks_joined": int(ood.joined.sum())}


# --------------------------------------------------------------------------- main
def build(kt: pd.DataFrame, pr: pd.DataFrame) -> dict:
    return {
        "problems": problem_summary(kt, pr),
        "sharp_check": sharp_check(kt),
        "ds1000_origin": ds1000_report(kt, pr),
        "provisional_diff": provisional_diff(kt, pr),
        "second_review_recount": recount(kt, pr),
        "cross_checks": {"multiple_names": multiple_names(kt), "mbpp_texts": mbpp_texts(kt),
                         "lcb_partners": lcb_partners(kt), "sup": sup_check(kt), "ood": ood_check(kt)},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    ap.add_argument("--write-problems", action="store_true", help="also write router/data/problems.csv")
    a = ap.parse_args(argv)
    kt = S.key_table()
    pr = S.derive(kt)
    if a.write_problems:
        S.write_problems(pr)
    rep = build(kt, pr)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=int) + "\n")
    sc = rep["sharp_check"]
    print(f"problems: {rep['problems']['problems']} over {rep['problems']['tasks']} tasks; "
          f"eligible pool {rep['problems']['eligible_pool']['problems']} problems")
    print(f"sharp check: {len(sc['content_unimplied_groups'])} unimplied content groups, "
          f"match S1.5 pairs = {sc['content_matches_s1_5_pairs']}")
    print(f"wrote {a.out}")
    return 0 if sc["content_matches_s1_5_pairs"] else 2


if __name__ == "__main__":
    sys.exit(main())
