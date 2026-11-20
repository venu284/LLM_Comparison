"""S1 batch A2 report: every number in the batch's Outcome block (S1.9, S1.10, S1.11).

Run from the repo root on the pinned data (scripts/fetch_data.py --verify-only must print
`fetched: 0, verified: 187`):

    pipeline/venv/bin/python router/experiments/s1_a2_report.py

The Phase 5 part reads the cached raw responses (external/phase5/raw_responses.jsonl, built by
router.data.phase5.fetch_raw_responses); it does not connect to the database. Results are
written to router/experiments/out/s1_a2_report.json. Headline counts are not frozen here: they
are derived after S1.4 and S1.7 and frozen in artefacts.yaml.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "pipeline"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from router.data import phase5 as p5  # noqa: E402
from router.data import prices as pr  # noqa: E402
from router.data import verdicts as vd  # noqa: E402
from router.experiments import p0_5_recount as rc  # noqa: E402
from router.experiments import p0_6_cellbest_bound as cb  # noqa: E402

OUT_JSON = REPO_ROOT / "router/experiments/out/s1_a2_report.json"


def verdict_section() -> dict:
    v = vd.load_verdicts()
    d = rc.load()
    tasks = v.drop_duplicates("task_id")
    out = {
        "rows": len(v),
        "tasks": int(v.task_id.nunique()),
        "tasks_by_split": tasks.split.value_counts().to_dict(),
        "cells_signal": int(v.signal.sum()),
        "cells_no_signal": int((~v.signal & ~v.unavailable).sum()),
        "cells_unavailable": int(v.unavailable.sum()),
        "observations": v.observation.value_counts(dropna=False)
                         .rename(index=lambda k: k if isinstance(k, str) else "unavailable")
                         .to_dict(),
        "equivalence_p0_5": {
            "complete_case_1907": vd.compare_with_p0_5(v, d, cb.complete_case(d).ids),
            "all_x13": vd.compare_with_p0_5(v, d, d["A"].index),
        },
    }
    t = vd.truth_matches_release()
    t.pop("examples")
    out["truth_vs_release"] = t
    ct = vd.hidden_vs_release()
    out["w2_kind_by_release_success"] = {
        k: {str(c): int(n) for c, n in row.items()} for k, row in ct.iterrows()}
    h = vd.load_hidden_tests()
    out["w2_kind_counts"] = h.w2_kind.value_counts().to_dict()
    raw = vd.load_raw()
    empty = raw["empty"] == "True"
    out["empty_cells_with_w2_kind_empty_response"] = int(
        (raw.loc[empty, "w2_kind"] == "empty_response").sum())
    return out


def price_section() -> dict:
    opus = pr.CRB_PRICES[pr.PAID_TOP_PUBLISHED.model]
    pub = pr.PAID_TOP_PUBLISHED
    return {
        "crb_source": pr.CRB_MODELS_URL,
        "crb_accessed": pr.CRB_ACCESSED,
        "cost_identity": pr.cost_identity(),
        "paid_top": {
            "table": [opus.input_per_m, opus.output_per_m],
            "published": [pub.input_per_m, pub.output_per_m],
            "published_url": pub.source_url, "accessed": pub.accessed,
            "differs": (opus.input_per_m, opus.output_per_m) != (pub.input_per_m,
                                                                pub.output_per_m),
        },
        "verifier": {**pr.verifier_run_cost(), "url": pr.LAMBDA_PRICING_URL,
                     "accessed": pr.LAMBDA_ACCESSED, "arch": pr.LAMBDA_ARCH,
                     "region": pr.LAMBDA_REGION},
        "kimi": {**pr.kimi_input_price_effect(
                     sorted(set(pr.KIMI_FINDING["third_party_input_per_m"].values()))),
                 "official": pr.KIMI_FINDING["official"],
                 "third_party_url": pr.KIMI_FINDING["third_party_url"]},
        "live_candidates": sorted(pr.LIVE_CANDIDATES),
    }


def phase5_section() -> dict:
    out = {"csv_sha256_ok": p5.csv_sha256() == p5.PHASE5_SHA256, "csv_rows": len(p5.load_runs())}
    if p5.RESPONSE_CACHE.exists():
        out["responses"] = p5.response_counts(p5.load_runs_with_responses())
    else:
        out["responses"] = "cache not built"
    return out


def main() -> dict:
    out = {"verdicts": verdict_section(), "prices": price_section(), "phase5": phase5_section()}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2, default=str) + "\n")
    print(json.dumps(out, indent=2, default=str))
    return out


if __name__ == "__main__":
    main()
