"""S3 checkpoint 1 outputs (`router/experiments/s3_descriptive.py`): the bridges to earlier figures,
the recounts against artefacts.yaml, and an independent recount of two headline families straight
from the release CSVs and the X13 file (no `load_release`, no `router.sim.profile`)."""

import json

import pandas as pd
import pytest

from router.data.verdicts import X13_VERDICTS
from router.settings import EXTERNAL_DIR, REPO_ROOT

CRB = EXTERNAL_DIR / "CodeRouterBench"
FILES = ("id_probing_results_long.csv", "id_test_results_long.csv")
OUT = REPO_ROOT / "router" / "experiments" / "out"
HAVE_DATA = X13_VERDICTS.exists() and all((CRB / f).exists() for f in FILES)
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external data not fetched")


@needs_data
def test_plan_verifier_bridge():
    from router.experiments.s3_descriptive import plan_verifier_bridge
    b = plan_verifier_bridge()
    vis, heu = b["visible_tests"], b["no_visible_tests_heuristic"]
    assert vis["cells"] == 8420 and round(vis["false_pass_pct"], 1) == 15.0
    assert round(vis["false_reject_pct"], 1) == 27.0
    assert heu["cells"] == 12472 and round(heu["false_pass_pct"], 1) == 66.5


@needs_data
def test_old_historical_figures_reproduce():
    from router.experiments.s3_descriptive import old_historical_row
    row, ids = old_historical_row()
    assert len(ids) == 2919
    assert (round(row["ladder_usd"], 1), round(row["always_usd"], 1),
            round(row["perfect_stop_usd"], 1)) == (41.0, 34.0, 6.0)
    assert round(100 * row["unsolved_share"], 1) == 39.3


def test_committed_checks_against_artefacts():
    res = json.loads((OUT / "s3_descriptive.json").read_text())
    checks = res["checks_against_artefacts"]
    assert checks and all(c["equal"] for c in checks.values())
    assert checks["filtered_unsolved_strict"]["here"] == 198
    assert checks["filtered_unsolved_lenient"]["here"] == 161
    assert checks["filtered_complete_case_no_signal_tasks"]["here"] == 1148


def _raw_family(prefix: str):
    L = pd.concat([pd.read_csv(CRB / f) for f in FILES], ignore_index=True)
    L = L[L.task_id.str.startswith(prefix)]
    x = pd.read_json(X13_VERDICTS, lines=True, dtype=str)
    x = x[x.task_id.str.startswith(prefix)]
    d = L.merge(x[["task_id", "model", "empty", "w1_n", "w1_p"]], on=["task_id", "model"], how="left")
    d["available"] = (d.cost_source != "missing_token_record_zero_total") & (d["empty"] != "True")
    d["ok"] = d.score.eq(1.0) & d.available
    return d


@needs_data
@pytest.mark.parametrize("prefix,family", [("dim4_he_completion_", "he_completion"),
                                           ("dim1_mbpp_", "mbpp")])
def test_independent_recount_headline_family(prefix, family):
    d = _raw_family(prefix)
    cc = d.groupby("task_id").available.all()
    cc = cc[cc].index
    unsolved = int((~d[d.task_id.isin(cc)].groupby("task_id").ok.any()).sum())
    u = pd.read_csv(OUT / "s3_3_unsolved.csv")
    u = u[(u.view == "filtered = headline") & (u.variant == "main") & (u.family == family)].iloc[0]
    assert (u.tasks, u.unsolved) == (len(cc), unsolved)

    s = d[d.available & d.w1_n.notna()]
    s = s[s.w1_n.astype(int) > 0]
    passed = s.w1_p.astype(int) == s.w1_n.astype(int)
    v = pd.read_csv(OUT / "s3_6_verifier_pooled.csv")
    v = v[(v.view == "filtered = headline") & (v.variant == "main") & (v.level == "family")
          & (v.key == family)].iloc[0]
    assert v.signal_cells == len(s)
    assert v.pass_failure == int((passed & ~s.ok).sum())
    assert v.fail_success == int((~passed & s.ok).sum())
    assert v.failures == int((~s.ok).sum())
