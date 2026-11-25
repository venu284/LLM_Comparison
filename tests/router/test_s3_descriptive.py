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
    # CLAIMS A4 (S1.7 Amendment 11, 2026-11-23): mbpp left the filtered, headline and sensitivity
    # views (headline 1,287 -> 1,062 tasks, complete case 1,264 -> 1,046). The values pinned here
    # before A4 were 198 / 161 unsolved, 1,148 no-signal tasks, and 1,264 / 1,707 complete case
    # with empty responses unavailable; all_eligible is unchanged.
    assert checks["filtered_complete_case_tasks"]["here"] == 1046
    assert checks["filtered_unsolved_strict"]["here"] == 42
    assert checks["filtered_unsolved_lenient"]["here"] == 5
    assert checks["filtered_complete_case_no_signal_tasks"]["here"] == 930
    assert checks["filtered_signal_cells"]["here"] == 956          # mbpp had no visible tests
    # the response_matrix-empty sensitivity uses the same cells as S1.7 (response_key mapping)
    assert checks["all_eligible_complete_case_if_empty_responses_unavailable"]["here"] == 4737
    assert checks["sensitivity_complete_case_if_empty_responses_unavailable"]["here"] == 1489
    assert checks["filtered = headline_complete_case_if_empty_responses_unavailable"]["here"] == 1046
    assert res["views"]["filtered = headline"]["tasks"] == 1062
    assert res["views"]["sensitivity"]["tasks"] == 1935
    assert res["views"]["all_eligible"]["tasks"] == 5626


S3_TABLES = ("s3_6_verifier_family_model.csv", "s3_6_verifier_pooled.csv", "s3_6_no_signal.csv",
             "s3_6_population.csv", "s3_6_family_mix.csv", "s3_6_model_acceptance.csv",
             "s3_3_unsolved.csv", "s3_3_rescue_pairs.csv", "s3_3_historical.csv",
             "s3_3_historical_price_sweep.csv")


@pytest.mark.parametrize("name", S3_TABLES)
def test_every_table_names_its_population(name):
    df = pd.read_csv(OUT / name)
    assert "population" in df.columns and df.population.notna().all()
    if "variant" in df.columns and name.startswith(("s3_6_pop", "s3_6_model", "s3_3_uns")):
        assert set(df[df.variant == "all_tasks"].population) == {"all tasks (available cells)"}
        assert set(df[df.variant != "all_tasks"].population) == {"complete case"}


def test_population_follows_variant():
    from router.experiments.s3_descriptive import ALL_TASKS, COMPLETE_CASE, VARIANTS, population
    assert [population(v) for v in VARIANTS] == [COMPLETE_CASE] * 3 + [ALL_TASKS]


def test_headline_verifier_counts_by_population():
    """Checkpoint-1 review: 3/63 and 36/893 count the available cells of all 1,062 headline tasks
    (spec §5.1); the complete case gives 3/57 and 26/871. Both rows are in the committed table."""
    v = pd.read_csv(OUT / "s3_6_verifier_pooled.csv")
    v = v[(v.view == "filtered = headline") & (v.variant == "main") & (v.level == "view")]
    got = {r.population: (r.pass_failure, r.failures, r.fail_success, r.successes, r.signal_cells)
           for r in v.itertuples()}
    assert got == {"all tasks (available cells)": (3, 63, 36, 893, 956),
                   "complete case": (3, 57, 26, 871, 928)}


def test_label_empty_rates():
    from router.experiments.s3_descriptive import label_empty_rates
    df = pd.DataFrame({"signal_cells": [0, 5, 5], "failures": [0, 0, 2], "successes": [0, 5, 3],
                       "false_pass_rate": [float("nan"), float("nan"), 0.5],
                       "false_reject_rate": [float("nan"), 0.2, float("nan")]})
    out = label_empty_rates(df)
    assert out.false_pass_rate.tolist() == ["no signal", "none", 0.5]
    assert out.false_reject_rate.tolist()[:2] == ["no signal", 0.2]


def _raw_family(prefix: str):
    L = pd.concat([pd.read_csv(CRB / f) for f in FILES], ignore_index=True)
    L = L[L.task_id.str.startswith(prefix)]
    x = pd.read_json(X13_VERDICTS, lines=True, dtype=str)
    x = x[x.task_id.str.startswith(prefix)]
    d = L.merge(x[["task_id", "model", "empty", "w1_n", "w1_p"]], on=["task_id", "model"], how="left")
    d["available"] = (d.cost_source != "missing_token_record_zero_total") & (d["empty"] != "True")
    d["ok"] = d.score.eq(1.0) & d.available
    return d


def _check_pooled(s, v):
    passed = s.w1_p.astype(int) == s.w1_n.astype(int)
    assert v.signal_cells == len(s)
    assert v.pass_failure == int((passed & ~s.ok).sum())
    assert v.fail_success == int((~passed & s.ok).sum())
    assert v.failures == int((~s.ok).sum())


@needs_data
# mbpp left the headline under CLAIMS A4; cruxeval (no visible tests, every unsolved task) replaces it
@pytest.mark.parametrize("prefix,family", [("dim4_he_completion_", "he_completion"),
                                           ("dim4_cruxeval_", "cruxeval")])
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
    v = pd.read_csv(OUT / "s3_6_verifier_pooled.csv")
    v = v[(v.view == "filtered = headline") & (v.variant == "main") & (v.level == "family")
          & (v.key == family)].set_index("population")
    _check_pooled(s, v.loc["all tasks (available cells)"])
    _check_pooled(s[s.task_id.isin(set(cc))], v.loc["complete case"])
