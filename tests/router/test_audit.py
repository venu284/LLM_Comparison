"""S1.7 checkpoint 1: the score-free and length-only helpers in `router.data.audit`.

Synthetic tests check each pre-registered rule (docs/research/S1.7_audit_criteria.md §5). The
data tests pin the counts reported at checkpoint 1 (2026-11-20) and skip without external/.
"""

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from router.data import audit
from router.experiments import e0_audit

HAVE_DATA = audit.RESPONSE_MATRIX.exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="external/ data not downloaded")


# --------------------------------------------------------------------------- §5.1
def test_flagship_is_top_three_by_output_price():
    assert audit.flagship_models() == ["claude-opus-4-6", "claude-sonnet-4-6", "gpt-5.4"]


def test_flagship_tie_at_third_place_includes_all_tied():
    p = {m: SimpleNamespace(output_per_m=v) for m, v in
         {"a": 30, "b": 20, "c": 10, "d": 10, "e": 5}.items()}
    assert audit.flagship_models(p, 3) == ["a", "b", "c", "d"]


# --------------------------------------------------------------------------- responses
def test_response_stats_keeps_no_text(tmp_path):
    f = tmp_path / "rm.json"
    f.write_text(json.dumps({"t1": {"k1": "abc", "k2": "  \n", "k3": None},
                             "t2": {"k1": "abc", "k2": "xyz"}}))
    s = audit.response_stats(f)
    assert list(s.columns) == ["task_id", "key", "is_empty", "n_chars", "sha256"]
    assert s.set_index(["task_id", "key"])["is_empty"].to_dict() == {
        ("t1", "k1"): False, ("t1", "k2"): True, ("t1", "k3"): True,
        ("t2", "k1"): False, ("t2", "k2"): False}
    assert not s.isin(["abc", "xyz"]).any().any()


def _stats(rows):
    return pd.DataFrame(rows, columns=["task_id", "key", "is_empty", "n_chars", "sha256"])


def test_alias_shares_and_mapping():
    rows = []
    for i in range(100):
        rows += [(f"t{i}", "m1", False, i, f"h{i}"), (f"t{i}", "m1_alias", False, i, f"h{i}"),
                 (f"t{i}", "m2", False, 100 - i, f"g{i}")]
    al = audit.alias_shares(_stats(rows))
    assert al.set_index(["key_a", "key_b"]).loc[("m1", "m1_alias"), "share"] == 1.0
    assert al.set_index(["key_a", "key_b"]).loc[("m1", "m2"), "share"] == 0.0
    corr = pd.DataFrame({"m1": [0.9, 0.9, 0.1], "m2": [0.2, 0.2, 0.8]},
                        index=["m1", "m1_alias", "m2"])
    mp = audit.map_keys(al, corr)
    assert mp["key_to_model"] == {"m1": "m1", "m1_alias": "m1", "m2": "m2"}
    assert mp["graded_key"] == {"m1": "m1", "m2": "m2"}


def test_name_equal_key_not_top_in_its_row_does_not_map():
    al = pd.DataFrame([{"key_a": "a", "key_b": "b", "both_nonempty": 10, "identical": 0, "share": 0.0}])
    corr = pd.DataFrame({"a": [0.3, 0.1], "b": [0.9, 0.9]}, index=["a", "b"])
    mp = audit.map_keys(al, corr)
    assert mp["key_to_model"]["a"] is None
    assert mp["graded_key"]["a"] is None and "a: no key maps" in mp["notes"]


# --------------------------------------------------------------------------- §5.2
def _table(family, shares_one, values_off=0, zero=False):
    models = [f"m{i}" for i in range(8)]
    return pd.DataFrame({"family": family, "model": models, "share_one": shares_one,
                         "share_zero": [1.0 if zero else 0.0] * 8,
                         "off_binary": [values_off] + [0] * 7})


def test_flags():
    t = pd.concat([
        _table("lcb", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], values_off=3),
        _table("swebench", [0.0] * 8, zero=True),
        _table("codexglue", [0.0] * 8, values_off=5),   # judge: never off-rule
    ])
    f = audit.family_flags(t, ["m0", "m7"]).set_index("family")
    assert f.at["lcb", "flagship_below_median"] == ["m0"] and f.at["lcb", "off_rule"]
    assert f.at["lcb", "off_rule_cells"] == 3
    assert f.at["swebench", "all_zero"] and f.at["swebench", "flagged"]
    assert not f.at["codexglue", "off_rule"] and not f.at["codexglue", "eligible"]


# --------------------------------------------------------------------------- §5.3
def test_truncation_screen():
    n = 20
    long = pd.DataFrame({"task_id": [f"t{i}" for i in range(n)], "model": "m", "family": "lcb",
                         "available": True,
                         "output_tokens": [4096] * 12 + [4060, 4000] + [100] * 6})
    s, cells = audit.truncation_screen(long)
    r = s.iloc[0]
    assert (r.max_output_tokens, r.cells_at_max, r.cells_screened, r.pileup) == (4096, 12, 13, True)
    assert len(cells) == 13


# --------------------------------------------------------------------------- data (pinned)
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("e0_audit")
    return e0_audit.main(out_dir=out), out


@pytest.fixture(scope="module")
def res(run):
    return run[0]


@needs_data
def test_e0_audit_reproduces_committed_outputs_byte_for_byte(run):
    _, out = run
    for name in e0_audit.OUTPUTS:
        assert (out / name).read_bytes() == (e0_audit.OUT / name).read_bytes(), name


@needs_data
def test_checkpoint1_counts(res):
    assert res["cells"] == {"total": 79992, "unavailable": 148}
    assert res["mapping"]["aliases"] == [["Qwen3-Max", "通义千问Max"]]
    assert res["mapping"]["graded_key"] == {
        "MiniMax-M2.7": None, "Qwen3-Max": "Qwen3-Max", "claude-opus-4-6": "claude-opus-4-6",
        "claude-sonnet-4-6": "claude-sonnet-4-6", "glm-5": None, "gpt-5.4": "gpt-5.4",
        "kimi-k2.5": "kimi-k2.5", "qwen3.5-plus": None}
    assert res["coverage_graded_keys_total"]["empty"] == 255
    assert res["coverage_name_match_total_not_the_rule"]["empty"] == 3478
    assert len(res["flagged_eligible"]) == 19 and "he_completion" not in res["flagged_eligible"]
    assert res["identical_prompt_bcb"]["bcb_tasks"] == 499
    assert res["identical_prompt_bcb"]["cells_outcome_differs"] == 638
    ctl = res["chars_per_token_control_key_x_model"]
    for m in ("glm-5", "MiniMax-M2.7", "qwen3.5-plus"):
        col = {k: ctl[k][m] for k in ctl}
        assert col[m] == 0.0 and sum(v < 0.01 for k, v in col.items() if k != m) >= 4
    const = res["constant_token_records"]
    assert len(const) == 29 and {c["model"] for c in const} == {"Qwen3-Max"}
    assert {f: n for f, n in ((k.split("|")[1], v) for k, v in res["key_tasks_by_family"].items()
                              if k.startswith("qwen3.6-plus|"))} == {
        "bcb": 675, "bcb_v014": 111, "humaneval": 100, "mbpp": 225}
