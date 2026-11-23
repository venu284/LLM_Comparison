"""S1.7 Amendment 11 on the data: the committed funnel, before/after table and mapping test,
and a recount of the mbpp and mbpp_completion cells from the stored responses and the re-run logs.
The recount skips without external/."""

import json

import pandas as pd
import pytest

from router.data import audit
from router.experiments import s1_7_entry_point as ep

OUT = ep.OUT
needs_data = pytest.mark.skipif(not audit.RESPONSE_MATRIX.exists(), reason="external/ data not downloaded")


def _csv(name):
    return pd.read_csv(OUT / name)


def test_funnel_pins_mbpp_and_mbpp_completion():
    f = _csv("s1_7_amend11_funnel.csv").set_index("sample")
    cols = ["drawn", "i_named_entry_point", "ii_name_not_stated", "iii_defines_other",
            "iv_released_fail", "v_undefined_in_log", "no_rerun"]
    assert list(f.loc["mbpp", cols]) == [97, 97, 93, 78, 78, 78, 0]
    assert list(f.loc["mbpp_completion", cols]) == [97, 97, 6, 4, 4, 4, 0]
    assert list(f.loc["allzero_mbpp", cols]) == [97, 97, 94, 86, 86, 85, 0]    # §3.5: evidence only


def test_families_whose_prompts_state_the_name_get_nothing():
    """Every other audited family: (ii) never holds, so the rule assigns no cell."""
    f = _csv("s1_7_amend11_funnel.csv").set_index("sample")
    rest = f.drop(index=["mbpp", "mbpp_completion", "allzero_mbpp"])
    assert (rest.ii_name_not_stated == 0).all() and (rest.v_undefined_in_log == 0).all()
    no_entry = {"ds1000", "cruxeval", "swebench", "lcb_kept"}             # (i) fails
    assert set(rest.index[rest.i_named_entry_point == 0]) == no_entry
    assert f.loc["lcb", "i_named_entry_point"] == 41                      # LeetCode cells only


def test_before_and_after_labels():
    t = _csv("s1_7_amend11_labels.csv").set_index("family")
    assert set(t.index[t.changed]) == {"mbpp", "mbpp_completion"}
    m = t.loc["mbpp"]
    assert (m.d_before, m.interim_label_before) == (0, "cleared")
    assert (m.n_exec_after, m.a_after, m.d_after, m.u_after) == (97, 1.0, 78, 0)
    assert m.interim_label_after == "confirmed defect" and m.r_wilson_lo_after >= 0.05
    c = t.loc["mbpp_completion"]
    assert (c.d_before, c.d_after, c.u_after) == (0, 4, 1)
    assert c.interim_label_before == c.interim_label_after == "undecided"
    assert c.r_wilson_lo_after < 0.05                                     # below §3.2's threshold
    same = t.drop(index=["mbpp", "mbpp_completion"])
    assert (same.interim_label_before == same.interim_label_after).all()


def test_amendment_cells_are_released_fail_and_rerun_fail_only():
    """Amendment 11 cells: released fail, re-run fail, NameError in the log; no other mechanism
    was replaced (every amendment cell agrees with the release)."""
    c = pd.concat([_csv(f) for f in ("s1_7_mechanisms_2a.csv", "s1_7_mechanisms_2b.csv",
                                     "s1_7_mechanisms_2c.csv", "s1_7_mechanisms_lcbk.csv")])
    hit = c[c.mechanism_detail == ep.SUBTYPE]
    assert (hit.mechanism == "extraction").all()
    assert (~hit.released_pass.astype(bool)).all() and (hit.rerun == "fail").all()
    assert hit.agree.astype(bool).all()
    assert set(hit.family) == {"mbpp", "mbpp_completion"}


def test_mapping_test_still_passes_every_model():
    m = json.loads((OUT / "s1_7_mapping_test.json").read_text())
    assert (m["a_C"], m["n_C"]) == (0.992, 623)
    assert {k: (v["n"], v["fails"]) for k, v in m.items() if isinstance(v, dict)} == \
        {"glm-5": (120, False), "MiniMax-M2.7": (129, False), "qwen3.5-plus": (120, False)}


@needs_data
def test_recount_of_the_mbpp_cells_from_responses_and_logs():
    """Re-apply the rule to the committed mbpp and mbpp_completion cell rows (stored responses
    read as text, S1.7's run-1 logs) and compare with the committed column."""
    c = _csv("s1_7_mechanisms_2a.csv")
    c = c[c["sample"].isin(["mbpp", "mbpp_completion", "allzero_mbpp"])]
    got = ep.apply(c.drop(columns=["a11", "ep_name", "c_i", "c_ii", "c_iii", "c_iv", "c_v"]))
    assert list(got.a11.fillna("")) == list(c.a11.fillna(""))
