"""S1.7 Amendment 11 (`extraction` / `entry_point_unstated`): toy cases for each condition, with
hand-checked values. The data tests that pin the per-family counts live in
test_s1_7_entry_point_data.py."""

import pandas as pd

from router.experiments import s1_7_entry_point as ep

PROMPT = "Write a python function to remove first and last occurrence of a given character."
RESP_OTHER = "Here it is.\n\n```python\ndef remove_char(s, ch):\n    return s\n```\n"
RESP_REF = "```python\ndef remove_Occ(s, ch):\n    return s\n```\n"
RESP_NO_DEF = "```python\nprint(1)\n```\n"
ERR = ("Traceback (most recent call last):\n  File \"/work/_run1.py\", line 49, in <module>\n"
       "    assertion(remove_Occ(*inp), exp, 0)\nNameError: name 'remove_Occ' is not defined\n")


def _c(**kw):
    base = dict(name="remove_Occ", content=PROMPT, response=RESP_OTHER, released_pass=False,
                err=ERR, has_rerun=True, lang="python")
    base.update(kw)
    return ep.conditions(**base)


# --------------------------------------------------------------------------- (i)-(v), one at a time
def test_all_five_hold():
    assert _c() == {"c_i": True, "c_ii": True, "c_iii": True, "c_iv": True, "c_v": True,
                    "verdict": ep.SUBTYPE}


def test_i_no_named_entry_point():
    assert _c(name=None) == {"c_i": False, "c_ii": None, "c_iii": None, "c_iv": None,
                             "c_v": None, "verdict": None}


def test_ii_name_stated_as_a_whole_word():
    r = _c(content=PROMPT + " Name it remove_Occ.")
    assert (r["c_ii"], r["c_iii"], r["verdict"]) == (False, None, None)


def test_ii_whole_word_boundaries():
    assert ep.name_stated("count", "use count(x)")
    assert not ep.name_stated("count", "count_vowels and recount")      # substrings do not count
    assert not ep.name_stated("remove_Occ", "remove_occ")                # case-sensitive
    assert ep.name_stated("f", "call f.")
    assert not ep.name_stated("f", None)


def test_iii_response_uses_the_reference_name():
    r = _c(response=RESP_REF)
    assert (r["c_iii"], r["verdict"]) == (False, None)


def test_iii_response_defines_no_function():
    r = _c(response=RESP_NO_DEF)
    assert (r["c_iii"], r["verdict"]) == (False, None)
    assert _c(response="")["c_iii"] is False
    assert _c(response=None)["c_iii"] is False


def test_iii_python_follows_the_reading_field():
    # a method counts as a definition; the reference name defined anywhere is a match
    assert ep.defines_other("```python\nclass A:\n    def go(self):\n        pass\n```", "remove_Occ", "python")
    assert not ep.defines_other("```python\ndef helper():\n    pass\ndef remove_Occ():\n    pass\n```",
                                "remove_Occ", "python")


def test_iv_released_pass():
    r = _c(released_pass=True)
    assert (r["c_iv"], r["c_v"], r["verdict"]) == (False, None, None)


def test_v_other_error_in_log():
    r = _c(err="AssertionError: out: 1, exp: 2")
    assert (r["c_v"], r["verdict"]) == (False, None)


def test_v_nameerror_on_another_name_does_not_count():
    r = _c(err="NameError: name 'helper' is not defined")
    assert (r["c_v"], r["verdict"]) == (False, None)


def test_v_did_you_mean_suffix_still_matches():
    assert ep.undefined_in_log("NameError: name 'count' is not defined. Did you mean: 'round'?",
                               "count", "python")


def test_v_no_rerun_is_counted_apart():
    r = _c(has_rerun=False, err=None)
    assert (r["c_iv"], r["c_v"], r["verdict"]) == (True, None, ep.NO_RERUN)


# --------------------------------------------------------------------------- other languages
def test_other_language_undefined_messages():
    cases = {"cpp": "error: 'add' was not declared in this scope",
             "java": "error: cannot find symbol\n    symbol:   method add(int,int)",
             "go": "./main_test.go:9:5: undefined: add",
             "rs": "error[E0425]: cannot find function `add` in this scope",
             "js": "ReferenceError: add is not defined",
             "ts": "main.ts(3,1): error TS2304: Cannot find name 'add'."}
    for lang, err in cases.items():
        assert ep.undefined_in_log(err, "add", lang), lang
        assert not ep.undefined_in_log(err, "sub", lang), lang


def test_other_language_defines_other():
    assert ep.defines_other("```go\nfunc Sum(a int) int {\n return a\n}\n```", "Add", "go")
    assert not ep.defines_other("```go\nfunc Add(a int) int {\n return a\n}\n```", "Add", "go")
    assert ep.defines_other("```rust\nfn total(x: i32) -> i32 { x }\n```", "add", "rs")
    assert ep.defines_other("```js\nconst total = (a) => a;\n```", "add", "js")
    assert ep.defines_other("```cpp\nint total(int a) {\n  return a;\n}\n```", "add", "cpp")
    # a control-flow block is not a function definition
    assert not ep.defines_other("```cpp\nif (x) {\n  y();\n}\n```", "add", "cpp")


# --------------------------------------------------------------------------- a family that states the name
class _Names:
    def __call__(self, family, upstream):
        return {"he::0": "has_close_elements", "mb::1": "remove_Occ", "ds::2": None}[upstream]


def test_family_whose_prompts_state_the_name_gets_nothing():
    """A HumanEval-like family: every prompt carries the entry point, so (ii) fails on every cell
    and the rule assigns no mechanism, whatever the responses and logs say."""
    c = pd.DataFrame({
        "sample": ["he", "he", "mb", "ds"], "family": ["humaneval", "humaneval", "mbpp", "ds1000"],
        "task_id": ["t0", "t0", "t1", "t2"], "key": ["m1", "m2", "m1", "m1"],
        "upstream": ["he::0", "he::0", "mb::1", "ds::2"], "released_pass": [False, False, False, False],
        "log": ["", "", "", ""], "key_role": "graded", "test_version": None})
    contents = pd.Series({"t0": "def has_close_elements(numbers, threshold):\n    ...",
                          "t1": PROMPT, "t2": "Problem: ..."})
    responses = {("t0", "m1"): RESP_OTHER, ("t0", "m2"): RESP_OTHER, ("t1", "m1"): RESP_OTHER}
    out = ep.apply(c, names=_Names(), contents=contents, responses=responses)
    assert list(out.a11) == ["", "", ep.NO_RERUN, ""]     # the mbpp cell has no log here
    assert list(out.c_ii) == [False, False, True, None]
    f = ep.funnel(out, ["he", "mb", "ds"]).set_index("sample")
    assert f.loc["he", "v_undefined_in_log"] == 0 and f.loc["he", "ii_name_not_stated"] == 0
    assert f.loc["mb", "no_rerun"] == 1
    assert f.loc["ds", "i_named_entry_point"] == 0


def test_amendment_never_replaces_an_earlier_mechanism(monkeypatch):
    import pytest
    from router.experiments import s1_7_mechanisms as mech
    c = pd.DataFrame({"mechanism": [None, "toolchain", None], "mechanism_detail": None})
    monkeypatch.setattr(ep, "apply", lambda c: c.assign(a11=[ep.SUBTYPE, "", ""]))
    out = mech.apply_amendment_11(c)
    as_list = lambda col: [v if isinstance(v, str) else None for v in col]
    assert as_list(out.mechanism) == ["extraction", "toolchain", None]
    assert as_list(out.mechanism_detail) == [ep.SUBTYPE, None, None]
    monkeypatch.setattr(ep, "apply", lambda c: c.assign(a11=["", ep.SUBTYPE, ""]))
    with pytest.raises(ValueError):
        mech.apply_amendment_11(c)
