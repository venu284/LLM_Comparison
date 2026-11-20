"""S1.4: the documented scoring rules, the success rule, the legacy threshold, and `annotate`."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from router.data import scoring, sources
from router.settings import SUCCESS_SCORE

REPO = Path(__file__).resolve().parents[2]
LEGACY_NAMES = {"LEGACY_ACCEPT_THRESHOLD", "legacy_accepted", "ACCEPT_THRESHOLD", "accepted"}
LEGACY_IMPORTERS = {"router/sim/metrics.py", "router/sim/verifier.py",
                    "router/experiments/m3_irt.py", "router/experiments/m3_difficulty.py"}
ID_FAMILIES = {f for f, v in sources.FAMILIES.items() if v.split_kind == "id"}
OOD_FAMILIES = {f for f, v in sources.FAMILIES.items() if v.split_kind == "ood"}


# --------------------------------------------------------------------------- rule table
def test_every_family_has_exactly_one_rule():
    assert set(scoring.RULES) == ID_FAMILIES
    assert set(scoring.OOD_RULES) == OOD_FAMILIES


@pytest.mark.parametrize("family", sorted(ID_FAMILIES | OOD_FAMILIES))
def test_every_rule_is_typed_and_cited(family):
    r = scoring.rule(family)
    assert r.score_type in scoring.SCORE_TYPES
    assert r.partial_credit in scoring.PARTIAL_CREDIT
    for cite in (r.crb_rule, r.source_rule, r.partial_credit_cite, r.test_checkable_cite):
        assert cite.strip()
    # Every CodeRouterBench citation names the paper version and quotes it.
    assert "arXiv:2606.22902v3" in r.crb_rule and '"' in r.crb_rule
    # A source citation either quotes a document or says plainly that none was found.
    assert '"' in r.source_rule or r.source_rule.startswith("NOT DOCUMENTED")


def test_crb_rule_follows_the_dimension():
    exec_dims = {"code_generation", "algorithm", "bug_fixing", "code_completion", "data_science",
                 "multi_language"}
    for f, r in scoring.RULES.items():
        dim = sources.FAMILIES[f].dimension
        assert (r.score_type == "exec") == (dim in exec_dims), f
        assert ('Eval "Exec"' in r.crb_rule) == (dim in exec_dims), f


def test_eligible_implies_exec_and_test_checkable():
    for f, r in scoring.RULES.items():
        assert r.eligible == (r.score_type == "exec" and r.test_checkable), f
    for r in scoring.OOD_RULES.values():
        assert "never eligible" in r.notes


def test_undocumented_families_are_never_eligible():
    assert not [f for f, r in scoring.RULES.items() if r.score_type == "undocumented" and r.eligible]


def test_frozen_eligible_families():
    assert scoring.eligible_families() == sorted([
        "bcb", "bcb_algo2", "bcb_completion", "bcb_ds", "bcb_v014", "cruxeval", "ds1000",
        "he_completion", "hex_cpp", "hex_go", "hex_java", "hex_js", "hex_python", "humaneval",
        "lcb", "mbpp", "mbpp_completion", "multiple_rs", "multiple_ts", "swebench"])


# --------------------------------------------------------------------------- success rule
def test_success_uses_success_score():
    assert SUCCESS_SCORE == 1.0
    s = pd.Series([0.0, 0.5, 0.999999999, np.nextafter(1.0, 0.0), 1.0])
    assert scoring.success(s).tolist() == [False, False, False, False, True]


def test_success_follows_the_setting(monkeypatch):
    monkeypatch.setattr(scoring, "SUCCESS_SCORE", 0.5)
    assert bool(scoring.success(0.5))


# --------------------------------------------------------------------------- legacy names
def _imported_names(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module == "router.data.scoring":
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.Attribute) and node.attr in LEGACY_NAMES:
            names.add(node.attr)
    return names


def test_legacy_names_are_confined_to_the_legacy_importers():
    offenders = {}
    for path in [*REPO.glob("router/**/*.py"), *REPO.glob("scripts/**/*.py")]:
        rel = path.relative_to(REPO).as_posix()
        if rel == "router/data/scoring.py":
            continue
        hit = _imported_names(path) & LEGACY_NAMES
        if hit and rel not in LEGACY_IMPORTERS:
            offenders[rel] = hit
    assert not offenders
    assert not hasattr(scoring, "ACCEPT_THRESHOLD") and not hasattr(scoring, "accepted")


def test_legacy_threshold_and_its_importers_are_unchanged():
    assert scoring.LEGACY_ACCEPT_THRESHOLD == 0.5
    assert bool(scoring.legacy_accepted(pd.Series([0.5])).iloc[0])
    for rel in LEGACY_IMPORTERS:
        assert "LEGACY_ACCEPT_THRESHOLD" in _imported_names(REPO / rel), rel
    from router.sim.verifier import Verifier
    assert Verifier().threshold == 0.5


# --------------------------------------------------------------------------- annotate
def _toy_long() -> pd.DataFrame:
    fams = sorted(ID_FAMILIES | OOD_FAMILIES)
    return pd.DataFrame({"task_id": [f"t{i}" for i in range(len(fams))], "family": fams,
                         "score": np.linspace(0, 1, len(fams))})


def test_annotate_fills_every_row():
    out = scoring.annotate(_toy_long())
    assert out.score_type.notna().all() and out.eligible.notna().all()
    for f, t, e in out[["family", "score_type", "eligible"]].itertuples(index=False):
        assert t == scoring.rule(f).score_type
        assert bool(e) == (f in scoring.RULES and scoring.RULES[f].eligible)


def test_annotate_reads_only_the_family():
    """R16: shuffling or blanking the observed scores leaves score_type and eligible unchanged."""
    base = scoring.annotate(_toy_long())[["score_type", "eligible"]]
    shuffled = _toy_long()
    shuffled["score"] = np.random.default_rng(0).permutation(shuffled.score.to_numpy())
    blank = _toy_long()
    blank["score"] = np.nan
    for df in (shuffled, blank):
        pd.testing.assert_frame_equal(scoring.annotate(df)[["score_type", "eligible"]], base)


def test_annotate_rejects_an_unknown_family():
    with pytest.raises(KeyError):
        scoring.annotate(pd.DataFrame({"family": ["nope"], "score": [1.0]}))
