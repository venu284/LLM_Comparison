"""The claim constants in router/settings.py equal the values S0 fixes, appear in the constants
table of docs/CLAIMS.md, and match the constants the P0.5 and P0.6 scripts used."""

import re

import pytest

from router import settings as S
from router.experiments import p0_5_recount as rc
from router.experiments import p0_6_cellbest_bound as cb

CLAIMS_MD = S.REPO_ROOT / "docs" / "CLAIMS.md"

# Name -> value as written in docs/IMPLEMENTATION_PLAN.md (source item in the comment).
DOCUMENTED = {
    "SUCCESS_SCORE": 1.0,                     # S0.1
    "CRUXEVAL_SUCCESS_SCORE": 0.8,            # S0.1 sensitivity threshold (amendments A2, A3)
    "VERIFIER_RUN_S": 0.200,                  # S0.2 (amendment A1)
    "VERIFIER_RUN_USD": 6.9e-6,               # S0.2 (amendment A1)
    "VERIFIER_COST_SWEEP": (1, 10, 100),      # S0.2
    "PRICE_SWEEP": (0.5, 2.0),                # S0.2 (O13)
    "GATE_SAVING": 20.0,                      # S0.3(a)
    "GATE_GAP": -2.0,                         # S0.3(b)
    "C_SAVING": 0.0,                          # S0.3(c)
    "C_GAP": -2.0,                            # S0.3(c)
    "CI_LEVEL": 0.95,                         # S0.3, S0.8
    "DEV_FRACTION": 0.20,                     # S0.6
    "DEV_SEED": 0,                            # S0.6 (O6)
    "N_FOLDS": 5,                             # S0.6
    "N_INNER_FOLDS": 4,                       # S0.6 (O7)
    "N_SEEDS": 5,                             # S0.8
    "AT_THRESHOLD_MARGIN": 0.1,               # S0.8
    "GATE_SEED": 0,                           # S0.8 (O5)
    "N_BOOT": 1000,                           # S0.8 (O4)
    "N_BOOT_GATE": 2000,                      # S0.8 (O4)
    "LIVE_BUDGET_USD": 20.0,                  # S0.9
    "PAID_TOP_MODEL": "claude-opus-4-6",      # S0.9
    "PAID_TOP_PRICE_USD_PER_MTOK": (5.0, 25.0),  # S0.9
    "MIN_RUNG_PRICE_RATIO": 3.0,              # S0.9
    "MIN_LIVE_FAMILIES": 2,                   # S0.9
    "MIN_LIVE_PROVIDERS": 2,                  # S0.9
    "LIVE_HELD_OUT_MODELS": (1, 2),           # S0.9
    "MAX_PROBES_PER_GROUP": 5,                # S0.13
    "MAX_PROBES_PER_MODEL": 100,              # S0.13
    "EXPLORATION_SHARE": 0.10,                # S0.13 (O8)
    "STOP_POSTERIOR_SD": 0.5,                 # S0.13
    "MIN_FAMILY_TASKS": 30,                   # S0.13
    "DP1A_MIN_LOWER": 3.0,                    # DP1a
    "DP2_K_GRID": (10, 25, 50, 100),          # DP2
    "DP3_GAP_SD_MULTIPLE": 2.0,               # DP3
    "DP3_MIN_RUNS_PER_TASK": 2,               # DP3
    "SC5_MIN_FREE_MODELS": 3,                 # SC5
    "WEBDEV_N_TASKS": 65,                     # SC5
}


@pytest.mark.parametrize("name,value", sorted(DOCUMENTED.items()))
def test_constant_has_documented_value(name, value):
    assert getattr(S, name) == value


def constants_table_names():
    text = CLAIMS_MD.read_text()
    section = text.split("## Constants", 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\| `([A-Z0-9_]+)` \|", section, flags=re.M))


def test_every_constant_is_in_the_claims_table():
    assert set(DOCUMENTED) <= constants_table_names()


def test_claims_table_names_exist_in_settings():
    assert all(hasattr(S, n) for n in constants_table_names())


@pytest.mark.parametrize("name,module", [
    ("GATE_SAVING", rc), ("GATE_GAP", rc), ("N_FOLDS", rc), ("N_SEEDS", rc), ("GATE_SEED", rc),
    ("C_SAVING", cb), ("C_GAP", cb)])
def test_p0_5_and_p0_6_used_the_same_constants(name, module):
    assert getattr(module, name) == getattr(S, name)


# The P0.5, P0.6 and E0 records were computed before amendment A1 (docs/CLAIMS.md) changed the
# verifier rate; they keep the value they used, and S3 re-scores on the current constant.
P0_RECORD_VERIFIER_RUN_USD = 1e-6


def test_p0_5_record_keeps_the_strict_success_rule():
    """The P0.5 record scores success as released score == 1.0 for every family, CRUXEval
    included. That is also the headline rule after amendment A3."""
    src = (S.REPO_ROOT / "router" / "experiments" / "p0_5_recount.py").read_text()
    assert '(d["S"].loc[self.ids] == 1.0)' in src
    assert "CRUXEVAL_SUCCESS_SCORE" not in src


def test_p0_records_keep_the_pre_amendment_verifier_rate():
    assert rc.VERIFIER_RUN_USD == P0_RECORD_VERIFIER_RUN_USD
    assert S.VERIFIER_RUN_USD != P0_RECORD_VERIFIER_RUN_USD


def test_e0_power_used_the_signed_off_development_seed():
    from router.experiments import e0_power
    assert e0_power.DEV_SEED_PROVISIONAL == S.DEV_SEED
