"""Checks for the single price table (router/data/prices.py, S1.10, R7)."""

import json

import numpy as np
import pandas as pd
import pytest

from router import settings
from router.data import prices as pr

MODELS_JSON = pr.CRB_DIR / "models.json"
HAVE_PRICES = MODELS_JSON.exists()
HAVE_LONG = all((pr.CRB_DIR / f).exists() for f in pr.CRB_LONG_FILES)

CRB_MODELS = ["claude-opus-4-6", "claude-sonnet-4-6", "gpt-5.4", "glm-5", "kimi-k2.5",
              "MiniMax-M2.7", "Qwen3-Max", "qwen3.5-plus"]


def test_every_entry_has_source_and_date():
    for p in list(pr.CRB_PRICES.values()) + list(pr.LIVE_CANDIDATES.values()):
        assert p.source_url.startswith("https://")
        assert len(p.accessed) == 10 and p.accessed[4] == "-"
        assert p.input_per_m >= 0 and p.output_per_m > 0


def test_crb_urls_use_the_pinned_revision():
    for p in pr.CRB_PRICES.values():
        assert pr.CRB_REVISION in p.source_url
    assert len(pr.CRB_REVISION) == 40


def test_crb_table_covers_the_eight_models():
    assert sorted(pr.CRB_PRICES) == sorted(CRB_MODELS)


@pytest.mark.skipif(not HAVE_PRICES, reason="external data not fetched")
def test_table_matches_models_json():
    released = {m["model"]: m for m in json.load(open(MODELS_JSON))["models"]}
    assert set(released) == set(pr.CRB_PRICES)
    for name, p in pr.CRB_PRICES.items():
        assert p.input_per_m == released[name]["input_per_1m"]
        assert p.output_per_m == released[name]["output_per_1m"]
        assert p.provider == released[name]["provider"]
        assert ("cny" in p.note.lower()) == ("_note" in released[name])
    assert pr.load_models_json() == {k: (v.input_per_m, v.output_per_m)
                                     for k, v in pr.CRB_PRICES.items()}


@pytest.mark.skipif(not HAVE_LONG, reason="external data not fetched")
def test_released_cost_equals_tokens_times_price():
    out = pr.cost_identity()
    assert out["n_token_log_pricing"] > 0
    assert out["n_mismatch"] == 0
    assert out["n_missing_token_record"] == 148
    assert out["n_other_source"] == 0


def test_paid_top_price_matches_claim_constant():
    p = pr.CRB_PRICES[settings.PAID_TOP_MODEL]
    assert (p.input_per_m, p.output_per_m) == settings.PAID_TOP_PRICE_USD_PER_MTOK
    pub = pr.PAID_TOP_PUBLISHED
    assert pub.model == settings.PAID_TOP_MODEL and pub.source_url.startswith("https://")


def test_live_candidates_are_candidates_only():
    for p in pr.LIVE_CANDIDATES.values():
        assert p.status == "candidate"
        assert p.model not in pr.CRB_PRICES


def test_price_order_unchanged():
    # Pinned output of the quick-check price_order before the S1.10 rebuild.
    assert pr.price_order(CRB_MODELS) == [
        "qwen3.5-plus", "Qwen3-Max", "kimi-k2.5", "glm-5", "MiniMax-M2.7",
        "gpt-5.4", "claude-sonnet-4-6", "claude-opus-4-6"]
    assert set(pr.PRICES) == set(CRB_MODELS)
    assert pr.PRICES["kimi-k2.5"][:2] == (0.01, 2.90)


def test_verifier_rate_arithmetic():
    r = pr.verifier_run_cost()
    gb_s = settings.VERIFIER_RUN_S * pr.LAMBDA_MEMORY_GB
    assert r["compute_usd"] == pytest.approx(gb_s * pr.LAMBDA_X86_USD_PER_GB_S)
    assert r["request_usd"] == pytest.approx(pr.LAMBDA_USD_PER_REQUEST)
    assert r["per_run_usd"] == pytest.approx(r["compute_usd"] + r["request_usd"])
    assert r["constant_usd"] == settings.VERIFIER_RUN_USD
    assert r["ratio_to_constant"] == pytest.approx(r["per_run_usd"] / settings.VERIFIER_RUN_USD)
    # Route Anything's quoted per-second rate is the 2 GB x86 price, rounded to 3 significant figures.
    assert pr.LAMBDA_MEMORY_GB * pr.LAMBDA_X86_USD_PER_GB_S == pytest.approx(pr.RA_C4_USD_PER_S,
                                                                            rel=2e-3)


def test_method_reproduces_route_anything_sandbox_cost():
    r = pr.verifier_run_cost(run_s=pr.RA_C4_SANDBOX_RUN_S)
    assert r["per_run_usd"] == pytest.approx(pr.RA_C4_SANDBOX_RUN_USD, rel=5e-3)
    assert r["compute_usd"] != pytest.approx(pr.RA_C4_SANDBOX_RUN_USD, rel=5e-3)


def test_claim_constants_untouched():
    assert settings.VERIFIER_RUN_USD == 1e-6
    assert settings.PAID_TOP_PRICE_USD_PER_MTOK == (5.0, 25.0)


@pytest.mark.skipif(not HAVE_LONG, reason="external data not fetched")
def test_kimi_effect_is_computed_without_changing_released_costs():
    eff = pr.kimi_input_price_effect([0.49, 0.60])
    assert eff["released_input_per_m"] == 0.01
    assert eff["released_total_usd"] > 0
    assert len(eff["alternatives"]) == 2
    for alt in eff["alternatives"]:
        assert alt["total_usd"] > eff["released_total_usd"]
