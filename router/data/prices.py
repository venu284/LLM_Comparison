"""Single source of prices (Rule R7; CLAIMS S0.2, S0.9, O11). USD per million tokens.

Four parts:

- `CRB_PRICES`: the eight CodeRouterBench models, as released in `models.json` at the pinned
  Hugging Face revision. Every released `cost_usd` with `cost_source == "token_log_pricing"` equals
  input_tokens x input price + output_tokens x output price (`cost_identity()`). Released costs are
  never changed here; the x0.5 / x2 sweep (`settings.PRICE_SWEEP`) covers price uncertainty.
- `PAID_TOP_PUBLISHED`: Anthropic's currently published price for the paid top model, recorded
  next to the S0.9 table value, which stays as the claim constant.
- `LIVE_CANDIDATES`: reference prices for possible live-pool models. Status `candidate` only: the
  pool is chosen in S6.4.
- The verifier rate basis (O11): AWS Lambda on-demand, as Route Anything prices verifier compute
  (2611.05275 v1, App. C.4). `verifier_run_cost()` recomputes the per-run dollar value next to
  `settings.VERIFIER_RUN_USD`, which is not changed here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from router.settings import EXTERNAL_DIR, VERIFIER_RUN_S, VERIFIER_RUN_USD

CRB_DIR = EXTERNAL_DIR / "CodeRouterBench"
CRB_LONG_FILES = ("id_probing_results_long.csv", "id_test_results_long.csv")
TOKEN_PRICED = "token_log_pricing"
MISSING_TOKEN_RECORD = "missing_token_record_zero_total"

# Pinned in scripts/fetch_data.py and external/CHECKSUMS (release dated 2026-06-23).
CRB_REVISION = "e567d89bdd569c9c74ffc7c7118e50d15e46b886"
CRB_MODELS_URL = (f"https://huggingface.co/datasets/Lance1573/CodeRouterBench/blob/"
                  f"{CRB_REVISION}/models.json")
CRB_ACCESSED = "2026-11-20"


@dataclass(frozen=True)
class Price:
    model: str
    provider: str
    tier: str | None
    input_per_m: float
    output_per_m: float
    source_url: str
    accessed: str
    note: str = ""
    status: str = "in_use"


def _crb(model, provider, tier, inp, out, note=""):
    return Price(model, provider, tier, inp, out, CRB_MODELS_URL, CRB_ACCESSED, note)


# Tier labels are CodeRouterBench's own (`models.json`), not the S0.9 live tiers.
_CNY = "converted from CNY by the dataset authors ({})"
CRB_PRICES: dict[str, Price] = {p.model: p for p in [
    _crb("claude-opus-4-6", "Anthropic", "premium", 5.00, 25.00),
    _crb("claude-sonnet-4-6", "Anthropic", "high", 3.00, 15.00),
    _crb("gpt-5.4", "OpenAI", "high", 2.50, 15.00),
    _crb("glm-5", "Zhipu", "low", 0.80, 3.00, _CNY.format("models.json note: pricing CNY-> USD")),
    _crb("kimi-k2.5", "Moonshot", "mid", 0.01, 2.90,
         _CNY.format("models.json note: kimi-k2.5 pricing CNY-> USD")
         + ". The $0.01/M input price looks like a conversion error: third-party providers on "
           "OpenRouter list kimi-k2.5 input at $0.49-0.60/M (accessed 2026-11-20); Moonshot's own "
           "price table did not render (see KIMI_FINDING). Released costs are not changed."),
    _crb("MiniMax-M2.7", "MiniMax", "mid", 1.00, 3.00,
         _CNY.format("models.json note: pricing CNY-> USD")),
    _crb("Qwen3-Max", "Alibaba", "mid", 0.34, 1.38,
         _CNY.format("models.json note: qwen-max-latest pricing CNY-> USD")),
    _crb("qwen3.5-plus", "Alibaba", "low", 0.11, 0.66,
         _CNY.format("models.json note: pricing CNY-> USD")),
]}

# Backward-compatible view used by price_order: model -> (input $/M, output $/M, note).
PRICES = {m: (p.input_per_m, p.output_per_m, p.note or "CodeRouterBench models.json")
          for m, p in CRB_PRICES.items()}


def price_order(models) -> list[str]:
    """Models from cheapest to most expensive by output price, then input price."""
    return sorted(models, key=lambda m: (PRICES[m][1], PRICES[m][0]))


# ---------------------------------------------------------------------------------- paid top (S0.9)

ANTHROPIC_PRICING_URL = "https://platform.claude.com/docs/en/about-claude/pricing"

# The table value above ($5 / $25) is the claim constant settings.PAID_TOP_PRICE_USD_PER_MTOK.
# This is the price Anthropic publishes today; a difference is reported, never applied here.
PAID_TOP_PUBLISHED = Price(
    "claude-opus-4-6", "Anthropic", "paid_frontier", 5.00, 25.00, ANTHROPIC_PRICING_URL,
    "2026-11-20",
    note="Base input / output, USD. Same page: cache hit $0.50/M, batch $2.50 / $12.50. "
         "Opus 4.7 and later use a tokenizer that yields about 30% more tokens for the same text.",
    status="published_check")


# ----------------------------------------------------------------------------- live pool (S0.9)

# Reference prices only. The pool, tiers and free-tier quotas are decided in S6.4 (S0.9), so `tier`
# is left unassigned. Each entry is a published paid price for the model.
LIVE_CANDIDATES: dict[str, Price] = {p.model: p for p in [
    Price("gemini-3.5-flash-lite", "Google", None, 0.30, 2.50,
          "https://aistudio.google.com/docs/pricing", "2026-11-20",
          note="Standard paid tier; output includes thinking tokens.", status="candidate"),
    Price("deepseek-flash", "DeepSeek", None, 0.30, 1.20,
          "https://api-docs.deepseek.com/quick_start/pricing", "2026-11-20",
          note="DeepSeek-V4.1-Flash-1017. Peak cache-miss input and output; off-peak is half; "
               "cache-hit input $0.006/M peak.", status="candidate"),
    Price("deepseek-v4-pro", "DeepSeek", None, 1.32, 3.96,
          "https://api-docs.deepseek.com/quick_start/pricing", "2026-11-20",
          note="DeepSeek-V4-Pro-0813. Peak cache-miss input and output; off-peak is half; "
               "cache-hit input $0.044/M peak.", status="candidate"),
]}


# ------------------------------------------------------------------------- verifier rate (O11)

LAMBDA_PRICING_URL = "https://aws.amazon.com/lambda/pricing/"
LAMBDA_ACCESSED = "2026-11-20"
LAMBDA_X86_USD_PER_GB_S = 0.0000166667   # x86, first tier, US East
LAMBDA_USD_PER_REQUEST = 0.20 / 1e6      # $0.20 per 1M requests
LAMBDA_BILLING_MS = 1                    # duration rounded up to 1 ms
# Route Anything 2611.05275 v1, App. C.4: "priced at AWS Lambda's x86 list price for a 2 GB
# function ($3.33x10^-5 per second, us-east-1, September 2026)". Memory, architecture and region
# are taken from there, not chosen here.
LAMBDA_MEMORY_GB = 2.0
LAMBDA_ARCH = "x86"
LAMBDA_REGION = "us-east-1"
RA_C4_USD_PER_S = 3.33e-5


def verifier_run_cost(run_s: float = VERIFIER_RUN_S, memory_gb: float = LAMBDA_MEMORY_GB) -> dict:
    """Per-run verifier cost on the O11 basis, next to the claim constant VERIFIER_RUN_USD."""
    import math
    billed_s = math.ceil(round(run_s * 1000, 6) / LAMBDA_BILLING_MS) * LAMBDA_BILLING_MS / 1000
    compute = billed_s * memory_gb * LAMBDA_X86_USD_PER_GB_S
    per_run = compute + LAMBDA_USD_PER_REQUEST
    ratio = per_run / VERIFIER_RUN_USD
    return {
        "run_s": run_s, "billed_s": billed_s, "memory_gb": memory_gb,
        "usd_per_gb_s": LAMBDA_X86_USD_PER_GB_S, "compute_usd": compute,
        "request_usd": LAMBDA_USD_PER_REQUEST, "per_run_usd": per_run,
        "constant_usd": VERIFIER_RUN_USD, "ratio_to_constant": ratio,
        "outside_x0.5_x2": not (0.5 <= ratio <= 2.0),
        "compute_only_ratio": compute / VERIFIER_RUN_USD,
    }


# ------------------------------------------------------------------------------ kimi-k2.5 check

KIMI_FINDING = {
    "released_input_per_m": 0.01,
    "released_output_per_m": 2.90,
    "official": ("Not obtained. platform.kimi.ai/docs/pricing/chat (and the .com mirror) render "
                 "their K2 price tables with JavaScript, so no figure was readable "
                 "(accessed 2026-11-20). https://platform.kimi.ai/docs/pricing/chat-k25 lists "
                 "kimi-k2.5 as Deprecated; K2-series models retire by end of December 2026."),
    "third_party_url": "https://openrouter.ai/moonshotai/kimi-k2.5/providers",
    "third_party_accessed": "2026-11-20",
    "third_party_input_per_m": {"AtlasCloud": 0.49, "SiliconFlow": 0.50, "NovitaAI": 0.57,
                                "Amazon Bedrock": 0.60},
    "third_party_output_per_m": {"AtlasCloud": 2.50, "SiliconFlow": 2.50, "NovitaAI": 2.85,
                                 "Amazon Bedrock": 3.00},
}


# ------------------------------------------------------------------------ released-cost checks

def load_models_json(root: Path = CRB_DIR) -> dict[str, tuple[float, float]]:
    rows = json.load(open(root / "models.json"))["models"]
    return {r["model"]: (r["input_per_1m"], r["output_per_1m"]) for r in rows}


def _long(root: Path = CRB_DIR) -> pd.DataFrame:
    return pd.concat([pd.read_csv(root / f) for f in CRB_LONG_FILES], ignore_index=True)


def recomputed_cost(df: pd.DataFrame, prices: dict[str, tuple[float, float]] | None = None
                    ) -> pd.Series:
    prices = prices or {m: (p.input_per_m, p.output_per_m) for m, p in CRB_PRICES.items()}
    inp = df.model.map(lambda m: prices[m][0])
    out = df.model.map(lambda m: prices[m][1])
    return (df.input_tokens * inp + df.output_tokens * out) / 1e6


def cost_identity(root: Path = CRB_DIR, rtol: float = 1e-6) -> dict:
    """Does every token-priced released cost equal tokens x the table prices?"""
    L = _long(root)
    tp = L[L.cost_source == TOKEN_PRICED]
    rec = recomputed_cost(tp)
    bad = ~((rec - tp.cost_usd).abs() <= rtol * tp.cost_usd.abs())
    return {
        "n_rows": len(L),
        "n_token_log_pricing": len(tp),
        "n_mismatch": int(bad.sum()),
        "max_rel_err": float(((rec - tp.cost_usd).abs() / tp.cost_usd.where(tp.cost_usd > 0)).max()),
        "n_missing_token_record": int((L.cost_source == MISSING_TOKEN_RECORD).sum()),
        "n_other_source": int((~L.cost_source.isin([TOKEN_PRICED, MISSING_TOKEN_RECORD])).sum()),
    }


def kimi_input_price_effect(alt_input_per_m, root: Path = CRB_DIR) -> dict:
    """Effect of other kimi-k2.5 input prices on its released total cost. Reporting only."""
    L = _long(root)
    k = L[(L.model == "kimi-k2.5") & (L.cost_source == TOKEN_PRICED)]
    base = float(k.cost_usd.sum())
    inp_tok, out_tok = float(k.input_tokens.sum()), float(k.output_tokens.sum())
    out_price = CRB_PRICES["kimi-k2.5"].output_per_m
    alts = []
    for a in alt_input_per_m:
        total = (inp_tok * a + out_tok * out_price) / 1e6
        order = sorted(CRB_PRICES, key=lambda m: (CRB_PRICES[m].output_per_m,
                                                   a if m == "kimi-k2.5" else CRB_PRICES[m].input_per_m))
        alts.append({"input_per_m": a, "total_usd": total, "ratio_to_released": total / base,
                     "price_order_changes": order != price_order(list(CRB_PRICES))})
    return {
        "n_rows": len(k), "input_tokens": inp_tok, "output_tokens": out_tok,
        "released_input_per_m": CRB_PRICES["kimi-k2.5"].input_per_m,
        "released_total_usd": base,
        "input_share_of_released": inp_tok * CRB_PRICES["kimi-k2.5"].input_per_m / 1e6 / base,
        "alternatives": alts,
    }
