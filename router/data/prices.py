"""Single price table (Rule R7). USD per million tokens.

CodeRouterBench prices come from its `models.json` (HF rev 2026-06-23); entries marked `cny`
were converted from CNY by the dataset authors.
"""

from __future__ import annotations

PRICES = {
    # model: (input $/M, output $/M, source note)
    "claude-opus-4-6": (5.00, 25.00, "CodeRouterBench models.json"),
    "claude-sonnet-4-6": (3.00, 15.00, "CodeRouterBench models.json"),
    "gpt-5.4": (2.50, 15.00, "CodeRouterBench models.json"),
    "glm-5": (0.80, 3.00, "CodeRouterBench models.json, cny"),
    "kimi-k2.5": (0.01, 2.90, "CodeRouterBench models.json, cny"),
    "MiniMax-M2.7": (1.00, 3.00, "CodeRouterBench models.json, cny"),
    "Qwen3-Max": (0.34, 1.38, "CodeRouterBench models.json, cny"),
    "qwen3.5-plus": (0.11, 0.66, "CodeRouterBench models.json, cny"),
}


def price_order(models) -> list[str]:
    """Models from cheapest to most expensive by output price, then input price."""
    return sorted(models, key=lambda m: (PRICES[m][1], PRICES[m][0]))
