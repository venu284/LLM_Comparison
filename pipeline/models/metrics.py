from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CategorySummary:
    model_name: str
    total_runs: int
    passed: int
    pass_rate: float
    avg_latency_ms: Optional[int]
    avg_tokens_output: Optional[float]
    avg_estimated_cost: Optional[float]
