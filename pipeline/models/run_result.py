from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class RunResult:
    task_id: str
    model_name: str
    model_id: str
    run_number: int
    timestamp: datetime
    raw_response: str
    extracted_code: str
    extraction_success: bool
    extraction_method: str
    latency_ttft_ms: Optional[int]
    latency_total_ms: int
    tokens_input: int
    tokens_output: int
    pass_fail: bool
    tests_passed: int
    tests_total: int
    test_assertions: List[Dict]
    eslint_warnings: int
    ts_any_count: int
    compiler_errors: int
    failure_mode: Optional[str] = None
    failure_notes: Optional[str] = None
    estimated_cost_usd: Optional[float] = None
    llm_error: Optional[str] = None
    runner_error: Optional[str] = None
