from __future__ import annotations

from datetime import datetime, timezone

from config import get_cost_rates
from components.code_extractor import ExtractionResult
from components.llm_gateway import LLMResponse
from components.test_runner import TestResult
from models.run_result import RunResult


class MetricsCollector:
    def assemble(
        self,
        task_id: str,
        run_number: int,
        llm_response: LLMResponse,
        extraction: ExtractionResult,
        test_result: TestResult,
    ) -> RunResult:
        rates = get_cost_rates().get(llm_response.model_name, {"input": 0.0, "output": 0.0})
        estimated_cost = (
            (llm_response.tokens_input * float(rates.get("input", 0.0)) / 1_000_000)
            + (llm_response.tokens_output * float(rates.get("output", 0.0)) / 1_000_000)
        )

        return RunResult(
            task_id=task_id,
            model_name=llm_response.model_name,
            model_id=llm_response.model_id,
            run_number=run_number,
            timestamp=datetime.now(timezone.utc),
            raw_response=llm_response.raw_response,
            extracted_code=extraction.code,
            extraction_success=extraction.success,
            extraction_method=extraction.method,
            latency_ttft_ms=llm_response.latency_ttft_ms,
            latency_total_ms=llm_response.latency_total_ms,
            tokens_input=llm_response.tokens_input,
            tokens_output=llm_response.tokens_output,
            pass_fail=test_result.overall_pass if extraction.success else False,
            tests_passed=test_result.tests_passed if extraction.success else 0,
            tests_total=test_result.tests_total,
            test_assertions=[
                {"name": item.name, "passed": item.passed, "error": item.error_message}
                for item in test_result.assertions
            ]
            if extraction.success
            else [],
            eslint_warnings=test_result.eslint_warnings,
            ts_any_count=test_result.ts_any_count,
            compiler_errors=test_result.compiler_errors,
            estimated_cost_usd=round(estimated_cost, 6),
            llm_error=llm_response.error_message,
            runner_error=test_result.runner_error,
        )
