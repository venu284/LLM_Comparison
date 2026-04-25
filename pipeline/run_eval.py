#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv

from components.code_extractor import CodeExtractor, ExtractionResult
from components.llm_gateway import LLMGateway, LLMResponse
from components.metrics_collector import MetricsCollector
from components.results_storage import ResultsStorage
from components.task_loader import TaskLoader
from components.test_runner import TestResult, TestRunner
from config import MODELS, ensure_runtime_dirs, get_database_url, get_experiment_settings
from models.run_result import RunResult


def configure_logging() -> logging.Logger:
    paths = ensure_runtime_dirs()
    log_path = paths["logs_dir"] / "run_eval.log"
    root = logging.getLogger()
    if not root.handlers:
        root.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        root.addHandler(console)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    return logging.getLogger(__name__)


logger = configure_logging()


def _build_failed_run_result(
    task_id: str,
    run_number: int,
    llm_response: LLMResponse,
    tests_total: int,
) -> RunResult:
    return RunResult(
        task_id=task_id,
        model_name=llm_response.model_name,
        model_id=llm_response.model_id,
        run_number=run_number,
        timestamp=datetime.now(timezone.utc),
        raw_response="",
        extracted_code="",
        extraction_success=False,
        extraction_method="none",
        latency_ttft_ms=None,
        latency_total_ms=llm_response.latency_total_ms,
        tokens_input=llm_response.tokens_input,
        tokens_output=llm_response.tokens_output,
        pass_fail=False,
        tests_passed=0,
        tests_total=tests_total,
        test_assertions=[],
        eslint_warnings=0,
        ts_any_count=0,
        compiler_errors=0,
        llm_error=llm_response.error_message,
    )


def _build_extraction_failure_result(
    task_id: str,
    run_number: int,
    llm_response: LLMResponse,
    extraction: ExtractionResult,
    tests_total: int,
) -> RunResult:
    return RunResult(
        task_id=task_id,
        model_name=llm_response.model_name,
        model_id=llm_response.model_id,
        run_number=run_number,
        timestamp=datetime.now(timezone.utc),
        raw_response=llm_response.raw_response,
        extracted_code="",
        extraction_success=False,
        extraction_method=extraction.method,
        latency_ttft_ms=llm_response.latency_ttft_ms,
        latency_total_ms=llm_response.latency_total_ms,
        tokens_input=llm_response.tokens_input,
        tokens_output=llm_response.tokens_output,
        pass_fail=False,
        tests_passed=0,
        tests_total=tests_total,
        test_assertions=[],
        eslint_warnings=0,
        ts_any_count=0,
        compiler_errors=0,
        failure_notes=extraction.failure_reason,
        llm_error=llm_response.error_message,
        runner_error="Code extraction failed",
    )


async def evaluate_task(
    task_id: str,
    model_names: Optional[List[str]] = None,
    runs: Optional[int] = None,
    store_results: bool = True,
) -> List[RunResult]:
    experiment = get_experiment_settings()
    run_count = runs or int(experiment.get("runs_per_task", 3))

    loader = TaskLoader()
    gateway = LLMGateway(
        models=[model for model in MODELS if not model_names or model.name in model_names],
        temperature=float(experiment.get("temperature", 0.2)),
        max_tokens=int(experiment.get("max_tokens", 4096)),
    )
    extractor = CodeExtractor()
    runner = TestRunner()
    collector = MetricsCollector()
    storage = None

    if store_results:
        storage = ResultsStorage(get_database_url() or "")
        storage.connect()
        storage.setup_database(str(loader.benchmark_dir))

    task = loader.load_task(task_id)
    logger.info("Loaded task: %s (%s/%s)", task.task_id, task.category, task.difficulty)
    logger.info("Prompt length: %s chars", len(task.formatted_prompt))
    logger.info("Expected output: %s", task.expected_output.extension)
    logger.info("Test file: %s (%s assertions)", task.test_file, task.test_count)

    collected_results: List[RunResult] = []

    try:
        for run_number in range(1, run_count + 1):
            logger.info("%s", "=" * 60)
            logger.info("RUN %s/%s for %s", run_number, run_count, task.task_id)
            logger.info("%s", "=" * 60)

            import asyncio as _asyncio

            responses = []
            for model in gateway.models:
                response = await gateway.query_single_with_retry(task.formatted_prompt, model)
                responses.append(response)
                await _asyncio.sleep(3)

            for response in responses:
                logger.info("--- %s ---", response.model_name)

                if not response.success:
                    logger.error("LLM call failed: %s", response.error_message)
                    failed_result = _build_failed_run_result(
                        task.task_id, run_number, response, task.test_count
                    )
                    collected_results.append(failed_result)
                    if storage:
                        storage.save_run(failed_result)
                    continue

                logger.info(
                    "Tokens: %s in, %s out | Latency: %sms",
                    response.tokens_input,
                    response.tokens_output,
                    response.latency_total_ms,
                )

                extraction = extractor.extract(
                    raw_response=response.raw_response,
                    task_id=task.task_id,
                    model_name=response.model_name,
                    expected_extension=task.expected_output.extension,
                )

                if not extraction.success or not extraction.file_path:
                    logger.warning("Code extraction failed: %s", extraction.failure_reason)
                    failed_result = _build_extraction_failure_result(
                        task.task_id, run_number, response, extraction, task.test_count
                    )
                    collected_results.append(failed_result)
                    if storage:
                        storage.save_run(failed_result)
                    continue

                logger.info(
                    "Code extracted (%s): %s chars",
                    extraction.method,
                    len(extraction.code),
                )

                test_result = runner.run(
                    task_id=task.task_id,
                    category=task.category,
                    code_file_path=extraction.file_path,
                    expected_extension=task.expected_output.extension,
                )
                logger.info(
                    "Tests: %s (%s/%s)",
                    "PASS" if test_result.overall_pass else "FAIL",
                    test_result.tests_passed,
                    test_result.tests_total,
                )

                run_result = collector.assemble(
                    task_id=task.task_id,
                    run_number=run_number,
                    llm_response=response,
                    extraction=extraction,
                    test_result=test_result,
                )
                collected_results.append(run_result)

                if storage:
                    row_id = storage.save_run(run_result)
                    logger.info("Saved to database (row %s)", row_id)
    finally:
        extractor.cleanup()
        if storage:
            storage.close()

    logger.info("Evaluation complete for %s", task.task_id)
    return collected_results


async def evaluate_all(
    model_names: Optional[List[str]] = None,
    runs: Optional[int] = None,
    store_results: bool = True,
) -> None:
    loader = TaskLoader()
    tasks = loader.load_all()
    logger.info("Loaded %s tasks", len(tasks))
    for index, task in enumerate(tasks, start=1):
        logger.info("%s", "#" * 60)
        logger.info("TASK %s/%s: %s", index, len(tasks), task.task_id)
        logger.info("%s", "#" * 60)
        await evaluate_task(task.task_id, model_names=model_names, runs=runs, store_results=store_results)


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")

    experiment = get_experiment_settings()
    parser = argparse.ArgumentParser(description="LLM Benchmark Evaluation Pipeline")
    parser.add_argument("--task", type=str, help="Single task ID (e.g. FE-001)")
    parser.add_argument("--category", type=str, help="Run all tasks in a category")
    parser.add_argument("--all", action="store_true", help="Run all benchmark tasks")
    parser.add_argument("--models", type=str, nargs="+", help="Filter to specific model names")
    parser.add_argument(
        "--runs",
        type=int,
        default=int(experiment.get("runs_per_task", 3)),
        help="Number of runs per task per model",
    )
    parser.add_argument("--warmup", action="store_true", help="Run warm-up requests first")
    parser.add_argument("--no-db", action="store_true", help="Skip database storage")
    args = parser.parse_args()

    if args.warmup:
        gateway = LLMGateway(
            temperature=float(experiment.get("temperature", 0.2)),
            max_tokens=int(experiment.get("max_tokens", 4096)),
        )
        asyncio.run(gateway.warmup(int(experiment.get("warmup_requests", 3))))
        if not any([args.task, args.category, args.all]):
            return

    store_results = not args.no_db

    if args.task:
        asyncio.run(
            evaluate_task(
                args.task,
                model_names=args.models,
                runs=args.runs,
                store_results=store_results,
            )
        )
        return

    if args.category:
        loader = TaskLoader()
        for task in loader.load_category(args.category):
            asyncio.run(
                evaluate_task(
                    task.task_id,
                    model_names=args.models,
                    runs=args.runs,
                    store_results=store_results,
                )
            )
        return

    if args.all:
        asyncio.run(evaluate_all(model_names=args.models, runs=args.runs, store_results=store_results))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
