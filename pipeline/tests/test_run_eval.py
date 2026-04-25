from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import List
from unittest.mock import patch

import run_eval
from components.llm_gateway import LLMResponse
from config import ModelConfig


@dataclass
class FakeExpectedOutput:
    extension: str = "jsx"


@dataclass
class FakeTask:
    task_id: str = "FE-001"
    category: str = "frontend"
    difficulty: str = "easy"
    formatted_prompt: str = "Build a button"
    expected_output: FakeExpectedOutput = field(default_factory=FakeExpectedOutput)
    test_file: str = "FE-001.test.jsx"
    test_count: int = 1


class FakeTaskLoader:
    benchmark_dir = "/tmp/benchmark"

    def load_task(self, task_id: str) -> FakeTask:
        return FakeTask(task_id=task_id)


class FakeGateway:
    calls: List[str] = []

    def __init__(self, models, temperature, max_tokens):
        self.models = models

    async def query_all(self, prompt: str):
        raise AssertionError("evaluate_task should not call query_all")

    async def query_single_with_retry(self, prompt: str, model: ModelConfig) -> LLMResponse:
        self.calls.append(model.name)
        return LLMResponse(
            model_name=model.name,
            model_id=model.model_id,
            raw_response="",
            latency_ttft_ms=None,
            latency_total_ms=10,
            tokens_input=1,
            tokens_output=0,
            success=False,
            error_message="simulated failure",
        )


class FakeExtractor:
    def cleanup(self) -> None:
        return None


class RunEvalSequencingTests(unittest.IsolatedAsyncioTestCase):
    async def test_evaluate_task_queries_models_sequentially_with_delay(self) -> None:
        model_a = run_eval.MODELS[0].name
        model_b = run_eval.MODELS[1].name
        sleeps = []
        FakeGateway.calls = []

        async def fake_sleep(seconds: int) -> None:
            sleeps.append(seconds)

        with (
            patch.object(run_eval, "TaskLoader", FakeTaskLoader),
            patch.object(run_eval, "LLMGateway", FakeGateway),
            patch.object(run_eval, "CodeExtractor", FakeExtractor),
            patch.object(run_eval.asyncio, "sleep", fake_sleep),
        ):
            results = await run_eval.evaluate_task(
                "FE-001",
                model_names=[model_a, model_b],
                runs=1,
                store_results=False,
            )

        self.assertEqual(FakeGateway.calls, [model_a, model_b])
        self.assertEqual(sleeps, [3, 3])
        self.assertEqual([result.model_name for result in results], [model_a, model_b])


if __name__ == "__main__":
    unittest.main()
