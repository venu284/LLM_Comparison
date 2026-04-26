from __future__ import annotations

import unittest

from components.llm_gateway import LLMGateway, LLMResponse
from config import ModelConfig


class LLMGatewayWarmupTests(unittest.IsolatedAsyncioTestCase):
    async def test_warmup_stops_repeating_a_model_after_rate_limit(self) -> None:
        model = ModelConfig(
            name="Llama-3.1-8B",
            model_id="groq/llama-3.1-8b-instant",
            provider="groq",
            api_key_env="GROQ_API_KEY",
        )
        gateway = LLMGateway(models=[model])
        calls = []

        async def fake_query(prompt, queried_model, max_retries=None):
            calls.append((prompt, queried_model.name, max_retries))
            return LLMResponse(
                model_name=queried_model.name,
                model_id=queried_model.model_id,
                raw_response="",
                latency_ttft_ms=None,
                latency_total_ms=0,
                tokens_input=0,
                tokens_output=0,
                success=False,
                error_message="Groq provider returned 429 rate limited",
            )

        gateway.query_single_with_retry = fake_query

        await gateway.warmup(3)

        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
