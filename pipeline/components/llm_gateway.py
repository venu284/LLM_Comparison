from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import litellm

from config import MODELS, ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    model_name: str
    model_id: str
    raw_response: str
    latency_ttft_ms: Optional[int]
    latency_total_ms: int
    tokens_input: int
    tokens_output: int
    success: bool
    error_message: Optional[str]


class LLMGateway:
    def __init__(
        self,
        models: List[ModelConfig] = MODELS,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ):
        self.models = models
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        litellm.drop_params = True
        self._configure_api_keys()

    def _configure_api_keys(self) -> None:
        for model in self.models:
            if not os.getenv(model.api_key_env):
                logger.warning(
                    "API key not found for %s (env: %s)",
                    model.name,
                    model.api_key_env,
                )

    async def query_single(self, prompt: str, model: ModelConfig) -> LLMResponse:
        start_time = time.perf_counter()
        api_key = os.getenv(model.api_key_env)
        if not api_key:
            return LLMResponse(
                model_name=model.name,
                model_id=model.model_id,
                raw_response="",
                latency_ttft_ms=None,
                latency_total_ms=0,
                tokens_input=0,
                tokens_output=0,
                success=False,
                error_message=f"Missing API key: {model.api_key_env}",
            )

        try:
            response = await litellm.acompletion(
                model=model.model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=api_key,
            )
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            usage = getattr(response, "usage", None)
            content = self._normalize_content(
                response.choices[0].message.content if response.choices else ""
            )

            return LLMResponse(
                model_name=model.name,
                model_id=model.model_id,
                raw_response=content,
                latency_ttft_ms=None,
                latency_total_ms=elapsed_ms,
                tokens_input=self._usage_value(usage, "prompt_tokens"),
                tokens_output=self._usage_value(usage, "completion_tokens"),
                success=True,
                error_message=None,
            )
        except Exception as exc:  # pragma: no cover - network/provider errors
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.error("Error querying %s: %s", model.name, exc)
            return LLMResponse(
                model_name=model.name,
                model_id=model.model_id,
                raw_response="",
                latency_ttft_ms=None,
                latency_total_ms=elapsed_ms,
                tokens_input=0,
                tokens_output=0,
                success=False,
                error_message=str(exc),
            )

    async def query_single_with_retry(
        self, prompt: str, model: ModelConfig, max_retries: Optional[int] = None
    ) -> LLMResponse:
        retries = max_retries if max_retries is not None else self.max_retries
        last_response: Optional[LLMResponse] = None

        for attempt in range(retries):
            response = await self.query_single(prompt, model)
            last_response = response
            if response.success:
                return response

            error_text = (response.error_message or "").lower()
            transient = any(token in error_text for token in ["rate", "429", "503", "timeout", "connection"])
            if not transient or attempt == retries - 1:
                return response

            wait_seconds = (2 ** attempt) * 5
            logger.warning(
                "Retrying %s in %ss (attempt %s/%s): %s",
                model.name,
                wait_seconds,
                attempt + 1,
                retries,
                response.error_message,
            )
            await asyncio.sleep(wait_seconds)

        return last_response or LLMResponse(
            model_name=model.name,
            model_id=model.model_id,
            raw_response="",
            latency_ttft_ms=None,
            latency_total_ms=0,
            tokens_input=0,
            tokens_output=0,
            success=False,
            error_message="Unknown gateway failure",
        )

    async def query_all(self, prompt: str) -> List[LLMResponse]:
        return await asyncio.gather(
            *(self.query_single_with_retry(prompt, model) for model in self.models)
        )

    async def query_selected(self, prompt: str, model_names: List[str]) -> List[LLMResponse]:
        selected_names = {name.strip() for name in model_names}
        selected = [model for model in self.models if model.name in selected_names]
        if not selected:
            logger.warning("No models matched names: %s", model_names)
            return []

        return await asyncio.gather(
            *(self.query_single_with_retry(prompt, model) for model in selected)
        )

    async def warmup(self, n: int = 3) -> None:
        logger.info("Warming up %s models with %s requests each...", len(self.models), n)
        for model in self.models:
            for index in range(1, n + 1):
                response = await self.query_single_with_retry("Say hello in one word.", model, max_retries=1)
                if response.success:
                    logger.info("Warmup %s/%s for %s: OK", index, n, model.name)
                else:
                    logger.warning(
                        "Warmup %s/%s for %s failed: %s",
                        index,
                        n,
                        model.name,
                        response.error_message,
                    )
                    if self._is_rate_limit_error(response.error_message):
                        remaining = n - index
                        if remaining > 0:
                            logger.warning(
                                "Skipping remaining %s warmup request(s) for %s after rate limit",
                                remaining,
                                model.name,
                            )
                        break
        logger.info("Warmup complete.")

    def _is_rate_limit_error(self, error_message: Optional[str]) -> bool:
        if not error_message:
            return False
        error_text = error_message.lower()
        return "429" in error_text or "rate limit" in error_text or "rate-limit" in error_text

    def _normalize_content(self, content: object) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content or "")

    def _usage_value(self, usage: object, field: str) -> int:
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(field, 0) or 0)
        return int(getattr(usage, field, 0) or 0)
