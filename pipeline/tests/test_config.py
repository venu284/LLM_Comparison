from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


MODEL_CONFIGS = [
    ("Llama-3.3-70B", "groq/llama-3.3-70b-versatile", "groq", "GROQ_API_KEY"),
    ("DeepSeek-R1-Distill-70B", "groq/deepseek-r1-distill-llama-70b", "groq", "GROQ_API_KEY"),
    ("Qwen-QwQ-32B", "groq/qwen-qwq-32b", "groq", "GROQ_API_KEY"),
    (
        "Llama-4-Scout",
        "groq/meta-llama/llama-4-scout-17b-16e-instruct",
        "groq",
        "GROQ_API_KEY",
    ),
    ("Mistral-Saba-24B", "groq/mistral-saba-24b", "groq", "GROQ_API_KEY"),
]


class ConfigDatabaseUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_database_url = os.environ.pop("DATABASE_URL", None)
        config.load_config.cache_clear()

    def tearDown(self) -> None:
        if self.previous_database_url is not None:
            os.environ["DATABASE_URL"] = self.previous_database_url
        else:
            os.environ.pop("DATABASE_URL", None)
        config.load_config.cache_clear()

    def test_unresolved_database_url_placeholder_returns_none(self) -> None:
        self.assertIsNone(config.get_database_url())


class ConfigModelTests(unittest.TestCase):
    def setUp(self) -> None:
        config.load_config.cache_clear()

    def tearDown(self) -> None:
        config.load_config.cache_clear()

    def test_models_use_expected_model_ids_providers_and_api_keys(self) -> None:
        self.assertEqual(
            [
                (model.name, model.model_id, model.provider, model.api_key_env)
                for model in config.MODELS
            ],
            MODEL_CONFIGS,
        )

    def test_cost_rates_are_zero_for_configured_models(self) -> None:
        rates = config.get_cost_rates()

        self.assertEqual(set(rates), {name for name, _, _, _ in MODEL_CONFIGS})
        for model_name, _, _, _ in MODEL_CONFIGS:
            self.assertEqual(rates[model_name]["input"], 0.0)
            self.assertEqual(rates[model_name]["output"], 0.0)


if __name__ == "__main__":
    unittest.main()
