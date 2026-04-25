from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


OPENROUTER_MODELS = [
    ("Nemotron-3-Super", "openrouter/nvidia/nemotron-3-super-120b-a12b:free"),
    ("GLM-4.5-Air", "openrouter/z-ai/glm-4.5-air:free"),
    ("GPT-OSS-120B", "openrouter/openai/gpt-oss-120b:free"),
    ("MiniMax-M2.5", "openrouter/minimax/minimax-m2.5:free"),
    ("Devstral-2", "openrouter/mistralai/devstral-2512:free"),
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


class ConfigOpenRouterModelTests(unittest.TestCase):
    def setUp(self) -> None:
        config.load_config.cache_clear()

    def tearDown(self) -> None:
        config.load_config.cache_clear()

    def test_models_use_openrouter_free_model_ids_and_single_api_key(self) -> None:
        self.assertEqual(
            [(model.name, model.model_id) for model in config.MODELS],
            OPENROUTER_MODELS,
        )
        self.assertEqual({model.provider for model in config.MODELS}, {"openrouter"})
        self.assertEqual({model.api_key_env for model in config.MODELS}, {"OPENROUTER_API_KEY"})

    def test_cost_rates_are_zero_for_openrouter_models(self) -> None:
        rates = config.get_cost_rates()

        self.assertEqual(set(rates), {name for name, _ in OPENROUTER_MODELS})
        for model_name, _ in OPENROUTER_MODELS:
            self.assertEqual(rates[model_name]["input"], 0.0)
            self.assertEqual(rates[model_name]["output"], 0.0)


if __name__ == "__main__":
    unittest.main()
