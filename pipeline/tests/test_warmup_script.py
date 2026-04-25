from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from scripts import warmup


class WarmupScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_api_key = os.environ.pop("OPENROUTER_API_KEY", None)

    def tearDown(self) -> None:
        if self.previous_api_key is not None:
            os.environ["OPENROUTER_API_KEY"] = self.previous_api_key
        else:
            os.environ.pop("OPENROUTER_API_KEY", None)

    def test_load_environment_uses_env_local_as_override(self) -> None:
        with TemporaryDirectory() as tmp:
            pipeline_dir = Path(tmp)
            (pipeline_dir / ".env").write_text("OPENROUTER_API_KEY=from-env\n", encoding="utf-8")
            (pipeline_dir / ".env.local").write_text(
                "OPENROUTER_API_KEY=from-env-local\n",
                encoding="utf-8",
            )

            warmup.load_environment(pipeline_dir)

        self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from-env-local")

    def test_parse_args_accepts_model_filter_and_request_count(self) -> None:
        args = warmup.parse_args(["--models", "Google: Gemma 4 26B A4B (free)", "--requests", "1"])

        self.assertEqual(args.models, ["Google: Gemma 4 26B A4B (free)"])
        self.assertEqual(args.requests, 1)


if __name__ == "__main__":
    unittest.main()
