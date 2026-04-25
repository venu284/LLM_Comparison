from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import config


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


if __name__ == "__main__":
    unittest.main()
