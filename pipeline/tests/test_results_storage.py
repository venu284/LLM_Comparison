from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


if "psycopg2" not in sys.modules:
    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = lambda *args, **kwargs: None
    extras_stub = types.ModuleType("psycopg2.extras")
    extras_stub.Json = lambda value: value
    extras_stub.RealDictCursor = object
    psycopg2_stub.extras = extras_stub
    sys.modules["psycopg2"] = psycopg2_stub
    sys.modules["psycopg2.extras"] = extras_stub


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from components.results_storage import ResultsStorage


OPENROUTER_MODELS = [
    ("Nemotron-3-Super", "openrouter/nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
    ("GLM-4.5-Air", "openrouter/z-ai/glm-4.5-air:free", "openrouter"),
    ("GPT-OSS-120B", "openrouter/openai/gpt-oss-120b:free", "openrouter"),
    ("MiniMax-M2.5", "openrouter/minimax/minimax-m2.5:free", "openrouter"),
    ("Qwen3-Coder-480B", "openrouter/qwen/qwen3-coder:free", "openrouter"),
]


class FakeCursor:
    def __init__(self) -> None:
        self.executed = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        self.executed.append((sql, params))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def cursor(self, *args, **kwargs):
        return self.cursor_instance


class ResultsStorageSetupTests(unittest.TestCase):
    def test_setup_database_creates_schema_and_seeds_models_and_tasks(self) -> None:
        with TemporaryDirectory() as tmp:
            benchmark_dir = Path(tmp)
            task_dir = benchmark_dir / "tasks" / "frontend"
            task_dir.mkdir(parents=True)
            (task_dir / "FE-001.json").write_text(
                json.dumps(
                    {
                        "task_id": "FE-001",
                        "category": "frontend",
                        "difficulty": "easy",
                        "title": "Render a button",
                        "test_count": 3,
                        "tags": ["react", "ui"],
                    }
                ),
                encoding="utf-8",
            )

            storage = ResultsStorage("postgresql://example")
            storage.conn = FakeConnection()

            storage.setup_database(str(benchmark_dir))

        executed = storage.conn.cursor_instance.executed
        self.assertGreaterEqual(len(executed), 7)
        self.assertIn("CREATE TABLE IF NOT EXISTS tasks", executed[0][0])
        self.assertEqual([statement[1] for statement in executed[1:6]], OPENROUTER_MODELS)
        self.assertEqual(
            executed[-1][1],
            ("FE-001", "frontend", "easy", "Render a button", 3, ["react", "ui"]),
        )

    def test_setup_database_requires_connection(self) -> None:
        storage = ResultsStorage("postgresql://example")

        with self.assertRaisesRegex(RuntimeError, "Database connection has not been established"):
            storage.setup_database()


if __name__ == "__main__":
    unittest.main()
