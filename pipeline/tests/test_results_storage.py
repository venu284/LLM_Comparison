from __future__ import annotations

import json
import sys
import types
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory


if "psycopg2" not in sys.modules:
    class OperationalError(Exception):
        pass

    psycopg2_stub = types.ModuleType("psycopg2")
    psycopg2_stub.connect = lambda *args, **kwargs: None
    psycopg2_stub.OperationalError = OperationalError
    extras_stub = types.ModuleType("psycopg2.extras")
    extras_stub.Json = lambda value: value
    extras_stub.RealDictCursor = object
    psycopg2_stub.extras = extras_stub
    sys.modules["psycopg2"] = psycopg2_stub
    sys.modules["psycopg2.extras"] = extras_stub


PIPELINE_DIR = Path(__file__).resolve().parents[1]
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

import components.results_storage as results_storage_module
from components.results_storage import ResultsStorage
from models.run_result import RunResult


MODEL_SEEDS = [
    ("Llama-3.3-70B", "groq/llama-3.3-70b-versatile", "groq"),
    ("DeepSeek-R1-Distill-70B", "groq/deepseek-r1-distill-llama-70b", "groq"),
    ("Qwen-QwQ-32B", "groq/qwen-qwq-32b", "groq"),
    ("Llama-4-Scout", "groq/meta-llama/llama-4-scout-17b-16e-instruct", "groq"),
    ("Mistral-Saba-24B", "groq/mistral-saba-24b", "groq"),
]


class FakeCursor:
    def __init__(self, row=None, execute_exception=None) -> None:
        self.executed = []
        self.row = row
        self.execute_exception = execute_exception

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, sql, params=None) -> None:
        self.executed.append((sql, params))
        if self.execute_exception:
            exception = self.execute_exception
            self.execute_exception = None
            raise exception

    def fetchone(self):
        return self.row


class FakeConnection:
    def __init__(self, cursor=None, closed=0) -> None:
        self.cursor_instance = cursor or FakeCursor()
        self.closed = closed

    def cursor(self, *args, **kwargs):
        return self.cursor_instance


def make_run_result() -> RunResult:
    return RunResult(
        task_id="FE-001",
        model_name="Llama-3.3-70B",
        model_id="groq/llama-3.3-70b-versatile",
        run_number=1,
        timestamp=datetime(2026, 4, 25, 12, 0, 0),
        raw_response="response",
        extracted_code="export default function App() { return null; }",
        extraction_success=True,
        extraction_method="markdown",
        latency_ttft_ms=100,
        latency_total_ms=500,
        tokens_input=10,
        tokens_output=20,
        pass_fail=True,
        tests_passed=3,
        tests_total=3,
        test_assertions=[],
        eslint_warnings=0,
        ts_any_count=0,
        compiler_errors=0,
        failure_mode=None,
        failure_notes=None,
        estimated_cost_usd=0.0,
    )


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
        self.assertEqual([statement[1] for statement in executed[1:6]], MODEL_SEEDS)
        self.assertEqual(
            executed[-1][1],
            ("FE-001", "frontend", "easy", "Render a button", 3, ["react", "ui"]),
        )

    def test_setup_database_requires_connection(self) -> None:
        storage = ResultsStorage("postgresql://example")

        with self.assertRaisesRegex(RuntimeError, "Database connection has not been established"):
            storage.setup_database()


class ResultsStorageSaveRunTests(unittest.TestCase):
    def test_save_run_connects_when_connection_missing(self) -> None:
        storage = ResultsStorage("postgresql://example")
        connection = FakeConnection(FakeCursor(row=(101,)))
        connect_calls = []

        def fake_connect() -> None:
            connect_calls.append("connect")
            storage.conn = connection

        storage.connect = fake_connect

        run_id = storage.save_run(make_run_result())

        self.assertEqual(run_id, 101)
        self.assertEqual(connect_calls, ["connect"])
        self.assertEqual(connection.cursor_instance.executed[0][1][1], "Llama-3.3-70B")

    def test_save_run_reconnects_when_connection_closed(self) -> None:
        storage = ResultsStorage("postgresql://example")
        storage.conn = FakeConnection(
            FakeCursor(execute_exception=AssertionError("closed connection used")),
            closed=1,
        )
        reconnected = FakeConnection(FakeCursor(row=(102,)))
        connect_calls = []

        def fake_connect() -> None:
            connect_calls.append("connect")
            storage.conn = reconnected

        storage.connect = fake_connect

        run_id = storage.save_run(make_run_result())

        self.assertEqual(run_id, 102)
        self.assertEqual(connect_calls, ["connect"])

    def test_save_run_reconnects_and_retries_after_operational_error(self) -> None:
        storage = ResultsStorage("postgresql://example")
        failure = results_storage_module.psycopg2.OperationalError("SSL connection closed")
        failing = FakeConnection(FakeCursor(execute_exception=failure), closed=0)
        reconnected = FakeConnection(FakeCursor(row=(103,)))
        storage.conn = failing
        connect_calls = []

        def fake_connect() -> None:
            connect_calls.append("connect")
            storage.conn = reconnected

        storage.connect = fake_connect

        run_id = storage.save_run(make_run_result())

        self.assertEqual(run_id, 103)
        self.assertEqual(connect_calls, ["connect"])
        self.assertEqual(len(failing.cursor_instance.executed), 1)
        self.assertEqual(len(reconnected.cursor_instance.executed), 1)


if __name__ == "__main__":
    unittest.main()
