"""Results storage for local PostgreSQL and Neon Serverless Postgres.

Neon uses the standard PostgreSQL wire protocol, so psycopg2 works unchanged.
The connection string is the only difference:

  Local: postgresql://postgres:password@localhost:5432/llm_benchmark
  Neon:  postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import List, Optional

import psycopg2
from psycopg2.extras import Json, RealDictCursor

from models.run_result import RunResult

logger = logging.getLogger(__name__)


class ResultsStorage:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.conn = None

    def connect(self) -> None:
        if not self.database_url:
            raise ValueError("DATABASE_URL is required for results storage")
        self.conn = psycopg2.connect(self.database_url)
        self.conn.autocommit = True
        logger.info("Database connected successfully")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def initialize_schema(self) -> None:
        """Create all tables and indexes. Idempotent and safe to call repeatedly."""
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        schema_sql = """
        CREATE TABLE IF NOT EXISTS tasks (
            id SERIAL PRIMARY KEY,
            task_id VARCHAR(10) UNIQUE NOT NULL,
            category VARCHAR(20) NOT NULL,
            difficulty VARCHAR(10) NOT NULL,
            title TEXT NOT NULL,
            test_count INTEGER NOT NULL,
            tags TEXT[],
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS models (
            id SERIAL PRIMARY KEY,
            name VARCHAR(50) UNIQUE NOT NULL,
            model_id VARCHAR(100) NOT NULL,
            provider VARCHAR(30) NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS runs (
            id SERIAL PRIMARY KEY,
            task_id INTEGER REFERENCES tasks(id) NOT NULL,
            model_id INTEGER REFERENCES models(id) NOT NULL,
            run_number INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            raw_response TEXT,
            extracted_code TEXT,
            extraction_success BOOLEAN NOT NULL DEFAULT FALSE,
            latency_ttft_ms INTEGER,
            latency_total_ms INTEGER NOT NULL,
            tokens_input INTEGER NOT NULL DEFAULT 0,
            tokens_output INTEGER NOT NULL DEFAULT 0,
            pass_fail BOOLEAN NOT NULL DEFAULT FALSE,
            test_results JSONB,
            tests_passed INTEGER NOT NULL DEFAULT 0,
            tests_total INTEGER NOT NULL DEFAULT 0,
            eslint_warnings INTEGER DEFAULT 0,
            ts_any_count INTEGER DEFAULT 0,
            compiler_errors INTEGER DEFAULT 0,
            failure_mode VARCHAR(50),
            failure_notes TEXT,
            estimated_cost_usd NUMERIC(10, 6),
            created_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(task_id, model_id, run_number)
        );

        CREATE TABLE IF NOT EXISTS human_votes (
            id SERIAL PRIMARY KEY,
            task_id INTEGER REFERENCES tasks(id) NOT NULL,
            model_a INTEGER REFERENCES models(id) NOT NULL,
            model_b INTEGER REFERENCES models(id) NOT NULL,
            winner VARCHAR(10) NOT NULL,
            evaluator_id VARCHAR(50),
            created_at TIMESTAMP DEFAULT NOW()
        );

        CREATE TABLE IF NOT EXISTS performance_profiles (
            id SERIAL PRIMARY KEY,
            model_id INTEGER REFERENCES models(id) NOT NULL,
            category VARCHAR(20) NOT NULL,
            pass_rate NUMERIC(5, 2),
            avg_latency_ms INTEGER,
            avg_tokens_output INTEGER,
            avg_estimated_cost NUMERIC(10, 6),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(model_id, category)
        );

        CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
        CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id);
        CREATE INDEX IF NOT EXISTS idx_runs_pass ON runs(pass_fail);
        """
        with self.conn.cursor() as cursor:
            cursor.execute(schema_sql)
        logger.info("Schema initialized")

    def seed_models(self) -> None:
        """Insert OpenRouter model records. Idempotent."""
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        models_data = [
            ("Nemotron-3-Super", "openrouter/nvidia/nemotron-3-super-120b-a12b:free", "openrouter"),
            ("GLM-4.5-Air", "openrouter/z-ai/glm-4.5-air:free", "openrouter"),
            ("GPT-OSS-120B", "openrouter/openai/gpt-oss-120b:free", "openrouter"),
            ("MiniMax-M2.5", "openrouter/minimax/minimax-m2.5:free", "openrouter"),
            (
                "Google: Gemma 4 26B A4B (free)",
                "openrouter/google/gemma-4-26b-a4b-it:free",
                "openrouter",
            ),
        ]
        with self.conn.cursor() as cursor:
            for name, model_id, provider in models_data:
                cursor.execute(
                    """INSERT INTO models (name, model_id, provider)
                       VALUES (%s, %s, %s) ON CONFLICT (name) DO NOTHING""",
                    (name, model_id, provider),
                )
        logger.info("Models seeded")

    def seed_tasks_from_benchmark(self, benchmark_dir: Optional[str] = None) -> None:
        """Seed tasks from benchmark/tasks/*.json files. Idempotent."""
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        if benchmark_dir:
            benchmark_path = Path(benchmark_dir)
        else:
            benchmark_path = Path(__file__).resolve().parent.parent.parent / "benchmark"

        if not benchmark_path.exists():
            logger.warning("Benchmark directory not found: %s", benchmark_path)
            return

        count = 0
        for category in ["frontend", "api", "css", "typescript", "bugfix"]:
            tasks_dir = benchmark_path / "tasks" / category
            if not tasks_dir.exists():
                continue
            for task_file in sorted(tasks_dir.glob("*.json")):
                with open(task_file, "r", encoding="utf-8") as handle:
                    task_data = json.load(handle)
                tags = task_data.get("tags", [])
                with self.conn.cursor() as cursor:
                    cursor.execute(
                        """INSERT INTO tasks (task_id, category, difficulty, title, test_count, tags)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT (task_id) DO NOTHING""",
                        (
                            task_data["task_id"],
                            task_data["category"],
                            task_data["difficulty"],
                            task_data["title"],
                            task_data["test_count"],
                            tags,
                        ),
                    )
                count += 1
        logger.info("Seeded %s tasks", count)

    def setup_database(self, benchmark_dir: Optional[str] = None) -> None:
        """Create schema, seed models, and seed tasks in one idempotent call."""
        self.initialize_schema()
        self.seed_models()
        self.seed_tasks_from_benchmark(benchmark_dir)
        logger.info("Database setup complete")

    def save_run(self, result: RunResult) -> int:
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        query = """
        WITH ids AS (
            SELECT t.id AS task_db_id, m.id AS model_db_id
            FROM tasks t
            JOIN models m ON TRUE
            WHERE t.task_id = %s AND m.name = %s
        )
        INSERT INTO runs (
            task_id, model_id, run_number, timestamp,
            raw_response, extracted_code, extraction_success,
            latency_ttft_ms, latency_total_ms,
            tokens_input, tokens_output,
            pass_fail, test_results, tests_passed, tests_total,
            eslint_warnings, ts_any_count, compiler_errors,
            failure_mode, failure_notes, estimated_cost_usd
        )
        SELECT
            ids.task_db_id, ids.model_db_id, %s, %s,
            %s, %s, %s,
            %s, %s,
            %s, %s,
            %s, %s, %s, %s,
            %s, %s, %s,
            %s, %s, %s
        FROM ids
        ON CONFLICT (task_id, model_id, run_number) DO UPDATE SET
            timestamp = EXCLUDED.timestamp,
            raw_response = EXCLUDED.raw_response,
            extracted_code = EXCLUDED.extracted_code,
            extraction_success = EXCLUDED.extraction_success,
            latency_ttft_ms = EXCLUDED.latency_ttft_ms,
            latency_total_ms = EXCLUDED.latency_total_ms,
            tokens_input = EXCLUDED.tokens_input,
            tokens_output = EXCLUDED.tokens_output,
            pass_fail = EXCLUDED.pass_fail,
            test_results = EXCLUDED.test_results,
            tests_passed = EXCLUDED.tests_passed,
            tests_total = EXCLUDED.tests_total,
            eslint_warnings = EXCLUDED.eslint_warnings,
            ts_any_count = EXCLUDED.ts_any_count,
            compiler_errors = EXCLUDED.compiler_errors,
            failure_mode = EXCLUDED.failure_mode,
            failure_notes = EXCLUDED.failure_notes,
            estimated_cost_usd = EXCLUDED.estimated_cost_usd
        RETURNING id;
        """
        with self.conn.cursor() as cursor:
            cursor.execute(
                query,
                (
                    result.task_id,
                    result.model_name,
                    result.run_number,
                    result.timestamp,
                    result.raw_response,
                    result.extracted_code,
                    result.extraction_success,
                    result.latency_ttft_ms,
                    result.latency_total_ms,
                    result.tokens_input,
                    result.tokens_output,
                    result.pass_fail,
                    Json(result.test_assertions),
                    result.tests_passed,
                    result.tests_total,
                    result.eslint_warnings,
                    result.ts_any_count,
                    result.compiler_errors,
                    result.failure_mode,
                    result.failure_notes,
                    result.estimated_cost_usd,
                ),
            )
            row = cursor.fetchone()
            return row[0] if row else -1

    def get_runs_for_task(self, task_id: str) -> List[dict]:
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        query = """
        SELECT r.*, t.task_id, t.category, t.difficulty, m.name AS model_name
        FROM runs r
        JOIN tasks t ON r.task_id = t.id
        JOIN models m ON r.model_id = m.id
        WHERE t.task_id = %s
        ORDER BY m.name, r.run_number;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (task_id,))
            return list(cursor.fetchall())

    def get_category_summary(self, category: str) -> List[dict]:
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        query = """
        SELECT
            m.name AS model_name,
            COUNT(*) AS total_runs,
            SUM(CASE WHEN r.pass_fail THEN 1 ELSE 0 END) AS passed,
            ROUND(AVG(CASE WHEN r.pass_fail THEN 1.0 ELSE 0.0 END) * 100, 1) AS pass_rate,
            ROUND(AVG(r.latency_total_ms)) AS avg_latency_ms,
            ROUND(AVG(r.tokens_output)) AS avg_tokens_output,
            ROUND(AVG(r.estimated_cost_usd)::numeric, 6) AS avg_cost
        FROM runs r
        JOIN tasks t ON r.task_id = t.id
        JOIN models m ON r.model_id = m.id
        WHERE t.category = %s
        GROUP BY m.name
        ORDER BY pass_rate DESC, m.name ASC;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, (category,))
            return list(cursor.fetchall())

    def get_full_summary(self) -> List[dict]:
        """Return aggregated results per model, category, and difficulty."""
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        query = """
        SELECT
            m.name AS model_name,
            t.category,
            t.difficulty,
            COUNT(*) AS total_runs,
            SUM(CASE WHEN r.pass_fail THEN 1 ELSE 0 END) AS passed,
            ROUND(AVG(CASE WHEN r.pass_fail THEN 1.0 ELSE 0.0 END) * 100, 1) AS pass_rate,
            ROUND(AVG(r.latency_total_ms)) AS avg_latency_ms,
            ROUND(AVG(r.tokens_output)) AS avg_tokens_output,
            ROUND(SUM(r.estimated_cost_usd)::numeric, 6) AS total_cost
        FROM runs r
        JOIN tasks t ON r.task_id = t.id
        JOIN models m ON r.model_id = m.id
        GROUP BY m.name, t.category, t.difficulty
        ORDER BY m.name, t.category, t.difficulty;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            return list(cursor.fetchall())

    def export_to_csv(self, output_path: str) -> None:
        if not self.conn:
            raise RuntimeError("Database connection has not been established")

        query = """
        SELECT
            t.task_id,
            t.category,
            t.difficulty,
            m.name AS model_name,
            r.run_number,
            r.pass_fail,
            r.latency_total_ms,
            r.tokens_input,
            r.tokens_output,
            r.estimated_cost_usd,
            r.extraction_success,
            r.tests_passed,
            r.tests_total,
            r.eslint_warnings,
            r.ts_any_count,
            r.compiler_errors,
            r.failure_mode,
            r.timestamp
        FROM runs r
        JOIN tasks t ON r.task_id = t.id
        JOIN models m ON r.model_id = m.id
        ORDER BY t.task_id, m.name, r.run_number;
        """
        with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            rows = list(cursor.fetchall())

        if not rows:
            logger.info("No rows available for CSV export")
            return

        with open(output_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        logger.info("Exported %s rows to %s", len(rows), output_path)
