from __future__ import annotations

import csv
import logging
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

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

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
