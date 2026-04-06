#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
import psycopg2

PIPELINE_DIR = Path(__file__).resolve().parent.parent
DB_DIR = Path(__file__).resolve().parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))
if str(DB_DIR) not in sys.path:
    sys.path.insert(0, str(DB_DIR))

from seed_tasks import generate_seed_sql


def _execute_file(cursor, filepath: Path) -> None:
    with open(filepath, "r", encoding="utf-8") as handle:
        sql = handle.read().strip()
    if sql:
        cursor.execute(sql)


def main() -> None:
    load_dotenv(PIPELINE_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required to run migrations")

    db_dir = Path(__file__).resolve().parent
    schema_path = db_dir / "schema.sql"
    seed_models_path = db_dir / "seed_models.sql"
    seed_tasks_path = generate_seed_sql(db_dir / "seed_tasks.sql")

    with psycopg2.connect(database_url) as conn:
        conn.autocommit = True
        with conn.cursor() as cursor:
            _execute_file(cursor, schema_path)
            _execute_file(cursor, seed_models_path)
            _execute_file(cursor, seed_tasks_path)

    print("Migrations completed successfully.")


if __name__ == "__main__":
    main()
