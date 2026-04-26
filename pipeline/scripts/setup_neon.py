#!/usr/bin/env python3
"""One-command database setup for Neon Serverless Postgres or any PostgreSQL.

Usage:
    python scripts/setup_neon.py

Prerequisites:
    1. Create a free Neon account at https://neon.tech (no credit card needed)
    2. Create a new project
    3. Copy the connection string into pipeline/.env:
       DATABASE_URL=postgresql://user:pass@ep-xxx.us-east-2.aws.neon.tech/neondb?sslmode=require
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from components.results_storage import ResultsStorage
from config import BENCHMARK_DIR, get_database_url


def main() -> None:
    load_dotenv(PIPELINE_DIR / ".env")

    database_url = get_database_url()
    if not database_url:
        print("ERROR: DATABASE_URL not set in .env file")
        print()
        print("Steps to fix:")
        print("  1. Go to https://neon.tech and create a free account")
        print("  2. Create a new project and copy the connection string")
        print("  3. Add to pipeline/.env:")
        print("     DATABASE_URL=postgresql://user:pass@ep-xxx.aws.neon.tech/neondb?sslmode=require")
        sys.exit(1)

    print("Connecting to database...")
    storage = ResultsStorage(database_url)
    storage.connect()

    try:
        print("Setting up schema, models, and tasks...")
        storage.setup_database(str(BENCHMARK_DIR))

        with storage.conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM tasks")
            task_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM models")
            model_count = cursor.fetchone()[0]

        print()
        print("Database ready:")
        print(f"  Tasks:  {task_count}")
        print(f"  Models: {model_count}")
        print("  Tables: tasks, models, runs, human_votes, performance_profiles")
    finally:
        storage.close()

    print()
    print("Done. You can now run:")
    print('  python run_eval.py --task FE-001 --models "Llama-3.1-8B" --runs 1')


if __name__ == "__main__":
    main()
