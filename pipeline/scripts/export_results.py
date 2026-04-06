#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from components.results_storage import ResultsStorage
from config import ensure_runtime_dirs, get_database_url


def main() -> None:
    parser = argparse.ArgumentParser(description="Export benchmark runs to CSV")
    parser.add_argument("--output", default="exports/results.csv", help="CSV output path")
    args = parser.parse_args()

    load_dotenv(PIPELINE_DIR / ".env")
    ensure_runtime_dirs()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (PIPELINE_DIR / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    storage = ResultsStorage(get_database_url() or "")
    storage.connect()
    try:
        storage.export_to_csv(str(output_path))
    finally:
        storage.close()


if __name__ == "__main__":
    main()
