#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from dotenv import load_dotenv

from components.llm_gateway import LLMGateway
from config import get_experiment_settings
from run_eval import evaluate_all


def main() -> None:
    load_dotenv(Path(__file__).resolve().parent / ".env")
    experiment = get_experiment_settings()

    parser = argparse.ArgumentParser(description="Batch runner for all benchmark tasks")
    parser.add_argument("--models", type=str, nargs="+", help="Filter to specific model names")
    parser.add_argument(
        "--runs",
        type=int,
        default=int(experiment.get("runs_per_task", 3)),
        help="Number of runs per task per model",
    )
    parser.add_argument("--warmup", action="store_true", help="Run warm-up requests first")
    parser.add_argument("--db", action="store_true", help="Store results in the database. Off by default: the Phase 5 runs table is the published record and upserts on (task, model, run_number) would overwrite it.")
    parser.add_argument("--no-db", action="store_true", help="Deprecated; database storage is already off by default")
    args = parser.parse_args()

    if args.warmup:
        gateway = LLMGateway(
            temperature=float(experiment.get("temperature", 0.2)),
            max_tokens=int(experiment.get("max_tokens", 4096)),
        )
        asyncio.run(gateway.warmup(int(experiment.get("warmup_requests", 3))))

    asyncio.run(
        evaluate_all(
            model_names=args.models,
            runs=args.runs,
            store_results=args.db,
        )
    )


if __name__ == "__main__":
    main()
