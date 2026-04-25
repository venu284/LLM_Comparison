#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from components.llm_gateway import LLMGateway
from config import MODELS, ModelConfig, get_experiment_settings


def load_environment(pipeline_dir: Path = PIPELINE_DIR) -> None:
    load_dotenv(pipeline_dir / ".env")
    load_dotenv(pipeline_dir / ".env.local", override=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm up configured LLM connections")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Warm up only these display model names",
    )
    parser.add_argument(
        "--requests",
        type=int,
        help="Number of warmup requests per selected model",
    )
    return parser.parse_args(argv)


def select_models(model_names: Optional[Sequence[str]]) -> list[ModelConfig]:
    if not model_names:
        return MODELS

    requested = {name.strip() for name in model_names}
    selected = [model for model in MODELS if model.name in requested]
    missing = sorted(requested - {model.name for model in selected})
    if missing:
        valid = ", ".join(model.name for model in MODELS)
        raise ValueError(f"Unknown model name(s): {', '.join(missing)}. Valid models: {valid}")
    return selected


async def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    load_environment(PIPELINE_DIR)
    configured_requests = int(get_experiment_settings().get("warmup_requests", 3))
    warmup_requests = args.requests if args.requests is not None else configured_requests
    gateway = LLMGateway(models=select_models(args.models))
    await gateway.warmup(warmup_requests)


if __name__ == "__main__":
    asyncio.run(main())
