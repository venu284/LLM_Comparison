#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from components.llm_gateway import LLMGateway
from config import get_experiment_settings


async def main() -> None:
    load_dotenv(PIPELINE_DIR / ".env")
    warmup_requests = int(get_experiment_settings().get("warmup_requests", 3))
    gateway = LLMGateway()
    await gateway.warmup(warmup_requests)


if __name__ == "__main__":
    asyncio.run(main())
