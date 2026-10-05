"""Paths and constants shared by the router.

Named `settings`, not `config`: pipeline/ owns the bare `config` module.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Frozen Phase 5 snapshot (325 runs). Read-only; never edit it.
PHASE5_CSV = REPO_ROOT / "data" / "phase5_results.csv"

# Third-party code and datasets (CodeRouterBench, ACRouter scripts). Git-ignored.
EXTERNAL_DIR = REPO_ROOT / "external"

# The 65 web-dev tasks and their Docker harness.
BENCHMARK_DIR = REPO_ROOT / "benchmark"
