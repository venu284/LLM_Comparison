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


# =============================================================================
# Claim constants (docs/CLAIMS.md)
#
# Every number the S0 claim definitions fix, defined once. Each constant is
# listed in the constants table of docs/CLAIMS.md with its source item in
# docs/IMPLEMENTATION_PLAN.md. A change needs a progress-log entry giving the
# reason, and every result already computed is re-scored (S0 Gate).
#
# Constants marked "pending sign-off" are used by the P0.5/P0.6 scripts but are
# not fixed by the plan text; they are listed as open items in docs/CLAIMS.md.
# =============================================================================

# S0.1 Acceptance: success means all tests pass (released score = 1.0).
SUCCESS_SCORE = 1.0

# S0.2 Cost: verifier runs priced at the X13 mean visible-test run time on a
# published on-demand vCPU-hour rate (about $1e-6 per run), swept x1, x10, x100.
VERIFIER_RUN_S = 0.074
VERIFIER_RUN_USD = 1e-6
VERIFIER_COST_SWEEP = (1, 10, 100)

# S0.3 Performance gate, judged on CI lower bounds of the cross-fitted scores.
GATE_SAVING = 20.0      # (a) saving vs the best single model, percent, lower bound >= this
GATE_GAP = -2.0         # (b) acceptance gap vs the best single model, points, lower bound >= this
C_SAVING = 0.0          # (c) saving vs CellBest with predicted cells, percent, lower bound > this (strict)
C_GAP = -2.0            # (c) acceptance gap vs CellBest with predicted cells, points, lower bound >= this
CI_LEVEL = 0.95         # S0.3, S0.8: 95% cluster-bootstrap CIs

# S0.6 Development split and nested cross-fitting (option (a)).
DEV_FRACTION = 0.20     # share of problems in the development split
N_FOLDS = 5             # outer folds over the evaluation population, by problem

# S0.8 Statistics.
N_SEEDS = 5             # the gate seed plus "the other four seeds"
AT_THRESHOLD_MARGIN = 0.1  # points: a point estimate within this of its threshold is "at the threshold"
GATE_SEED = 0           # pending sign-off: S0.8 says "a single pre-registered seed"; 0 is P0.5/P0.6's choice
N_BOOT = 1000           # pending sign-off: resample count not set in S0.8; P0.5/P0.6 used 1,000

# S0.9 Live pool and budget.
LIVE_BUDGET_USD = 20.0
PAID_TOP_MODEL = "claude-opus-4-6"
PAID_TOP_PRICE_USD_PER_MTOK = (5.0, 25.0)   # (input, output), per million tokens
MIN_RUNG_PRICE_RATIO = 3.0                  # at least a 3x price gap between neighbouring rungs
MIN_LIVE_FAMILIES = 2                       # at least 2 model families ...
MIN_LIVE_PROVIDERS = 2                      # ... from at least 2 providers
LIVE_HELD_OUT_MODELS = (1, 2)               # models held out of calibration, arriving mid-stream

# S0.13 Observations, probes and pinned thresholds.
MAX_PROBES_PER_GROUP = 5
MAX_PROBES_PER_MODEL = 100
EXPLORATION_SHARE = 0.10   # pending sign-off: the plan says "about 10% per group"
STOP_POSTERIOR_SD = 0.5    # stop exploring a group when the posterior SD of its offset (logit) <= this
MIN_FAMILY_TASKS = 30      # no family under 30 tasks

# Decision points (section 1).
DP1A_MIN_LOWER = 3.0              # DP1a: CI lower bound of headroom over the best single model, points
DP2_K_GRID = (10, 25, 50, 100)    # DP2: fixed k grid
DP3_GAP_SD_MULTIPLE = 2.0         # DP3: routing gap > 2 x run-to-run SD ...
DP3_MIN_RUNS_PER_TASK = 2         # ... measured on >= 2 runs per task

# SC5: at least 3 free-tier models plus the paid top model x 65 web-dev tasks.
SC5_MIN_FREE_MODELS = 3
WEBDEV_N_TASKS = 65
