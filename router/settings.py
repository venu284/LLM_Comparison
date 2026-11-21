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
# Values the plan text left open (open items O1-O19 of the draft) were decided
# by the user when docs/CLAIMS.md was signed off on 2026-11-20.
# =============================================================================

# S0.1 Acceptance: success means all tests pass (released score = 1.0).
SUCCESS_SCORE = 1.0
# S0.1 sensitivity threshold for CRUXEval (CLAIMS amendments A2 and A3, 2026-11-21). A2 made 0.8
# CRUXEval's success threshold. A3 reversed that for the headline, because the S1.7 audit could
# confirm a stated correct value in only part of the 0.8 cells. The headline uses SUCCESS_SCORE;
# this value is used only by the sensitivity check (scoring.success(..., cruxeval_lenient=True)).
CRUXEVAL_SUCCESS_SCORE = 0.8
CRUXEVAL_FAMILY = "cruxeval"

# S0.2 Cost: verifier runs priced at the X13 mean visible-test run time over the
# cells where a verifier ran (0.200 s; the earlier 0.074 s averaged in the cells
# with no visible tests as 0 s), on AWS Lambda x86, 2 GB, us-east-1, plus the
# per-request charge (O11; source and date in router/data/prices.py), about
# $6.9e-6 per run, swept x1, x10, x100. Amended 2026-11-20 (CLAIMS amendment A1);
# the P0.5, P0.6 and E0 records used the earlier $1e-6.
VERIFIER_RUN_S = 0.200
VERIFIER_RUN_USD = 6.9e-6
VERIFIER_COST_SWEEP = (1, 10, 100)
# S0.2 price-sensitivity sweep (O13): each model's price x0.5 and x2, one model at a time.
PRICE_SWEEP = (0.5, 2.0)

# S0.3 Performance gate, judged on CI lower bounds of the cross-fitted scores.
GATE_SAVING = 20.0      # (a) saving vs the best single model, percent, lower bound >= this
GATE_GAP = -2.0         # (b) acceptance gap vs the best single model, points, lower bound >= this
C_SAVING = 0.0          # (c) saving vs CellBest with predicted cells, percent, lower bound > this (strict)
C_GAP = -2.0            # (c) acceptance gap vs CellBest with predicted cells, points, lower bound >= this
CI_LEVEL = 0.95         # S0.3, S0.8: 95% cluster-bootstrap CIs

# S0.6 Development split and nested cross-fitting (option (a)).
DEV_FRACTION = 0.20     # share of problems in the development split
DEV_SEED = 0            # seed of the one development-split draw (O6)
N_FOLDS = 5             # outer folds over the evaluation population, by problem
N_INNER_FOLDS = 4       # inner folds: one per remaining outer fold (O7)

# S0.8 Statistics.
N_SEEDS = 5             # the gate seed plus "the other four seeds"
AT_THRESHOLD_MARGIN = 0.1  # points (percentage points for savings); a reporting label only (O12)
GATE_SEED = 0           # the single pre-registered gate seed (O5)
N_BOOT = 1000           # cluster-bootstrap resamples outside gate scoring (O4)
N_BOOT_GATE = 2000      # cluster-bootstrap resamples for gate scoring (O4)

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
EXPLORATION_SHARE = 0.10   # exactly 10% of each group's requests (a group is an S1.1 family, O8, O9)
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
