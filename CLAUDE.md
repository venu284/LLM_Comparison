# LLM_Comparison — notes for Claude sessions

**Before doing any work, read `docs/IMPLEMENTATION_PLAN.md`.** It is the living plan for the
PROBE-CASCADE router, the current phase of this project. It contains the current status, the next
action, the rules that must not be broken (§6), and the session protocol (§9). Update its Status
block and progress log at the end of every session.

Background:
- `Reports/Research_Progress_Report_7.md` covers the design rationale and the literature.
- `Reports/Research_Progress_Report_6.md` covers the Phase 5–7 results.
- `docs/history/handover/` holds earlier session handovers, including the 2026-08-18 adversarial review.

Layout:
- `benchmark/` holds the 65 web-dev tasks, their tests, and the Docker harness (`benchmark/docker-run.sh`).
- `pipeline/` holds data collection (LiteLLM gateway, test runner) and the Phase 6 analysis.
- `data/phase5_results.csv` is the frozen Phase 5 snapshot. Never edit it.
- `router/` (new) holds the PROBE-CASCADE implementation.

Working rules:
- Run tests from the repo root: `pipeline/venv/bin/python -m pytest` (configured in `pyproject.toml`).
- Evaluation runs write to the database only with `--db`. Never overwrite Phase 5 rows.
- `.claude/` is git-ignored local state. Anything worth keeping goes in `docs/`.
