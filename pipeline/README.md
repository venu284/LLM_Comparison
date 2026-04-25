# Phase 4 Evaluation Pipeline

This module builds the automated evaluation pipeline on top of the Phase 3 benchmark in [`../benchmark`](../benchmark).

## What It Does

- Loads benchmark task JSON files directly from the Phase 3 suite
- Sends each task prompt to one or more LLMs through LiteLLM
- Extracts runnable code from raw LLM responses
- Evaluates the extracted code through the benchmark Docker runner
- Collects metrics such as pass rate, latency, token usage, and estimated cost
- Stores run results in PostgreSQL and exports them to CSV

## Structure

```text
pipeline/
├── run_eval.py
├── run_eval_all.py
├── config.py
├── config.yaml
├── requirements.txt
├── components/
├── models/
├── db/
├── scripts/
├── temp/
├── logs/
└── exports/
```

## Setup

```bash
cd pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your Neon database URL and OpenRouter API key:

```bash
DATABASE_URL=postgresql://user:pass@ep-cool-night-123456.us-east-2.aws.neon.tech/neondb?sslmode=require
OPENROUTER_API_KEY=sk-or-v1-...
```

Then prepare the database and prerequisite benchmark image:

```bash
python scripts/setup_neon.py
cd ../benchmark && ./docker-run.sh build && ./docker-run.sh validate && cd ../pipeline
```

## Common Commands

```bash
# Warm up model connections
python scripts/warmup.py

# Dry run: one task, one run, no DB writes
python run_eval.py --task FE-001 --runs 1 --no-db

# Single task with selected models
python run_eval.py --task FE-001 --models "Nemotron-3-Super" "Gemma-4-31B" --runs 1 --no-db

# Category batch
python run_eval.py --category frontend --runs 3

# Full experiment
python run_eval_all.py --runs 3

# Export stored results
python scripts/export_results.py --output exports/results.csv
```

## Notes

- The task loader uses each task JSON `prompt` exactly as stored. It does not re-wrap or rebuild prompts.
- The test runner delegates all execution to [`../benchmark/docker-run.sh`](../benchmark/docker-run.sh).
- Bugfix evaluation uses the benchmark's `.fixed.<ext>` naming convention automatically.
- Cost values are estimates derived from configured token pricing rates in `config.yaml`.
- Runtime artifacts in `temp/`, `logs/`, and `exports/` are git-ignored except for the `.gitkeep` placeholders.

## Model Names

```text
Nemotron-3-Super
GLM-4.5-Air
GPT-OSS-120B
MiniMax-M2.5
Gemma-4-31B
```
