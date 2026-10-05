# Comparative Evaluation of Large Language Models for Web Development Tasks

This repository contains a major project for comparing large language models on practical web-development tasks. It combines a curated benchmark suite with an automated evaluation pipeline that prompts selected LLMs, extracts generated code, runs task-specific tests in Docker, records metrics, and exports results for analysis.

The project is organized around two main modules:

- `benchmark/` - a web-development benchmark with tasks, tests, fixtures, and reference solutions.
- `pipeline/` - a Python evaluation pipeline that runs LLM experiments against the benchmark and stores the results.

## Project Status

Phases 1–6 are complete: benchmark, pipeline, 325 evaluation runs, and statistical analysis.
Phase 6 found that routing between these five models has little headroom (+8.6% oracle ceiling),
because their successes are nested. The project has therefore moved on to **PROBE-CASCADE**: a
cost-aware router for coding tasks that onboards newly added models with a small measured probe.
The 65 web-dev tasks here serve as its showcase domain.

- Current plan, status and next steps: [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md)
- Design rationale and literature: [Reports/Research_Progress_Report_7.md](Reports/Research_Progress_Report_7.md)
- Phase 5–7 results: [Reports/Research_Progress_Report_6.md](Reports/Research_Progress_Report_6.md)
- Frozen Phase 5 data: [data/phase5_results.csv](data/README.md)

## Project Objectives

The goal of this project is to evaluate how well different LLMs can solve realistic development tasks across common web engineering areas:

- React frontend component implementation
- REST API and backend logic
- CSS layout and styling
- TypeScript data modeling
- Bug fixing and refactoring

The evaluation focuses on executable correctness rather than subjective scoring. Each model response is converted into code, placed into the benchmark environment, and validated with automated tests. The pipeline records pass/fail status, test counts, latency, token usage, extraction success, and estimated cost.

## System Architecture

```text
benchmark task JSON
        |
        v
pipeline task loader
        |
        v
LLM prompt through LiteLLM / Groq
        |
        v
raw model response
        |
        v
code extractor
        |
        v
Docker benchmark runner
        |
        v
metrics collector
        |
        v
PostgreSQL / Neon database, CSV exports, charts
```

The benchmark module owns the task definitions and validation logic. The pipeline module owns experiment execution, model access, result collection, database storage, and reporting.

## Benchmark Suite

The benchmark contains 65 total tasks across five categories:

| Category | Task Prefix | Count | Difficulty Split |
| --- | --- | ---: | --- |
| Frontend / React | `FE-*` | 13 | 4 easy, 4 medium, 5 hard |
| REST API / Backend | `API-*` | 13 | 4 easy, 4 medium, 5 hard |
| CSS / Layout | `CSS-*` | 13 | 4 easy, 4 medium, 5 hard |
| TypeScript | `TS-*` | 13 | 4 easy, 4 medium, 5 hard |
| Bug Fixing | `BF-*` | 13 | 4 easy, 4 medium, 5 hard |

Each task includes metadata, a prompt, expected output details, automated tests, and a reference solution. Bug-fix tasks also include buggy source fixtures.

See [benchmark/README.md](benchmark/README.md) for benchmark-specific details.

## Evaluation Pipeline

The pipeline performs the full model evaluation workflow:

1. Loads one or more benchmark tasks from `benchmark/tasks/`.
2. Sends each task prompt to configured models through LiteLLM.
3. Extracts runnable code from fenced or raw LLM responses.
4. Runs the extracted code against benchmark tests through `benchmark/docker-run.sh`.
5. Collects correctness, latency, token, compiler, and extraction metrics.
6. Stores run data in PostgreSQL or Neon.
7. Exports CSV files and generates result charts.

Configured models are defined in `pipeline/config.py`:

- `Llama-3.3-70B`
- `Llama-4-Scout`
- `GPT-OSS-120B`
- `Qwen3-32B`
- `Llama-3.1-8B`

All configured models currently use Groq API credentials through `GROQ_API_KEY`.

See [pipeline/README.md](pipeline/README.md) for pipeline-specific details.

## Prerequisites

Install the following before running the full project:

- Docker
- Node.js 20 or compatible Node.js runtime
- Python 3.10 or newer
- A Groq API key
- A PostgreSQL or Neon database URL, if storing results

Docker is recommended for benchmark validation because it gives a consistent Linux test environment and installs Playwright Chromium inside the container.

## Setup

Clone the repository and enter the project root:

```bash
cd LLM_Comparison
```

Install benchmark dependencies:

```bash
cd benchmark
npm install
npx playwright install chromium
cd ..
```

Set up the Python pipeline:

```bash
cd pipeline
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `pipeline/.env` with your own credentials:

```bash
DATABASE_URL=postgresql://user:pass@host/dbname?sslmode=require
GROQ_API_KEY=gsk_...
```

The real `.env` file is ignored by Git and should not be committed.

Prepare the database:

```bash
python scripts/setup_neon.py
cd ..
```

Build and validate the benchmark Docker image:

```bash
cd benchmark
./docker-run.sh build
./docker-run.sh validate
cd ..
```

## Running the Benchmark

Run all benchmark validation tests:

```bash
cd benchmark
npm run validate
```

Run a single task locally:

```bash
bash scripts/run-task.sh FE-001
```

Run validation through Docker:

```bash
./docker-run.sh task FE-001
```

Evaluate a candidate solution file against one task:

```bash
./docker-run.sh eval FE-001 path/to/output.jsx
```

## Running LLM Evaluations

Results are **not** written to the database unless `--db` is passed. The Phase 5 `runs` table is
the published record, and database writes upsert on `(task, model, run_number)`, so a new run
could overwrite a Phase 5 row. Store new runs in a separate table or database.

Run a single task against all configured models:

```bash
cd pipeline
source venv/bin/activate
python run_eval.py --task FE-001 --runs 1
```

Run one task against selected models:

```bash
python run_eval.py --task FE-001 --models "Llama-3.1-8B" "GPT-OSS-120B" --runs 1
```

Run a full category:

```bash
python run_eval.py --category frontend --runs 3
```

Run the full experiment (Phase 5 used one run per task):

```bash
python run_eval_all.py --runs 3
```

Warm up model connections before an experiment:

```bash
python run_eval.py --warmup --task FE-001 --runs 1
```

## Exporting and Visualizing Results

Export stored database results to CSV:

```bash
cd pipeline
python scripts/export_results.py --output exports/results.csv
```

Generate charts and a summary from stored results:

```bash
python scripts/visualize_results.py
```

Runtime outputs are written under:

- `pipeline/logs/` - pipeline logs
- `pipeline/temp/` - extracted temporary model outputs
- `pipeline/exports/` - CSV exports, summaries, and charts
- `benchmark/results/` - benchmark runner results

These runtime artifacts are ignored by Git except for placeholder files.

## Repository Structure

```text
LLM_Comparison/
|-- README.md
|-- CLAUDE.md              # pointer for Claude Code sessions
|-- pyproject.toml         # pytest configuration
|-- benchmark/             # 65 web-dev tasks, tests, solutions, Docker harness
|-- data/                  # frozen Phase 5 results snapshot
|-- docs/
|   |-- IMPLEMENTATION_PLAN.md   # living plan for the router
|   |-- history/handover/        # session handover notes
|   `-- superpowers/specs/       # design specs and the router charter
|-- human_eval/            # Phase 7 pairwise voting app and Bradley-Terry
|-- pipeline/
|   |-- analysis/          # Phase 6 statistics, recommender, failure taxonomy
|   |   `-- exploratory/   # Phase 8 cost and probe exploration scripts
|   |-- components/        # LLM gateway, code extractor, test runner, storage
|   |-- db/
|   |-- models/
|   |-- scripts/
|   |-- tests/
|   |-- config.py
|   |-- config.yaml
|   |-- run_eval.py
|   `-- run_eval_all.py
`-- Reports/               # progress reports and figures
```

Tests run from the repository root with `pipeline/venv/bin/python -m pytest`.

## Key Metrics

The pipeline captures the following metrics for each run:

- Model name and model ID
- Task ID, category, and difficulty
- Run number and timestamp
- Extraction success and extraction method
- Test pass/fail result
- Tests passed and total tests
- Latency in milliseconds
- Input and output token counts
- TypeScript `any` usage count for TypeScript tasks
- Compiler error count
- Estimated cost in USD
- Failure mode (defined in the schema but not yet populated: it is empty in all Phase 5 rows, and `pipeline/analysis/failure_taxonomy.py` derives it at analysis time instead)

## Notes and Limitations

- The benchmark evaluates functional correctness through automated tests; it does not replace human review for maintainability, readability, or security.
- Cost values are estimates based on rates configured in `pipeline/config.yaml`. The current configuration uses zero-cost Groq free-tier estimates.
- API availability, rate limits, and model behavior can affect repeated experiment results.
- Docker must be available for the pipeline's automated candidate evaluation step.
- The real `pipeline/.env` file may contain credentials and must remain private.

## Related Documentation

- [Benchmark README](benchmark/README.md)
- [Pipeline README](pipeline/README.md)
