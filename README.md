# Comparative Evaluation of Large Language Models for Web Development Tasks

This repository contains a major project for comparing large language models on practical web-development tasks. It combines a curated benchmark suite with an automated evaluation pipeline that prompts selected LLMs, extracts generated code, runs task-specific tests in Docker, records metrics, and exports results for analysis.

The project is organized around two main modules:

- `benchmark/` - a web-development benchmark with tasks, tests, fixtures, and reference solutions.
- `pipeline/` - a Python evaluation pipeline that runs LLM experiments against the benchmark and stores the results.

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

Run a single task against all configured models without database writes:

```bash
cd pipeline
source venv/bin/activate
python run_eval.py --task FE-001 --runs 1 --no-db
```

Run one task against selected models:

```bash
python run_eval.py --task FE-001 --models "Llama-3.1-8B" "GPT-OSS-120B" --runs 1
```

Run a full category:

```bash
python run_eval.py --category frontend --runs 3
```

Run the full experiment:

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
|-- benchmark/
|   |-- tasks/
|   |-- tests/
|   |-- solutions/
|   |-- fixtures/
|   |-- scripts/
|   |-- Dockerfile
|   |-- docker-run.sh
|   `-- README.md
`-- pipeline/
    |-- components/
    |-- db/
    |-- models/
    |-- scripts/
    |-- tests/
    |-- config.py
    |-- config.yaml
    |-- run_eval.py
    |-- run_eval_all.py
    `-- README.md
```

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
- Failure mode and notes, when available

## Notes and Limitations

- The benchmark evaluates functional correctness through automated tests; it does not replace human review for maintainability, readability, or security.
- Cost values are estimates based on rates configured in `pipeline/config.yaml`. The current configuration uses zero-cost Groq free-tier estimates.
- API availability, rate limits, and model behavior can affect repeated experiment results.
- Docker must be available for the pipeline's automated candidate evaluation step.
- The real `pipeline/.env` file may contain credentials and must remain private.

## Related Documentation

- [Benchmark README](benchmark/README.md)
- [Pipeline README](pipeline/README.md)
