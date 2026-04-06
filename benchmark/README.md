# LLM Web Development Benchmark

This module follows the Phase 3 construction guide and sets up a benchmark workspace for 65 web-development tasks across five categories:

- Frontend / React Components
- REST API / Backend
- CSS / Layout & Styling
- TypeScript / Data Modeling
- Bug Fixing / Refactoring

Current status:

- Benchmark project scaffold is in place
- All 65 tasks are present across the five planned categories
- Each category has 13 tasks with the required 4 easy, 4 medium, and 5 hard split
- Matching task metadata, automated tests, and reference solutions are present for every task
- Bug-fix tasks include buggy fixture files and fixed reference solutions
- Structural verification is complete
- Runtime validation now passes end-to-end through the repo-local Docker workflow in `benchmark/`

## Structure

```text
benchmark/
├── fixtures/
│   ├── buggy-components/
│   ├── mock-api-data.json
│   └── sample-data/
├── scripts/
├── solutions/
│   ├── api/
│   ├── bugfix/
│   ├── css/
│   ├── frontend/
│   └── typescript/
├── tasks/
│   ├── api/
│   ├── bugfix/
│   ├── css/
│   ├── frontend/
│   └── typescript/
└── tests/
    ├── api/
    ├── bugfix/
    ├── css/
    ├── frontend/
    ├── setup/
    └── typescript/
```

## Scripts

- `npm run test:frontend` runs the React benchmark tests
- `npm run test:api` runs the Express/Supertest benchmark tests
- `npm run test:css` runs the static CSS tests and Playwright specs
- `npm run test:typescript` runs the TypeScript benchmark checks
- `npm run test:bugfix` runs the bug-fix benchmark tests for the selected `BUGFIX_VARIANT` (`fixed` by default)
- `npm run validate` runs every category that currently has test files and checks bug-fix tasks against both fixed and buggy variants
- `bash scripts/run-task.sh FE-001` runs a single task by id

## Docker Workflow

- `./docker-run.sh build` builds a Linux validation image from the `benchmark/` directory
- `./docker-run.sh validate` runs the full Phase 2 validation flow inside Docker
- `./docker-run.sh task API-009` runs a single benchmark task inside Docker
- `./docker-run.sh eval FE-001 path/to/output.jsx` overlays one candidate solution into the container and runs that task's tests
- `docker compose run --rm benchmark` provides an optional live-mounted validation workflow

## Notes

- The benchmark tree now contains the full planned suite: 65 tasks, 65 tests, and 65 reference solutions.
- The folder layout matches the construction guide closely, with bug-fix source fixtures stored in `fixtures/buggy-components/`.
- Jest is configured through `ts-jest` so JSX test files can run without adding a separate Babel layer.
- Bug-fix validation now uses the same test files against both `solutions/bugfix/` and `fixtures/buggy-components/`.
- The Docker image installs Node.js dependencies and Playwright Chromium, and `./docker-run.sh validate` now passes end-to-end.
