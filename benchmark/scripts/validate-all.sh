#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

run_if_present() {
  local label="$1"
  local script_name="$2"
  local path="$3"

  if find "$path" -type f \( -name '*.test.js' -o -name '*.test.jsx' -o -name '*.test.ts' -o -name '*.spec.js' \) | grep -q .; then
    echo "Running ${label}..."
    npm run "$script_name"
  else
    echo "Skipping ${label}: no matching files yet."
  fi
}

run_if_present "frontend tests" "test:frontend" "tests/frontend"
run_if_present "api tests" "test:api" "tests/api"
run_if_present "css tests" "test:css" "tests/css"
run_if_present "typescript tests" "test:typescript" "tests/typescript"

if find "tests/bugfix" -type f \( -name '*.test.js' -o -name '*.test.jsx' -o -name '*.test.ts' -o -name '*.spec.js' \) | grep -q .; then
  echo "Running bugfix tests against fixed solutions..."
  BUGFIX_VARIANT=fixed npm run test:bugfix

  echo "Running bugfix tests against buggy fixtures (expected to fail)..."
  if BUGFIX_VARIANT=buggy npm run test:bugfix; then
    echo "Bugfix tests unexpectedly passed against buggy fixtures."
    exit 1
  fi

  echo "Bugfix tests correctly failed against buggy fixtures."
else
  echo "Skipping bugfix tests: no matching files yet."
fi
