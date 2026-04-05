#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TASK_ID="${1:-}"

if [[ -z "$TASK_ID" ]]; then
  echo "Usage: bash scripts/run-task.sh <TASK_ID>"
  exit 1
fi

case "$TASK_ID" in
  FE-*)
    npx jest "tests/frontend/${TASK_ID}.test.jsx" --runInBand
    ;;
  API-*)
    npx jest "tests/api/${TASK_ID}.test.js" --runInBand
    ;;
  CSS-*)
    if [[ -f "tests/css/${TASK_ID}.spec.js" ]]; then
      npx playwright test "tests/css/${TASK_ID}.spec.js"
    else
      npx jest "tests/css/${TASK_ID}.test.js" --runInBand
    fi
    ;;
  TS-*)
    npx jest "tests/typescript/${TASK_ID}.test.ts" --runInBand
    ;;
  BF-*)
    for ext in js jsx ts tsx; do
      if [[ -f "tests/bugfix/${TASK_ID}.test.${ext}" ]]; then
        BUGFIX_VARIANT="${BUGFIX_VARIANT:-fixed}" npx jest "tests/bugfix/${TASK_ID}.test.${ext}" --runInBand
        exit 0
      fi
    done
    echo "No bugfix test file found for ${TASK_ID}"
    exit 1
    ;;
  *)
    echo "Unsupported task id: $TASK_ID"
    exit 1
    ;;
esac
