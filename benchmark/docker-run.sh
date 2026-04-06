#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-llm-webdev-benchmark:latest}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ensure_results_dir() {
  mkdir -p results
}

build_image() {
  echo "Building Docker image..."
  docker build -t "$IMAGE" -f Dockerfile .
}

ensure_image() {
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    build_image
  fi
}

resolve_task() {
  local task_id="$1"
  case "$task_id" in
    FE-*) echo "frontend" ;;
    API-*) echo "api" ;;
    CSS-*) echo "css" ;;
    TS-*) echo "typescript" ;;
    BF-*) echo "bugfix" ;;
    *) echo "unknown" ;;
  esac
}

resolve_ext() {
  local task_id="$1"
  local category
  category="$(resolve_task "$task_id")"
  local task_file="tasks/${category}/${task_id}.json"

  if [[ -f "$task_file" ]]; then
    python3 -c "import json; print(json.load(open('$task_file'))['expected_output']['extension'])" 2>/dev/null && return
  fi

  case "$category" in
    frontend) echo ".jsx" ;;
    api) echo ".js" ;;
    css) echo ".css" ;;
    typescript) echo ".ts" ;;
    bugfix) echo ".jsx" ;;
    *) echo ".js" ;;
  esac
}

resolve_target() {
  local task_id="$1"
  local category
  category="$(resolve_task "$task_id")"
  local ext
  ext="$(resolve_ext "$task_id")"

  if [[ "$category" == "bugfix" ]]; then
    echo "/benchmark/solutions/bugfix/${task_id}.fixed${ext}"
  else
    echo "/benchmark/solutions/${category}/${task_id}${ext}"
  fi
}

docker_run() {
  docker run --rm --shm-size=1gb -e CI=1 "$@"
}

cmd_build() {
  build_image
}

cmd_validate() {
  ensure_image
  ensure_results_dir
  echo "Running full benchmark validation inside Docker..."
  docker_run "$IMAGE"
}

cmd_task() {
  local task_id="${1:?Usage: ./docker-run.sh task <TASK_ID>}"
  ensure_image
  echo "Running task ${task_id} inside Docker..."
  docker_run -e BUGFIX_VARIANT="${BUGFIX_VARIANT:-fixed}" "$IMAGE" bash scripts/run-task.sh "$task_id"
}

cmd_eval() {
  local task_id="${1:?Usage: ./docker-run.sh eval <TASK_ID> <FILE>}"
  local llm_file="${2:?Usage: ./docker-run.sh eval <TASK_ID> <FILE>}"

  if [[ ! -f "$llm_file" ]]; then
    echo "Error: file not found: $llm_file"
    exit 1
  fi

  local category
  category="$(resolve_task "$task_id")"
  if [[ "$category" == "unknown" ]]; then
    echo "Error: unsupported task id: $task_id"
    exit 1
  fi

  ensure_image

  local target
  target="$(resolve_target "$task_id")"

  echo "Evaluating ${llm_file} against ${task_id}..."
  docker_run \
    -v "$(realpath "$llm_file"):/tmp/llm-output:ro" \
    "$IMAGE" \
    bash -lc "cp /tmp/llm-output '$target' && bash scripts/run-task.sh '$task_id'"
}

cmd_eval_all() {
  local dir="${1:?Usage: ./docker-run.sh eval-all <DIR>}"

  if [[ ! -d "$dir" ]]; then
    echo "Error: directory not found: $dir"
    exit 1
  fi

  echo "task_id,pass,total,status" > "$dir/results.csv"

  local file
  for file in "$dir"/*; do
    [[ -f "$file" ]] || continue

    local basename
    basename="$(basename "$file")"
    local task_id
    task_id="$(echo "$basename" | sed -E 's/\.(fixed\.)?(jsx|js|ts|tsx|css|html)$//')"

    local category
    category="$(resolve_task "$task_id")"
    [[ "$category" == "unknown" ]] && continue

    echo "Evaluating ${task_id}..."
    local output
    output="$(cmd_eval "$task_id" "$file" 2>&1)" || true

    local tests_passed
    tests_passed="$(echo "$output" | grep -oE '[0-9]+ passed' | tail -1 | cut -d' ' -f1 || echo 0)"
    local tests_total
    tests_total="$(echo "$output" | grep -oE '[0-9]+ total' | tail -1 | cut -d' ' -f1 || echo 0)"
    local status="FAIL"

    if [[ "$output" != *"FAIL"* ]] && [[ "$output" != *"Error:"* ]]; then
      status="PASS"
    fi

    echo "${task_id},${tests_passed:-0},${tests_total:-0},${status}" >> "$dir/results.csv"
    echo "  -> ${status} (${tests_passed:-0}/${tests_total:-0})"
  done

  echo "Results saved to $dir/results.csv"
}

case "${1:-help}" in
  build)
    cmd_build
    ;;
  validate)
    cmd_validate
    ;;
  task)
    shift
    cmd_task "$@"
    ;;
  eval)
    shift
    cmd_eval "$@"
    ;;
  eval-all)
    shift
    cmd_eval_all "$@"
    ;;
  *)
    echo "LLM Web Benchmark Docker Runner"
    echo
    echo "Usage:"
    echo "  ./docker-run.sh build"
    echo "  ./docker-run.sh validate"
    echo "  ./docker-run.sh task <TASK_ID>"
    echo "  ./docker-run.sh eval <TASK_ID> <FILE>"
    echo "  ./docker-run.sh eval-all <DIR>"
    ;;
esac
