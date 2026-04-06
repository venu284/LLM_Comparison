#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
OUTPUT_PATH = Path(__file__).resolve().parent / "seed_tasks.sql"


def _escape_sql(value: str) -> str:
    return value.replace("'", "''")


def _pg_array(items: Iterable[str]) -> str:
    escaped = []
    for item in items:
        escaped_item = item.replace("\\", "\\\\").replace('"', '\\"')
        escaped.append(f'"{escaped_item}"')
    return "{" + ",".join(escaped) + "}"


def generate_seed_sql(output_path: Optional[Path] = None) -> Path:
    lines: List[str] = []
    for category in ["frontend", "api", "css", "typescript", "bugfix"]:
        task_dir = BENCHMARK_DIR / "tasks" / category
        for task_file in sorted(task_dir.glob("*.json")):
            with open(task_file, "r", encoding="utf-8") as handle:
                data = json.load(handle)

            title = _escape_sql(data["title"])
            tags_literal = _pg_array(data.get("tags", []))
            lines.append(
                "INSERT INTO tasks (task_id, category, difficulty, title, test_count, tags) "
                f"VALUES ('{data['task_id']}', '{data['category']}', '{data['difficulty']}', "
                f"'{title}', {int(data['test_count'])}, '{tags_literal}') "
                "ON CONFLICT (task_id) DO NOTHING;"
            )

    destination = output_path or OUTPUT_PATH
    with open(destination, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Generated {len(lines)} task inserts at {destination}")
    return destination


if __name__ == "__main__":
    generate_seed_sql()
