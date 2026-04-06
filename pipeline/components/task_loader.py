from __future__ import annotations

import json
from pathlib import Path
from typing import List

from config import BENCHMARK_DIR
from models.task import Task, TaskExpectedOutput


class TaskLoader:
    def __init__(self, benchmark_dir: Path = BENCHMARK_DIR):
        self.benchmark_dir = benchmark_dir
        self.tasks_dir = benchmark_dir / "tasks"

    def load_task(self, task_id: str) -> Task:
        category = self._category_from_id(task_id)
        filepath = self.tasks_dir / category / f"{task_id}.json"
        if not filepath.exists():
            raise FileNotFoundError(f"Task file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return self._parse_task(data)

    def load_category(self, category: str) -> List[Task]:
        category_dir = self.tasks_dir / category
        if not category_dir.exists():
            raise FileNotFoundError(f"Task category not found: {category_dir}")

        tasks: List[Task] = []
        for filepath in sorted(category_dir.glob("*.json")):
            with open(filepath, "r", encoding="utf-8") as handle:
                tasks.append(self._parse_task(json.load(handle)))
        return tasks

    def load_all(self) -> List[Task]:
        tasks: List[Task] = []
        for category in ["frontend", "api", "css", "typescript", "bugfix"]:
            tasks.extend(self.load_category(category))
        return tasks

    def _category_from_id(self, task_id: str) -> str:
        prefix_map = {
            "FE": "frontend",
            "API": "api",
            "CSS": "css",
            "TS": "typescript",
            "BF": "bugfix",
        }
        prefix = task_id.split("-")[0]
        if prefix not in prefix_map:
            raise ValueError(f"Unknown task prefix: {prefix}")
        return prefix_map[prefix]

    def _parse_task(self, data: dict) -> Task:
        expected_output = data.get("expected_output", {})
        return Task(
            task_id=data["task_id"],
            category=data["category"],
            difficulty=data["difficulty"],
            title=data["title"],
            prompt=data["prompt"],
            constraints=data.get("constraints", []),
            expected_output=TaskExpectedOutput(
                type=expected_output.get("type", "single_file"),
                extension=expected_output.get("extension", ".js"),
                export_type=expected_output.get("export_type"),
            ),
            test_file=data["test_file"],
            test_count=data["test_count"],
            tags=data.get("tags", []),
            reference_solution=data["reference_solution"],
            buggy_file=data.get("buggy_file"),
            bug_description=data.get("bug_description"),
        )
