from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TaskExpectedOutput:
    type: str
    extension: str
    export_type: Optional[str]


@dataclass
class Task:
    task_id: str
    category: str
    difficulty: str
    title: str
    prompt: str
    constraints: List[str] = field(default_factory=list)
    expected_output: TaskExpectedOutput = field(
        default_factory=lambda: TaskExpectedOutput(
            type="single_file", extension=".js", export_type=None
        )
    )
    test_file: str = ""
    test_count: int = 0
    tags: List[str] = field(default_factory=list)
    reference_solution: str = ""
    buggy_file: Optional[str] = None
    bug_description: Optional[str] = None

    @property
    def formatted_prompt(self) -> str:
        return self.prompt
