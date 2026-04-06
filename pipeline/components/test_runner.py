from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

from config import BENCHMARK_DIR, get_experiment_settings

logger = logging.getLogger(__name__)


@dataclass
class TestAssertion:
    name: str
    passed: bool
    error_message: Optional[str] = None


@dataclass
class TestResult:
    task_id: str
    model_name: str
    overall_pass: bool
    tests_passed: int
    tests_total: int
    assertions: List[TestAssertion] = field(default_factory=list)
    execution_time_ms: int = 0
    runner_error: Optional[str] = None
    eslint_warnings: int = 0
    ts_any_count: int = 0
    compiler_errors: int = 0


class TestRunner:
    def __init__(self, benchmark_dir: Path = BENCHMARK_DIR):
        self.benchmark_dir = benchmark_dir
        self.docker_script = benchmark_dir / "docker-run.sh"
        self.timeout_seconds = int(get_experiment_settings().get("test_timeout_seconds", 60))

    def run(
        self,
        task_id: str,
        category: str,
        code_file_path: str,
        expected_extension: str,
    ) -> TestResult:
        try:
            result = subprocess.run(
                ["bash", str(self.docker_script), "eval", task_id, code_file_path],
                capture_output=True,
                text=True,
                cwd=str(self.benchmark_dir),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return TestResult(
                task_id=task_id,
                model_name="unknown",
                overall_pass=False,
                tests_passed=0,
                tests_total=0,
                runner_error=f"Docker test execution timed out ({self.timeout_seconds}s)",
            )
        except Exception as exc:  # pragma: no cover - subprocess failures
            return TestResult(
                task_id=task_id,
                model_name="unknown",
                overall_pass=False,
                tests_passed=0,
                tests_total=0,
                runner_error=str(exc),
            )

        return self._parse_output(
            task_id=task_id,
            category=category,
            code_file_path=code_file_path,
            stdout=result.stdout,
            stderr=result.stderr,
            returncode=result.returncode,
        )

    def _parse_output(
        self,
        task_id: str,
        category: str,
        code_file_path: str,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> TestResult:
        combined = (stdout or "") + ("\n" + stderr if stderr else "")
        tests_passed, tests_failed, tests_total = self._parse_test_counts(combined)
        execution_time_ms = self._parse_execution_time_ms(combined)
        compiler_errors = len(re.findall(r"\berror TS\d+:", combined))
        ts_any_count = self._count_any_usage(category, code_file_path)
        assertions = self._parse_assertions(combined, tests_passed, tests_failed, tests_total)

        runner_error = None
        if returncode != 0 and tests_total == 0:
            runner_error = (stderr or stdout).strip() or "Docker runner failed"

        return TestResult(
            task_id=task_id,
            model_name="unknown",
            overall_pass=(returncode == 0 and tests_failed == 0 and tests_total > 0),
            tests_passed=tests_passed,
            tests_total=tests_total,
            assertions=assertions,
            execution_time_ms=execution_time_ms,
            runner_error=runner_error,
            eslint_warnings=0,
            ts_any_count=ts_any_count,
            compiler_errors=compiler_errors,
        )

    def _parse_test_counts(self, output: str) -> Tuple[int, int, int]:
        tests_line = re.search(r"Tests:\s+([^\n]+)", output)
        if tests_line:
            line = tests_line.group(1)
            passed = int(re.search(r"(\d+)\s+passed", line).group(1)) if re.search(r"(\d+)\s+passed", line) else 0
            failed = int(re.search(r"(\d+)\s+failed", line).group(1)) if re.search(r"(\d+)\s+failed", line) else 0
            total = int(re.search(r"(\d+)\s+total", line).group(1)) if re.search(r"(\d+)\s+total", line) else passed + failed
            return passed, failed, total

        passed_matches = re.findall(r"^\s*(\d+)\s+passed(?:\s+\(|$)", output, re.MULTILINE)
        failed_matches = re.findall(r"^\s*(\d+)\s+failed(?:\s+\(|$)", output, re.MULTILINE)
        total = 0
        if passed_matches or failed_matches:
            passed = int(passed_matches[-1]) if passed_matches else 0
            failed = int(failed_matches[-1]) if failed_matches else 0
            total = passed + failed
            return passed, failed, total

        return 0, 0, 0

    def _parse_execution_time_ms(self, output: str) -> int:
        match = re.search(r"Time:\s+([\d.]+)\s*s", output)
        if match:
            return int(float(match.group(1)) * 1000)

        matches = re.findall(r"\(([\d.]+)s\)", output)
        if matches:
            return int(float(matches[-1]) * 1000)

        return 0

    def _parse_assertions(
        self, output: str, tests_passed: int, tests_failed: int, tests_total: int
    ) -> List[TestAssertion]:
        assertions: List[TestAssertion] = []
        for line in output.splitlines():
            stripped = line.strip()
            pass_match = re.match(r"^[✓✔]\s+(.*)$", stripped)
            fail_match = re.match(r"^[✕x]\s+(.*)$", stripped)
            playwright_fail = re.match(r"^\d+\)\s+.*?›\s+(.*)$", stripped)

            if pass_match:
                assertions.append(TestAssertion(name=pass_match.group(1), passed=True))
            elif fail_match:
                assertions.append(TestAssertion(name=fail_match.group(1), passed=False))
            elif playwright_fail:
                assertions.append(TestAssertion(name=playwright_fail.group(1), passed=False))

        if len(assertions) >= tests_total > 0:
            return assertions[:tests_total]

        remaining_passed = max(0, tests_passed - sum(1 for item in assertions if item.passed))
        remaining_failed = max(0, tests_failed - sum(1 for item in assertions if not item.passed))

        for index in range(remaining_passed):
            assertions.append(TestAssertion(name=f"passed_{index + 1}", passed=True))
        for index in range(remaining_failed):
            assertions.append(TestAssertion(name=f"failed_{index + 1}", passed=False))

        return assertions

    def _count_any_usage(self, category: str, code_file_path: str) -> int:
        if category != "typescript":
            return 0

        try:
            with open(code_file_path, "r", encoding="utf-8") as handle:
                code = handle.read()
        except OSError:
            return 0

        return len(re.findall(r"\bany\b", code))
