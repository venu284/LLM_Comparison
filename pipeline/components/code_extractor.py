from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from config import ensure_runtime_dirs

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    success: bool
    code: str
    file_path: Optional[str]
    method: str
    failure_reason: Optional[str]


class CodeExtractor:
    FENCE_PATTERN = re.compile(
        r"```(?:jsx?|tsx?|typescript|javascript|css|html|react)?\s*\n(.*?)\n```",
        re.DOTALL | re.IGNORECASE,
    )
    GENERIC_FENCE = re.compile(r"```\w*\s*\n(.*?)\n```", re.DOTALL)

    def __init__(self, temp_dir: Optional[Path] = None):
        paths = ensure_runtime_dirs()
        self.temp_dir = temp_dir or paths["temp_dir"]
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        raw_response: str,
        task_id: str,
        model_name: str,
        expected_extension: str = ".js",
    ) -> ExtractionResult:
        if not raw_response or not raw_response.strip():
            return ExtractionResult(
                success=False,
                code="",
                file_path=None,
                method="none",
                failure_reason="Empty response",
            )

        code, method = self._extract_fenced(raw_response)
        if not code:
            code, method = self._extract_raw(raw_response, expected_extension)
        if not code:
            code, method = self._extract_stripped(raw_response)

        if not code or not code.strip():
            return ExtractionResult(
                success=False,
                code="",
                file_path=None,
                method=method,
                failure_reason="No extractable code found",
            )

        validation_error = self._validate(code, expected_extension)
        if validation_error:
            logger.warning(
                "Extracted code has issues for %s/%s: %s",
                task_id,
                model_name,
                validation_error,
            )

        safe_model = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name)
        filepath = self.temp_dir / f"{task_id}_{safe_model}{expected_extension}"
        with open(filepath, "w", encoding="utf-8") as handle:
            handle.write(code)

        return ExtractionResult(
            success=True,
            code=code,
            file_path=str(filepath),
            method=method,
            failure_reason=None,
        )

    def _extract_fenced(self, response: str) -> Tuple[Optional[str], str]:
        matches = self.FENCE_PATTERN.findall(response)
        if matches:
            return max(matches, key=len).strip(), "fenced_specific"

        matches = self.GENERIC_FENCE.findall(response)
        if matches:
            return max(matches, key=len).strip(), "fenced_generic"

        return None, "no_fences"

    def _extract_raw(self, response: str, extension: str) -> Tuple[Optional[str], str]:
        indicators = {
            ".jsx": ["import ", "export ", "function ", "const ", "return ("],
            ".js": ["const ", "module.exports", "require(", "function ", "app."],
            ".ts": ["interface ", "type ", "export ", "import "],
            ".tsx": ["interface ", "type ", "export ", "import ", "return ("],
            ".css": ["{", "}", "display:", "margin:", "padding:"],
            ".html": ["<!DOCTYPE", "<html", "<div", "<style", "<body"],
        }

        checks = indicators.get(extension, ["function ", "const ", "import "])
        lines = [line for line in response.strip().splitlines() if line.strip()]
        if not lines:
            return None, "empty_raw_response"

        code_like = sum(1 for line in lines if any(token in line for token in checks))
        if code_like / len(lines) >= 0.3:
            return response.strip(), "raw_response"
        return None, "not_code_like"

    def _extract_stripped(self, response: str) -> Tuple[Optional[str], str]:
        code_lines = []
        for line in response.splitlines():
            stripped = line.strip()
            if stripped.startswith("Here") or stripped.startswith("This") or stripped.startswith("Explanation:"):
                if not any(token in stripped for token in ["=", "(", "{", ";", "<", ":"]):
                    continue
            if stripped.startswith("Note:"):
                continue
            code_lines.append(line)

        code = "\n".join(code_lines).strip()
        if code:
            return code, "stripped"
        return None, "strip_failed"

    def _validate(self, code: str, extension: str) -> Optional[str]:
        if extension in {".jsx", ".js", ".ts", ".tsx"}:
            opens = code.count("{") + code.count("(") + code.count("[")
            closes = code.count("}") + code.count(")") + code.count("]")
            if abs(opens - closes) > 2:
                return f"Unbalanced brackets: {opens} opens, {closes} closes"

        if extension == ".css" and "{" not in code:
            return "No CSS rule blocks found"

        if code.rstrip().endswith("...") or code.rstrip().endswith("// ..."):
            return "Response appears truncated"

        return None

    def cleanup(self) -> None:
        for filepath in self.temp_dir.glob("*"):
            if filepath.name != ".gitkeep":
                filepath.unlink()
