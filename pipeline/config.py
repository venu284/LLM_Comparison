from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - optional until requirements are installed
    yaml = None

PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent
BENCHMARK_DIR = REPO_ROOT / "benchmark"
DEFAULT_CONFIG_PATH = PIPELINE_DIR / "config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_id: str
    provider: str
    api_key_env: str


MODELS: List[ModelConfig] = [
    ModelConfig(
        name="DeepSeek-R1-0528",
        model_id="deepseek/deepseek-reasoner",
        provider="deepseek",
        api_key_env="DEEPSEEK_API_KEY",
    ),
    ModelConfig(
        name="Qwen3-Coder",
        model_id="dashscope/qwen3-coder",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
    ),
    ModelConfig(
        name="GLM-4",
        model_id="zhipuai/glm-4",
        provider="zhipuai",
        api_key_env="GLM_API_KEY",
    ),
    ModelConfig(
        name="Gemini-2.5-Flash",
        model_id="gemini/gemini-2.5-flash",
        provider="google",
        api_key_env="GOOGLE_API_KEY",
    ),
    ModelConfig(
        name="Qwen2.5-Coder-32B",
        model_id="dashscope/qwen2.5-coder-32b-instruct",
        provider="dashscope",
        api_key_env="DASHSCOPE_API_KEY",
    ),
]

_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _default_config() -> Dict[str, Any]:
    return {
        "experiment": {
            "name": "Web Dev LLM Benchmark v1.0",
            "runs_per_task": 3,
            "temperature": 0.2,
            "max_tokens": 4096,
            "warmup_requests": 3,
            "test_timeout_seconds": 60,
        },
        "paths": {
            "benchmark_dir": "../benchmark",
            "temp_dir": "temp",
            "logs_dir": "logs",
            "exports_dir": "exports",
        },
        "database": {"url": os.getenv("DATABASE_URL", "")},
        "cost_rates": {
            "DeepSeek-R1-0528": {
                "input": 0.55,
                "output": 2.19,
                "source": "https://platform.deepseek.com/api-docs/pricing",
                "accessed": "2026-04-06",
            },
            "Qwen3-Coder": {
                "input": 0.50,
                "output": 2.00,
                "source": "https://dashscope.console.aliyun.com",
                "accessed": "2026-04-06",
            },
            "GLM-4": {
                "input": 0.40,
                "output": 1.60,
                "source": "https://open.bigmodel.cn/pricing",
                "accessed": "2026-04-06",
            },
            "Gemini-2.5-Flash": {
                "input": 0.15,
                "output": 0.60,
                "source": "https://ai.google.dev/pricing",
                "accessed": "2026-04-06",
            },
            "Qwen2.5-Coder-32B": {
                "input": 0.30,
                "output": 1.20,
                "source": "https://dashscope.console.aliyun.com",
                "accessed": "2026-04-06",
            },
        },
    }


def _merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return _ENV_PATTERN.sub(lambda match: os.getenv(match.group(1), match.group(0)), value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


@lru_cache(maxsize=1)
def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    raw: Dict[str, Any] = {}
    if path.exists() and yaml is not None:
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
    merged = _merge_dicts(_default_config(), raw)
    return _expand_env(merged)


@lru_cache(maxsize=1)
def get_paths() -> Dict[str, Path]:
    configured = load_config()["paths"]
    resolved: Dict[str, Path] = {}
    for key, raw_value in configured.items():
        path = Path(raw_value)
        if not path.is_absolute():
            path = (PIPELINE_DIR / path).resolve()
        resolved[key] = path
    return resolved


def ensure_runtime_dirs() -> Dict[str, Path]:
    paths = get_paths()
    for key in ("temp_dir", "logs_dir", "exports_dir"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def get_database_url() -> Optional[str]:
    url = load_config().get("database", {}).get("url") or os.getenv("DATABASE_URL")
    if isinstance(url, str):
        url = url.strip()
        if not url or _ENV_PATTERN.search(url):
            return None
    return url or None


def get_experiment_settings() -> Dict[str, Any]:
    return load_config()["experiment"]


def get_cost_rates() -> Dict[str, Dict[str, Any]]:
    return load_config()["cost_rates"]
