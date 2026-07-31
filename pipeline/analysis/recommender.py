"""Task-aware model recommendation (RQ3, H4).

Two stages, as specified in the Phase 2 framework:

1. Classify a free-text prompt into one of the five benchmark categories using
   rule-based keyword matching.
2. Score every model against empirical performance profiles and rank them.

The scoring formula is the one fixed in the framework:

    Score = w1*CategoryPassRate + w2*DifficultyPassRate
          + w3*(1/NormalizedLatency) + w4*(1/NormalizedTokens)

Normalization divides by the best (lowest) value across models, so the fastest
model scores exactly 1.0 on the latency term and everything else falls in (0, 1].
This makes the speed and efficiency terms unit-free: measuring latency in
seconds instead of milliseconds cannot change a ranking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from analysis.data import CATEGORIES, DIFFICULTIES, ExperimentData

DEFAULT_WEIGHTS = {"category": 0.5, "difficulty": 0.2, "latency": 0.15, "tokens": 0.15}

# Bug fixing is an *intent*, not a language. A bug-fix prompt embeds the broken
# React/Express/TypeScript source, so it necessarily contains more language
# keywords than repair keywords. Counting hits across all five categories at
# once therefore classifies almost every bug-fix task as whatever language it
# happens to be written in. Repair intent is tested first and decides on its
# own; only if it is absent do the language categories compete.
BUGFIX_INTENT_MARKERS: Sequence[str] = (
    # Benchmark template phrasing.
    "fix the bug", "bug description", "buggy", "the following code",
    "existing code",
    # Generic repair language a developer would actually type. Kept broad on
    # purpose: these are how bugs get described outside this benchmark.
    "is broken", "does not work", "doesn't work", "not working",
    "isn't working", "debug this", "refactor", "fix this", "fix it", "fix my",
    "what's wrong", "whats wrong", "why does", "why is", "incorrect behavior",
    "memory leak", "race condition", "infinite loop", "never updates",
    "never re-renders", "fails to", "should instead",
    "wrong output", "stopped working", "returns duplicate",
)
# Deliberately excluded after a false-positive audit against the 65 benchmark
# prompts: bare "debug" (fires on TS-003, which merely mentions debugging) and
# "instead of" (fires on FE-010, a design prompt). Both cost precision on
# non-bug-fix tasks without catching any bug-fix task the markers above miss.

LANGUAGE_KEYWORDS: List[Tuple[str, Sequence[str]]] = [
    (
        "css",
        (
            "css", "stylesheet", "flexbox", "flex-", "grid", "grid-template",
            "layout", "responsive", "breakpoint", "media query", "center the",
            "sticky", "z-index", "styling", "margin", "padding", "viewport",
            "column", "align", "spacing", "mobile", "tablet",
        ),
    ),
    (
        "typescript",
        (
            "typescript", "interface", "type definition", "generic", "tsc",
            "strongly typed", "type-safe", "discriminated union", "enum",
            "declare a type", "utility type", ".ts file",
        ),
    ),
    (
        "api",
        (
            "express", "endpoint", "rest api", "http", "status code", "middleware",
            "route", "jwt", "req, res", "request body", "pagination", "server",
            "get /", "post /", "put /", "delete /", "supertest",
        ),
    ),
    (
        "frontend",
        (
            "react", "component", "jsx", "usestate", "useeffect", "hook", "render",
            "props", "data-testid", "button", "form", "modal", "navbar", "dropdown",
            "controlled input",
        ),
    ),
]


@dataclass
class ModelProfile:
    """Empirical performance profile, built from training tasks only."""

    model_name: str
    category_pass_rate: Dict[str, float]
    difficulty_pass_rate: Dict[str, float]
    mean_latency_ms: float
    mean_tokens_output: float
    overall_pass_rate: float

    def rate_for_category(self, category: str) -> float:
        return self.category_pass_rate.get(category, self.overall_pass_rate)

    def rate_for_difficulty(self, difficulty: str) -> float:
        return self.difficulty_pass_rate.get(difficulty, self.overall_pass_rate)


@dataclass
class Recommendation:
    model_name: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


def classify_prompt(prompt: str) -> str:
    """Map free text to a benchmark category.

    Stage 1 tests repair intent, which wins outright when present. Stage 2
    picks the language category with the most keyword hits, breaking ties by
    the priority order in LANGUAGE_KEYWORDS. Falls back to 'frontend', the
    largest generative category, when nothing matches.
    """
    text = prompt.lower()

    if any(marker in text for marker in BUGFIX_INTENT_MARKERS):
        return "bugfix"

    best_category = "frontend"
    best_hits = 0
    for category, keywords in LANGUAGE_KEYWORDS:
        hits = sum(1 for keyword in keywords if keyword in text)
        if hits > best_hits:
            best_category, best_hits = category, hits

    return best_category


def build_profiles(runs: pd.DataFrame) -> Dict[str, ModelProfile]:
    """Build one profile per model from the supplied runs.

    Callers doing cross-validation must pass training rows only. This function
    deliberately has no knowledge of held-out tasks.
    """
    profiles: Dict[str, ModelProfile] = {}

    for model_name, group in runs.groupby("model_name"):
        category_rates = group.groupby("category")["pass_fail"].mean().to_dict()
        difficulty_rates = group.groupby("difficulty")["pass_fail"].mean().to_dict()
        profiles[str(model_name)] = ModelProfile(
            model_name=str(model_name),
            category_pass_rate={key: float(value) for key, value in category_rates.items()},
            difficulty_pass_rate={key: float(value) for key, value in difficulty_rates.items()},
            mean_latency_ms=float(group["latency_total_ms"].mean()),
            mean_tokens_output=float(group["tokens_output"].mean()),
            overall_pass_rate=float(group["pass_fail"].mean()),
        )

    return profiles


def _inverse_normalized(values: Dict[str, float]) -> Dict[str, float]:
    """Map raw cost values to (0, 1], where 1.0 is the best (lowest) cost."""
    positive = [value for value in values.values() if value > 0]
    if not positive:
        return {key: 1.0 for key in values}

    best = min(positive)
    return {
        key: (best / value if value > 0 else 1.0)
        for key, value in values.items()
    }


class Recommender:
    def __init__(
        self,
        profiles: Dict[str, ModelProfile],
        weights: Optional[Dict[str, float]] = None,
    ):
        self.profiles = profiles
        self.weights = dict(weights or DEFAULT_WEIGHTS)

        self._latency_scores = _inverse_normalized(
            {name: profile.mean_latency_ms for name, profile in profiles.items()}
        )
        self._token_scores = _inverse_normalized(
            {name: profile.mean_tokens_output for name, profile in profiles.items()}
        )

    def score(self, model_name: str, category: str, difficulty: str) -> Recommendation:
        profile = self.profiles[model_name]
        components = {
            "category": self.weights["category"] * profile.rate_for_category(category),
            "difficulty": self.weights["difficulty"] * profile.rate_for_difficulty(difficulty),
            "latency": self.weights["latency"] * self._latency_scores[model_name],
            "tokens": self.weights["tokens"] * self._token_scores[model_name],
        }
        return Recommendation(model_name, float(sum(components.values())), components)

    def rank(self, category: str, difficulty: str) -> List[Recommendation]:
        ranked = [self.score(name, category, difficulty) for name in self.profiles]
        # Deterministic ordering: score descending, then name, so ties never
        # depend on dict insertion order.
        return sorted(ranked, key=lambda item: (-item.score, item.model_name))

    def recommend(self, category: str, difficulty: str) -> str:
        return self.rank(category, difficulty)[0].model_name

    def recommend_for_prompt(self, prompt: str, difficulty: str = "medium") -> Tuple[str, str]:
        category = classify_prompt(prompt)
        return self.recommend(category, difficulty), category


def load_benchmark_prompts(benchmark_dir: Path) -> pd.DataFrame:
    """Load every task's prompt with its true category, for classifier scoring."""
    records = []
    for category in CATEGORIES:
        directory = benchmark_dir / "tasks" / category
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            records.append(
                {
                    "task_id": data["task_id"],
                    "true_category": data["category"],
                    "difficulty": data["difficulty"],
                    "prompt": data["prompt"],
                }
            )
    return pd.DataFrame(records)


def evaluate_classifier(prompts: pd.DataFrame) -> Dict[str, object]:
    """Accuracy of the keyword classifier against the 65 ground-truth labels."""
    predictions = prompts["prompt"].apply(classify_prompt)
    correct = predictions == prompts["true_category"]

    confusion = pd.crosstab(
        prompts["true_category"], predictions, rownames=["true"], colnames=["predicted"]
    ).reindex(index=CATEGORIES, columns=CATEGORIES, fill_value=0)

    per_category = (
        pd.DataFrame({"true": prompts["true_category"], "correct": correct})
        .groupby("true")["correct"]
        .mean()
        .to_dict()
    )

    return {
        "accuracy": float(correct.mean()),
        "n": int(len(prompts)),
        "per_category_accuracy": {key: float(value) for key, value in per_category.items()},
        "confusion": confusion,
        "misclassified": prompts.loc[~correct, ["task_id", "true_category"]]
        .assign(predicted=predictions[~correct])
        .to_dict("records"),
    }
