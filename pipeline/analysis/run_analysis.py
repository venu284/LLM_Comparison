#!/usr/bin/env python3
"""Phase 6 orchestrator.

Runs every analysis, writes machine-readable JSON plus charts, and prints a
summary. The report quotes these artifacts rather than hand-copied numbers.

    cd pipeline && python analysis/run_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

from analysis.data import load_experiment_data  # noqa: E402
from analysis.recommender import (  # noqa: E402
    evaluate_classifier,
    load_benchmark_prompts,
)
from analysis.stats import build_stats_report  # noqa: E402
from analysis.validate import (  # noqa: E402
    bootstrap_difference,
    leave_one_task_out,
    weight_sensitivity,
)
from config import BENCHMARK_DIR, ensure_runtime_dirs  # noqa: E402

logger = logging.getLogger("phase6")

PALETTE = {
    "GPT-OSS-120B": "#4C72B0",
    "Llama-3.3-70B": "#DD8452",
    "Llama-4-Scout": "#55A868",
    "Llama-3.1-8B": "#C44E52",
    "Qwen3-32B": "#8172B3",
}


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, pd.DataFrame):
        return json.loads(value.to_json(orient="split"))
    if isinstance(value, pd.Series):
        return json.loads(value.to_json())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def chart_strategy_comparison(scores: Dict[str, float], output: Path) -> None:
    order = ["random", "recommender", "best_single_model", "oracle"]
    labels = ["Random", "Recommender", "Best single\n(GPT-OSS-120B)", "Oracle\n(upper bound)"]
    values = [scores[key] * 100 for key in order]
    colors = ["#BBBBBB", "#C44E52", "#4C72B0", "#55A868"]

    figure, axes = plt.subplots(figsize=(8, 5))
    bars = axes.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    for bar, value in zip(bars, values):
        axes.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
        )

    axes.axhline(scores["oracle"] * 100, linestyle="--", color="#55A868", linewidth=1, alpha=0.7)
    axes.set_ylabel("Pass rate (%)")
    axes.set_ylim(0, max(values) * 1.25)
    axes.set_title(
        "Routing strategies under leave-one-task-out CV\n"
        "Headroom between best-single and oracle is only "
        f"{(scores['oracle'] - scores['best_single_model']) * 100:.1f} points",
        fontsize=11,
    )
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def chart_mcnemar(stats_report, output: Path) -> None:
    if not stats_report.mcnemar:
        return

    items = sorted(stats_report.mcnemar, key=lambda item: item.p_value_holm)
    labels = [f"{item.model_a}\nvs {item.model_b}" for item in items]
    values = [item.p_value_holm for item in items]
    colors = ["#C44E52" if item.significant_holm else "#BBBBBB" for item in items]

    figure, axes = plt.subplots(figsize=(11, 5))
    axes.bar(labels, values, color=colors, edgecolor="black", linewidth=0.6)
    axes.axhline(0.05, linestyle="--", color="black", linewidth=1)
    axes.text(len(labels) - 0.5, 0.06, "alpha = 0.05", ha="right", fontsize=9)
    axes.set_ylabel("Holm-adjusted p-value")
    axes.set_title(
        "Pairwise McNemar tests. Grey pairs are statistically indistinguishable.",
        fontsize=11,
    )
    axes.tick_params(axis="x", labelsize=7.5)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def chart_sensitivity(full_rates: pd.Series, filtered: pd.DataFrame, output: Path) -> None:
    models = list(full_rates.index)
    positions = np.arange(len(models))
    width = 0.38

    figure, axes = plt.subplots(figsize=(9, 5))
    axes.bar(
        positions - width / 2,
        [full_rates[model] * 100 for model in models],
        width,
        label="All runs (published treatment)",
        color="#4C72B0",
        edgecolor="black",
        linewidth=0.5,
    )
    axes.bar(
        positions + width / 2,
        [filtered.loc[model, "rate"] * 100 for model in models],
        width,
        label="Zero-test runs dropped",
        color="#DD8452",
        edgecolor="black",
        linewidth=0.5,
    )

    for index, model in enumerate(models):
        axes.text(
            index + width / 2,
            filtered.loc[model, "rate"] * 100 + 1.5,
            f"n={int(filtered.loc[model, 'n'])}",
            ha="center",
            fontsize=8,
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(models, rotation=15, ha="right", fontsize=8)
    axes.set_ylabel("Pass rate (%)")
    axes.set_title(
        "Zero-test sensitivity. Dropping those runs leaves unequal denominators\n"
        "(Qwen3-32B retains 25 of 65 attempts), so the two bars are not directly comparable.",
        fontsize=10,
    )
    axes.legend(fontsize=9)
    axes.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6 analysis")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    parser.add_argument("--bootstrap", type=int, default=10000, help="Bootstrap iterations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_dotenv(PIPELINE_DIR / ".env")

    paths = ensure_runtime_dirs()
    analysis_dir = paths["exports_dir"] / "analysis"
    charts_dir = paths["exports_dir"] / "charts"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    charts_dir.mkdir(parents=True, exist_ok=True)

    data = load_experiment_data()
    prompts = load_benchmark_prompts(BENCHMARK_DIR)

    logger.info("Loaded %s runs from %s", len(data.runs), data.source)
    logger.info("Fully paired: %s", data.is_fully_paired())

    stats_report = build_stats_report(data)
    classifier = evaluate_classifier(prompts)
    validation = leave_one_task_out(data, prompts=prompts)
    bootstrap = bootstrap_difference(validation, iterations=args.bootstrap)
    sensitivity_grid = weight_sensitivity(data, prompts=prompts)

    filtered = load_experiment_data(exclude_zero_tests=True)
    filtered_stats = build_stats_report(filtered)
    filtered_summary = (
        filtered.runs.groupby("model_name")["pass_fail"]
        .agg(passes="sum", n="size")
        .assign(rate=lambda frame: frame["passes"] / frame["n"])
    )

    payload = {
        "source": data.source,
        "n_runs": int(len(data.runs)),
        "zero_test_rows": int(data.zero_test_rows),
        "overall_pass_rates": _jsonable(data.overall_pass_rates()),
        "stats": {
            "overall_chi2": _jsonable(stats_report.overall_chi2),
            "per_category_chi2": _jsonable(stats_report.per_category_chi2),
            "difficulty_chi2": _jsonable(stats_report.difficulty_chi2),
            "spearman_params": _jsonable(stats_report.spearman_params),
            "mcnemar": _jsonable(stats_report.mcnemar),
            "notes": stats_report.notes,
        },
        "classifier": {
            "accuracy": classifier["accuracy"],
            "n": classifier["n"],
            "per_category_accuracy": classifier["per_category_accuracy"],
            "misclassified": classifier["misclassified"],
        },
        "validation": {
            "strategy_scores": _jsonable(validation.strategy_scores),
            "h4_improvement_pct": validation.h4_improvement_pct,
            "h4_supported": validation.h4_supported,
            "weights": validation.weights,
            "per_category": _jsonable(validation.per_category),
            "bootstrap": _jsonable(bootstrap),
            "max_possible_improvement_pct": float(
                (validation.strategy_scores["oracle"] - validation.strategy_scores["best_single_model"])
                / validation.strategy_scores["best_single_model"]
                * 100
            ),
        },
        "weight_sensitivity": _jsonable(sensitivity_grid),
        "zero_test_sensitivity": {
            "filtered_n": int(len(filtered.runs)),
            "filtered_fully_paired": filtered.is_fully_paired(),
            "filtered_attempts": _jsonable(filtered.attempt_counts()),
            "filtered_rates": _jsonable(filtered_summary),
            "filtered_overall_chi2": _jsonable(filtered_stats.overall_chi2),
            "notes": filtered_stats.notes,
        },
    }

    output_path = analysis_dir / "phase6_results.json"
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Wrote %s", output_path)

    if not args.no_charts:
        chart_strategy_comparison(validation.strategy_scores, charts_dir / "phase6_01_strategies.png")
        chart_mcnemar(stats_report, charts_dir / "phase6_02_mcnemar.png")
        chart_sensitivity(
            data.overall_pass_rates(), filtered_summary, charts_dir / "phase6_03_zero_test_sensitivity.png"
        )
        logger.info("Wrote 3 charts to %s", charts_dir)

    scores = validation.strategy_scores
    logger.info("\n%s", "=" * 62)
    logger.info("PHASE 6 SUMMARY")
    logger.info("%s", "=" * 62)
    logger.info("Classifier accuracy      : %.1f%% on 65 benchmark prompts", classifier["accuracy"] * 100)
    logger.info("Oracle (upper bound)     : %.1f%%", scores["oracle"] * 100)
    logger.info("Best single model        : %.1f%%", scores["best_single_model"] * 100)
    logger.info("Recommender              : %.1f%%", scores["recommender"] * 100)
    logger.info("Random                   : %.1f%%", scores["random"] * 100)
    logger.info(
        "H4 (>15%% improvement)    : %s (%+.1f%%; ceiling for any router is %+.1f%%)",
        "SUPPORTED" if validation.h4_supported else "NOT SUPPORTED",
        validation.h4_improvement_pct,
        payload["validation"]["max_possible_improvement_pct"],
    )


if __name__ == "__main__":
    main()
