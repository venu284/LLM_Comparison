#!/usr/bin/env python3
"""
LLM Benchmark Results Visualizer
=================================
Generates publication-ready charts from benchmark results stored in Neon/PostgreSQL.

Usage:
    python scripts/visualize_results.py

Output:
    exports/charts/  — all PNG charts saved here
    exports/results_summary.csv — tabular summary
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent.parent
if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))

load_dotenv(PIPELINE_DIR / ".env")

from components.results_storage import ResultsStorage

# ── Config ──────────────────────────────────────────────

CHARTS_DIR = PIPELINE_DIR / "exports" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Color palette — distinct, colorblind-friendly
MODEL_COLORS = {
    "GPT-OSS-120B": "#2563EB",
    "Llama-3.3-70B": "#16A34A",
    "Llama-4-Scout": "#DC2626",
    "Llama-3.1-8B": "#F59E0B",
    "Qwen3-32B": "#8B5CF6",
}

CATEGORY_ORDER = ["frontend", "api", "css", "typescript", "bugfix"]
DIFFICULTY_ORDER = ["easy", "medium", "hard"]

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.family"] = "sans-serif"


# ── Data Loading ────────────────────────────────────────

def load_data() -> pd.DataFrame:
    """Load all run results from database into a DataFrame."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set in .env")
        sys.exit(1)

    storage = ResultsStorage(db_url)
    storage.connect()

    csv_path = PIPELINE_DIR / "exports" / "results_full.csv"
    storage.export_to_csv(str(csv_path))
    storage.close()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} results from database")
    print(f"Models: {df['model_name'].nunique()} | Tasks: {df['task_id'].nunique()}")
    print(f"Categories: {sorted(df['category'].unique())}")
    return df


# ── Chart 1: Overall Pass Rate per Model ───────────────

def chart_overall_pass_rate(df: pd.DataFrame):
    """Bar chart: overall pass rate per model, sorted descending."""
    summary = (
        df.groupby("model_name")["pass_fail"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index()
    )
    summary.columns = ["Model", "Pass Rate (%)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [MODEL_COLORS.get(m, "#999") for m in summary["Model"]]
    bars = ax.bar(summary["Model"], summary["Pass Rate (%)"], color=colors, width=0.6, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, summary["Pass Rate (%)"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Overall Pass Rate by Model", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(summary["Pass Rate (%)"]) + 12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "01_overall_pass_rate.png")
    plt.close(fig)
    print("  ✓ 01_overall_pass_rate.png")


# ── Chart 2: Pass Rate by Category (grouped bar) ──────

def chart_pass_rate_by_category(df: pd.DataFrame):
    """Grouped bar chart: pass rate per model per category."""
    pivot = (
        df.groupby(["category", "model_name"])["pass_fail"]
        .mean()
        .mul(100)
        .reset_index()
    )
    pivot.columns = ["Category", "Model", "Pass Rate"]

    # Order categories
    pivot["Category"] = pd.Categorical(pivot["Category"], categories=CATEGORY_ORDER, ordered=True)
    pivot = pivot.sort_values("Category")

    fig, ax = plt.subplots(figsize=(14, 6))
    models = sorted(df["model_name"].unique())
    n_models = len(models)
    x = np.arange(len(CATEGORY_ORDER))
    width = 0.15

    for i, model in enumerate(models):
        model_data = pivot[pivot["Model"] == model]
        vals = []
        for cat in CATEGORY_ORDER:
            row = model_data[model_data["Category"] == cat]
            vals.append(row["Pass Rate"].values[0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=MODEL_COLORS.get(model, "#999"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Category")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate by Category and Model", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([c.title() for c in CATEGORY_ORDER])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "02_pass_rate_by_category.png")
    plt.close(fig)
    print("  ✓ 02_pass_rate_by_category.png")


# ── Chart 3: Pass Rate by Difficulty ───────────────────

def chart_pass_rate_by_difficulty(df: pd.DataFrame):
    """Grouped bar chart: pass rate per model per difficulty level."""
    pivot = (
        df.groupby(["difficulty", "model_name"])["pass_fail"]
        .mean()
        .mul(100)
        .reset_index()
    )
    pivot.columns = ["Difficulty", "Model", "Pass Rate"]
    pivot["Difficulty"] = pd.Categorical(pivot["Difficulty"], categories=DIFFICULTY_ORDER, ordered=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    models = sorted(df["model_name"].unique())
    n_models = len(models)
    x = np.arange(len(DIFFICULTY_ORDER))
    width = 0.15

    for i, model in enumerate(models):
        model_data = pivot[pivot["Model"] == model]
        vals = []
        for diff in DIFFICULTY_ORDER:
            row = model_data[model_data["Difficulty"] == diff]
            vals.append(row["Pass Rate"].values[0] if len(row) > 0 else 0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=model,
               color=MODEL_COLORS.get(model, "#999"), edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Pass Rate by Difficulty Level", fontsize=14, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in DIFFICULTY_ORDER])
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "03_pass_rate_by_difficulty.png")
    plt.close(fig)
    print("  ✓ 03_pass_rate_by_difficulty.png")


# ── Chart 4: Latency Comparison (box plot) ─────────────

def chart_latency_comparison(df: pd.DataFrame):
    """Box plot: latency distribution per model."""
    df_valid = df[df["latency_total_ms"] > 0].copy()

    order = (
        df_valid.groupby("model_name")["latency_total_ms"]
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = [MODEL_COLORS.get(m, "#999") for m in order]

    sns.boxplot(
        data=df_valid, x="model_name", y="latency_total_ms", order=order,
        palette=palette, ax=ax, showfliers=False, width=0.5,
    )

    medians = df_valid.groupby("model_name")["latency_total_ms"].median()
    for i, model in enumerate(order):
        med = medians[model]
        ax.text(i, med + 50, f"{med:.0f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Response Latency Distribution by Model", fontsize=14, fontweight="bold", pad=15)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "04_latency_comparison.png")
    plt.close(fig)
    print("  ✓ 04_latency_comparison.png")


# ── Chart 5: Token Efficiency ──────────────────────────

def chart_token_efficiency(df: pd.DataFrame):
    """Scatter plot: tokens output vs pass rate — shows efficiency."""
    summary = (
        df.groupby("model_name")
        .agg(
            pass_rate=("pass_fail", lambda x: x.mean() * 100),
            avg_tokens=("tokens_output", "mean"),
            avg_latency=("latency_total_ms", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in summary.iterrows():
        color = MODEL_COLORS.get(row["model_name"], "#999")
        ax.scatter(row["avg_tokens"], row["pass_rate"], s=200, c=color,
                   edgecolors="white", linewidth=1.5, zorder=5)
        ax.annotate(row["model_name"], (row["avg_tokens"], row["pass_rate"]),
                    textcoords="offset points", xytext=(10, 5), fontsize=9, fontweight="bold")

    ax.set_xlabel("Average Tokens Output")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Token Efficiency: Output Length vs Accuracy", fontsize=14, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "05_token_efficiency.png")
    plt.close(fig)
    print("  ✓ 05_token_efficiency.png")


# ── Chart 6: Heatmap (Model × Category) ───────────────

def chart_heatmap(df: pd.DataFrame):
    """Heatmap: pass rate for every model × category combination."""
    pivot = (
        df.groupby(["model_name", "category"])["pass_fail"]
        .mean()
        .mul(100)
        .unstack(fill_value=0)
    )
    pivot = pivot.reindex(columns=CATEGORY_ORDER)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100,
        linewidths=0.5, linecolor="white", ax=ax,
        cbar_kws={"label": "Pass Rate (%)", "shrink": 0.8},
        annot_kws={"fontsize": 12, "fontweight": "bold"},
    )
    ax.set_title("Pass Rate Heatmap: Model × Category", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels([c.title() for c in CATEGORY_ORDER], rotation=0)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "06_heatmap_model_category.png")
    plt.close(fig)
    print("  ✓ 06_heatmap_model_category.png")


# ── Chart 7: Radar Chart (Model Profiles) ─────────────

def chart_radar(df: pd.DataFrame):
    """Radar chart: multi-dimensional comparison of each model."""
    summary = (
        df.groupby("model_name")
        .agg(
            pass_rate=("pass_fail", lambda x: x.mean() * 100),
            avg_latency=("latency_total_ms", "mean"),
            avg_tokens_out=("tokens_output", "mean"),
            tests_passed=("tests_passed", "mean"),
        )
        .reset_index()
    )

    # Normalize metrics to 0-100 scale (higher = better)
    summary["speed_score"] = 100 * (1 - (summary["avg_latency"] - summary["avg_latency"].min()) /
                                    (summary["avg_latency"].max() - summary["avg_latency"].min() + 1))
    summary["efficiency_score"] = 100 * (1 - (summary["avg_tokens_out"] - summary["avg_tokens_out"].min()) /
                                         (summary["avg_tokens_out"].max() - summary["avg_tokens_out"].min() + 1))
    summary["test_score"] = 100 * (summary["tests_passed"] / (summary["tests_passed"].max() + 1))

    categories_radar = ["Pass Rate", "Speed", "Token Efficiency", "Tests Passed"]
    n_cats = len(categories_radar)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]

    for _, row in summary.iterrows():
        values = [row["pass_rate"], row["speed_score"], row["efficiency_score"], row["test_score"]]
        values += values[:1]
        color = MODEL_COLORS.get(row["model_name"], "#999")
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=row["model_name"])
        ax.fill(angles, values, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories_radar, fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Model Performance Profiles", fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "07_radar_profiles.png")
    plt.close(fig)
    print("  ✓ 07_radar_profiles.png")


# ── Chart 8: Category Breakdown (stacked) ─────────────

def chart_category_stacked(df: pd.DataFrame):
    """Stacked bar: pass vs fail count per model per category."""
    summary = (
        df.groupby(["model_name", "category"])
        .agg(
            passed=("pass_fail", "sum"),
            total=("pass_fail", "count"),
        )
        .reset_index()
    )
    summary["failed"] = summary["total"] - summary["passed"]

    models = sorted(df["model_name"].unique())
    fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

    for idx, cat in enumerate(CATEGORY_ORDER):
        ax = axes[idx]
        cat_data = summary[summary["category"] == cat].set_index("model_name")
        cat_data = cat_data.reindex(models).fillna(0)

        colors_pass = [MODEL_COLORS.get(m, "#999") for m in models]
        colors_fail = ["#E5E7EB"] * len(models)

        ax.barh(models, cat_data["passed"], color=colors_pass, edgecolor="white", linewidth=0.5, label="Passed")
        ax.barh(models, cat_data["failed"], left=cat_data["passed"], color=colors_fail,
                edgecolor="white", linewidth=0.5, label="Failed")

        for i, m in enumerate(models):
            p = int(cat_data.loc[m, "passed"])
            t = int(cat_data.loc[m, "total"])
            ax.text(t + 0.2, i, f"{p}/{t}", va="center", fontsize=9)

        ax.set_title(cat.title(), fontweight="bold")
        ax.set_xlim(0, 16)

    axes[0].set_ylabel("")
    fig.suptitle("Pass/Fail Breakdown by Category", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "08_category_breakdown.png")
    plt.close(fig)
    print("  ✓ 08_category_breakdown.png")


# ── Chart 9: Latency vs Accuracy Tradeoff ──────────────

def chart_latency_vs_accuracy(df: pd.DataFrame):
    """Bubble chart: latency vs pass rate, bubble size = tokens output."""
    summary = (
        df.groupby("model_name")
        .agg(
            pass_rate=("pass_fail", lambda x: x.mean() * 100),
            avg_latency=("latency_total_ms", "mean"),
            avg_tokens=("tokens_output", "mean"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in summary.iterrows():
        color = MODEL_COLORS.get(row["model_name"], "#999")
        size = max(row["avg_tokens"] / 3, 80)
        ax.scatter(row["avg_latency"], row["pass_rate"], s=size, c=color,
                   alpha=0.8, edgecolors="black", linewidth=1, zorder=5)
        ax.annotate(row["model_name"], (row["avg_latency"], row["pass_rate"]),
                    textcoords="offset points", xytext=(12, 5), fontsize=9, fontweight="bold")

    ax.set_xlabel("Average Latency (ms)")
    ax.set_ylabel("Pass Rate (%)")
    ax.set_title("Latency vs Accuracy Tradeoff (bubble size = tokens output)",
                 fontsize=13, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "09_latency_vs_accuracy.png")
    plt.close(fig)
    print("  ✓ 09_latency_vs_accuracy.png")


# ── Chart 10: Summary Table as Image ──────────────────

def chart_summary_table(df: pd.DataFrame):
    """Render a summary statistics table as an image."""
    summary = (
        df.groupby("model_name")
        .agg(
            tasks=("task_id", "nunique"),
            pass_rate=("pass_fail", lambda x: f"{x.mean() * 100:.1f}%"),
            passed=("pass_fail", "sum"),
            total=("pass_fail", "count"),
            avg_latency=("latency_total_ms", lambda x: f"{x.mean():.0f}ms"),
            avg_tokens_in=("tokens_input", lambda x: f"{x.mean():.0f}"),
            avg_tokens_out=("tokens_output", lambda x: f"{x.mean():.0f}"),
        )
        .reset_index()
    )
    summary.columns = ["Model", "Tasks", "Pass Rate", "Passed", "Total",
                       "Avg Latency", "Avg Tokens In", "Avg Tokens Out"]
    summary = summary.sort_values("Pass Rate", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 3))
    ax.axis("off")

    table = ax.table(
        cellText=summary.values,
        colLabels=summary.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Style header
    for j in range(len(summary.columns)):
        table[0, j].set_facecolor("#1F2937")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(summary) + 1):
        color = "#F9FAFB" if i % 2 == 0 else "white"
        for j in range(len(summary.columns)):
            table[i, j].set_facecolor(color)

    ax.set_title("Summary Statistics", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    fig.savefig(CHARTS_DIR / "10_summary_table.png")
    plt.close(fig)
    print("  ✓ 10_summary_table.png")

    # Also save as CSV
    csv_path = PIPELINE_DIR / "exports" / "results_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"  ✓ results_summary.csv")


# ── Main ───────────────────────────────────────────────

def main():
    print("=" * 50)
    print("LLM Benchmark Results Visualizer")
    print("=" * 50)

    df = load_data()

    if df.empty:
        print("No data found. Run evaluations first.")
        sys.exit(1)

    print(f"\nGenerating charts in {CHARTS_DIR}/\n")

    chart_overall_pass_rate(df)
    chart_pass_rate_by_category(df)
    chart_pass_rate_by_difficulty(df)
    chart_latency_comparison(df)
    chart_token_efficiency(df)
    chart_heatmap(df)
    chart_radar(df)
    chart_category_stacked(df)
    chart_latency_vs_accuracy(df)
    chart_summary_table(df)

    print(f"\nDone! {len(list(CHARTS_DIR.glob('*.png')))} charts saved to exports/charts/")
    print(f"Open the folder: open {CHARTS_DIR}")


if __name__ == "__main__":
    main()