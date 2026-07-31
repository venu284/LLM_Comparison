"""Publication figures for the Phase 6/7 report.

Design rules applied throughout (see the project's dataviz guidance):

- Form is chosen by the data's job, not by habit. Where one number is the point,
  the figure uses *emphasis* -- one accent hue, everything else gray -- rather
  than giving every bar its own color.
- The categorical palette is fixed-order and was validated with the palette
  checker rather than eyeballed. Both the 4-slot set used for score components
  and the 3-slot set used for strategy comparison pass the lightness, chroma,
  CVD-separation, and normal-vision gates on a white surface.
- Two slots (aqua, yellow) fall below 3:1 contrast against white, so every
  chart that uses them carries visible direct labels. That relief is required,
  not optional.
- Thin marks, hairline grid, no dashed gridlines, 2px surface gaps between
  stacked segments, and selective direct labels rather than a number on every
  mark.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

# Validated categorical slots, fixed order. Never cycled, never re-ordered.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE, ORANGE, AQUA, YELLOW = SERIES

# Chart chrome.
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
DEEMPHASIS = "#d5d4cf"

STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"

FONT = ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"]


def _base_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.labelsize": 9.5,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.linewidth": 0.8,
        }
    )


def _clean(axes, grid_axis: Optional[str] = "y") -> None:
    """Hairline recessive chrome: no top/right spines, solid grid only."""
    axes.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        axes.spines[spine].set_color(BASELINE)
    if grid_axis:
        axes.grid(axis=grid_axis, color=GRID, linewidth=0.7, linestyle="-", zorder=0)
        axes.set_axisbelow(True)


def _caption(figure, text: str) -> None:
    figure.text(
        0.5, 0.008, text, ha="center", va="bottom",
        fontsize=8, color=MUTED, style="italic", wrap=True,
    )


# ---------------------------------------------------------------- formula ----

def figure_formula_anatomy(output: Path, weights: Dict[str, float]) -> None:
    """The scoring formula, with each term colored to match Figure 2.

    A formula is not data, so this is a labelled diagram rather than a chart.
    The colors are the payload: they let the reader carry each term straight
    into the decomposition figure that follows.
    """
    _base_style()
    figure, axes = plt.subplots(figsize=(10.4, 3.5))
    axes.set_xlim(0, 10.4)
    axes.set_ylim(0, 3.5)
    axes.axis("off")

    terms = [
        ("w₁", weights["category"], "CategoryPassRate", BLUE, "how often this model\npasses in this category"),
        ("w₂", weights["difficulty"], "DifficultyPassRate", ORANGE, "how often it passes\nat this difficulty"),
        ("w₃", weights["latency"], "1 / NormLatency", AQUA, "speed, normalized so\nthe fastest model = 1.0"),
        ("w₄", weights["tokens"], "1 / NormTokens", YELLOW, "output brevity,\nsame normalization"),
    ]

    axes.text(0.15, 2.94, "Score (model, task)  =", fontsize=13, va="center",
              color=INK, fontweight="bold")

    width, gap, left = 2.28, 0.34, 0.15
    for index, (symbol, weight, name, color, blurb) in enumerate(terms):
        x = left + index * (width + gap)

        box = FancyBboxPatch(
            (x, 1.62), width, 0.92,
            boxstyle="round,pad=0.045,rounding_size=0.07",
            facecolor=color, edgecolor="none", alpha=0.16, zorder=1,
        )
        axes.add_patch(box)
        axes.add_patch(
            FancyBboxPatch(
                (x, 1.62), 0.075, 0.92,
                boxstyle="round,pad=0,rounding_size=0.02",
                facecolor=color, edgecolor="none", zorder=2,
            )
        )

        axes.text(x + width / 2, 2.30, f"{symbol} = {weight:g}", fontsize=11.5,
                  ha="center", va="center", color=INK, fontweight="bold")
        axes.text(x + width / 2, 1.90, f"×  {name}", fontsize=9.6,
                  ha="center", va="center", color=INK_SECONDARY)
        axes.text(x + width / 2, 1.30, blurb, fontsize=8.2, ha="center",
                  va="top", color=MUTED, linespacing=1.45)

        if index < len(terms) - 1:
            axes.text(x + width + gap / 2, 2.08, "+", fontsize=14,
                      ha="center", va="center", color=MUTED)

    accuracy = weights["category"] + weights["difficulty"]
    efficiency = weights["latency"] + weights["tokens"]

    span_y = 0.72
    axes.plot([left, left + 2 * width + gap], [span_y, span_y], color=INK_SECONDARY, lw=1.1)
    axes.plot([left + 2 * (width + gap), left + 4 * width + 3 * gap], [span_y, span_y],
              color=STATUS_CRITICAL, lw=1.1)
    axes.text(left + width + gap / 2, span_y - 0.16,
              f"accuracy terms — {accuracy:.0%} of the score",
              fontsize=9, ha="center", va="top", color=INK_SECONDARY)
    axes.text(left + 3 * width + 2.5 * gap, span_y - 0.16,
              f"speed & brevity terms — {efficiency:.0%} of the score",
              fontsize=9, ha="center", va="top", color=STATUS_CRITICAL, fontweight="bold")

    axes.text(5.2, 3.42, "The recommendation scoring formula (Phase 2 framework, pre-registered weights)",
              fontsize=11, ha="center", va="top", color=INK, fontweight="bold")

    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)


def figure_score_decomposition(
    output: Path, framework: pd.DataFrame, accuracy_only: pd.DataFrame
) -> None:
    """Why the pre-registered weights pick the wrong model.

    Stacked horizontal bars: each model's score broken into its four weighted
    terms, under the framework weights (left) and pure category weighting
    (right). Direct labels on every segment wide enough to hold one -- required
    relief, since two of the four hues sit below 3:1 on white.
    """
    _base_style()
    figure, axes_pair = plt.subplots(1, 2, figsize=(11.6, 4.5), sharey=True)

    labels = ["Category", "Difficulty", "Speed", "Brevity"]
    keys = ["category", "difficulty", "latency", "tokens"]

    # One fixed row order for both panels. The axes share y, so a per-panel sort
    # would let the second panel's tick labels overwrite the first panel's and
    # mislabel its bars. A common order also makes the two panels comparable
    # row by row, which is the whole point of showing them together.
    order = list(accuracy_only.sort_values("total").index)

    for axes, source, title in (
        (axes_pair[0], framework, "Framework weights  (0.5 / 0.2 / 0.15 / 0.15)"),
        (axes_pair[1], accuracy_only, "Accuracy only  (1.0 / 0 / 0 / 0)"),
    ):
        frame = source.loc[order]
        positions = np.arange(len(frame))
        left = np.zeros(len(frame))

        for key, label, color in zip(keys, labels, SERIES):
            values = frame[key].to_numpy()
            axes.barh(positions, values, left=left, height=0.6,
                      color=color, label=label, zorder=3,
                      edgecolor=SURFACE, linewidth=1.6)  # 2px surface gap
            for position, value, start in zip(positions, values, left):
                if value > 0.055:
                    axes.text(start + value / 2, position, f"{value:.2f}",
                              ha="center", va="center", fontsize=7.6,
                              color=INK, zorder=4)
            left += values

        # The selected model is whichever scores highest, not whichever sits
        # at the top of a fixed row order.
        selected = str(frame["total"].idxmax())
        selected_row = order.index(selected)

        for position, (total, model) in enumerate(zip(frame["total"], frame.index)):
            winner = model == selected
            suffix = "   ◂ selected" if winner else ""
            axes.text(total + 0.014, position, f"{total:.3f}{suffix}",
                      va="center", ha="left", fontsize=8.4,
                      color=STATUS_CRITICAL if winner else INK_SECONDARY,
                      fontweight="bold" if winner else "normal")

        axes.set_yticks(positions)
        tick_labels = axes.set_yticklabels(order, fontsize=8.8, color=INK_SECONDARY)
        if tick_labels:
            tick_labels[selected_row].set_color(INK)
            tick_labels[selected_row].set_fontweight("bold")

        axes.set_xlim(0, max(0.92, float(left.max()) * 1.42))
        axes.set_xlabel("Recommendation score")
        axes.set_title(title, fontsize=10, pad=9)
        _clean(axes, grid_axis="x")

    axes_pair[0].legend(
        loc="lower right", frameon=False, ncol=2, fontsize=8.4,
        handlelength=1.1, handleheight=0.9, borderpad=0.3,
        bbox_to_anchor=(1.0, 0.13),
    )

    figure.suptitle(
        "Score decomposition: the speed and brevity terms decide the winner",
        fontsize=12, y=0.99, color=INK, fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    _caption(
        figure,
        "Mean weighted contribution per term across all 65 tasks. GPT-OSS-120B has the largest accuracy "
        "term yet places fourth; the top two totals differ by 0.0005, so the formula barely discriminates.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


# ---------------------------------------------------------------- results ----

def figure_headroom(output: Path, scores: Dict[str, float]) -> None:
    """The headline: H4 was unreachable, and the gap shows why.

    Emphasis form. The recommender is the subject and carries the accent; the
    baselines are context and stay gray. The oracle ceiling and the level H4
    demanded are drawn as reference rules, so the reader sees that the target
    sits above what any router could reach.
    """
    _base_style()
    figure, axes = plt.subplots(figsize=(9.2, 5.0))

    order = ["random", "recommender", "best_single_model", "oracle"]
    labels = ["Random\nselection", "Recommender\n(this work)", "Best single model\n(GPT-OSS-120B)", "Oracle\n(perfect router)"]
    values = [scores[key] * 100 for key in order]
    colors = [DEEMPHASIS, BLUE, DEEMPHASIS, DEEMPHASIS]

    positions = np.arange(len(order))
    axes.bar(positions, values, width=0.56, color=colors, zorder=3)

    for position, value in zip(positions, values):
        axes.text(position, value + 1.1, f"{value:.1f}%", ha="center",
                  va="bottom", fontsize=10.5, color=INK, fontweight="bold")

    best = scores["best_single_model"] * 100
    oracle = scores["oracle"] * 100
    h4_target = best * 1.15

    # Reference rules are labelled inside the plot on the left, where the two
    # shortest bars leave clear space -- keeping them off the value labels.
    axes.axhline(oracle, color=INK_SECONDARY, lw=1.0, zorder=2)
    axes.text(-0.34, oracle + 0.9, "ceiling for any router", va="bottom", ha="left",
              fontsize=8.4, color=INK_SECONDARY)

    axes.axhline(h4_target, color=STATUS_CRITICAL, lw=1.2, zorder=2)
    axes.text(-0.34, h4_target + 0.9, "H4 target (+15%)", va="bottom", ha="left",
              fontsize=8.4, color=STATUS_CRITICAL, fontweight="bold")

    # The gap between the baseline and the ceiling is the whole finding, so it
    # gets its own measured callout in the empty column between the two bars.
    axes.annotate(
        "", xy=(2.5, oracle), xytext=(2.5, best),
        arrowprops=dict(arrowstyle="<->", color=INK, lw=1.2),
    )
    axes.text(2.5, best - 2.4,
              f"all available headroom\n{oracle - best:.1f} points",
              ha="center", va="top", fontsize=8.6, color=INK,
              linespacing=1.45, fontweight="bold")

    axes.set_xticks(positions)
    axes.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY)
    axes.set_ylabel("Tasks passed (%)")
    axes.set_ylim(0, 78)
    axes.set_title(
        "H4 asked for more improvement than the data can supply",
        fontsize=12, pad=12, color=INK, fontweight="bold",
    )
    _clean(axes)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    _caption(
        figure,
        "Leave-one-task-out cross-validation, 65 tasks. The H4 target sits above the oracle, "
        "so no routing algorithm could have reached it on this model set.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_per_category(output: Path, per_category: pd.DataFrame) -> None:
    """Three strategies across the five categories. Grouped bars, 3 series."""
    _base_style()
    figure, axes = plt.subplots(figsize=(10.2, 4.8))

    frame = per_category.copy()
    pretty = {
        "bugfix": "Bug Fixing", "typescript": "TypeScript", "api": "REST API",
        "css": "CSS / Layout", "frontend": "Frontend",
    }
    frame = frame.sort_values("oracle", ascending=False)
    categories = [pretty.get(name, name) for name in frame.index]

    series = [
        ("recommender", "Recommender", BLUE),
        ("best_single", "Best single model", ORANGE),
        ("oracle", "Oracle (ceiling)", AQUA),
    ]

    positions = np.arange(len(frame))
    width = 0.26

    for index, (key, label, color) in enumerate(series):
        offset = (index - 1) * width
        values = frame[key].to_numpy() * 100
        axes.bar(positions + offset, values, width * 0.92, label=label,
                 color=color, zorder=3)
        for position, value in zip(positions, values):
            axes.text(position + offset, value + 1.4, f"{value:.0f}",
                      ha="center", va="bottom", fontsize=7.6, color=INK_SECONDARY)

    axes.set_xticks(positions)
    axes.set_xticklabels(categories, fontsize=9.2, color=INK_SECONDARY)
    axes.set_ylabel("Tasks passed (%)")
    axes.set_ylim(0, 100)
    axes.legend(frameon=False, ncol=3, loc="upper right", fontsize=8.6,
                handlelength=1.1, handleheight=0.9)
    axes.set_title(
        "The recommender trails the single-model baseline in every category",
        fontsize=12, pad=12, color=INK, fontweight="bold",
    )
    _clean(axes)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _caption(
        figure,
        "Where the orange and aqua bars meet, the best single model already achieves everything "
        "routing could achieve -- true for REST API and Bug Fixing.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_weight_sensitivity(output: Path, grid: pd.DataFrame) -> None:
    """Every weighting tested, against the baseline. Emphasis on the best case."""
    _base_style()
    figure, axes = plt.subplots(figsize=(9.6, 4.4))

    frame = grid.copy().sort_values("recommender")
    labels = [
        f"{row.w_category:g} / {row.w_difficulty:g} / {row.w_latency:g} / {row.w_tokens:g}"
        for row in frame.itertuples()
    ]
    values = frame["recommender"].to_numpy() * 100
    baseline = float(frame["best_single"].iloc[0]) * 100

    best = values.max()
    colors = [BLUE if abs(value - best) < 1e-9 else DEEMPHASIS for value in values]

    positions = np.arange(len(frame))
    axes.barh(positions, values, height=0.56, color=colors, zorder=3)

    for position, value in zip(positions, values):
        delta = (value - baseline) / baseline * 100
        axes.text(value + 0.7, position, f"{value:.1f}%   ({delta:+.1f}%)",
                  va="center", ha="left", fontsize=8.4,
                  color=INK if abs(value - best) < 1e-9 else INK_SECONDARY,
                  fontweight="bold" if abs(value - best) < 1e-9 else "normal")

    axes.axvline(baseline, color=STATUS_CRITICAL, lw=1.2, zorder=4)
    axes.text(baseline, len(frame) - 0.28,
              f"best single model {baseline:.1f}%", va="bottom", ha="center",
              fontsize=8.2, color=STATUS_CRITICAL, fontweight="bold")

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8.4, color=INK_SECONDARY,
                         fontfamily="monospace")
    axes.set_xlabel("Tasks passed (%)")
    axes.set_ylabel("w₁ / w₂ / w₃ / w₄", labelpad=8)
    axes.set_xlim(0, 72)
    axes.set_ylim(-0.6, len(frame) - 0.05)
    axes.set_title(
        "No weighting beats the baseline; only pure accuracy weighting matches it",
        fontsize=11.5, pad=12, color=INK, fontweight="bold",
    )
    _clean(axes, grid_axis="x")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _caption(
        figure,
        "Weights were fixed in Phase 2 before any data existed and were not tuned afterwards. "
        "Every configuration carrying speed or brevity weight loses accuracy.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_mcnemar(output: Path, comparisons: List[dict]) -> None:
    """Which model differences survive correction. Status color = decision."""
    _base_style()
    figure, axes = plt.subplots(figsize=(9.8, 4.6))

    items = sorted(comparisons, key=lambda item: item["p_value_holm"], reverse=True)
    labels = [f"{item['model_a']}  vs  {item['model_b']}" for item in items]
    values = [item["p_value_holm"] for item in items]
    colors = [STATUS_CRITICAL if item["significant_holm"] else DEEMPHASIS for item in items]

    positions = np.arange(len(items))
    axes.barh(positions, values, height=0.58, color=colors, zorder=3)

    for position, (value, item) in enumerate(zip(values, items)):
        text = "<0.0001" if value < 0.0001 else f"{value:.3f}"
        axes.text(value + 0.012, position, text, va="center", ha="left",
                  fontsize=8, color=INK_SECONDARY)

    axes.axvline(0.05, color=INK, lw=1.0, zorder=4)
    axes.text(0.062, -0.72, "α = 0.05", fontsize=8.4, color=INK, va="bottom")

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=8.4, color=INK_SECONDARY)
    axes.set_xlabel("Holm-adjusted p-value")
    axes.set_xlim(0, 1.13)
    axes.set_title(
        "Only five of ten model pairs are distinguishable; the top three are not",
        fontsize=11.5, pad=12, color=INK, fontweight="bold",
    )
    axes.legend(
        handles=[
            Patch(facecolor=STATUS_CRITICAL, label="significant after correction"),
            Patch(facecolor=DEEMPHASIS, label="not distinguishable"),
        ],
        frameon=False, loc="upper right", fontsize=8.4,
        handlelength=1.1, handleheight=0.9,
    )
    _clean(axes, grid_axis="x")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    _caption(
        figure,
        "Paired McNemar tests on the same 65 tasks, Holm-corrected across all ten comparisons.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_bradley_terry(output: Path, ranking: Sequence[dict]) -> None:
    """Synthetic Bradley-Terry ratings with bootstrap intervals.

    Labelled SYNTHETIC in the title, not only in the caption: this figure would
    otherwise be mistaken for a human-preference result.
    """
    _base_style()
    figure, axes = plt.subplots(figsize=(9.4, 4.3))

    items = list(reversed(list(ranking)))
    positions = np.arange(len(items))
    ratings = [item["rating"] for item in items]
    lower = [item["rating"] - (item["ci_lower"] or item["rating"]) for item in items]
    upper = [(item["ci_upper"] or item["rating"]) - item["rating"] for item in items]

    axes.errorbar(
        ratings, positions, xerr=[lower, upper], fmt="o",
        color=BLUE, ecolor=BASELINE, elinewidth=1.6, capsize=4,
        markersize=9, markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=4,
    )

    # x in axes fraction, y in data coordinates -- so the pass-rate column lines
    # up with its own row rather than being spread evenly down the axis.
    row_transform = axes.get_yaxis_transform()

    for position, item in zip(positions, items):
        axes.text(item["rating"], position + 0.2, f"{item['rating']:.0f}",
                  ha="center", va="bottom", fontsize=8.4, color=INK)
        axes.text(1.03, position, f"{item['auto_pass_rate'] * 100:.1f}%",
                  transform=row_transform, ha="left", va="center",
                  fontsize=8.2, color=MUTED)

    axes.set_yticks(positions)
    axes.set_yticklabels([item["model"] for item in items], fontsize=9,
                         color=INK_SECONDARY)
    axes.set_ylim(-0.55, len(items) - 0.3)
    axes.set_xlabel("Bradley-Terry rating (bootstrap 95% CI)")
    axes.set_title(
        "SYNTHETIC pilot — pipeline validation, not evidence for H3",
        fontsize=11.5, pad=20, color=STATUS_CRITICAL, fontweight="bold",
    )
    axes.text(1.03, len(items) - 0.22, "automated\npass rate",
              transform=row_transform, ha="left", va="bottom",
              fontsize=7.6, color=MUTED, linespacing=1.3)
    _clean(axes, grid_axis="x")
    figure.tight_layout(rect=(0, 0.06, 0.92, 1))
    _caption(
        figure,
        "Fitted to 1,200 generated votes. Because those votes were generated from the automated "
        "pass rates, agreement with them is circular. No human votes have been collected.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_zero_test(output: Path, full_rates: pd.Series, filtered: pd.DataFrame) -> None:
    """The sensitivity that flips RQ1, with denominators made visible."""
    _base_style()
    figure, axes = plt.subplots(figsize=(9.8, 4.7))

    models = list(full_rates.index)
    positions = np.arange(len(models))
    width = 0.36

    published = [full_rates[model] * 100 for model in models]
    dropped = [filtered.loc[model, "rate"] * 100 for model in models]

    axes.bar(positions - width / 2, published, width * 0.94, label="All runs (as published)",
             color=BLUE, zorder=3)
    axes.bar(positions + width / 2, dropped, width * 0.94, label="Zero-test runs dropped",
             color=ORANGE, zorder=3)

    for position, (value, model) in enumerate(zip(published, models)):
        axes.text(position - width / 2, value + 1.3, f"{value:.1f}", ha="center",
                  va="bottom", fontsize=7.8, color=INK_SECONDARY)
    for position, model in enumerate(models):
        value = filtered.loc[model, "rate"] * 100
        attempts = int(filtered.loc[model, "n"])
        axes.text(position + width / 2, value + 1.3, f"{value:.1f}", ha="center",
                  va="bottom", fontsize=7.8, color=INK_SECONDARY)
        # Inside the bar, so white rather than muted gray -- the denominators
        # are the reason the two bars are not comparable and have to be legible.
        axes.text(position + width / 2, 1.8, f"n={attempts}", ha="center",
                  va="bottom", fontsize=7.6, color=SURFACE,
                  fontweight="bold" if attempts < 40 else "normal")

    axes.set_xticks(positions)
    axes.set_xticklabels(models, fontsize=8.6, color=INK_SECONDARY, rotation=12, ha="right")
    axes.set_ylabel("Tasks passed (%)")
    axes.set_ylim(0, 72)
    axes.legend(frameon=False, loc="upper right", fontsize=8.6,
                handlelength=1.1, handleheight=0.9)
    axes.set_title(
        "Dropping the zero-test runs changes χ² p from 0.00002 to 0.37",
        fontsize=11.5, pad=12, color=INK, fontweight="bold",
    )
    _clean(axes)
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    _caption(
        figure,
        "The two bars are not directly comparable: dropping those runs leaves unequal denominators, "
        "and Qwen3-32B retains only 25 of its 65 attempts.",
    )
    figure.savefig(output, dpi=200, facecolor=SURFACE)
    plt.close(figure)


def figure_classifier_pipeline(output: Path, benchmark_accuracy: float, holdout_accuracy: float) -> None:
    """Two-stage classifier as a flow diagram, with both accuracy figures."""
    _base_style()
    figure, axes = plt.subplots(figsize=(10.4, 3.5))
    axes.set_xlim(0, 10.4)
    axes.set_ylim(0, 3.5)
    axes.axis("off")

    def box(x, y, width, height, color, alpha=0.16):
        axes.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.05,rounding_size=0.08",
                facecolor=color, edgecolor=color, alpha=alpha, lw=1.2, zorder=1,
            )
        )

    def arrow(x1, y1, x2, y2, color=MUTED, label=None, label_offset=0.16):
        axes.annotate("", xy=(x2, y2), xytext=(x1, y1),
                      arrowprops=dict(arrowstyle="->", color=color, lw=1.3))
        if label:
            axes.text((x1 + x2) / 2, (y1 + y2) / 2 + label_offset, label,
                      ha="center", va="bottom", fontsize=8, color=color)

    box(0.1, 1.42, 1.85, 0.86, MUTED, 0.12)
    axes.text(1.02, 1.85, "Free-text\nprompt", ha="center", va="center",
              fontsize=9.4, color=INK, linespacing=1.4)

    box(2.55, 1.42, 2.45, 0.86, ORANGE)
    axes.text(3.78, 1.97, "Stage 1", ha="center", va="center", fontsize=8.2,
              color=ORANGE, fontweight="bold")
    axes.text(3.78, 1.66, "repair intent?", ha="center", va="center",
              fontsize=9.6, color=INK)

    box(5.6, 2.42, 2.25, 0.76, AQUA)
    axes.text(6.72, 2.80, "bugfix", ha="center", va="center", fontsize=9.6,
              color=INK, fontweight="bold")

    box(5.6, 0.52, 2.25, 1.42, BLUE)
    axes.text(6.72, 1.72, "Stage 2", ha="center", va="center", fontsize=8.2,
              color=BLUE, fontweight="bold")
    axes.text(6.72, 1.16, "most keyword hits:\nfrontend / api /\ncss / typescript",
              ha="center", va="center", fontsize=8.8, color=INK, linespacing=1.5)

    arrow(2.0, 1.85, 2.5, 1.85)
    arrow(5.05, 2.0, 5.55, 2.62, color=AQUA, label="yes")
    arrow(5.05, 1.7, 5.55, 1.30, color=BLUE, label="no", label_offset=-0.34)

    box(8.4, 1.42, 1.9, 0.86, MUTED, 0.12)
    axes.text(9.35, 1.85, "category\n→ profiles", ha="center", va="center",
              fontsize=9.2, color=INK, linespacing=1.4)
    arrow(7.9, 2.62, 8.42, 2.05, color=MUTED)
    arrow(7.9, 1.30, 8.42, 1.72, color=MUTED)

    axes.text(5.2, 3.42,
              "Two-stage task classifier: repair intent is tested first",
              ha="center", va="top", fontsize=11.5, color=INK, fontweight="bold")
    axes.text(5.2, 0.30,
              f"{benchmark_accuracy * 100:.0f}% on the 65 templated benchmark prompts   ·   "
              f"{holdout_accuracy * 100:.0f}% on freshly written paraphrases",
              ha="center", va="center", fontsize=9, color=INK_SECONDARY)
    axes.text(5.2, 0.03,
              "Testing intent first is what fixes bug-fix classification: those prompts embed the broken "
              "source, so they contain more language keywords than repair keywords.",
              ha="center", va="bottom", fontsize=7.8, color=MUTED, style="italic")

    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
