"""Visualization utilities for ranker evaluation outputs.

Styling notes:
- Colors are semantic, not decorative: XGBRanker and LGBMRanker get bold,
  distinct hues; the qualifying baseline is deliberately muted grey so it
  visually recedes as the "floor" rather than competing for attention.
- Every bar is value-labeled so readers don't have to eyeball gridlines.
- The top-performing model per metric gets a small crown marker.
- Per-race trend lines shade the area under the leading model to make
  "who's winning, and by how much" readable at a glance.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "xgb_ranker": "#E10600",          # F1 red
    "lgbm_ranker": "#00A3E0",         # circuit-blue
    "qualifying_baseline": "#9CA3AF",  # neutral grey (deliberately muted)
}
MODEL_LABELS = {
    "xgb_ranker": "XGBRanker",
    "lgbm_ranker": "LGBMRanker",
    "qualifying_baseline": "Qualifying Baseline",
}

BG_COLOR = "#0F1115"
PANEL_COLOR = "#171A21"
GRID_COLOR = "#2A2F3A"
TEXT_COLOR = "#E8E9ED"
MUTED_TEXT = "#9AA0AC"
ACCENT = "#E10600"

METRIC_LABELS = {
    "global_ndcg": "NDCG",
    "ndcg_at_3": "NDCG@3",
    "mrr": "MRR",
    "map": "MAP",
    "map_at_3": "MAP@3",
    "precision_at_3": "Precision@3",
}


def _color_for(model_name: str) -> str:
    return MODEL_COLORS.get(model_name, "#C9CDD4")


def _label_for(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def _apply_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": PANEL_COLOR,
            "savefig.facecolor": BG_COLOR,
            "text.color": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": MUTED_TEXT,
            "ytick.color": MUTED_TEXT,
            "axes.edgecolor": GRID_COLOR,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
        }
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _style_axes(ax, title: str, y_label: str, subtitle: str | None = None) -> None:
    ax.set_title(title, fontsize=13.5, fontweight="bold", color=TEXT_COLOR, pad=22 if subtitle else 12, loc="left")
    if subtitle:
        ax.text(
            0.0, 1.03, subtitle, transform=ax.transAxes,
            fontsize=9, color=MUTED_TEXT, ha="left", va="bottom",
        )
    ax.set_ylabel(y_label, fontsize=10, color=MUTED_TEXT)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.9, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("top", "right"):
            spine.set_visible(False)
        else:
            spine.set_color(GRID_COLOR)
    ax.tick_params(length=0)


def _sorted_model_order(metrics_df: pd.DataFrame) -> list[str]:
    ranked = metrics_df.sort_values("global_ndcg", ascending=False)
    return ranked["model"].tolist()


def _grouped_bars(ax, plot_df: pd.DataFrame, metric_cols: list[str], annotate: bool = True) -> None:
    """Draw grouped bars with per-model colors, rounded caps, and value labels."""
    models = plot_df.index.tolist()
    n_models = len(models)
    n_metrics = len(metric_cols)
    group_width = 0.8
    bar_width = group_width / n_models
    x = np.arange(n_metrics)

    # mark the winner (max value) for each metric with a small crown
    winners = {m: plot_df[m].idxmax() for m in metric_cols}

    for i, model in enumerate(models):
        offsets = x - group_width / 2 + bar_width * i + bar_width / 2
        values = plot_df.loc[model, metric_cols].to_numpy(dtype=float)
        color = _color_for(model)
        bars = ax.bar(
            offsets, values, width=bar_width * 0.88,
            color=color, edgecolor=BG_COLOR, linewidth=1.2,
            label=_label_for(model), zorder=3,
        )
        if annotate:
            for rect, val, metric in zip(bars, values, metric_cols):
                ax.text(
                    rect.get_x() + rect.get_width() / 2, val + 0.015,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7.6, color=TEXT_COLOR, fontweight="bold" if winners[metric] == model else "normal",
                )
                if winners[metric] == model:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2, val + 0.065,
                        "\u2605", ha="center", va="bottom", fontsize=8, color="#FFD24C",
                    )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_cols], fontsize=9.5)
    ax.set_ylim(0, max(1.0, plot_df[metric_cols].to_numpy(dtype=float).max() * 1.18))


def plot_overall_metric_bars(metrics_df: pd.DataFrame, output_path: str) -> str:
    _apply_theme()
    _ensure_dir(os.path.dirname(output_path) or ".")

    metric_cols = ["global_ndcg", "ndcg_at_3", "mrr", "map", "map_at_3", "precision_at_3"]
    plot_df = metrics_df.set_index("model")[metric_cols].copy()
    plot_df = plot_df.loc[_sorted_model_order(metrics_df)]

    fig, ax = plt.subplots(figsize=(12, 6.2))
    _grouped_bars(ax, plot_df, metric_cols)
    _style_axes(
        ax, "Model Comparison: Core Ranking Metrics", "Score",
        subtitle="\u2605 marks the best model per metric  \u00b7  higher is better",
    )
    ax.set_xlabel("")
    legend = ax.legend(loc="upper right", ncol=1, fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_topk_focus(metrics_df: pd.DataFrame, output_path: str) -> str:
    _apply_theme()
    _ensure_dir(os.path.dirname(output_path) or ".")

    metric_cols = ["ndcg_at_3", "map_at_3", "precision_at_3"]
    plot_df = metrics_df.set_index("model")[metric_cols].copy()
    plot_df = plot_df.loc[_sorted_model_order(metrics_df)]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    _grouped_bars(ax, plot_df, metric_cols)
    _style_axes(
        ax, "Top-3 Quality Focus", "Score",
        subtitle="How well each model nails the podium specifically",
    )
    ax.set_xlabel("")
    legend = ax.legend(loc="upper right", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_per_race_trends(per_race_df: pd.DataFrame, output_path: str) -> str:
    _apply_theme()
    _ensure_dir(os.path.dirname(output_path) or ".")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), sharex=True)
    metrics = ["global_ndcg", "mrr", "precision_at_3"]
    titles = ["Per-Race NDCG", "Per-Race MRR", "Per-Race Precision@3"]

    # Determine, race by race, which model leads on global_ndcg (used to shade "current leader")
    ndcg_pivot = per_race_df.pivot(index="race_id", columns="model", values="global_ndcg").sort_index()
    leader_by_race = ndcg_pivot.idxmax(axis=1)

    for ax, metric, title in zip(axes, metrics, titles):
        pivot = per_race_df.pivot(index="race_id", columns="model", values=metric).sort_index()
        race_ids = pivot.index.to_numpy()

        for model_name in pivot.columns:
            color = _color_for(model_name)
            values = pivot[model_name].to_numpy(dtype=float)
            ax.plot(
                race_ids, values, marker="o", markersize=4.2, linewidth=2.0,
                color=color, label=_label_for(model_name), zorder=3,
                markeredgecolor=BG_COLOR, markeredgewidth=0.6,
            )

        # shade a thin band under the metric's own leading line for readability
        top_model = pivot.mean(axis=0).idxmax()
        ax.fill_between(
            race_ids, 0, pivot[top_model].to_numpy(dtype=float),
            color=_color_for(top_model), alpha=0.08, zorder=1,
        )

        _style_axes(ax, title, "Score")
        ax.set_xlabel("Race ID", fontsize=9.5, color=MUTED_TEXT)
        ax.set_ylim(bottom=0)

        # Thin out x-tick labels when there are many races, and angle them
        # so they never overlap regardless of label length.
        n_races = len(race_ids)
        step = max(1, n_races // 10)
        tick_positions = race_ids[::step]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions, rotation=45, ha="right", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08),
        frameon=False, fontsize=10,
    )
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)

    fig.suptitle(
        "Consistency Across the Season", fontsize=13, fontweight="bold",
        color=TEXT_COLOR, x=0.01, y=1.14, ha="left",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_win_rate_breakdown(per_race_df: pd.DataFrame, output_path: str) -> str:
    """For each metric, what fraction of races did each model post the best score in.

    Averages can hide a model that's inconsistent-but-occasionally-brilliant vs.
    one that's steady-but-rarely-best. This answers "who actually wins races,
    head to head" rather than "who has the higher mean."
    """
    _apply_theme()
    _ensure_dir(os.path.dirname(output_path) or ".")

    metric_cols = ["global_ndcg", "mrr", "precision_at_3"]
    models = per_race_df["model"].unique().tolist()
    models = sorted(models, key=lambda m: 0 if m == "qualifying_baseline" else -1)

    win_rates = pd.DataFrame(index=models, columns=metric_cols, dtype=float)
    for metric in metric_cols:
        pivot = per_race_df.pivot(index="race_id", columns="model", values=metric)
        winners = pivot.idxmax(axis=1)
        counts = winners.value_counts(normalize=True)
        for model in models:
            win_rates.loc[model, metric] = float(counts.get(model, 0.0)) * 100.0

    # order models by overall win-rate strength for a cleaner legend/bar order
    order = win_rates.mean(axis=1).sort_values(ascending=False).index.tolist()
    win_rates = win_rates.loc[order]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    n_models = len(order)
    n_metrics = len(metric_cols)
    group_width = 0.8
    bar_width = group_width / n_models
    x = np.arange(n_metrics)

    for i, model in enumerate(order):
        offsets = x - group_width / 2 + bar_width * i + bar_width / 2
        values = win_rates.loc[model, metric_cols].to_numpy(dtype=float)
        bars = ax.bar(
            offsets, values, width=bar_width * 0.88,
            color=_color_for(model), edgecolor=BG_COLOR, linewidth=1.2,
            label=_label_for(model), zorder=3,
        )
        for rect, val in zip(bars, values):
            ax.text(
                rect.get_x() + rect.get_width() / 2, val + 1.5,
                f"{val:.0f}%", ha="center", va="bottom",
                fontsize=8, color=TEXT_COLOR,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_cols], fontsize=9.5)
    ax.set_ylim(0, 100)
    _style_axes(
        ax, "Race Win-Rate by Model", "% of races won",
        subtitle="Which model posts the single best score, race by race \u00b7 ties split",
    )
    legend = ax.legend(loc="upper right", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_improvement_over_baseline(metrics_df: pd.DataFrame, output_path: str) -> str:
    """Show XGB/LGBM's score deltas vs. the qualifying baseline directly.

    Reframes the comparison around the actual question an evaluation like this
    is trying to answer: how much value does the trained ranker add over just
    trusting qualifying order? Bars above the zero line = beats baseline;
    below = the baseline would have done better.
    """
    _apply_theme()
    _ensure_dir(os.path.dirname(output_path) or ".")

    metric_cols = ["global_ndcg", "ndcg_at_3", "mrr", "map", "map_at_3", "precision_at_3"]
    indexed = metrics_df.set_index("model")[metric_cols]

    if "qualifying_baseline" not in indexed.index:
        raise ValueError("qualifying_baseline row not found in metrics_df.")

    baseline = indexed.loc["qualifying_baseline"]
    challengers = [m for m in indexed.index if m != "qualifying_baseline"]
    delta_df = indexed.loc[challengers].subtract(baseline, axis=1)
    delta_df = delta_df.loc[delta_df.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    n_models = len(challengers)
    n_metrics = len(metric_cols)
    group_width = 0.7
    bar_width = group_width / max(n_models, 1)
    x = np.arange(n_metrics)

    # Scale label offset (and axis padding) to the actual data range so labels
    # stay attached to their bars regardless of whether deltas are ~0.2 or ~0.01.
    all_values = delta_df[metric_cols].to_numpy(dtype=float)
    data_min, data_max = float(all_values.min()), float(all_values.max())
    data_range = max(data_max - data_min, 1e-6)
    label_offset = max(data_range * 0.045, 0.0015)
    pad = data_range * 0.18
    ax.set_ylim(min(data_min - pad, -pad), max(data_max + pad, pad))

    for i, model in enumerate(delta_df.index):
        offsets = x - group_width / 2 + bar_width * i + bar_width / 2
        values = delta_df.loc[model, metric_cols].to_numpy(dtype=float)
        colors = [_color_for(model) if v >= 0 else "#5A2A2A" for v in values]
        bars = ax.bar(
            offsets, values, width=bar_width * 0.85,
            color=colors, edgecolor=BG_COLOR, linewidth=1.2,
            label=_label_for(model), zorder=3,
        )
        for rect, val in zip(bars, values):
            va = "bottom" if val >= 0 else "top"
            offset = label_offset if val >= 0 else -label_offset
            ax.text(
                rect.get_x() + rect.get_width() / 2, val + offset,
                f"{val:+.3f}", ha="center", va=va,
                fontsize=7.8, color=TEXT_COLOR,
            )

    ax.axhline(0, color=MUTED_TEXT, linewidth=1.2, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metric_cols], fontsize=9.5)
    _style_axes(
        ax, "Improvement Over Qualifying Baseline", "\u0394 Score vs. baseline",
        subtitle="Positive = model beats just trusting grid order \u00b7 negative = baseline wins",
    )
    legend = ax.legend(loc="best", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_metric_correlation_scatter(
    per_race_df: pd.DataFrame,
    output_path: str,
    x_metric: str = "global_ndcg",
    y_metric: str = "precision_at_3",
) -> str:
    """Scatter of two per-race metrics, colored by model.

    Averages can mask cases where a metric pair tells different stories, e.g.
    a model that ranks the full field well (high NDCG) but doesn't specifically
    nail the podium (lower Precision@3), or vice versa. Points clustering
    tightly along a diagonal mean the two metrics broadly agree for that model;
    a scattered cloud means they disagree race to race.
    """
    _apply_theme()
    _ensure_dir(os.path.dirname(output_path) or ".")

    markers = {"xgb_ranker": "o", "lgbm_ranker": "^", "qualifying_baseline": "s"}

    fig, ax = plt.subplots(figsize=(8.5, 7))
    for model_name, model_df in per_race_df.groupby("model", sort=False):
        ax.scatter(
            model_df[x_metric], model_df[y_metric],
            s=55, color=_color_for(model_name), alpha=0.75,
            marker=markers.get(model_name, "o"),
            edgecolor=BG_COLOR, linewidth=0.6,
            label=_label_for(model_name), zorder=3,
        )

    ax.set_xlabel(METRIC_LABELS.get(x_metric, x_metric), fontsize=10, color=MUTED_TEXT)
    _style_axes(
        ax, "Metric Agreement Across Races",
        METRIC_LABELS.get(y_metric, y_metric),
        subtitle=f"Each point is one race \u00b7 {METRIC_LABELS.get(x_metric, x_metric)} vs. {METRIC_LABELS.get(y_metric, y_metric)}",
    )
    legend = ax.legend(loc="lower right", fontsize=9, frameon=False)
    for text in legend.get_texts():
        text.set_color(TEXT_COLOR)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def generate_evaluation_plots(
    metrics_df: pd.DataFrame,
    per_race_df: pd.DataFrame,
    output_dir: str = "outputs/reports",
) -> list[str]:
    _ensure_dir(output_dir)

    paths = [
        plot_overall_metric_bars(metrics_df, os.path.join(output_dir, "model_comparison_bars.png")),
        plot_topk_focus(metrics_df, os.path.join(output_dir, "topk_focus_bars.png")),
        plot_per_race_trends(per_race_df, os.path.join(output_dir, "per_race_metric_trends.png")),
        plot_win_rate_breakdown(per_race_df, os.path.join(output_dir, "win_rate_breakdown.png")),
        plot_improvement_over_baseline(metrics_df, os.path.join(output_dir, "improvement_over_baseline.png")),
        plot_metric_correlation_scatter(per_race_df, os.path.join(output_dir, "metric_correlation_scatter.png")),
    ]
    return paths


def plot_results(results):
    """Backward-compatible wrapper retained for old imports."""
    if isinstance(results, dict):
        return results
    return {"results": results}