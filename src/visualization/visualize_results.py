"""Visualization utilities for ranker evaluation outputs."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _style_axes(ax, title: str, y_label: str) -> None:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label)
    ax.grid(axis="y", alpha=0.25)


def _sorted_model_order(metrics_df: pd.DataFrame) -> list[str]:
    ranked = metrics_df.sort_values("global_ndcg", ascending=False)
    return ranked["model"].tolist()


def plot_overall_metric_bars(metrics_df: pd.DataFrame, output_path: str) -> str:
    _ensure_dir(os.path.dirname(output_path) or ".")

    metric_cols = ["global_ndcg", "ndcg_at_3", "mrr", "map", "map_at_3", "precision_at_3"]
    plot_df = metrics_df.set_index("model")[metric_cols].copy()
    plot_df = plot_df.loc[_sorted_model_order(metrics_df)]

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", ax=ax)
    _style_axes(ax, "Model Comparison: Core Ranking Metrics", "Score")
    ax.set_xlabel("Model")
    ax.legend(loc="best", ncol=3, fontsize=8)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_topk_focus(metrics_df: pd.DataFrame, output_path: str) -> str:
    _ensure_dir(os.path.dirname(output_path) or ".")

    metric_cols = ["ndcg_at_3", "map_at_3", "precision_at_3"]
    plot_df = metrics_df.set_index("model")[metric_cols].copy()
    plot_df = plot_df.loc[_sorted_model_order(metrics_df)]

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax)
    _style_axes(ax, "Top-3 Quality Focus", "Score")
    ax.set_xlabel("Model")
    ax.legend(loc="best")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def plot_per_race_trends(per_race_df: pd.DataFrame, output_path: str) -> str:
    _ensure_dir(os.path.dirname(output_path) or ".")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    metrics = ["global_ndcg", "mrr", "precision_at_3"]
    titles = ["Per-Race NDCG", "Per-Race MRR", "Per-Race Precision@3"]

    for ax, metric, title in zip(axes, metrics, titles):
        for model_name, model_df in per_race_df.groupby("model", sort=False):
            ordered = model_df.sort_values("race_id")
            ax.plot(ordered["race_id"], ordered[metric], marker="o", linewidth=1.8, label=model_name)
        _style_axes(ax, title, "Score")
        ax.set_xlabel("Race ID")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
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
    ]
    return paths


def plot_results(results):
    """Backward-compatible wrapper retained for old imports."""
    if isinstance(results, dict):
        return results
    return {"results": results}
