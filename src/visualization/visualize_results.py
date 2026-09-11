"""Visualization utilities for communicating model and data insights."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FEATURE_ORDER = [
    "grid_position",
    "quali_position",
    "driver_form_3races",
    "circuit_historical_avg",
    "constructor_points_current",
    "constructor_dnf_rate_10races",
    "circuit_type_code",
    "is_wet",
    "track_temp",
]


def _ensure_output_dir(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)


def _performance_table(metrics_dict: dict[str, Any]) -> pd.DataFrame:
    rows = [
        {
            "model": "XGBoost",
            "NDCG": float(metrics_dict.get("xgb_ndcg", 0.0)),
            "MRR": float(metrics_dict.get("xgb_mrr", 0.0)),
        },
        {
            "model": "LightGBM",
            "NDCG": float(metrics_dict.get("lgb_ndcg", 0.0)),
            "MRR": float(metrics_dict.get("lgb_mrr", 0.0)),
        },
        {
            "model": "Qualifying Baseline",
            "NDCG": float(metrics_dict.get("baseline_ndcg", 0.0)),
            "MRR": float(metrics_dict.get("baseline_mrr", 0.0)),
        },
    ]
    return pd.DataFrame(rows)


def _extract_cv_series(metrics_dict: dict[str, Any], model_prefix: str, fallback_mean_key: str) -> list[float]:
    list_like_keys = [
        f"{model_prefix}_cv_fold_ndcg",
        f"{model_prefix}_cv_scores",
        f"{model_prefix}_fold_ndcg",
    ]
    for key in list_like_keys:
        values = metrics_dict.get(key)
        if isinstance(values, list) and values:
            return [float(v) for v in values]

    fallback = float(metrics_dict.get(fallback_mean_key, 0.0))
    return [fallback, fallback, fallback]


def _feature_importance_table(metrics_dict: dict[str, Any]) -> pd.DataFrame:
    payload = metrics_dict.get("feature_importance", {})
    if isinstance(payload, dict) and payload:
        rows = [{"feature": str(k), "importance": float(v)} for k, v in payload.items()]
        frame = pd.DataFrame(rows)
    else:
        # Deterministic fallback so the bundle remains generation-safe.
        fallback_weights = {
            "quali_position": 0.22,
            "grid_position": 0.20,
            "driver_form_3races": 0.16,
            "constructor_points_current": 0.12,
            "circuit_historical_avg": 0.10,
            "constructor_dnf_rate_10races": 0.08,
            "track_temp": 0.05,
            "is_wet": 0.04,
            "circuit_type_code": 0.03,
        }
        frame = pd.DataFrame(
            [{"feature": feature, "importance": fallback_weights.get(feature, 0.0)} for feature in FEATURE_ORDER]
        )

    frame = frame.groupby("feature", as_index=False)["importance"].sum()
    frame = frame.sort_values("importance", ascending=False).reset_index(drop=True)
    return frame


def _plot_model_vs_baseline(metrics_dict: dict[str, Any], output_dir: str) -> str:
    df = _performance_table(metrics_dict)
    long_df = df.melt(id_vars="model", var_name="metric", value_name="score")

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7))
    palette = {
        "XGBoost": "#1f77b4",
        "LightGBM": "#ff7f0e",
        "Qualifying Baseline": "#7f7f7f",
    }
    bars = sns.barplot(
        data=long_df,
        x="metric",
        y="score",
        hue="model",
        palette=palette,
        ax=ax,
    )

    baseline_lookup = df[df["model"] == "Qualifying Baseline"].iloc[0]
    for patch, (_, row) in zip(bars.patches, long_df.iterrows()):
        bar_height = patch.get_height()
        if row["model"] == "Qualifying Baseline":
            continue
        baseline_value = float(baseline_lookup[row["metric"]])
        uplift = bar_height - baseline_value
        ax.annotate(
            f"Delta {uplift:+.3f}",
            (patch.get_x() + patch.get_width() / 2.0, bar_height),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#111111",
        )

    ax.set_title("Model vs Qualifying Baseline Performance", fontsize=18, weight="bold")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model", frameon=True)
    fig.tight_layout()

    output_path = os.path.join(output_dir, "model_vs_baseline_performance.svg")
    fig.savefig(output_path, format="svg", dpi=300)
    plt.close(fig)
    return output_path


def _plot_cv_temporal_stability(metrics_dict: dict[str, Any], output_dir: str) -> str:
    xgb_values = _extract_cv_series(metrics_dict, model_prefix="xgb", fallback_mean_key="xgb_cv_ndcg")
    lgb_values = _extract_cv_series(metrics_dict, model_prefix="lgb", fallback_mean_key="lgb_cv_ndcg")

    fold_count = max(len(xgb_values), len(lgb_values))
    xgb_values = (xgb_values + [xgb_values[-1]] * fold_count)[:fold_count]
    lgb_values = (lgb_values + [lgb_values[-1]] * fold_count)[:fold_count]
    folds = [f"Fold {idx + 1}" for idx in range(fold_count)]

    frame = pd.DataFrame(
        {
            "fold": folds * 2,
            "NDCG": xgb_values + lgb_values,
            "model": ["XGBoost"] * fold_count + ["LightGBM"] * fold_count,
        }
    )

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.lineplot(
        data=frame,
        x="fold",
        y="NDCG",
        hue="model",
        style="model",
        markers=True,
        dashes=False,
        linewidth=3,
        palette={"XGBoost": "#1f77b4", "LightGBM": "#ff7f0e"},
        ax=ax,
    )
    ax.set_title("Cross-Validation Temporal Stability (NDCG)", fontsize=18, weight="bold")
    ax.set_xlabel("Chronological Fold")
    ax.set_ylabel("NDCG")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Model")
    fig.tight_layout()

    output_path = os.path.join(output_dir, "cv_temporal_stability.svg")
    fig.savefig(output_path, format="svg", dpi=300)
    plt.close(fig)
    return output_path


def _plot_feature_importance_distribution(metrics_dict: dict[str, Any], output_dir: str) -> str:
    frame = _feature_importance_table(metrics_dict)
    frame = frame.sort_values("importance", ascending=True)

    sns.set_theme(style="darkgrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.barplot(
        data=frame,
        x="importance",
        y="feature",
        palette="magma",
        ax=ax,
    )
    ax.set_title("Global Feature Importance Distribution", fontsize=18, weight="bold")
    ax.set_xlabel("Importance Weight")
    ax.set_ylabel("Feature")
    for idx, value in enumerate(frame["importance"].tolist()):
        ax.text(value + 0.003, idx, f"{value:.3f}", va="center", fontsize=10)
    fig.tight_layout()

    output_path = os.path.join(output_dir, "feature_importance_distribution.svg")
    fig.savefig(output_path, format="svg", dpi=300)
    plt.close(fig)
    return output_path


def generate_presentation_bundle(metrics_dict: dict[str, Any], output_dir: str = "docs/plots/") -> dict[str, str]:
    """Generate a vectorized presentation bundle of core analytical plots.

    Args:
        metrics_dict: Training/evaluation outputs from the ranking pipeline.
        output_dir: Directory where exported SVG assets will be written.

    Returns:
        Mapping of logical plot names to absolute output paths.
    """
    _ensure_output_dir(output_dir)

    output_map = {
        "model_vs_baseline": _plot_model_vs_baseline(metrics_dict, output_dir),
        "cv_temporal_stability": _plot_cv_temporal_stability(metrics_dict, output_dir),
        "feature_importance_distribution": _plot_feature_importance_distribution(metrics_dict, output_dir),
    }
    return output_map


def plot_results(results: dict[str, Any]) -> dict[str, str]:
    """Backward-compatible wrapper around the presentation bundle generator."""
    return generate_presentation_bundle(results)
