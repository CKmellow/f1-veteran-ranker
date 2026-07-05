"""Side-by-side evaluation for veteran F1 rankers.

This script compares:
- XGBRanker
- LGBMRanker
- Qualifying baseline

Outputs:
- Tabular metrics CSV for quick reporting.
- Per-race breakdown CSV for diagnostic analysis.
- Visualization PNGs in outputs/reports.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

try:
    from src.visualization.visualize_results import generate_evaluation_plots
except ModuleNotFoundError:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from src.visualization.visualize_results import generate_evaluation_plots

INPUT_PATH = "data/processed/veteran_training_matrix.csv"
XGB_MODEL_PATH = "models/f1_xgb_ranker.pkl"
LGB_MODEL_PATH = "models/f1_lgb_ranker.pkl"

METRICS_CSV_PATH = "outputs/reports/model_comparison_metrics.csv"
PER_RACE_CSV_PATH = "outputs/reports/model_per_race_breakdown.csv"
SUMMARY_JSON_PATH = "outputs/reports/model_comparison_summary.json"

FEATURE_COLUMNS = [
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

TARGET_COLUMN = "finish_position"
RELEVANCE_COLUMN = "relevance_score"
MODEL_TARGET_COLUMN = "model_relevance_score"


def _ensure_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _extract_estimator(obj: Any) -> Any:
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as file_obj:
        raw = pickle.load(file_obj)
    return _extract_estimator(raw)


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(FEATURE_COLUMNS + [TARGET_COLUMN, "season", "race_id", "driver_id", "quali_position"])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for evaluation: {missing}")


def _load_training_frame(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature matrix not found at: {path}")

    df = pd.read_csv(path)
    _validate_columns(df)

    numeric_cols = FEATURE_COLUMNS + [TARGET_COLUMN, "season"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, "season", "race_id"]).copy()
    df["season"] = df["season"].astype(int)
    df[RELEVANCE_COLUMN] = 15 - df[TARGET_COLUMN]
    df[MODEL_TARGET_COLUMN] = df[RELEVANCE_COLUMN].clip(lower=0)
    df = df.sort_values(["season", "race_id", "driver_id"]).reset_index(drop=True)
    return df


def _latest_season_test_split(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    if len(seasons) < 1:
        raise ValueError("No valid seasons in dataset.")
    test_season = int(seasons[-1])
    test_df = df[df["season"] == test_season].copy()
    if test_df.empty:
        raise ValueError(f"No rows available for latest season {test_season}.")
    test_df = test_df.sort_values(["race_id", "driver_id"]).reset_index(drop=True)
    return test_df, test_season


def _grouped_ndcg(df: pd.DataFrame, scores: np.ndarray, k: int | None = None) -> float:
    scored = df.copy()
    scored["pred_score"] = scores
    values: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        y_true = race_df[MODEL_TARGET_COLUMN].to_numpy(dtype=float)
        y_pred = race_df["pred_score"].to_numpy(dtype=float)
        race_k = min(k, len(race_df)) if k is not None else len(race_df)
        values.append(float(ndcg_score([y_true], [y_pred], k=race_k)))

    return float(np.mean(values))


def _grouped_mrr(df: pd.DataFrame, scores: np.ndarray) -> float:
    scored = df.copy()
    scored["pred_score"] = scores
    reciprocal_ranks: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        predicted = race_df.sort_values("pred_score", ascending=False).reset_index(drop=True)
        winner_index = predicted.index[predicted[RELEVANCE_COLUMN] == 14]
        reciprocal_ranks.append(0.0 if len(winner_index) == 0 else 1.0 / float(winner_index[0] + 1))

    return float(np.mean(reciprocal_ranks))


def _precision_at_3(df: pd.DataFrame, scores: np.ndarray) -> float:
    scored = df.copy()
    scored["pred_score"] = scores
    precisions: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        predicted_top3 = set(race_df.sort_values("pred_score", ascending=False).head(3)["driver_id"].tolist())
        actual_top3 = set(race_df.sort_values(TARGET_COLUMN, ascending=True).head(3)["driver_id"].tolist())
        precisions.append(len(predicted_top3.intersection(actual_top3)) / 3.0)

    return float(np.mean(precisions))


def _map_score(df: pd.DataFrame, scores: np.ndarray, k: int | None = None) -> float:
    scored = df.copy()
    scored["pred_score"] = scores
    average_precisions: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        ranked = race_df.sort_values("pred_score", ascending=False).reset_index(drop=True)
        relevant_ids = set(race_df.sort_values(TARGET_COLUMN, ascending=True)["driver_id"].tolist())

        if k is not None:
            ranked = ranked.head(k)
            max_relevant = min(k, len(relevant_ids))
        else:
            max_relevant = len(relevant_ids)

        if max_relevant == 0:
            average_precisions.append(0.0)
            continue

        hit_count = 0
        precision_sum = 0.0
        for idx, driver_id in enumerate(ranked["driver_id"].tolist(), start=1):
            if driver_id in relevant_ids:
                hit_count += 1
                precision_sum += hit_count / float(idx)

        average_precisions.append(precision_sum / float(max_relevant))

    return float(np.mean(average_precisions))


def _qualifying_baseline_scores(df: pd.DataFrame) -> np.ndarray:
    return -df["quali_position"].to_numpy(dtype=float)


def _evaluate_scores(df: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    return {
        "global_ndcg": _grouped_ndcg(df, scores),
        "ndcg_at_3": _grouped_ndcg(df, scores, k=3),
        "mrr": _grouped_mrr(df, scores),
        "map": _map_score(df, scores),
        "map_at_3": _map_score(df, scores, k=3),
        "precision_at_3": _precision_at_3(df, scores),
    }


def _per_race_breakdown(df: pd.DataFrame, score_map: dict[str, np.ndarray]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    race_groups = list(df.groupby("race_id", sort=False))

    for race_id, race_df in race_groups:
        for model_name, scores in score_map.items():
            race_scores = scores[race_df.index.to_numpy()]
            metrics = _evaluate_scores(race_df, race_scores)
            rows.append(
                {
                    "race_id": race_id,
                    "season": int(race_df["season"].iloc[0]),
                    "rows": int(len(race_df)),
                    "model": model_name,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def evaluate_models(
    input_path: str = INPUT_PATH,
    xgb_model_path: str = XGB_MODEL_PATH,
    lgb_model_path: str = LGB_MODEL_PATH,
) -> dict[str, Any]:
    df = _load_training_frame(input_path)
    test_df, test_season = _latest_season_test_split(df)

    xgb_model = _load_model(xgb_model_path)
    lgb_model = _load_model(lgb_model_path)

    x_frame = test_df[FEATURE_COLUMNS]
    xgb_scores = xgb_model.predict(x_frame)
    lgb_scores = lgb_model.predict(x_frame)
    baseline_scores = _qualifying_baseline_scores(test_df)

    metrics = {
        "xgb_ranker": _evaluate_scores(test_df, xgb_scores),
        "lgbm_ranker": _evaluate_scores(test_df, lgb_scores),
        "qualifying_baseline": _evaluate_scores(test_df, baseline_scores),
    }

    metrics_rows = []
    for model_name, model_metrics in metrics.items():
        row = {"model": model_name, **model_metrics}
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("global_ndcg", ascending=False).reset_index(drop=True)

    per_race_df = _per_race_breakdown(
        test_df,
        {
            "xgb_ranker": np.asarray(xgb_scores),
            "lgbm_ranker": np.asarray(lgb_scores),
            "qualifying_baseline": np.asarray(baseline_scores),
        },
    )

    _ensure_dir(METRICS_CSV_PATH)
    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    per_race_df.to_csv(PER_RACE_CSV_PATH, index=False)

    summary = {
        "test_season": test_season,
        "rows_evaluated": int(len(test_df)),
        "races_evaluated": int(test_df["race_id"].nunique()),
        "metrics": metrics,
        "artifacts": {
            "metrics_csv": METRICS_CSV_PATH,
            "per_race_csv": PER_RACE_CSV_PATH,
        },
    }

    _ensure_dir(SUMMARY_JSON_PATH)
    with open(SUMMARY_JSON_PATH, "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2)

    generated_plots = generate_evaluation_plots(metrics_df, per_race_df, output_dir="outputs/reports")
    summary["artifacts"]["plots"] = generated_plots

    print("Model comparison complete.")
    print(f"Evaluated season: {test_season}")
    print(f"Metrics table: {METRICS_CSV_PATH}")
    print(f"Per-race table: {PER_RACE_CSV_PATH}")
    print(f"Summary JSON: {SUMMARY_JSON_PATH}")
    for plot_path in generated_plots:
        print(f"Plot: {plot_path}")

    return summary


if __name__ == "__main__":
    evaluate_models()
