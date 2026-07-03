"""Production-grade training script for F1 veteran rankers.

This module implements:
- Temporal recency weighting for training samples.
- Purged, chronological, group-aware expanding CV across available historical seasons.
- Automated hyperparameter search for XGBRanker and LGBMRanker.
- Final out-of-sample evaluation on the latest available season with top-heavy metrics.
- Persisted artifacts for both optimized models and SHAP summaries.
"""

from __future__ import annotations

import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMRanker
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker

INPUT_PATH = "data/processed/veteran_training_matrix.csv"

XGB_MODEL_PATH = "models/f1_xgb_ranker.pkl"
LGB_MODEL_PATH = "models/f1_lgb_ranker.pkl"

XGB_SHAP_PATH = "outputs/reports/xgb_shap_summary.png"
LGB_SHAP_PATH = "outputs/reports/lgb_shap_summary.png"

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
WEIGHT_COLUMN = "temporal_sample_weight"

MIN_CV_FOLDS = 1


def _ensure_dirs(*paths: str) -> None:
    for path in paths:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)


def _validate_columns(df: pd.DataFrame) -> None:
    required = set(
        FEATURE_COLUMNS
        + [TARGET_COLUMN, "season", "race_id", "driver_id"]
    )
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns for training: {missing}")


def _compute_temporal_weights(df: pd.DataFrame, decay_alpha: float = 0.5) -> pd.Series:
    season_max = int(df["season"].max())
    season_gap = season_max - df["season"].astype(int)
    weights = np.exp(-decay_alpha * season_gap)
    return pd.Series(weights, index=df.index, dtype=float)


def _prepare_training_frame(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training matrix not found at {path}")

    df = pd.read_csv(path)
    _validate_columns(df)

    numeric_columns = FEATURE_COLUMNS + [TARGET_COLUMN, "season"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, "season"]).copy()
    df["season"] = df["season"].astype(int)
    df[RELEVANCE_COLUMN] = 15 - df[TARGET_COLUMN]
    # LightGBM lambdarank requires non-negative labels.
    df[MODEL_TARGET_COLUMN] = df[RELEVANCE_COLUMN].clip(lower=0)

    # Keep race rows contiguous and chronologically ordered.
    df = df.sort_values(["season", "race_id", "driver_id"]).reset_index(drop=True)

    # Temporal sample weighting layer: down-weight older seasons exponentially.
    df[WEIGHT_COLUMN] = _compute_temporal_weights(df)
    return df


def _get_train_and_test_frames(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int, list[int]]:
    seasons = sorted(df["season"].dropna().astype(int).unique().tolist())
    if len(seasons) < 2:
        raise ValueError("Need at least two seasons to create train/test split.")

    max_season = int(seasons[-1])
    train_seasons = seasons[:-1]

    train_df = df[df["season"] < max_season].copy()
    test_df = df[df["season"] == max_season].copy()

    if train_df.empty:
        raise ValueError(f"No training rows found for seasons before {max_season}.")
    if test_df.empty:
        raise ValueError(
            f"No test rows found for latest season {max_season}. "
            "Ingest latest rounds, rebuild matrix, and rerun training."
        )
    if len(train_seasons) < 2:
        raise ValueError(
            f"Insufficient historical seasons for expanding CV before {max_season}. "
            f"Need at least 2 train seasons, found {len(train_seasons)}."
        )

    train_df = train_df.sort_values(["season", "race_id", "driver_id"]).reset_index(drop=True)
    test_df = test_df.sort_values(["season", "race_id", "driver_id"]).reset_index(drop=True)
    return train_df, test_df, max_season, train_seasons


def _group_sizes(df: pd.DataFrame) -> list[int]:
    return df.groupby("race_id", sort=False).size().astype(int).tolist()


def _group_temporal_weights(df: pd.DataFrame) -> np.ndarray:
    # XGBoost ranker with group list expects one weight per query group.
    return (
        df.groupby("race_id", sort=False)[WEIGHT_COLUMN]
        .mean()
        .to_numpy(dtype=float)
    )


def _chronological_cv_folds(
    train_df: pd.DataFrame,
    train_seasons: list[int],
    max_season: int,
) -> list[tuple[pd.DataFrame, pd.DataFrame, str]]:
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]] = []
    validation_seasons = train_seasons[1:]

    for valid_season in validation_seasons:
        fold_train = train_df[train_df["season"] < valid_season].copy()
        fold_valid = train_df[train_df["season"] == valid_season].copy()
        if fold_train.empty or fold_valid.empty:
            continue

        fold_train = fold_train.sort_values(["season", "race_id", "driver_id"]).reset_index(drop=True)
        fold_valid = fold_valid.sort_values(["season", "race_id", "driver_id"]).reset_index(drop=True)
        fold_name = f"train_{fold_train['season'].min()}_{fold_train['season'].max()}_val_{valid_season}"
        folds.append((fold_train, fold_valid, fold_name))

    if len(folds) < MIN_CV_FOLDS:
        raise ValueError(
            "Insufficient data for required expanding folds. "
            f"Expected at least {MIN_CV_FOLDS} folds before test season {max_season}, got {len(folds)}."
        )

    if folds:
        final_valid_season = int(folds[-1][1]["season"].iloc[0])
        expected_final = max_season - 1
        if final_valid_season != expected_final:
            raise ValueError(
                "Temporal CV alignment error: "
                f"final validation season is {final_valid_season}, expected {expected_final}."
            )

    return folds


def _compute_grouped_ndcg(df: pd.DataFrame, pred_scores: np.ndarray, k: int | None = None) -> float:
    scored = df.copy()
    scored["pred_score"] = pred_scores
    ndcgs: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        y_true = race_df[MODEL_TARGET_COLUMN].to_numpy(dtype=float)
        y_pred = race_df["pred_score"].to_numpy(dtype=float)
        race_k = min(k, len(race_df)) if k is not None else len(race_df)
        ndcgs.append(float(ndcg_score([y_true], [y_pred], k=race_k)))

    return float(np.mean(ndcgs))


def _compute_grouped_mrr(df: pd.DataFrame, pred_scores: np.ndarray) -> float:
    scored = df.copy()
    scored["pred_score"] = pred_scores
    reciprocal_ranks: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        predicted = race_df.sort_values("pred_score", ascending=False).reset_index(drop=True)
        winner_index = predicted.index[predicted[RELEVANCE_COLUMN] == 14]
        if len(winner_index) == 0:
            reciprocal_ranks.append(0.0)
        else:
            reciprocal_ranks.append(1.0 / float(winner_index[0] + 1))

    return float(np.mean(reciprocal_ranks))


def _compute_precision_at_k(df: pd.DataFrame, pred_scores: np.ndarray, k: int = 3) -> float:
    scored = df.copy()
    scored["pred_score"] = pred_scores
    precisions: list[float] = []

    for _, race_df in scored.groupby("race_id", sort=False):
        top_k = race_df.sort_values("pred_score", ascending=False).head(k)
        true_podium_ids = set(race_df.sort_values(TARGET_COLUMN, ascending=True).head(k)["driver_id"].tolist())
        predicted_ids = set(top_k["driver_id"].tolist())
        hit_count = len(predicted_ids.intersection(true_podium_ids))
        precisions.append(hit_count / float(k))

    return float(np.mean(precisions))


def _qualifying_baseline_scores(df: pd.DataFrame) -> np.ndarray:
    # Lower qualifying position is better, so negate for descending rank score.
    return -df["quali_position"].to_numpy(dtype=float)


def _xgb_param_grid() -> list[dict[str, float | int]]:
    return [
        {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "reg_lambda": reg_lambda,
        }
        for max_depth, learning_rate, reg_lambda in product(
            [3, 5, 7],
            [0.01, 0.05, 0.1],
            [1, 10, 100],
        )
    ]


def _lgb_param_grid() -> list[dict[str, float | int]]:
    return [
        {
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
        }
        for num_leaves, learning_rate, min_child_samples in product(
            [7, 15, 31],
            [0.01, 0.05, 0.1],
            [5, 10, 20],
        )
    ]


def _fit_xgb(
    train_df: pd.DataFrame,
    params: dict[str, float | int],
) -> XGBRanker:
    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=300,
        random_state=42,
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        reg_lambda=float(params["reg_lambda"]),
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[MODEL_TARGET_COLUMN],
        group=_group_sizes(train_df),
        sample_weight=_group_temporal_weights(train_df),
        verbose=False,
    )
    return model


def _fit_lgb(
    train_df: pd.DataFrame,
    params: dict[str, float | int],
) -> LGBMRanker:
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=300,
        random_state=42,
        num_leaves=int(params["num_leaves"]),
        learning_rate=float(params["learning_rate"]),
        min_child_samples=int(params["min_child_samples"]),
    )
    model.fit(
        train_df[FEATURE_COLUMNS],
        train_df[MODEL_TARGET_COLUMN],
        group=_group_sizes(train_df),
        sample_weight=train_df[WEIGHT_COLUMN].to_numpy(dtype=float),
    )
    return model


def _search_best_params(
    model_name: str,
    folds: list[tuple[pd.DataFrame, pd.DataFrame, str]],
) -> tuple[dict[str, float | int], float]:
    if model_name == "xgb":
        param_grid = _xgb_param_grid()
    elif model_name == "lgb":
        param_grid = _lgb_param_grid()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    best_params: dict[str, float | int] | None = None
    best_score = -np.inf

    for params in param_grid:
        fold_scores: list[float] = []

        for fold_train, fold_valid, _ in folds:
            if model_name == "xgb":
                model = _fit_xgb(fold_train, params)
            else:
                model = _fit_lgb(fold_train, params)

            valid_pred = model.predict(fold_valid[FEATURE_COLUMNS])
            fold_score = _compute_grouped_ndcg(fold_valid, valid_pred)
            fold_scores.append(fold_score)

        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params

    if best_params is None:
        raise RuntimeError(f"No best parameters found for model {model_name}")

    return best_params, best_score


def _save_shap_summary(model, x_frame: pd.DataFrame, output_path: str, model_name: str) -> None:
    _ensure_dirs(output_path)

    # Use a bounded sample to keep SHAP generation stable in CI.
    x_sample = x_frame.sample(n=min(len(x_frame), 1000), random_state=42) if len(x_frame) > 1000 else x_frame

    if model_name == "xgb":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, x_sample, show=False)
    else:
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_sample)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, x_sample, show=False)
        except Exception:
            explainer = shap.Explainer(model.predict, x_sample)
            explanation = explainer(x_sample)
            plt.figure(figsize=(10, 6))
            shap.plots.beeswarm(explanation, max_display=len(FEATURE_COLUMNS), show=False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def train_rankers(input_path: str = INPUT_PATH) -> dict[str, float]:
    df = _prepare_training_frame(input_path)
    train_df, test_df, test_season, train_seasons = _get_train_and_test_frames(df)
    folds = _chronological_cv_folds(train_df, train_seasons, test_season)

    xgb_best_params, xgb_cv_ndcg = _search_best_params("xgb", folds)
    lgb_best_params, lgb_cv_ndcg = _search_best_params("lgb", folds)

    xgb_model = _fit_xgb(train_df, xgb_best_params)
    lgb_model = _fit_lgb(train_df, lgb_best_params)

    xgb_test_pred = xgb_model.predict(test_df[FEATURE_COLUMNS])
    lgb_test_pred = lgb_model.predict(test_df[FEATURE_COLUMNS])
    baseline_test_pred = _qualifying_baseline_scores(test_df)

    xgb_ndcg = _compute_grouped_ndcg(test_df, xgb_test_pred)
    lgb_ndcg = _compute_grouped_ndcg(test_df, lgb_test_pred)
    baseline_ndcg = _compute_grouped_ndcg(test_df, baseline_test_pred)

    xgb_mrr = _compute_grouped_mrr(test_df, xgb_test_pred)
    lgb_mrr = _compute_grouped_mrr(test_df, lgb_test_pred)
    baseline_mrr = _compute_grouped_mrr(test_df, baseline_test_pred)

    xgb_ndcg_at_3 = _compute_grouped_ndcg(test_df, xgb_test_pred, k=3)
    lgb_ndcg_at_3 = _compute_grouped_ndcg(test_df, lgb_test_pred, k=3)

    xgb_precision_at_3 = _compute_precision_at_k(test_df, xgb_test_pred, k=3)
    lgb_precision_at_3 = _compute_precision_at_k(test_df, lgb_test_pred, k=3)

    _save_shap_summary(xgb_model, test_df[FEATURE_COLUMNS], XGB_SHAP_PATH, model_name="xgb")
    _save_shap_summary(lgb_model, test_df[FEATURE_COLUMNS], LGB_SHAP_PATH, model_name="lgb")

    _ensure_dirs(XGB_MODEL_PATH, LGB_MODEL_PATH)
    with open(XGB_MODEL_PATH, "wb") as file_obj:
        pickle.dump(
            {
                "model": xgb_model,
                "features": FEATURE_COLUMNS,
                "target": RELEVANCE_COLUMN,
                "model_target": MODEL_TARGET_COLUMN,
                "best_params": xgb_best_params,
                "cv_group_ndcg": xgb_cv_ndcg,
            },
            file_obj,
        )

    with open(LGB_MODEL_PATH, "wb") as file_obj:
        pickle.dump(
            {
                "model": lgb_model,
                "features": FEATURE_COLUMNS,
                "target": RELEVANCE_COLUMN,
                "model_target": MODEL_TARGET_COLUMN,
                "best_params": lgb_best_params,
                "cv_group_ndcg": lgb_cv_ndcg,
            },
            file_obj,
        )

    fold_labels = [fold_name for _, _, fold_name in folds]
    print("Chronological CV complete over folds:")
    print(", ".join(fold_labels))
    print("Best CV NDCG by model:")
    print(f"XGBRanker params={xgb_best_params} | mean_group_ndcg={xgb_cv_ndcg:.6f}")
    print(f"LGBMRanker params={lgb_best_params} | mean_group_ndcg={lgb_cv_ndcg:.6f}")
    print(f"Out-of-sample test metrics (season {test_season}):")
    print(f"Qualifying Baseline Global NDCG: {baseline_ndcg:.6f} | Global MRR: {baseline_mrr:.6f}")
    print(f"XGBRanker Global NDCG: {xgb_ndcg:.6f} | Global MRR: {xgb_mrr:.6f}")
    print(
        f"XGB vs Qualifying Baseline Uplift -> NDCG: {xgb_ndcg - baseline_ndcg:+.6f} | "
        f"MRR: {xgb_mrr - baseline_mrr:+.6f}"
    )
    print(f"XGBRanker NDCG@3: {xgb_ndcg_at_3:.6f} | Precision@3: {xgb_precision_at_3:.6f}")
    print(f"LGBMRanker Global NDCG: {lgb_ndcg:.6f} | Global MRR: {lgb_mrr:.6f}")
    print(
        f"LGBM vs Qualifying Baseline Uplift -> NDCG: {lgb_ndcg - baseline_ndcg:+.6f} | "
        f"MRR: {lgb_mrr - baseline_mrr:+.6f}"
    )
    print(f"LGBMRanker NDCG@3: {lgb_ndcg_at_3:.6f} | Precision@3: {lgb_precision_at_3:.6f}")
    print(f"XGB model saved to: {XGB_MODEL_PATH}")
    print(f"LGB model saved to: {LGB_MODEL_PATH}")
    print(f"XGB SHAP summary saved to: {XGB_SHAP_PATH}")
    print(f"LGB SHAP summary saved to: {LGB_SHAP_PATH}")

    return {
        "xgb_cv_ndcg": xgb_cv_ndcg,
        "lgb_cv_ndcg": lgb_cv_ndcg,
        "baseline_ndcg": baseline_ndcg,
        "baseline_mrr": baseline_mrr,
        "xgb_ndcg": xgb_ndcg,
        "xgb_mrr": xgb_mrr,
        "xgb_ndcg_at_3": xgb_ndcg_at_3,
        "xgb_precision_at_3": xgb_precision_at_3,
        "lgb_ndcg": lgb_ndcg,
        "lgb_mrr": lgb_mrr,
        "lgb_ndcg_at_3": lgb_ndcg_at_3,
        "lgb_precision_at_3": lgb_precision_at_3,
    }


if __name__ == "__main__":
    custom_input_path = os.environ.get("VETERAN_MATRIX_INPUT_PATH", INPUT_PATH)
    train_rankers(input_path=custom_input_path)
