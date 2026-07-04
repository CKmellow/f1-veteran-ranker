import os
import pickle

import pandas as pd
import streamlit as st

APP_TITLE = "Explainable Rank Prediction of F1 Veterans"

XGB_MODEL_PATH = "models/f1_xgb_ranker.pkl"
LGB_MODEL_PATH = "models/f1_lgb_ranker.pkl"
FEATURE_MATRIX_PATH = "data/processed/veteran_training_matrix.csv"
FULL_GRID_PATH = "data/raw/f1_canonical_master.csv"

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

ROLLING_COLUMNS = [
    "driver_form_3races",
    "circuit_historical_avg",
    "constructor_points_current",
    "constructor_dnf_rate_10races",
]

CIRCUIT_TYPE_CODE = {
    "Power/High-Speed": 0,
    "High-Downforce": 1,
    "Balanced/Technical": 2,
    "Street Circuit": 3,
}

DRIVER_ORDER = [
    "verstappen",
    "hamilton",
    "russell",
    "leclerc",
    "norris",
    "sainz",
    "albon",
    "ocon",
    "gasly",
    "hulkenberg",
    "alonso",
    "stroll",
    "bottas",
    "perez",
]

DRIVER_LABELS = {
    "max_verstappen": "M. VERSTAPPEN",
    "verstappen": "M. VERSTAPPEN",
    "hamilton": "L. HAMILTON",
    "russell": "G. RUSSELL",
    "leclerc": "C. LECLERC",
    "norris": "L. NORRIS",
    "sainz": "C. SAINZ",
    "albon": "A. ALBON",
    "ocon": "E. OCON",
    "gasly": "P. GASLY",
    "hulkenberg": "N. HULKENBERG",
    "alonso": "F. ALONSO",
    "stroll": "L. STROLL",
    "bottas": "V. BOTTAS",
    "perez": "S. PEREZ",
}

TEAM_META = {
    "verstappen": {"team": "Red Bull Racing", "prefix": "RBR", "brand": "RED BULL", "color": "#0600EF"},
    "perez": {"team": "Red Bull Racing", "prefix": "RBR", "brand": "RED BULL", "color": "#0600EF"},
    "hamilton": {"team": "Scuderia Ferrari", "prefix": "FER", "brand": "FERRARI", "color": "#EF1A2D"},
    "leclerc": {"team": "Scuderia Ferrari", "prefix": "FER", "brand": "FERRARI", "color": "#EF1A2D"},
    "russell": {"team": "Mercedes-AMG", "prefix": "MER", "brand": "MERCEDES", "color": "#27F4D2"},
    "norris": {"team": "McLaren", "prefix": "MCL", "brand": "MCLAREN", "color": "#FF8000"},
    "alonso": {"team": "Aston Martin", "prefix": "AST", "brand": "ASTON MARTIN", "color": "#229971"},
    "stroll": {"team": "Aston Martin", "prefix": "AST", "brand": "ASTON MARTIN", "color": "#229971"},
    "albon": {"team": "Williams", "prefix": "WIL", "brand": "WILLIAMS", "color": "#47C7FC"},
    "sainz": {"team": "Williams", "prefix": "WIL", "brand": "WILLIAMS", "color": "#47C7FC"},
    "gasly": {"team": "Alpine", "prefix": "ALP", "brand": "ALPINE", "color": "#0093CC"},
    "ocon": {"team": "Haas", "prefix": "HAA", "brand": "HAAS", "color": "#B6BABD"},
    "hulkenberg": {"team": "Audi", "prefix": "AUD", "brand": "AUDI", "color": "#C92D4B"},
    "bottas": {"team": "Cadillac", "prefix": "CAD", "brand": "CADILLAC", "color": "#1D3557"},
}

RACE_2026_MAP = {
    1: "Australia",
    2: "China",
    3: "Japan",
    4: "Miami",
    5: "Canada",
    6: "Monaco",
    7: "Barcelona",
    8: "Austria",
}

UPCOMING_GP_PROFILES = {
    "Silverstone - British GP": {
        "circuit_label": "Power/High-Speed",
        "is_wet": 0,
        "track_temp": 22.0,
    },
    "Spa - Belgian GP": {
        "circuit_label": "Power/High-Speed",
        "is_wet": 1,
        "track_temp": 18.0,
    },
    "Zandvoort - Dutch GP": {
        "circuit_label": "Balanced/Technical",
        "is_wet": 0,
        "track_temp": 21.0,
    },
    "Monza - Italian GP": {
        "circuit_label": "Power/High-Speed",
        "is_wet": 0,
        "track_temp": 27.0,
    },
    "Singapore - Singapore GP": {
        "circuit_label": "Street Circuit",
        "is_wet": 0,
        "track_temp": 30.0,
    },
}


def style_app():
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;700&family=Rajdhani:wght@500;700&display=swap');

html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif;
    color: var(--text-color);
    background-color: #060b14;
}

.stApp {
    background: radial-gradient(circle at 8% 8%, #12253d 0%, #0a1321 40%, #060b14 100%) !important;
    color: var(--text-color) !important;
}

h1, h2, h3 {
    color: #FFFFFF !important;
  letter-spacing: 0.04em;
}

p, label, span, div {
    color: var(--text-color);
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p,
[data-testid="stHeader"] {
        color: #FFFFFF !important;
}

[data-testid="stSidebar"] {
    background: #0b1320 !important;
    border-right: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
}

[data-testid="stSidebar"] * {
    color: var(--text-color) !important;
}

[data-testid="stDataFrame"] {
    background: color-mix(in srgb, #0f1726 90%, black) !important;
    border: 1px solid color-mix(in srgb, var(--text-color) 14%, transparent);
    border-radius: 10px;
}

.telemetry-card {
  border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
    background: color-mix(in srgb, #111a2a 86%, black) !important;
  border-radius: 12px;
  padding: 0.95rem 1rem;
}

.telemetry-badge {
  display: inline-block;
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--text-color) 24%, transparent);
    background: color-mix(in srgb, #17263b 85%, black) !important;
  padding: 0.24rem 0.62rem;
  font-size: 0.78rem;
  letter-spacing: 0.04em;
  font-weight: 700;
    color: #FFFFFF !important;
}

.podium-card {
  border-radius: 12px;
  overflow: hidden;
}

.podium-body {
  padding: 0.78rem 0.82rem 0.9rem;
}

.podium-rank {
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.03em;
  font-weight: 700;
    color: #FFFFFF !important;
}

.podium-driver {
  font-size: 1.08rem;
  letter-spacing: 0.02em;
  font-weight: 700;
    color: #FFFFFF !important;
}

.podium-team {
  font-size: 0.9rem;
    color: #FFFFFF !important;
}

.podium-medal {
  margin-top: 0.42rem;
  display: inline-block;
  border-radius: 999px;
  padding: 0.2rem 0.56rem;
  font-size: 0.78rem;
  font-weight: 700;
    color: #FFFFFF !important;
}

.broadcast-row {
  display: grid;
    grid-template-columns: 86px minmax(180px, 1.2fr) minmax(220px, 1.2fr) minmax(130px, 0.9fr) minmax(130px, 0.9fr) minmax(130px, 0.9fr);
  gap: 10px;
  align-items: center;
  border-radius: 10px;
  border: 1px solid color-mix(in srgb, var(--text-color) 16%, transparent);
    background: color-mix(in srgb, #121d2f 88%, black) !important;
  padding: 9px 10px;
  margin: 6px 0;
}

.broadcast-header {
  font-weight: 700;
    background: color-mix(in srgb, #1b2b44 82%, black) !important;
    color: #FFFFFF !important;
}

.broadcast-header * {
    color: #FFFFFF !important;
    opacity: 1 !important;
}

.rank-badge {
  width: 44px;
  height: 44px;
  border-radius: 999px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.02rem;
  font-weight: 700;
  border: 1px solid color-mix(in srgb, var(--text-color) 24%, transparent);
  background: color-mix(in srgb, var(--secondary-background-color) 80%, var(--background-color));
  margin: 0 auto;
    color: #FFFFFF !important;
}

.driver-cell {
  font-weight: 700;
  letter-spacing: 0.03em;
        color: #FFFFFF !important;
}

.team-cell {
  font-size: 0.92rem;
  letter-spacing: 0.01em;
        color: #FFFFFF !important;
}

.brand-cell {
  display: flex;
  align-items: center;
  gap: 8px;
}

.brand-bar {
  width: 12px;
  height: 24px;
  border-radius: 3px;
}

.brand-label {
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.03em;
        color: #FFFFFF !important;
}

.score-cell {
  text-align: right;
  font-family: 'Barlow Condensed', sans-serif;
  font-size: 1.0rem;
        color: #FFFFFF !important;
}

.delta-pill {
  display: inline-block;
  border-radius: 999px;
  padding: 0.18rem 0.56rem;
  border: 1px solid color-mix(in srgb, var(--text-color) 24%, transparent);
  font-weight: 700;
  font-size: 0.82rem;
}

.pill-good {
  color: #27b66c;
  background: color-mix(in srgb, #27b66c 14%, var(--secondary-background-color));
}

.pill-neutral {
  color: var(--text-color);
  background: color-mix(in srgb, var(--text-color) 10%, var(--secondary-background-color));
}

.pill-high {
  color: #ff5757;
  background: color-mix(in srgb, #ff5757 14%, var(--secondary-background-color));
}

.section-subtle {
        color: #FFFFFF !important;
  font-size: 0.95rem;
    font-weight: 600;
}

@media (max-width: 1200px) {
  .broadcast-row {
        grid-template-columns: 78px minmax(150px, 1fr) minmax(180px, 1fr) minmax(110px, 0.8fr) minmax(110px, 0.8fr) minmax(110px, 0.8fr);
  }
}

@media (max-width: 900px) {
  .broadcast-row {
    grid-template-columns: 72px minmax(145px, 1.2fr) minmax(120px, 1fr) minmax(120px, 1fr);
    row-gap: 8px;
  }

  .score-cell {
    text-align: left;
  }
}
</style>
        """,
        unsafe_allow_html=True,
    )


def _extract_estimator(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def normalize_driver_id(driver_id):
    normalized = str(driver_id).strip().lower()
    if normalized == "max_verstappen":
        return "verstappen"
    return normalized


def prettify_driver(driver_id):
    normalized = normalize_driver_id(driver_id)
    return DRIVER_LABELS.get(normalized, normalized.replace("_", " ").upper())


def team_meta(driver_id):
    normalized = normalize_driver_id(driver_id)
    return TEAM_META.get(
        normalized,
        {
            "team": "Unknown Team",
            "prefix": "UNK",
            "brand": "UNKNOWN",
            "color": "#6f7a87",
        },
    )


def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(model_path)
        with open(model_path, "rb") as file_obj:
            raw = pickle.load(file_obj)
        return _extract_estimator(raw), None
    except FileNotFoundError:
        return None, (
            f"Model file missing: {model_path}. "
            "Run training: python src/models/train_ranker.py"
        )
    except Exception as exc:
        return None, f"Unable to load model at {model_path}: {exc}"


def load_feature_matrix(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required_columns = {
            "driver_id",
            "season",
            "round",
            "race_id",
            "finish_position",
            "grid_position",
            "quali_position",
            "circuit_type_code",
            "is_wet",
            "track_temp",
        }.union(set(ROLLING_COLUMNS))
        missing = sorted(required_columns.difference(df.columns))
        if missing:
            raise ValueError(f"Feature matrix missing required columns: {missing}")
        return df, None
    except FileNotFoundError:
        return None, (
            f"Processed feature matrix missing: {path}. "
            "Run preprocessing: python src/preprocessing/build_veteran_features.py"
        )
    except Exception as exc:
        return None, f"Unable to load feature matrix at {path}: {exc}"


def load_full_grid(path):
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"season", "round", "driver_id", "finish_position", "status_str"}
        missing = sorted(required.difference(df.columns))
        if missing:
            raise ValueError(f"Full grid file missing required columns: {missing}")
        return df, None
    except FileNotFoundError:
        return None, (
            f"Full grid file missing: {path}. "
            "Run canonical build: python src/preprocessing/ingest_jolpica_qualifying.py"
        )
    except Exception as exc:
        return None, f"Unable to load full grid file at {path}: {exc}"


def latest_driver_states(df):
    work = df.copy()
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["round"] = pd.to_numeric(work["round"], errors="coerce")
    work["driver_id"] = work["driver_id"].map(normalize_driver_id)

    for col in ROLLING_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.sort_values(["season", "round", "driver_id"])
    latest = work.groupby("driver_id", as_index=False).tail(1).copy()
    latest = latest[latest["driver_id"].isin(DRIVER_ORDER)].copy()
    latest[ROLLING_COLUMNS] = latest[ROLLING_COLUMNS].fillna(0.0)

    latest["driver_id"] = pd.Categorical(latest["driver_id"], categories=DRIVER_ORDER, ordered=True)
    latest = latest.sort_values("driver_id").reset_index(drop=True)
    latest["driver_id"] = latest["driver_id"].astype(str)
    return latest[["driver_id"] + ROLLING_COLUMNS]


def build_default_input(driver_state_df):
    defaults = driver_state_df[["driver_id"]].copy()
    defaults["driver"] = defaults["driver_id"].map(prettify_driver)
    defaults["team"] = defaults["driver_id"].map(lambda x: team_meta(x)["team"])
    defaults["quali_position"] = list(range(1, len(defaults) + 1))
    defaults["grid_position"] = list(range(1, len(defaults) + 1))
    return defaults[["driver_id", "driver", "team", "quali_position", "grid_position"]]


def simulate_scores(model, frame):
    x = frame[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return model.predict(x)


def render_podium(leaderboard):
    top3 = leaderboard.head(3).reset_index(drop=True)
    if len(top3) < 3:
        return

    cols = st.columns(3)
    labels = ["P1", "P2", "P3"]
    medals = ["GOLD", "SILVER", "BRONZE"]

    for idx in range(3):
        color = top3.loc[idx, "team_color"]
        with cols[idx]:
            st.markdown(
                (
                    f"<div class='podium-card' style='background: linear-gradient(145deg, {color} 0%, color-mix(in srgb, {color} 62%, #070b12) 100%) !important; border: 1px solid color-mix(in srgb, #FFFFFF 24%, transparent) !important; box-shadow: inset 0 0 0 1px color-mix(in srgb, #FFFFFF 10%, transparent); color: #FFFFFF !important;'>"
                    f"<div style='height: 8px; background: color-mix(in srgb, #FFFFFF 26%, {color}); color: #FFFFFF !important;'></div>"
                    "<div class='podium-body' style='color: #FFFFFF !important;'>"
                    f"<div class='podium-rank' style='color: #FFFFFF !important;'>{labels[idx]}</div>"
                    f"<div class='podium-driver' style='color: #FFFFFF !important;'>{top3.loc[idx, 'driver_label']}</div>"
                    f"<div class='podium-team' style='color: #FFFFFF !important;'>{top3.loc[idx, 'team_label']}</div>"
                    f"<span class='podium-medal' style='color: #FFFFFF !important; border: 1px solid color-mix(in srgb, #FFFFFF 35%, transparent); background: color-mix(in srgb, #0A0F17 62%, transparent);'>{medals[idx]}</span>"
                    "</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )


def render_broadcast_header():
    st.markdown(
        (
            "<div class='broadcast-row broadcast-header' style='color: #EAF2FF !important;'>"
            "<div style='color: #EAF2FF !important; text-align:center; font-weight:700;'>PRED RANK</div>"
            "<div style='color: #EAF2FF !important; font-weight:700;'>DRIVER</div>"
            "<div style='color: #EAF2FF !important; font-weight:700;'>TEAM</div>"
            "<div style='color: #EAF2FF !important; text-align:right; font-weight:700;'>GLOBAL FINISH</div>"
            "<div style='color: #EAF2FF !important; text-align:right; font-weight:700;'>COHORT RANK</div>"
            "<div style='color: #EAF2FF !important; text-align:right; font-weight:700;'>ERROR DELTA</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def render_live_standings_rows(leaderboard):
    render_broadcast_header()
    for _, row in leaderboard.iterrows():
        st.markdown(
            (
                "<div class='broadcast-row' style='color: #EAF2FF !important;'>"
                f"<div style='color: #EAF2FF !important;'><div class='rank-badge' style='color: #EAF2FF !important;'>P{int(row['predicted_position'])}</div></div>"
                f"<div class='driver-cell' style='color: #EAF2FF !important;'>{row['driver_label']}</div>"
                "<div class='brand-cell team-cell' style='color: #EAF2FF !important;'>"
                f"<span class='brand-bar' style='background: {row['team_color']}; color: #EAF2FF !important;'></span>"
                f"<span class='brand-label' style='color: #EAF2FF !important;'>{row['team_label']}</span>"
                "</div>"
                "<div class='score-cell' style='color: #EAF2FF !important;'>-</div>"
                "<div class='score-cell' style='color: #EAF2FF !important;'>-</div>"
                "<div class='score-cell' style='color: #EAF2FF !important;'><span class='delta-pill pill-neutral' style='color: #EAF2FF !important;'>d=-</span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )


def render_retrospective_rows(retro):
    st.markdown(
        "<div class='telemetry-card' style='color: var(--text-color) !important;'>",
        unsafe_allow_html=True,
    )
    render_broadcast_header()

    for _, row in retro.iterrows():
        delta = int(row["error_delta"])
        if delta == 0:
            pill_class = "pill-good"
        elif delta >= 5:
            pill_class = "pill-high"
        else:
            pill_class = "pill-neutral"

        st.markdown(
            (
                "<div class='broadcast-row' style='color: #EAF2FF !important;'>"
                f"<div style='color: #EAF2FF !important;'><div class='rank-badge' style='color: #EAF2FF !important;'>P{int(row['predicted_rank'])}</div></div>"
                f"<div class='driver-cell' style='color: #EAF2FF !important;'>{row['driver_label']}</div>"
                "<div class='brand-cell team-cell' style='color: #EAF2FF !important;'>"
                f"<span class='brand-bar' style='background: {row['team_color']}; color: #EAF2FF !important;'></span>"
                f"<span class='brand-label' style='color: #EAF2FF !important;'>{row['team_label']}</span>"
                "</div>"
                f"<div class='score-cell' style='color: #EAF2FF !important;'>P{int(row['actual_finish'])} / {int(row['full_grid_size'])}</div>"
                f"<div class='score-cell' style='color: #EAF2FF !important;'>{int(row['actual_relative_rank'])} of 14</div>"
                f"<div class='score-cell' style='color: #EAF2FF !important;'><span class='delta-pill {pill_class}' style='color: #EAF2FF !important;'>d={delta}</span></div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def build_simulated_leaderboard(model, edited_inputs, state_df):
    merged = edited_inputs.merge(state_df, on="driver_id", how="left")
    for col in FEATURE_COLUMNS:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged[ROLLING_COLUMNS] = merged[ROLLING_COLUMNS].fillna(0.0)

    merged["predicted_score"] = simulate_scores(model, merged)
    leaderboard = merged.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    leaderboard["predicted_position"] = range(1, len(leaderboard) + 1)
    leaderboard["driver_label"] = leaderboard["driver_id"].map(prettify_driver)
    leaderboard["team_label"] = leaderboard["driver_id"].map(lambda x: team_meta(x)["team"])
    leaderboard["team_color"] = leaderboard["driver_id"].map(lambda x: team_meta(x)["color"])
    return leaderboard


def get_completed_2026(df):
    work = df.copy()
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["round"] = pd.to_numeric(work["round"], errors="coerce")
    work["finish_position"] = pd.to_numeric(work["finish_position"], errors="coerce")
    work["driver_id"] = work["driver_id"].map(normalize_driver_id)
    work = work[(work["season"] == 2026) & (work["round"].isin(RACE_2026_MAP.keys()))].copy()
    work = work[work["driver_id"].isin(DRIVER_ORDER)].copy()
    return work


def build_retro_predictions(model, data_2026, selected_round, full_grid_df):
    race_df = data_2026[data_2026["round"] == selected_round].copy()
    race_df = race_df.sort_values("driver_id").reset_index(drop=True)

    for col in FEATURE_COLUMNS:
        race_df[col] = pd.to_numeric(race_df[col], errors="coerce")
    race_df[FEATURE_COLUMNS] = race_df[FEATURE_COLUMNS].fillna(0.0)

    race_df["predicted_score"] = simulate_scores(model, race_df)
    race_df = race_df.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    race_df["predicted_rank"] = range(1, len(race_df) + 1)

    race_df["driver_label"] = race_df["driver_id"].map(prettify_driver)
    race_df["team_label"] = race_df["driver_id"].map(lambda x: team_meta(x)["team"])
    race_df["team_color"] = race_df["driver_id"].map(lambda x: team_meta(x)["color"])

    race_df["actual_finish"] = pd.to_numeric(race_df["finish_position"], errors="coerce")

    relative = race_df.sort_values("actual_finish").reset_index(drop=True)
    relative["actual_relative_rank"] = range(1, len(relative) + 1)
    race_df = race_df.merge(
        relative[["driver_id", "actual_relative_rank"]],
        on="driver_id",
        how="left",
    )

    race_df["error_delta"] = (race_df["predicted_rank"] - race_df["actual_relative_rank"]).abs()

    full_race = full_grid_df[
        (pd.to_numeric(full_grid_df["season"], errors="coerce") == 2026)
        & (pd.to_numeric(full_grid_df["round"], errors="coerce") == selected_round)
    ].copy()
    race_df["full_grid_size"] = max(len(full_race), len(race_df))
    return race_df


def sidebar_model_picker(xgb_model, lgb_model):
    st.sidebar.header("Telemetry Controls")
    st.sidebar.caption("Model switch and profile-driven race environment configuration.")
    model_choice = st.sidebar.radio(
        "Model Artifact",
        ["XGBoost Ranker (Recommended)", "LightGBM Ranker (Benchmark)"],
    )
    selected = xgb_model if model_choice.startswith("XGBoost") else lgb_model
    return model_choice, selected


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    style_app()

    st.title(APP_TITLE)
    st.caption(
        "Broadcast graphics dashboard for live race simulation and 2026 retrospective audits "
        "across the veteran 14-driver cohort."
    )

    xgb_model, xgb_err = load_model(XGB_MODEL_PATH)
    lgb_model, lgb_err = load_model(LGB_MODEL_PATH)
    feature_df, feature_err = load_feature_matrix(FEATURE_MATRIX_PATH)
    full_grid_df, full_grid_err = load_full_grid(FULL_GRID_PATH)

    if xgb_err:
        st.error(xgb_err)
    if lgb_err:
        st.error(lgb_err)
    if feature_err:
        st.error(feature_err)
    if full_grid_err:
        st.error(full_grid_err)

    if any([xgb_err, lgb_err, feature_err, full_grid_err]):
        st.info(
            "Pipeline order: 1) ingest_jolpica_results.py, 2) ingest_jolpica_qualifying.py, "
            "3) build_veteran_features.py, 4) train_ranker.py"
        )
        return

    model_choice, selected_model = sidebar_model_picker(xgb_model, lgb_model)
    st.sidebar.success(f"Active model: {model_choice}")

    selected_gp = st.sidebar.selectbox("Select Upcoming Grand Prix", list(UPCOMING_GP_PROFILES.keys()))
    profile = UPCOMING_GP_PROFILES[selected_gp]
    sim_circuit_type = profile["circuit_label"]
    sim_is_wet = int(profile["is_wet"])
    sim_track_temp = float(profile["track_temp"])

    st.sidebar.caption(
        f"Auto profile: {sim_circuit_type}, is_wet={sim_is_wet}, track_temp={sim_track_temp:.1f}C"
    )

    enable_manual_overrides = st.sidebar.checkbox("Enable Manual Environmental Overrides", value=False)
    if enable_manual_overrides:
        sim_is_wet = int(st.sidebar.checkbox("Manual Wet Conditions", value=bool(sim_is_wet)))
        sim_track_temp = float(
            st.sidebar.slider(
                "Manual Track Temperature (C)",
                min_value=10.0,
                max_value=50.0,
                value=float(sim_track_temp),
                step=0.5,
            )
        )
        sim_circuit_type = st.sidebar.selectbox(
            "Manual Circuit Type",
            list(CIRCUIT_TYPE_CODE.keys()),
            index=list(CIRCUIT_TYPE_CODE.keys()).index(sim_circuit_type),
        )

    state_df = latest_driver_states(feature_df)
    base_input = build_default_input(state_df)

    tab1, tab2 = st.tabs(["Live Grand Prix Simulator", "2026 Retrospective: Predicted vs Actual"])

    with tab1:
        st.subheader("Upcoming Weekend Forecast")
        st.markdown(
            "<div class='telemetry-card' style='color: var(--text-color) !important;'>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div class='telemetry-badge' style='color: #FFFFFF !important; background: #1b2b44 !important;'>LIVE SIMULATION INPUTS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            (
                "<p class='section-subtle' style='color: #FFFFFF !important;'>"
                f"Selected Event: {selected_gp} | Circuit: {sim_circuit_type} | Wet: {sim_is_wet} | Track Temp: {sim_track_temp:.1f}C"
                "</p>"
            ),
            unsafe_allow_html=True,
        )

        edited = st.data_editor(
            base_input,
            hide_index=True,
            num_rows="fixed",
            use_container_width=True,
            column_config={
                "driver_id": st.column_config.TextColumn("driver_id", disabled=True, width="small"),
                "driver": st.column_config.TextColumn("DRIVER", disabled=True),
                "team": st.column_config.TextColumn("TEAM", disabled=True),
                "quali_position": st.column_config.NumberColumn("QUALI", min_value=1, max_value=20, step=1),
                "grid_position": st.column_config.NumberColumn("GRID", min_value=1, max_value=20, step=1),
            },
        )

        run_live = st.button("Simulate Grand Prix", type="primary", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if run_live:
            sim_inputs = edited[["driver_id", "quali_position", "grid_position"]].copy()
            sim_inputs["is_wet"] = int(sim_is_wet)
            sim_inputs["track_temp"] = float(sim_track_temp)
            sim_inputs["circuit_type_code"] = int(CIRCUIT_TYPE_CODE[sim_circuit_type])

            simulated = build_simulated_leaderboard(selected_model, sim_inputs, state_df)
            st.subheader("Predicted Podium")
            render_podium(simulated)

            st.subheader("Predicted Full Order")
            st.markdown(
                "<p class='section-subtle' style='color: #FFFFFF !important;'>Six-column broadcast split with dedicated team color markers and contrast-safe delta pills.</p>",
                unsafe_allow_html=True,
            )
            render_live_standings_rows(simulated)

    with tab2:
        st.subheader("2026 Season Performance Retrospective")
        st.markdown(
            "<div class='telemetry-badge' style='color: #FFFFFF !important; background: #1b2b44 !important;'>PREDICTED VS ACTUAL AUDIT</div>",
            unsafe_allow_html=True,
        )

        data_2026 = get_completed_2026(feature_df)
        if data_2026.empty:
            st.warning("No completed 2026 rounds detected in the processed matrix.")
            return

        available_rounds = sorted(set(int(r) for r in data_2026["round"].dropna().unique()))
        available_rounds = [r for r in available_rounds if r in RACE_2026_MAP]
        if not available_rounds:
            st.warning("No supported rounds found for retrospective view (expected rounds 1-8).")
            return

        selected_label = st.selectbox(
            "Completed 2026 Race Round",
            [f"Round {r} - {RACE_2026_MAP[r]}" for r in available_rounds],
        )
        selected_round = int(selected_label.split(" - ")[0].replace("Round ", ""))

        retro = build_retro_predictions(selected_model, data_2026, selected_round, full_grid_df)
        render_retrospective_rows(retro)

        with st.expander("View Unfiltered Full Race Grid Context"):
            full_race = full_grid_df[
                (pd.to_numeric(full_grid_df["season"], errors="coerce") == 2026)
                & (pd.to_numeric(full_grid_df["round"], errors="coerce") == selected_round)
            ].copy()
            if full_race.empty:
                st.warning("No unfiltered race rows found for the selected round.")
            else:
                full_race["finish_position"] = pd.to_numeric(full_race["finish_position"], errors="coerce")
                full_race["driver_label"] = full_race["driver_id"].map(prettify_driver)
                full_race = full_race.sort_values("finish_position", na_position="last").reset_index(drop=True)
                full_race["global_order"] = range(1, len(full_race) + 1)
                st.dataframe(
                    full_race[
                        [
                            "global_order",
                            "driver_label",
                            "driver_id",
                            "constructor_id",
                            "finish_position",
                            "status_str",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )


if __name__ == "__main__":
    main()
