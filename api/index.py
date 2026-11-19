import csv
import json
import math
import os
import pickle
import time
from datetime import UTC, datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import requests

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except ModuleNotFoundError:
    pass

APP_TITLE = "F1 Veteran Ranker API"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TMP_ARTIFACT_ROOT = Path("/tmp/f1-artifacts")

MODEL_REGISTRY_REPO = os.environ.get("MODEL_REGISTRY_REPO", "CKmellow/f1-veteran-ranker")
MODEL_REGISTRY_BRANCH = os.environ.get("MODEL_REGISTRY_BRANCH", "model-registry")

MODEL_PATHS = {
    "xgb": "models/f1_xgb_ranker.pkl",
    "lgb": "models/f1_lgb_ranker.pkl",
}
MODEL_LABELS = {
    "xgb": "XGBoost Ranker (Recommended)",
    "lgb": "LightGBM Ranker (Benchmark)",
}
DEFAULT_MODEL = "xgb"

FEATURE_MATRIX_PATH = "data/processed/veteran_training_matrix.csv"
FULL_GRID_PATH = "data/raw/f1_canonical_master.csv"
REFRESH_METADATA_PATH = "outputs/reports/refresh_metadata.json"

SCHEDULE_TTL_SECONDS = 6 * 3600

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
    "verstappen": {"team": "Red Bull Racing", "color": "#0600EF"},
    "perez": {"team": "Red Bull Racing", "color": "#0600EF"},
    "hamilton": {"team": "Scuderia Ferrari", "color": "#EF1A2D"},
    "leclerc": {"team": "Scuderia Ferrari", "color": "#EF1A2D"},
    "russell": {"team": "Mercedes-AMG", "color": "#27F4D2"},
    "norris": {"team": "McLaren", "color": "#FF8000"},
    "alonso": {"team": "Aston Martin", "color": "#229971"},
    "stroll": {"team": "Aston Martin", "color": "#229971"},
    "albon": {"team": "Williams", "color": "#47C7FC"},
    "sainz": {"team": "Williams", "color": "#47C7FC"},
    "gasly": {"team": "Alpine", "color": "#0093CC"},
    "ocon": {"team": "Haas", "color": "#B6BABD"},
    "hulkenberg": {"team": "Audi", "color": "#C92D4B"},
    "bottas": {"team": "Cadillac", "color": "#1D3557"},
}
UNKNOWN_TEAM = {"team": "Unknown Team", "color": "#6f7a87"}

CIRCUIT_TYPE_CODE = {
    "Power/High-Speed": 0,
    "High-Downforce": 1,
    "Balanced/Technical": 2,
    "Street Circuit": 3,
}

ALLOWED_CIRCUITS = set(CIRCUIT_TYPE_CODE.keys())

DEFAULT_SIM_PROFILE = {
    "circuit_label": "Balanced/Technical",
    "is_wet": 0,
    "track_temp": 25.0,
}

CIRCUIT_PROFILE_BY_CIRCUIT_ID = {
    "spa": {"circuit_label": "Power/High-Speed", "is_wet": 1, "track_temp": 18.0},
    "zandvoort": {"circuit_label": "Balanced/Technical", "is_wet": 0, "track_temp": 21.0},
    "monza": {"circuit_label": "Power/High-Speed", "is_wet": 0, "track_temp": 27.0},
    "marina_bay": {"circuit_label": "Street Circuit", "is_wet": 0, "track_temp": 30.0},
    "silverstone": {"circuit_label": "Power/High-Speed", "is_wet": 0, "track_temp": 22.0},
    "hungaroring": {"circuit_label": "High-Downforce", "is_wet": 0, "track_temp": 29.0},
    "monaco": {"circuit_label": "High-Downforce", "is_wet": 0, "track_temp": 24.0},
    "red_bull_ring": {"circuit_label": "Power/High-Speed", "is_wet": 0, "track_temp": 23.0},
    "suzuka": {"circuit_label": "Balanced/Technical", "is_wet": 0, "track_temp": 23.0},
    "jeddah": {"circuit_label": "Street Circuit", "is_wet": 0, "track_temp": 31.0},
    "baku": {"circuit_label": "Street Circuit", "is_wet": 0, "track_temp": 26.0},
    "miami": {"circuit_label": "Street Circuit", "is_wet": 1, "track_temp": 30.0},
    "circuit_americas": {"circuit_label": "Balanced/Technical", "is_wet": 0, "track_temp": 28.0},
    "catalunya": {"circuit_label": "Balanced/Technical", "is_wet": 0, "track_temp": 27.0},
}

ARTIFACT_ERRORS = (
    FileNotFoundError,
    ValueError,
    OSError,
    requests.RequestException,
    pickle.UnpicklingError,
)


def normalize_driver_id(driver_id):
    normalized = str(driver_id).strip().lower()
    if normalized == "max_verstappen":
        return "verstappen"
    return normalized


def driver_label(driver_id):
    normalized = normalize_driver_id(driver_id)
    return DRIVER_LABELS.get(normalized, normalized.replace("_", " ").upper())


def team_meta(driver_id):
    return TEAM_META.get(normalize_driver_id(driver_id), UNKNOWN_TEAM)


def _extract_estimator(obj):
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj


def _registry_raw_url(relative_path):
    normalized = str(relative_path).replace("\\", "/").lstrip("/")
    return (
        f"https://raw.githubusercontent.com/{MODEL_REGISTRY_REPO}/"
        f"{MODEL_REGISTRY_BRANCH}/{normalized}"
    )


def _resolve_artifact_path(relative_path):
    in_repo = PROJECT_ROOT / relative_path
    if in_repo.exists():
        return in_repo

    in_tmp = TMP_ARTIFACT_ROOT / relative_path
    if in_tmp.exists():
        return in_tmp

    in_tmp.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(_registry_raw_url(relative_path), timeout=25)
    if response.status_code != 200:
        raise FileNotFoundError(
            f"Unable to locate artifact: {relative_path}. "
            f"Checked repository and model registry branch '{MODEL_REGISTRY_BRANCH}'."
        )
    in_tmp.write_bytes(response.content)
    return in_tmp


def _parse_model_key(value):
    key = str(value or DEFAULT_MODEL).strip().lower()
    if key not in MODEL_PATHS:
        raise ValueError(f"model must be one of: {sorted(MODEL_PATHS)}")
    return key


@lru_cache(maxsize=len(MODEL_PATHS))
def _load_model(model_key=DEFAULT_MODEL):
    model_path = _resolve_artifact_path(MODEL_PATHS[model_key])
    with model_path.open("rb") as file_obj:
        raw = pickle.load(file_obj)
    return _extract_estimator(raw)


def _predict_scores(model, features_df):
    """Predict safely across environments with or without scikit-learn installed."""
    try:
        return model.predict(features_df)
    except (AttributeError, TypeError):
        booster = getattr(model, "booster_", None)
        if booster is None:
            booster = getattr(model, "_Booster", None)
        if booster is None:
            raise
        return booster.predict(features_df)


def _to_float(value):
    """Parse a CSV cell to float; blanks and junk become None (like pandas' NaN)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _read_csv(relative_path, required, numeric):
    with _resolve_artifact_path(relative_path).open(newline="", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        missing = sorted(set(required).difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{relative_path} missing required columns: {missing}")
        rows = list(reader)
    for row in rows:
        row["driver_id"] = normalize_driver_id(row["driver_id"])
        for col in numeric:
            row[col] = _to_float(row.get(col))
    return rows


@lru_cache(maxsize=1)
def _load_feature_matrix():
    numeric = ["season", "round", "finish_position"] + FEATURE_COLUMNS
    return _read_csv(FEATURE_MATRIX_PATH, ["driver_id"] + numeric, numeric)


@lru_cache(maxsize=1)
def _load_full_grid():
    numeric = ["season", "round", "finish_position"]
    required = ["driver_id", "status_str"] + numeric
    return _read_csv(FULL_GRID_PATH, required, numeric)


def _feature_array(rows):
    return np.array(
        [[row.get(col) if row.get(col) is not None else 0.0 for col in FEATURE_COLUMNS] for row in rows],
        dtype=float,
    )


def _missing_last(value):
    return (value is None, value if value is not None else 0.0)


def _read_refresh_metadata():
    try:
        payload = json.loads(_resolve_artifact_path(REFRESH_METADATA_PATH).read_text("utf-8"))
        return {
            "generated_at_utc": payload.get("generated_at_utc"),
            "source": payload.get("source", "unknown"),
        }
    except (*ARTIFACT_ERRORS, TypeError):
        return {"generated_at_utc": None, "source": None}


_SCHEDULE_CACHE = {}


def _fetch_season_schedule(season):
    cached = _SCHEDULE_CACHE.get(season)
    if cached and time.time() - cached[0] < SCHEDULE_TTL_SECONDS:
        return cached[1]

    url = f"https://api.jolpi.ca/ergast/f1/{int(season)}.json?limit=100"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        races = response.json().get("MRData", {}).get("RaceTable", {}).get("Races", [])
        schedule = []
        for race in races if isinstance(races, list) else []:
            circuit = race.get("Circuit", {}) if isinstance(race, dict) else {}
            schedule.append(
                {
                    "round": int(race.get("round")),
                    "race_name": str(race.get("raceName", "")).strip(),
                    "circuit_id": str(circuit.get("circuitId", "")).strip().lower(),
                    "date": race.get("date"),
                }
            )
        schedule.sort(key=lambda item: item["round"])
    except (requests.RequestException, ValueError, TypeError, KeyError):
        # Don't cache failures, so the next request retries.
        return []

    _SCHEDULE_CACHE[season] = (time.time(), schedule)
    return schedule


def _profile_from_circuit(circuit_id):
    profile = CIRCUIT_PROFILE_BY_CIRCUIT_ID.get(str(circuit_id).strip().lower())
    return dict(profile if profile is not None else DEFAULT_SIM_PROFILE)


def _latest_season(rows):
    seasons = [row["season"] for row in rows if row["season"] is not None]
    return int(max(seasons)) if seasons else datetime.now(UTC).year


def _completed_rounds(full_grid_rows, season):
    return sorted(
        {
            int(row["round"])
            for row in full_grid_rows
            if row["season"] == season and row["finish_position"] is not None and row["round"] is not None
        }
    )


def _upcoming_events(season, completed_rounds, schedule):
    if not schedule:
        return [
            {
                "round": None,
                "label": f"Round ? - Next Race ({season})",
                "profile": dict(DEFAULT_SIM_PROFILE),
            }
        ]

    max_completed = max(completed_rounds) if completed_rounds else 0
    remaining = [item for item in schedule if int(item["round"]) > max_completed]
    if not remaining:
        last = schedule[-1]
        return [
            {
                "round": int(last["round"]),
                "label": f"Round {int(last['round'])} - {last['race_name']} (Season Complete)",
                "profile": _profile_from_circuit(last.get("circuit_id", "")),
            }
        ]

    return [
        {
            "round": int(item["round"]),
            "label": f"Round {int(item['round'])} - {item['race_name']}",
            "date": item.get("date"),
            "profile": _profile_from_circuit(item.get("circuit_id", "")),
        }
        for item in remaining
    ]


def _build_meta():
    feature_df = _load_feature_matrix()
    full_grid_df = _load_full_grid()
    season = _latest_season(feature_df)
    completed = _completed_rounds(full_grid_df, season)
    schedule = _fetch_season_schedule(season)
    round_names = {int(item["round"]): item["race_name"] for item in schedule}

    retro_rounds = sorted(
        {
            int(row["round"])
            for row in feature_df
            if row["season"] == season and row["round"] in completed and row["driver_id"] in DRIVER_ORDER
        }
    )

    return {
        "season": season,
        "refresh": _read_refresh_metadata(),
        "models": [{"key": key, "label": MODEL_LABELS[key]} for key in MODEL_PATHS],
        "default_model": DEFAULT_MODEL,
        "circuit_types": list(CIRCUIT_TYPE_CODE.keys()),
        "upcoming_events": _upcoming_events(season, completed, schedule),
        "completed_rounds": [
            {"round": r, "label": f"Round {r} - {round_names.get(r, f'Race {r}')}"}
            for r in retro_rounds
        ],
    }


def _load_latest_driver_states():
    """Latest rolling-feature snapshot per veteran driver, keyed by driver_id."""
    latest = {}
    ordered = sorted(
        _load_feature_matrix(),
        key=lambda r: (_missing_last(r["season"]), _missing_last(r["round"]), r["driver_id"]),
    )
    for row in ordered:
        if row["driver_id"] in DRIVER_ORDER:
            latest[row["driver_id"]] = {col: row[col] or 0.0 for col in ROLLING_COLUMNS}
    return latest


def _build_default_grid():
    """One row per veteran with default grid/quali order and latest rolling features."""
    states = _load_latest_driver_states()
    grid = []
    for position, driver_id in enumerate(DRIVER_ORDER, start=1):
        row = {"driver_id": driver_id, "grid_position": position, "quali_position": position}
        row.update(states.get(driver_id, {col: 0.0 for col in ROLLING_COLUMNS}))
        grid.append(row)
    return grid


def _send_json(req, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    req.send_response(status)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(body)))
    req.send_header("Cache-Control", "no-store")
    req.send_header("Access-Control-Allow-Origin", "*")
    req.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    req.send_header("Access-Control-Allow-Headers", "Content-Type")
    req.end_headers()
    req.wfile.write(body)


def _read_json_body(req):
    length = int(req.headers.get("Content-Length", "0"))
    raw = req.rfile.read(length) if length > 0 else b"{}"
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Invalid JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object.")
    return payload


def _validate_predict_payload(payload):
    model_key = _parse_model_key(payload.get("model"))

    circuit_label = payload.get("circuit_label", "Balanced/Technical")
    if circuit_label not in ALLOWED_CIRCUITS:
        raise ValueError("circuit_label is invalid.")

    is_wet = payload.get("is_wet", 0)
    if isinstance(is_wet, bool):
        is_wet = int(is_wet)
    if not isinstance(is_wet, int) or is_wet not in (0, 1):
        raise ValueError("is_wet must be 0 or 1.")

    track_temp = payload.get("track_temp", 25.0)
    try:
        track_temp = float(track_temp)
    except (TypeError, ValueError) as exc:
        raise ValueError("track_temp must be numeric.") from exc
    if track_temp < -20.0 or track_temp > 70.0:
        raise ValueError("track_temp must be between -20 and 70.")

    overrides = payload.get("overrides", [])
    if not isinstance(overrides, list):
        raise ValueError("overrides must be a list.")

    normalized_overrides = []
    seen = set()
    for item in overrides:
        if not isinstance(item, dict):
            raise ValueError("Each override must be an object.")

        driver_id = normalize_driver_id(item.get("driver_id", ""))
        if driver_id not in DRIVER_ORDER:
            raise ValueError(f"Unknown driver_id: {item.get('driver_id')}")
        if driver_id in seen:
            raise ValueError(f"Duplicate override for driver_id: {driver_id}")
        seen.add(driver_id)

        try:
            grid_position = int(item.get("grid_position"))
            quali_position = int(item.get("quali_position"))
        except (TypeError, ValueError) as exc:
            raise ValueError("grid_position and quali_position must be integers.") from exc

        if grid_position < 1 or grid_position > 20:
            raise ValueError("grid_position must be between 1 and 20.")
        if quali_position < 1 or quali_position > 20:
            raise ValueError("quali_position must be between 1 and 20.")

        normalized_overrides.append(
            {
                "driver_id": driver_id,
                "grid_position": grid_position,
                "quali_position": quali_position,
            }
        )

    return {
        "model": model_key,
        "circuit_label": circuit_label,
        "is_wet": is_wet,
        "track_temp": track_temp,
        "overrides": normalized_overrides,
    }


def _build_driver_payload():
    drivers = []
    for row in _build_default_grid():
        driver_id = row["driver_id"]
        meta = team_meta(driver_id)
        drivers.append(
            {
                "driver_id": driver_id,
                "driver_label": driver_label(driver_id),
                "team": meta["team"],
                "team_color": meta["color"],
                "default_grid_position": row["grid_position"],
                "default_quali_position": row["quali_position"],
                **{col: float(row[col]) for col in ROLLING_COLUMNS},
            }
        )
    return drivers


def _predict_live(validated_payload):
    model = _load_model(validated_payload["model"])
    grid = _build_default_grid()
    overrides = {item["driver_id"]: item for item in validated_payload["overrides"]}

    for row in grid:
        override = overrides.get(row["driver_id"])
        if override:
            row["grid_position"] = override["grid_position"]
            row["quali_position"] = override["quali_position"]
        row["circuit_type_code"] = CIRCUIT_TYPE_CODE[validated_payload["circuit_label"]]
        row["is_wet"] = validated_payload["is_wet"]
        row["track_temp"] = validated_payload["track_temp"]

    scores = _predict_scores(model, _feature_array(grid))
    for row, score in zip(grid, scores):
        row["predicted_score"] = float(score)
    grid.sort(key=lambda r: r["predicted_score"], reverse=True)

    leaderboard = []
    for position, row in enumerate(grid, start=1):
        meta = team_meta(row["driver_id"])
        leaderboard.append(
            {
                "predicted_position": position,
                "driver_id": row["driver_id"],
                "driver_label": driver_label(row["driver_id"]),
                "team": meta["team"],
                "team_color": meta["color"],
                "grid_position": int(row["grid_position"]),
                "quali_position": int(row["quali_position"]),
                "predicted_score": row["predicted_score"],
            }
        )
    return leaderboard


def _retrospective(model_key, round_number):
    feature_df = _load_feature_matrix()
    full_grid_df = _load_full_grid()
    season = _latest_season(feature_df)
    completed = _completed_rounds(full_grid_df, season)
    if round_number not in completed:
        raise LookupError(f"Round {round_number} of {season} has no completed results.")

    race = sorted(
        (
            dict(row)
            for row in feature_df
            if row["season"] == season and row["round"] == round_number and row["driver_id"] in DRIVER_ORDER
        ),
        key=lambda r: r["driver_id"],
    )
    if not race:
        raise LookupError(f"No veteran rows for round {round_number} of {season}.")

    scores = _predict_scores(_load_model(model_key), _feature_array(race))
    for row, score in zip(race, scores):
        row["predicted_score"] = float(score)
    race.sort(key=lambda r: r["predicted_score"], reverse=True)
    for rank, row in enumerate(race, start=1):
        row["predicted_rank"] = rank
    by_actual = sorted(race, key=lambda r: _missing_last(r["finish_position"]))
    for rank, row in enumerate(by_actual, start=1):
        row["actual_relative_rank"] = rank
        row["error_delta"] = abs(row["predicted_rank"] - rank)

    full_race = sorted(
        (row for row in full_grid_df if row["season"] == season and row["round"] == round_number),
        key=lambda r: _missing_last(r["finish_position"]),
    )
    full_grid_size = max(len(full_race), len(race))
    cohort_size = len(race)

    rows = []
    for row in race:
        meta = team_meta(row["driver_id"])
        finish = row["finish_position"]
        rows.append(
            {
                "predicted_rank": row["predicted_rank"],
                "driver_id": row["driver_id"],
                "driver_label": driver_label(row["driver_id"]),
                "team": meta["team"],
                "team_color": meta["color"],
                "actual_finish": None if finish is None else int(finish),
                "actual_relative_rank": row["actual_relative_rank"],
                "error_delta": row["error_delta"],
                "predicted_score": row["predicted_score"],
            }
        )

    full_grid = []
    for order, row in enumerate(full_race, start=1):
        finish = row["finish_position"]
        full_grid.append(
            {
                "global_order": order,
                "driver_id": row["driver_id"],
                "driver_label": driver_label(row["driver_id"]),
                "constructor_id": row.get("constructor_id") or None,
                "finish_position": None if finish is None else int(finish),
                "status": row.get("status_str") or None,
            }
        )

    deltas = [r["error_delta"] for r in rows]
    return {
        "season": season,
        "round": round_number,
        "model": model_key,
        "full_grid_size": full_grid_size,
        "cohort_size": cohort_size,
        "summary": {
            "exact_hits": sum(1 for d in deltas if d == 0),
            "mean_abs_error": round(sum(deltas) / len(deltas), 2) if deltas else None,
        },
        "rows": rows,
        "full_grid": full_grid,
    }


def _normalized_path(raw_path):
    path = urlparse(raw_path).path.rstrip("/")
    if path.startswith("/api"):
        path = path[len("/api"):]
    return path


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        _send_json(self, {"ok": True}, 200)

    def do_GET(self):
        path = _normalized_path(self.path)
        query = parse_qs(urlparse(self.path).query)

        if path == "":
            return _send_json(
                self,
                {
                    "service": APP_TITLE,
                    "status": "ok",
                    "routes": [
                        "/api/health",
                        "/api/meta",
                        "/api/drivers",
                        "/api/retro?round=<n>&model=<xgb|lgb>",
                        "POST /api/predict/live",
                    ],
                },
                200,
            )

        if path == "/health":
            try:
                for key in MODEL_PATHS:
                    _load_model(key)
                _load_feature_matrix()
                _load_full_grid()
                return _send_json(
                    self,
                    {
                        "status": "ok",
                        "models": list(MODEL_PATHS),
                        "checked_at_utc": datetime.now(UTC).isoformat(),
                    },
                    200,
                )
            except ARTIFACT_ERRORS as exc:
                return _send_json(
                    self,
                    {
                        "status": "degraded",
                        "checked_at_utc": datetime.now(UTC).isoformat(),
                        "detail": str(exc),
                    },
                    200,
                )

        if path == "/meta":
            try:
                return _send_json(self, _build_meta(), 200)
            except ARTIFACT_ERRORS as exc:
                return _send_json(self, {"error": f"Failed to load metadata: {exc}"}, 500)

        if path == "/drivers":
            try:
                drivers = _build_driver_payload()
            except ARTIFACT_ERRORS as exc:
                return _send_json(self, {"error": f"Failed to load driver state: {exc}"}, 500)
            return _send_json(self, {"count": len(drivers), "drivers": drivers}, 200)

        if path == "/retro":
            try:
                model_key = _parse_model_key(query.get("model", [None])[0])
                round_number = int(query.get("round", [""])[0])
            except ValueError as exc:
                return _send_json(self, {"error": f"Invalid query: {exc}"}, 400)
            try:
                return _send_json(self, _retrospective(model_key, round_number), 200)
            except LookupError as exc:
                return _send_json(self, {"error": str(exc)}, 404)
            except ARTIFACT_ERRORS as exc:
                return _send_json(self, {"error": f"Inference artifacts unavailable: {exc}"}, 500)

        return _send_json(self, {"error": "Not found"}, 404)

    def do_POST(self):
        path = _normalized_path(self.path)
        if path != "/predict/live":
            return _send_json(self, {"error": "Not found"}, 404)

        try:
            raw_payload = _read_json_body(self)
            validated = _validate_predict_payload(raw_payload)
            leaderboard = _predict_live(validated)
        except ValueError as exc:
            return _send_json(self, {"error": str(exc)}, 400)
        except ARTIFACT_ERRORS as exc:
            return _send_json(self, {"error": f"Inference artifacts unavailable: {exc}"}, 500)

        return _send_json(
            self,
            {
                "model": validated["model"],
                "model_label": MODEL_LABELS[validated["model"]],
                "generated_at_utc": datetime.now(UTC).isoformat(),
                "inputs": {
                    "circuit_label": validated["circuit_label"],
                    "is_wet": validated["is_wet"],
                    "track_temp": validated["track_temp"],
                    "overrides_count": len(validated["overrides"]),
                },
                "leaderboard": leaderboard,
            },
            200,
        )


handler = Handler
