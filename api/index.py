import json
import os
import pickle
from datetime import UTC, datetime
from functools import lru_cache
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
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

XGB_MODEL_PATH = "models/f1_xgb_ranker.pkl"
FEATURE_MATRIX_PATH = "data/processed/veteran_training_matrix.csv"

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

CIRCUIT_TYPE_CODE = {
    "Power/High-Speed": 0,
    "High-Downforce": 1,
    "Balanced/Technical": 2,
    "Street Circuit": 3,
}

ALLOWED_CIRCUITS = set(CIRCUIT_TYPE_CODE.keys())


def normalize_driver_id(driver_id):
    normalized = str(driver_id).strip().lower()
    if normalized == "max_verstappen":
        return "verstappen"
    return normalized


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


@lru_cache(maxsize=1)
def _load_model():
    model_path = _resolve_artifact_path(XGB_MODEL_PATH)
    with model_path.open("rb") as file_obj:
        raw = pickle.load(file_obj)
    return _extract_estimator(raw)


@lru_cache(maxsize=1)
def _load_latest_driver_states():
    matrix_path = _resolve_artifact_path(FEATURE_MATRIX_PATH)
    df = pd.read_csv(matrix_path)

    required_columns = {"driver_id", "season", "round"}.union(set(ROLLING_COLUMNS))
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Feature matrix missing required columns: {missing}")

    work = df.copy()
    work["driver_id"] = work["driver_id"].map(normalize_driver_id)
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work["round"] = pd.to_numeric(work["round"], errors="coerce")

    for col in ROLLING_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.sort_values(["season", "round", "driver_id"])
    latest = work.groupby("driver_id", as_index=False).tail(1).copy()
    latest = latest[latest["driver_id"].isin(DRIVER_ORDER)].copy()
    latest[ROLLING_COLUMNS] = latest[ROLLING_COLUMNS].fillna(0.0)

    latest["driver_id"] = pd.Categorical(
        latest["driver_id"], categories=DRIVER_ORDER, ordered=True
    )
    latest = latest.sort_values("driver_id").reset_index(drop=True)
    latest["driver_id"] = latest["driver_id"].astype(str)
    return latest[["driver_id"] + ROLLING_COLUMNS]


def _build_default_grid():
    frame = pd.DataFrame({"driver_id": DRIVER_ORDER})
    frame["grid_position"] = list(range(1, len(frame) + 1))
    frame["quali_position"] = list(range(1, len(frame) + 1))
    return frame


def _send_json(req, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    req.send_response(status)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(body)))
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
    circuit_label = payload.get("circuit_label", "Balanced/Technical")
    if circuit_label not in ALLOWED_CIRCUITS:
        raise ValueError("circuit_label is invalid.")

    is_wet = payload.get("is_wet", 0)
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
        "circuit_label": circuit_label,
        "is_wet": is_wet,
        "track_temp": track_temp,
        "overrides": normalized_overrides,
    }


def _build_driver_payload():
    state_df = _load_latest_driver_states()
    merged = _build_default_grid().merge(state_df, on="driver_id", how="left")
    merged[ROLLING_COLUMNS] = merged[ROLLING_COLUMNS].fillna(0.0)

    drivers = []
    for _, row in merged.iterrows():
        driver_id = normalize_driver_id(row["driver_id"])
        drivers.append(
            {
                "driver_id": driver_id,
                "driver_label": DRIVER_LABELS.get(driver_id, driver_id.replace("_", " ").upper()),
                "team": TEAM_META.get(driver_id, {"team": "Unknown Team"})["team"],
                "default_grid_position": int(row["grid_position"]),
                "default_quali_position": int(row["quali_position"]),
                "driver_form_3races": float(row["driver_form_3races"]),
                "circuit_historical_avg": float(row["circuit_historical_avg"]),
                "constructor_points_current": float(row["constructor_points_current"]),
                "constructor_dnf_rate_10races": float(row["constructor_dnf_rate_10races"]),
            }
        )
    return drivers


def _predict_live(validated_payload):
    model = _load_model()
    state_df = _load_latest_driver_states()

    merged = _build_default_grid().merge(state_df, on="driver_id", how="left")
    merged[ROLLING_COLUMNS] = merged[ROLLING_COLUMNS].fillna(0.0)

    for item in validated_payload["overrides"]:
        mask = merged["driver_id"] == item["driver_id"]
        merged.loc[mask, "grid_position"] = item["grid_position"]
        merged.loc[mask, "quali_position"] = item["quali_position"]

    merged["circuit_type_code"] = CIRCUIT_TYPE_CODE[validated_payload["circuit_label"]]
    merged["is_wet"] = validated_payload["is_wet"]
    merged["track_temp"] = validated_payload["track_temp"]

    x = merged[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    merged["predicted_score"] = model.predict(x)

    leaderboard_df = merged.sort_values("predicted_score", ascending=False).reset_index(drop=True)
    leaderboard_df["predicted_position"] = range(1, len(leaderboard_df) + 1)

    leaderboard = []
    for _, row in leaderboard_df.iterrows():
        driver_id = normalize_driver_id(row["driver_id"])
        meta = TEAM_META.get(driver_id, {"team": "Unknown Team", "color": "#6f7a87"})
        leaderboard.append(
            {
                "predicted_position": int(row["predicted_position"]),
                "driver_id": driver_id,
                "driver_label": DRIVER_LABELS.get(driver_id, driver_id.replace("_", " ").upper()),
                "team": meta["team"],
                "team_color": meta["color"],
                "predicted_score": float(row["predicted_score"]),
            }
        )

    return leaderboard


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        _send_json(self, {"ok": True}, 200)

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")
        if path in ("", "/api"):
            return _send_json(
                self,
                {
                    "service": APP_TITLE,
                    "status": "ok",
                    "routes": ["/api", "/api/health", "/api/drivers", "/api/predict/live"],
                },
                200,
            )

        if path in ("/health", "/api/health"):
            try:
                _load_model()
                _load_latest_driver_states()
                return _send_json(
                    self,
                    {
                        "status": "ok",
                        "model": "xgboost",
                        "checked_at_utc": datetime.now(UTC).isoformat(),
                    },
                    200,
                )
            except (FileNotFoundError, ValueError, OSError, requests.RequestException, pickle.UnpicklingError) as exc:
                return _send_json(
                    self,
                    {
                        "status": "degraded",
                        "model": "xgboost",
                        "checked_at_utc": datetime.now(UTC).isoformat(),
                        "detail": str(exc),
                    },
                    200,
                )

        if path in ("/drivers", "/api/drivers"):
            try:
                drivers = _build_driver_payload()
            except (FileNotFoundError, ValueError, OSError, requests.RequestException, pickle.UnpicklingError) as exc:
                return _send_json(self, {"error": f"Failed to load driver state: {exc}"}, 500)
            return _send_json(self, {"count": len(drivers), "drivers": drivers}, 200)

        return _send_json(self, {"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        if path not in ("/predict/live", "/api/predict/live"):
            return _send_json(self, {"error": "Not found"}, 404)

        try:
            raw_payload = _read_json_body(self)
            validated = _validate_predict_payload(raw_payload)
            leaderboard = _predict_live(validated)
        except ValueError as exc:
            return _send_json(self, {"error": str(exc)}, 400)
        except (FileNotFoundError, OSError, requests.RequestException, pickle.UnpicklingError) as exc:
            return _send_json(self, {"error": f"Inference artifacts unavailable: {exc}"}, 500)

        return _send_json(
            self,
            {
                "model": "xgboost",
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
