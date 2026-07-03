"""Jolpica F1 race-results ingestion pipeline.

This script implements dual-mode ingestion:
1) Historical bulk fetch across a year range.
2) Incremental fetch for a specific year/round.

The output schema is normalized to one row per driver result with the columns:
- race_id
- season
- round
- circuit_id
- driver_id
- constructor_id
- grid_position
- finish_position
- status_str
"""

import os
import time
import datetime

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 3
REQUEST_DELAY_SECONDS = 1.0
DEFAULT_START_YEAR = 2022

OUTPUT_COLUMNS = [
    "race_id",
    "season",
    "round",
    "circuit_id",
    "driver_id",
    "constructor_id",
    "grid_position",
    "finish_position",
    "points",
    "status_str",
    "is_wet",
    "track_temp",
]

WET_STATUS_KEYWORDS = (
    "rain",
    "wet",
    "aquaplan",
    "water",
)


def _safe_int(value):
    """Convert value to int, returning None when conversion fails."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value):
    """Convert value to float, returning 0.0 when conversion fails."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _infer_is_wet(status_text):
    """Infer wet race conditions from available status text tokens."""
    normalized = str(status_text).lower() if status_text is not None else ""
    return int(any(keyword in normalized for keyword in WET_STATUS_KEYWORDS))


def _infer_track_temp(is_wet):
    """Fallback ambient track temperature estimate when API temperature is unavailable."""
    return 20.0 if int(is_wet) == 1 else 25.0


def _build_bulk_url(year, limit=1000, offset=0):
    """Build the Jolpica bulk endpoint URL for a season."""
    return f"{BASE_URL}/{year}/results.json?limit={int(limit)}&offset={int(offset)}"


def _build_incremental_url(year, round_number):
    """Build the Jolpica incremental endpoint URL for a specific race."""
    return f"{BASE_URL}/{year}/{round_number}/results.json?limit=100"


def fetch_raw_payload(url, session=None, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=DEFAULT_MAX_RETRIES):
    """Fetch JSON payload with retry/backoff and strict HTTP handling."""
    active_session = session if session is not None else requests.Session()
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = active_session.get(url, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(REQUEST_DELAY_SECONDS)
            else:
                raise RuntimeError(f"Failed to fetch payload from {url}") from last_error


def _extract_races(payload):
    """Defensively extract race list from Jolpica/Ergast JSON envelope."""
    mr_data = payload.get("MRData", {}) if isinstance(payload, dict) else {}
    race_table = mr_data.get("RaceTable", {}) if isinstance(mr_data, dict) else {}
    races = race_table.get("Races", []) if isinstance(race_table, dict) else []
    if not isinstance(races, list):
        return []
    return races


def _extract_pagination(payload):
    """Read total/limit/offset metadata from MRData safely."""
    mr_data = payload.get("MRData", {}) if isinstance(payload, dict) else {}
    total = _safe_int(mr_data.get("total")) if isinstance(mr_data, dict) else None
    limit = _safe_int(mr_data.get("limit")) if isinstance(mr_data, dict) else None
    offset = _safe_int(mr_data.get("offset")) if isinstance(mr_data, dict) else None
    return total, limit, offset


def flatten_json_to_df(payload):
    """Flatten nested API response into a normalized pandas DataFrame."""
    rows = []

    for race in _extract_races(payload):
        season = _safe_int(race.get("season"))
        round_number = _safe_int(race.get("round"))
        race_id = f"{season}_{round_number}"

        circuit = race.get("Circuit", {}) if isinstance(race, dict) else {}
        circuit_id = circuit.get("circuitId") if isinstance(circuit, dict) else None

        results = race.get("Results", []) if isinstance(race, dict) else []
        if not isinstance(results, list):
            results = []

        for result in results:
            driver = result.get("Driver", {}) if isinstance(result, dict) else {}
            constructor = result.get("Constructor", {}) if isinstance(result, dict) else {}

            row = {
                "race_id": race_id,
                "season": season,
                "round": round_number,
                "circuit_id": circuit_id,
                "driver_id": driver.get("driverId") if isinstance(driver, dict) else None,
                "constructor_id": constructor.get("constructorId") if isinstance(constructor, dict) else None,
                "grid_position": _safe_int(result.get("grid")) if isinstance(result, dict) else None,
                "finish_position": _safe_int(result.get("positionOrder"))
                if isinstance(result, dict) and result.get("positionOrder") is not None
                else (_safe_int(result.get("position")) if isinstance(result, dict) else None),
                "points": _safe_float(result.get("points")) if isinstance(result, dict) else 0.0,
                "status_str": result.get("status") if isinstance(result, dict) else None,
            }
            row["is_wet"] = _infer_is_wet(row["status_str"])
            row["track_temp"] = _infer_track_temp(row["is_wet"])
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[OUTPUT_COLUMNS].copy()
    return df


def fetch_historical_results(start_year, end_year):
    """Fetch and combine seasonal bulk payloads from start_year to end_year."""
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year.")

    frames = []
    session = requests.Session()
    years = list(range(start_year, end_year + 1))

    for index, year in enumerate(years):
        page_offset = 0
        season_total = None

        while True:
            url = _build_bulk_url(year=year, limit=1000, offset=page_offset)
            payload = fetch_raw_payload(url, session=session)
            page_df = flatten_json_to_df(payload)
            frames.append(page_df)

            total, api_limit, _ = _extract_pagination(payload)
            if season_total is None:
                season_total = total

            retrieved_rows = len(page_df)
            if retrieved_rows == 0:
                break

            if season_total is not None and (page_offset + retrieved_rows) >= season_total:
                break

            if api_limit is not None and api_limit > 0:
                page_offset += api_limit
            else:
                page_offset += retrieved_rows

            time.sleep(REQUEST_DELAY_SECONDS)

        if index < len(years) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    combined = combined.sort_values(["season", "round", "finish_position", "driver_id"], na_position="last")
    combined = combined.reset_index(drop=True)
    return combined


def fetch_incremental_results(year, round_number):
    """Fetch one race payload and validate complete 20-driver record set."""
    url = _build_incremental_url(int(year), int(round_number))
    payload = fetch_raw_payload(url)
    race_df = flatten_json_to_df(payload)

    if len(race_df) != 20:
        raise ValueError("Incomplete data payload from API. Aborting execution.")

    race_df = race_df.sort_values(["finish_position", "driver_id"], na_position="last").reset_index(drop=True)
    return race_df


def append_to_master_dataset(master_csv_path, new_df):
    """Append new records to master dataset with deduplication safeguards."""
    master_dir = os.path.dirname(master_csv_path)
    if master_dir:
        os.makedirs(master_dir, exist_ok=True)

    if os.path.exists(master_csv_path):
        existing_df = pd.read_csv(master_csv_path)
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    merged = merged.drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    merged = merged.sort_values(["season", "round", "finish_position", "driver_id"], na_position="last")
    merged = merged.reset_index(drop=True)
    merged.to_csv(master_csv_path, index=False)
    return merged


def run_ingestion(mode, output_csv_path, start_year=None, end_year=None, year=None, round_number=None):
    """Main orchestrator for historical and incremental ingestion modes."""
    normalized_mode = str(mode).strip().lower()

    if normalized_mode == "historical":
        if start_year is None or end_year is None:
            raise ValueError("Historical mode requires start_year and end_year.")
        output_dir = os.path.dirname(output_csv_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        historical_df = fetch_historical_results(start_year=start_year, end_year=end_year)
        historical_df.to_csv(output_csv_path, index=False)
        return historical_df

    if normalized_mode == "incremental":
        if year is None or round_number is None:
            raise ValueError("Incremental mode requires year and round_number.")
        incremental_df = fetch_incremental_results(year=year, round_number=round_number)
        return append_to_master_dataset(output_csv_path, incremental_df)

    raise ValueError("Unsupported mode. Use 'historical' or 'incremental'.")


if __name__ == "__main__":
    # Required variables by mode:
    # - INGEST_MODE=historical and START_YEAR, END_YEAR
    # - INGEST_MODE=incremental and YEAR, ROUND
    # Optional variables:
    # - OUTPUT_CSV_PATH (default: data/raw/jolpica_results_master.csv)
    ingest_mode = os.environ.get("INGEST_MODE", "historical").strip().lower()
    output_path = os.environ.get("OUTPUT_CSV_PATH", "data/raw/jolpica_results_master.csv")

    if ingest_mode == "historical":
        default_end_year = datetime.datetime.now(datetime.UTC).year
        start = os.environ.get("START_YEAR", str(DEFAULT_START_YEAR))
        end = os.environ.get("END_YEAR", str(default_end_year))
        result_df = run_ingestion(
            mode="historical",
            output_csv_path=output_path,
            start_year=int(start),
            end_year=int(end),
        )
    elif ingest_mode == "incremental":
        yval = os.environ.get("YEAR")
        rval = os.environ.get("ROUND")
        if yval is None or rval is None:
            raise ValueError("Set YEAR and ROUND environment variables for incremental mode.")
        result_df = run_ingestion(
            mode="incremental",
            output_csv_path=output_path,
            year=int(yval),
            round_number=int(rval),
        )
    else:
        raise ValueError("INGEST_MODE must be either 'historical' or 'incremental'.")

    print(f"Ingestion completed in {ingest_mode} mode.")
    print(f"Rows in output dataframe: {len(result_df)}")
    print(f"Output file: {output_path}")