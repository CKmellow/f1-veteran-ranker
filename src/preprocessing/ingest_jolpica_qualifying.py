"""Jolpica F1 qualifying ingestion pipeline.

This script mirrors the operational model from ingest_jolpica_results.py:
1) Historical bulk fetch across a year range with pagination.
2) Incremental fetch for a specific year/round with payload validation.

It also merges qualifying rows into the existing results master table to
produce a canonical dataset.
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

QUALI_OUTPUT_COLUMNS = [
    "race_id",
    "driver_id",
    "quali_position",
]


def _safe_int(value):
    """Convert value to int, returning None if conversion is not possible."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_bulk_url(year, limit=1000, offset=0):
    """Build the Jolpica bulk qualifying endpoint URL for a season."""
    return f"{BASE_URL}/{int(year)}/qualifying.json?limit={int(limit)}&offset={int(offset)}"


def _build_incremental_url(year, round_number):
    """Build the Jolpica incremental qualifying endpoint URL for a race."""
    return f"{BASE_URL}/{int(year)}/{int(round_number)}/qualifying.json?limit=100"


def fetch_raw_payload(url, session=None, timeout=DEFAULT_TIMEOUT_SECONDS, max_retries=DEFAULT_MAX_RETRIES):
    """Fetch JSON payload with retry/backoff and strict HTTP checks."""
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
    """Defensively extract the races list from the MRData envelope."""
    mr_data = payload.get("MRData", {}) if isinstance(payload, dict) else {}
    race_table = mr_data.get("RaceTable", {}) if isinstance(mr_data, dict) else {}
    races = race_table.get("Races", []) if isinstance(race_table, dict) else []
    if not isinstance(races, list):
        return []
    return races


def _extract_pagination(payload):
    """Read total/limit/offset pagination fields from MRData safely."""
    mr_data = payload.get("MRData", {}) if isinstance(payload, dict) else {}
    total = _safe_int(mr_data.get("total")) if isinstance(mr_data, dict) else None
    limit = _safe_int(mr_data.get("limit")) if isinstance(mr_data, dict) else None
    offset = _safe_int(mr_data.get("offset")) if isinstance(mr_data, dict) else None
    return total, limit, offset


def flatten_json_to_df(payload):
    """Flatten nested qualifying payload into a normalized pandas DataFrame."""
    rows = []

    for race in _extract_races(payload):
        season = _safe_int(race.get("season"))
        round_number = _safe_int(race.get("round"))
        race_id = f"{season}_{round_number}"

        qualifying_results = race.get("QualifyingResults", []) if isinstance(race, dict) else []
        if not isinstance(qualifying_results, list):
            qualifying_results = []

        for result in qualifying_results:
            driver = result.get("Driver", {}) if isinstance(result, dict) else {}
            row = {
                "race_id": race_id,
                "driver_id": driver.get("driverId") if isinstance(driver, dict) else None,
                "quali_position": _safe_int(result.get("position")) if isinstance(result, dict) else None,
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=QUALI_OUTPUT_COLUMNS)

    for column in QUALI_OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = pd.NA

    df = df[QUALI_OUTPUT_COLUMNS].copy()
    return df


def fetch_historical_qualifying(start_year, end_year):
    """Fetch and combine seasonal qualifying payloads with robust pagination."""
    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        raise ValueError("start_year must be less than or equal to end_year.")

    frames = []
    session = requests.Session()
    years = list(range(start_year, end_year + 1))

    for year_index, year in enumerate(years):
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

        if year_index < len(years) - 1:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        return pd.DataFrame(columns=QUALI_OUTPUT_COLUMNS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    combined = combined.sort_values(["race_id", "quali_position", "driver_id"], na_position="last")
    combined = combined.reset_index(drop=True)
    return combined


def fetch_incremental_qualifying(year, round_number):
    """Fetch one qualifying payload and enforce complete 20-driver rows."""
    url = _build_incremental_url(year=year, round_number=round_number)
    payload = fetch_raw_payload(url)
    race_df = flatten_json_to_df(payload)

    if len(race_df) != 20:
        raise ValueError("Incomplete data payload from API. Aborting execution.")

    race_df = race_df.sort_values(["quali_position", "driver_id"], na_position="last").reset_index(drop=True)
    return race_df


def append_to_qualifying_master(qualifying_master_csv_path, new_df):
    """Append qualifying rows to the local master file with deduplication."""
    output_dir = os.path.dirname(qualifying_master_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(qualifying_master_csv_path):
        existing_df = pd.read_csv(qualifying_master_csv_path)
        merged = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        merged = new_df.copy()

    merged = merged.drop_duplicates(subset=["race_id", "driver_id"], keep="last")
    merged = merged.sort_values(["race_id", "quali_position", "driver_id"], na_position="last")
    merged = merged.reset_index(drop=True)
    merged.to_csv(qualifying_master_csv_path, index=False)
    return merged


def load_results_master(results_master_csv_path):
    """Load race-results master table, failing fast if unavailable."""
    if not os.path.exists(results_master_csv_path):
        raise FileNotFoundError(
            f"Results master file not found: {results_master_csv_path}. "
            "Run ingest_jolpica_results.py first."
        )
    return pd.read_csv(results_master_csv_path)


def merge_results_and_qualifying(results_df, qualifying_df):
    """Left-join qualifying position onto results with composite key."""
    right = qualifying_df[["race_id", "driver_id", "quali_position"]].copy()
    left = results_df.drop(columns=["quali_position"], errors="ignore")
    merged = left.merge(right, how="left", on=["race_id", "driver_id"])
    return merged


def save_dataframe(df, output_csv_path):
    """Save dataframe to CSV after ensuring destination directory exists."""
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    df.to_csv(output_csv_path, index=False)


def run_ingestion(
    mode,
    results_master_csv_path,
    canonical_output_csv_path,
    qualifying_master_csv_path="data/raw/jolpica_qualifying_master.csv",
    start_year=None,
    end_year=None,
    year=None,
    round_number=None,
):
    """Orchestrate qualifying ingestion and canonical merge output."""
    normalized_mode = str(mode).strip().lower()

    if normalized_mode == "historical":
        if start_year is None or end_year is None:
            raise ValueError("Historical mode requires start_year and end_year.")
        qualifying_df = fetch_historical_qualifying(start_year=start_year, end_year=end_year)
        save_dataframe(qualifying_df, qualifying_master_csv_path)
    elif normalized_mode == "incremental":
        if year is None or round_number is None:
            raise ValueError("Incremental mode requires year and round_number.")
        incremental_df = fetch_incremental_qualifying(year=year, round_number=round_number)
        qualifying_df = append_to_qualifying_master(qualifying_master_csv_path, incremental_df)
    else:
        raise ValueError("Unsupported mode. Use 'historical' or 'incremental'.")

    results_df = load_results_master(results_master_csv_path)
    canonical_df = merge_results_and_qualifying(results_df=results_df, qualifying_df=qualifying_df)
    save_dataframe(canonical_df, canonical_output_csv_path)
    return canonical_df


if __name__ == "__main__":
    """Execute qualifying ingestion with environment-variable configuration.

    Required variables by mode:
    - INGEST_MODE=historical and START_YEAR, END_YEAR
    - INGEST_MODE=incremental and YEAR, ROUND

    Optional variables:
    - RESULTS_MASTER_CSV_PATH (default: data/raw/jolpica_results_master.csv)
    - QUALIFYING_MASTER_CSV_PATH (default: data/raw/jolpica_qualifying_master.csv)
    - CANONICAL_OUTPUT_CSV_PATH (default: data/raw/f1_canonical_master.csv)
    """
    ingest_mode = os.environ.get("INGEST_MODE", "historical").strip().lower()
    results_master_path = os.environ.get("RESULTS_MASTER_CSV_PATH", "data/raw/jolpica_results_master.csv")
    qualifying_master_path = os.environ.get("QUALIFYING_MASTER_CSV_PATH", "data/raw/jolpica_qualifying_master.csv")
    canonical_output_path = os.environ.get("CANONICAL_OUTPUT_CSV_PATH", "data/raw/f1_canonical_master.csv")

    if ingest_mode == "historical":
        default_end_year = datetime.datetime.now(datetime.UTC).year
        start = os.environ.get("START_YEAR", str(DEFAULT_START_YEAR))
        end = os.environ.get("END_YEAR", str(default_end_year))
        output_df = run_ingestion(
            mode="historical",
            results_master_csv_path=results_master_path,
            canonical_output_csv_path=canonical_output_path,
            qualifying_master_csv_path=qualifying_master_path,
            start_year=int(start),
            end_year=int(end),
        )
    elif ingest_mode == "incremental":
        yval = os.environ.get("YEAR")
        rval = os.environ.get("ROUND")
        if yval is None or rval is None:
            raise ValueError("Set YEAR and ROUND environment variables for incremental mode.")
        output_df = run_ingestion(
            mode="incremental",
            results_master_csv_path=results_master_path,
            canonical_output_csv_path=canonical_output_path,
            qualifying_master_csv_path=qualifying_master_path,
            year=int(yval),
            round_number=int(rval),
        )
    else:
        raise ValueError("INGEST_MODE must be either 'historical' or 'incremental'.")

    print(f"Qualifying ingestion completed in {ingest_mode} mode.")
    print(f"Rows in canonical output dataframe: {len(output_df)}")
    print(f"Canonical output file: {canonical_output_path}")