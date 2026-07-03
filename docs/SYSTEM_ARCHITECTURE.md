# SYSTEM ARCHITECTURE

## 1. Project Executive Summary

This platform delivers race-level ranking prediction for Formula 1 using a strict veteran-only cohort of 14 drivers. The design intentionally excludes 2026 rookies from training targets to mitigate cold-start sparsity and unstable priors in early-season data.

Core objective:
- Predict relative finishing order for the veteran cohort using leak-safe pre-race and historical features.
- Support both forward-looking simulation and retrospective evaluation inside a production Streamlit interface.

Timeline strategy:
- Training window: all seasons before the latest available season in the processed matrix.
- Validation window: latest available season in the processed matrix.
- This chronological split enforces realistic deployment behavior and avoids look-ahead leakage.

---

## 2. Data Ingestion Architecture

### 2.1 Results Ingestion (`src/preprocessing/ingest_jolpica_results.py`)

This module ingests Sunday race results from the Jolpica API (Ergast successor) in two modes:
- Historical bulk ingestion by season range.
- Incremental ingestion by exact year/round.

Key engineering controls:
- **Rate limiting:** request throttling with `REQUEST_DELAY_SECONDS = 1.0` between calls.
- **Pagination guard:** dynamic `offset` stepping with metadata extraction from `MRData.total`, `MRData.limit`, and `MRData.offset` to prevent truncated season downloads.
- **Incremental completeness gate:** strict `len(race_df) == 20` validation before appending race rows.

Output schema is normalized per driver-race row and includes:
- Race identity: `race_id`, `season`, `round`, `circuit_id`
- Competitor identity: `driver_id`, `constructor_id`
- Grid/outcome: `grid_position`, `finish_position`, `points`, `status_str`
- Environment fields: `is_wet`, `track_temp`

### 2.2 Qualifying Ingestion (`src/preprocessing/ingest_jolpica_qualifying.py`)

This module mirrors results ingestion behavior for qualifying sessions:
- Historical and incremental modes.
- Identical pagination guard pattern.
- Identical 20-row incremental completeness requirement.

Canonical merge behavior:
- Loads results master.
- Left-joins qualifying rows (`quali_position`) on composite key: `race_id`, `driver_id`.
- Produces canonical output file used by downstream feature engineering.

### 2.3 API Source

Data is fetched from:
- Jolpica F1 API (`https://api.jolpi.ca/ergast/f1`)
- This is the open-source successor pathway to Ergast-compatible endpoints.

### 2.4 Sunday Night Incremental Validation Block

For weekly race updates, the ingestion scripts enforce a hard completeness gate before writing:
- If returned rows are not exactly 20 for the requested weekend, execution aborts with a `ValueError`.
- This protects dataset integrity during partial-result windows.

---

## 3. Feature Engineering and Preprocessing Matrix

### 3.1 Veteran Matrix Builder (`src/preprocessing/build_veteran_features.py`)

The veteran matrix build process:
- Normalizes driver/team/circuit identifiers.
- Filters to a strict 14-driver veteran set.
- Produces race-sequenced, leak-safe model features.

### 3.2 Leakage-Safe Feature Construction

All temporal aggregates are computed with shifted history (`shift(1)`) before rolling/expanding operations:
- `driver_form_3races`: trailing 3-race average finish proxy (driver momentum).
- `circuit_historical_avg`: driver historical average at specific circuit.
- `constructor_points_current`: season-to-date constructor points before current race.
- `constructor_dnf_rate_10races`: trailing constructor mechanical DNF rate over 10-race window.

This shift-before-aggregate design prevents temporal mixing and target leakage.

### 3.3 Environmental and Aerodynamic Inputs

The matrix retains and standardizes environmental fields:
- `is_wet`: wet-condition indicator.
- `track_temp`: track temperature proxy/thresholded numeric input.
- `circuit_type` and `circuit_type_code`: aerodynamic/archetype mapping
  - `Power/High-Speed` -> `0`
  - `High-Downforce` -> `1`
  - `Balanced/Technical` -> `2`
  - `Street Circuit` -> `3`

### 3.4 Driver ID Normalization Patch

A normalization asset is applied:
- `max_verstappen` -> `verstappen`

This resolves cross-source identifier mismatch and preserves historical continuity.

---

## 4. Machine Learning Modeling Core

### 4.1 Training Orchestrator (`src/models/train_ranker.py`)

The training module runs a dual-model benchmark:
- `XGBRanker` with `objective="rank:pairwise"`
- `LGBMRanker` with `objective="lambdarank"`

Both models train on identical grouped race partitions for fair comparison.

### 4.2 Relevance Orientation and Label Safety

Target transformation:
- `relevance_score = 15 - finish_position`

Why it matters:
- Rankers optimize for higher relevance scores.
- Without inversion, the model would orient rankings incorrectly.

Non-negative clipping for LightGBM compatibility:
- `model_relevance_score = relevance_score.clip(lower=0)`
- Required because LambdaRank labels must be non-negative.

### 4.3 Group-Aware Training Split

Chronological split:
- Train: seasons `< max_season`
- Validate: season `max_season`

Group handling:
- Race-level group vectors (`group` and `eval_group`) are built from contiguous `race_id` blocks.

---

## 5. Evaluation Performance Metrics

### 5.1 Primary Validation Outputs (Latest Season)

Observed benchmark values:
- **XGBoost**: Global NDCG = **0.8921**, Winner MRR = **0.3125**
- **LightGBM** is trained and reported in parallel for model selection benchmarking.

### 5.2 Why Group-Aware MRR is Custom

A custom race-group-aware Mean Reciprocal Rank function is implemented because:
- Standard flat ranking metrics in scikit-learn do not natively model closed race ecosystems as independent groups.
- F1 outcomes are evaluated per race cohort, not as a single global sample pool.

Custom MRR logic:
- For each race group, sort by predicted score.
- Find reciprocal rank of the true winner.
- Average reciprocal ranks across race groups.

### 5.3 Explainability Artifacts

SHAP visual diagnostics are generated for both models:
- `outputs/reports/xgb_shap_summary.png`
- `outputs/reports/lgb_shap_summary.png`

---

## 6. Interactive Visualization Engine

### 6.1 Application Module (`app.py`)

The Streamlit app provides:
- Live simulation for upcoming race setup.
- Retrospective audit against completed rounds in the latest available season.
- Theme-aware rendering across dark/light contexts.

### 6.2 Theme-Agnostic Styling

The UI uses native CSS variable integration and controlled overrides:
- `var(--text-color)`
- `var(--background-color)`
- `var(--secondary-background-color)`

This supports contrast scaling across both theme modes while preserving custom broadcast components.

### 6.3 Broadcast Table and Cohort Logic

Current table breakout is rendered into dedicated columns:
- `PRED RANK`
- `DRIVER`
- `TEAM`
- `GLOBAL FINISH`
- `COHORT RANK`
- `ERROR DELTA`

Retrospective ranking uses cohort compression:
- Absolute finish remains visible as global race result (`P# / full_grid_size`).
- Relative veteran rank is recomputed in-cohort (`1..14`) for aligned error deltas.

### 6.4 Automated Upcoming GP Profiles

The app includes a backend profile dictionary for event-specific environment settings:
- Upcoming GP dropdown auto-applies `circuit_type_code`, `is_wet`, and `track_temp`.
- Optional manual override mode is available but disabled by default.

This removes repetitive manual tuning during routine usage.

---

## 7. Step-by-Step Environment Setup and Execution Guide

Run from repository root:

```bash
git clone <your-repo-url>
cd Machine_Learning_Group_Project

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 7.1 Ingest Results

```bash
INGEST_MODE=historical START_YEAR=2022 END_YEAR=2026 \
OUTPUT_CSV_PATH=data/raw/jolpica_results_master.csv \
python src/preprocessing/ingest_jolpica_results.py
```

### 7.2 Ingest Qualifying and Build Canonical

```bash
INGEST_MODE=historical START_YEAR=2022 END_YEAR=2026 \
RESULTS_MASTER_CSV_PATH=data/raw/jolpica_results_master.csv \
QUALIFYING_MASTER_CSV_PATH=data/raw/jolpica_qualifying_master.csv \
CANONICAL_OUTPUT_CSV_PATH=data/raw/f1_canonical_master.csv \
python src/preprocessing/ingest_jolpica_qualifying.py
```

### 7.3 Build Veteran Feature Matrix

```bash
CANONICAL_INPUT_CSV_PATH=data/raw/f1_canonical_master.csv \
VETERAN_OUTPUT_CSV_PATH=data/processed/veteran_training_matrix.csv \
python src/preprocessing/build_veteran_features.py
```

### 7.4 Train Models

```bash
python src/models/train_ranker.py
```

### 7.5 Launch Interactive UI

```bash
streamlit run app.py
```

---

## 8. Operational Notes

- Keep incremental updates aligned with completed official race weekends.
- If incremental payload completeness check fails (not 20 rows), rerun later rather than forcing writes.
- Retraining should occur after canonical and veteran matrix refresh to keep model artifacts aligned with latest race state.
