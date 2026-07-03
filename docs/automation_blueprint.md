# Automation Blueprint

## End-to-End Pipeline Commands

Run these commands from repository root.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 1) Results ingestion (historical bulk build)
INGEST_MODE=historical START_YEAR=2022 END_YEAR=$(date +%Y) \
OUTPUT_CSV_PATH=data/raw/jolpica_results_master.csv \
python src/preprocessing/ingest_jolpica_results.py

# 2) Qualifying ingestion + canonical merge
INGEST_MODE=historical START_YEAR=2022 END_YEAR=$(date +%Y) \
RESULTS_MASTER_CSV_PATH=data/raw/jolpica_results_master.csv \
QUALIFYING_MASTER_CSV_PATH=data/raw/jolpica_qualifying_master.csv \
CANONICAL_OUTPUT_CSV_PATH=data/raw/f1_canonical_master.csv \
python src/preprocessing/ingest_jolpica_qualifying.py

# 3) Veteran feature engineering matrix
CANONICAL_INPUT_CSV_PATH=data/raw/f1_canonical_master.csv \
VETERAN_OUTPUT_CSV_PATH=data/processed/veteran_training_matrix.csv \
python src/preprocessing/build_veteran_features.py
```

## Incremental Weekly Update Commands

Use these after each completed race weekend once final classified results are stable.

```bash
# Example: YEAR=2026 ROUND=11
INGEST_MODE=incremental YEAR=2026 ROUND=11 \
OUTPUT_CSV_PATH=data/raw/jolpica_results_master.csv \
python src/preprocessing/ingest_jolpica_results.py

INGEST_MODE=incremental YEAR=2026 ROUND=11 \
RESULTS_MASTER_CSV_PATH=data/raw/jolpica_results_master.csv \
QUALIFYING_MASTER_CSV_PATH=data/raw/jolpica_qualifying_master.csv \
CANONICAL_OUTPUT_CSV_PATH=data/raw/f1_canonical_master.csv \
python src/preprocessing/ingest_jolpica_qualifying.py

CANONICAL_INPUT_CSV_PATH=data/raw/f1_canonical_master.csv \
VETERAN_OUTPUT_CSV_PATH=data/processed/veteran_training_matrix.csv \
python src/preprocessing/build_veteran_features.py
```

## GitHub Actions Workflow Template

Create `.github/workflows/pipeline.yml` with this template:

```yaml
name: f1-veteran-pipeline

on:
  schedule:
    - cron: "0 21 * * 0" # Every Sunday at 21:00 UTC
  workflow_dispatch:

jobs:
  run-pipeline:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Ingest Results (incremental)
        env:
          INGEST_MODE: incremental
          YEAR: ${{ vars.F1_YEAR }}
          ROUND: ${{ vars.F1_ROUND }}
          OUTPUT_CSV_PATH: data/raw/jolpica_results_master.csv
        run: python src/preprocessing/ingest_jolpica_results.py

      - name: Ingest Qualifying and Build Canonical
        env:
          INGEST_MODE: incremental
          YEAR: ${{ vars.F1_YEAR }}
          ROUND: ${{ vars.F1_ROUND }}
          RESULTS_MASTER_CSV_PATH: data/raw/jolpica_results_master.csv
          QUALIFYING_MASTER_CSV_PATH: data/raw/jolpica_qualifying_master.csv
          CANONICAL_OUTPUT_CSV_PATH: data/raw/f1_canonical_master.csv
        run: python src/preprocessing/ingest_jolpica_qualifying.py

      - name: Build Veteran Feature Matrix
        env:
          CANONICAL_INPUT_CSV_PATH: data/raw/f1_canonical_master.csv
          VETERAN_OUTPUT_CSV_PATH: data/processed/veteran_training_matrix.csv
        run: python src/preprocessing/build_veteran_features.py

      - name: Upload Pipeline Artifacts
        uses: actions/upload-artifact@v4
        with:
          name: f1-veteran-data-artifacts
          path: |
            data/raw/jolpica_results_master.csv
            data/raw/jolpica_qualifying_master.csv
            data/raw/f1_canonical_master.csv
            data/processed/veteran_training_matrix.csv
```

## Recommended Repository Variables

Set these in GitHub repository settings under Variables:

- `F1_YEAR` (example: `2026`)
- `F1_ROUND` (example: `11`)

Update `F1_ROUND` each race weekend before the scheduled run.