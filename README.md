# F1 Veteran Explainable Race Ranker

## Project Overview

This repository delivers a production-oriented, explainable machine learning ranking system focused on a fixed cohort of 14 Formula 1 veteran drivers. The pipeline ingests race and qualifying records, constructs leakage-safe pre-race features, trains list-wise ranking models, and evaluates ranking quality with information-retrieval metrics that reflect race-order accuracy rather than simple win/loss classification. The system is designed to answer two operational questions with rigor: whether the model can reliably beat a free qualifying-order baseline, and which pre-race variables most strongly influence predicted finishing order.

The deployed architecture uses two gradient-boosted rankers, `XGBRanker` and `LGBMRanker`, trained with chronological controls and race-level grouping constraints so that every experiment respects real-world temporal causality. The evaluation layer emphasizes top-heavy quality (podium accuracy), global ordering quality, and reciprocal rank behavior of winner prediction. In practical terms, this means the framework can be used as a transparent race-weekend decision support system, not just a notebook experiment.

Core evaluation outputs include Global NDCG, NDCG@3, MRR, MAP, MAP@3, and Precision@3, reported with qualifying-baseline uplift deltas for direct operational benchmarking.

## Repository Structure Map

```text
.
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   │   ├── f1_canonical_master.csv
│   │   ├── jolpica_qualifying_master.csv
│   │   └── jolpica_results_master.csv
│   └── processed/
│       └── veteran_training_matrix.csv
├── docs/
│   ├── MODEL_METRICS_AND_METHODOLOGY.md
│   ├── SYSTEM_ARCHITECTURE.md
│   └── automation_blueprint.md
├── models/
│   ├── f1_xgb_ranker.pkl
│   └── f1_lgb_ranker.pkl
├── notebooks/
├── outputs/
│   └── reports/
│       ├── xgb_shap_summary.png
│       └── lgb_shap_summary.png
├── reports/
└── src/
    ├── preprocessing/
    │   ├── ingest_jolpica_results.py
    │   ├── ingest_jolpica_qualifying.py
    │   └── build_veteran_features.py
    ├── models/
    │   └── train_ranker.py
    ├── features/
    ├── evaluation/
    └── visualization/
```

## Production Setup Guide

### 1) Clone the repository

```bash
git clone <your-repository-url>
cd Machine_Learning_Group_Project
```

### 2) Provision Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Optional environment verification

```bash
python --version
pip --version
```

## Operational Execution Runbook

Run the full system sequentially in the exact order below.

### Step 1: Ingest race results

```bash
python src/preprocessing/ingest_jolpica_results.py
```

### Step 2: Ingest qualifying and build canonical table

```bash
python src/preprocessing/ingest_jolpica_qualifying.py
```

### Step 3: Build veteran feature matrix

```bash
python src/preprocessing/build_veteran_features.py
```

Optional sanity check before training (confirm latest season is present):

<!-- ```bash
python -c "import pandas as pd; d=pd.read_csv('data/processed/veteran_training_matrix.csv'); print(d['season'].value_counts().sort_index())"
``` -->

### Step 4: Train rankers and evaluate against baseline

```bash
python src/models/train_ranker.py
```

### Step 5: Launch interactive Streamlit app

```bash
streamlit run app.py
```

## GitHub Actions Automation (Weekly + Race-Week)

This repository includes a scheduled workflow at `.github/workflows/pipeline.yml` that keeps the model refreshed automatically.

### Automated schedule

- Runs daily at 02:15 UTC.
- Runs additional race-window refreshes every 6 hours from Friday through Monday (UTC).
- Uses `historical` ingestion mode from `START_YEAR=2022` through the current year.
- Rebuilds datasets, retrains rankers, and uploads artifacts.

### Manual trigger options

In GitHub: `Actions` -> `f1-veteran-pipeline` -> `Run workflow`

- `ingest_mode=historical`:
    - Full refresh and retraining.
- `ingest_mode=incremental`:
    - Set both `year` and `round` to ingest a specific race weekend, then retrain.

Example manual run values:

- `ingest_mode`: `incremental`
- `year`: `2026`
- `round`: `15`

### Workflow outputs

Each run uploads the following artifacts:

- `data/raw/jolpica_results_master.csv`
- `data/raw/jolpica_qualifying_master.csv`
- `data/raw/f1_canonical_master.csv`
- `data/processed/veteran_training_matrix.csv`
- `models/f1_xgb_ranker.pkl`
- `models/f1_lgb_ranker.pkl`
- `outputs/reports/xgb_shap_summary.png`
- `outputs/reports/lgb_shap_summary.png`

### Streamlit deployment note (missing model files)

If Streamlit shows missing artifacts such as `models/f1_xgb_ranker.pkl` or
`models/f1_lgb_ranker.pkl`, use the in-app recovery button:

- Open the deployed app.
- Click `Build Missing Artifacts Now`.
- Wait for the full pipeline to finish (results ingestion -> qualifying merge ->
    veteran feature matrix build -> ranker training).
- The app reloads automatically with the rebuilt model artifacts.

This recovery flow is useful because GitHub Actions artifacts are not a
persistent model registry for Streamlit deployments.

### Durable model storage

The workflow now writes a durable snapshot branch named `model-registry` that
stores the latest deployable artifacts:

- `models/f1_xgb_ranker.pkl`
- `models/f1_lgb_ranker.pkl`
- `data/raw/f1_canonical_master.csv`
- `data/processed/veteran_training_matrix.csv`
- `outputs/reports/xgb_shap_summary.png`
- `outputs/reports/lgb_shap_summary.png`
- `outputs/reports/refresh_metadata.json`

The Streamlit app uses this branch as a fallback source if local artifacts are
missing during startup.

### Streamlit race-week auto-refresh behavior

The app now derives race-week context dynamically from refreshed data:

- Retrospective rounds are populated from completed rounds in the latest
    available season data.
- Upcoming GP options are pulled from the current season schedule and move
    forward automatically as rounds are completed.

After a race weekend, run the data/training pipeline (or wait for scheduled
GitHub Actions) and restart/reload Streamlit to reflect the new current vs
historical split.

The app sidebar also displays `Last data/model refresh` using
`outputs/reports/refresh_metadata.json` (or filesystem timestamps as fallback)
for run-to-run trust and traceability.

## Primary Artifacts Produced

- `models/f1_xgb_ranker.pkl`: optimized XGBoost ranker package with metadata.
- `models/f1_lgb_ranker.pkl`: optimized LightGBM ranker package with metadata.
- `outputs/reports/xgb_shap_summary.png`: SHAP global importance view for XGBoost.
- `outputs/reports/lgb_shap_summary.png`: SHAP global importance view for LightGBM.

## Engineering Notes

- Historical ingestion defaults span from 2022 through the current calendar year.
- Train/test split is dynamic: train uses all seasons before the latest available season, and test uses the latest available season.
- Race groups remain contiguous by `race_id` during fitting and scoring.
- Expanding chronological CV folds are generated dynamically and always end at `latest_season - 1`.
- The training logger prints NDCG, MRR, MAP, MAP@3, NDCG@3, and Precision@3 with direct Qualifying Baseline uplift comparisons.
