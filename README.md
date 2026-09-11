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

### 3.1) Configure local environment variables

This repo includes a local `.env` file for runtime configuration. The file is
gitignored and loaded automatically by both `app.py` and `api/index.py`.

If needed, copy from the template and edit values:

```bash
cp .env.example .env
```

Current keys used:

- `MODEL_REGISTRY_REPO`
- `MODEL_REGISTRY_BRANCH`
- `START_YEAR` (optional pipeline override)
- `SIM_DEFAULT_IS_WET` (optional)
- `SIM_DEFAULT_TRACK_TEMP` (optional)

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

## Vercel Deployment Guide (Recommended For Always-On Hosting)

This repository is now Vercel-ready using a Python serverless API in `api/index.py`.
The Streamlit app remains available for local usage, while Vercel hosts a stable
JSON API that does not auto-sleep like Streamlit Community Cloud.

### What was added for Vercel

- `api/index.py`: FastAPI inference service
- `vercel.json`: Vercel build/runtime config
- `requirements-vercel.txt`: lean dependency set for serverless builds
- `.vercelignore`: trims large folders from deployment bundle

### 1) Push repository updates

```bash
git add .
git commit -m "Add Vercel-ready inference API deployment"
git push
```

### 2) Import project into Vercel

- Go to Vercel dashboard -> `Add New...` -> `Project`
- Import this GitHub repository
- Framework preset: `Other`
- Keep default root directory (repository root)

### 3) Configure environment variables in Vercel

Set these in Project Settings -> Environment Variables:

- `MODEL_REGISTRY_REPO` (default: `CKmellow/f1-veteran-ranker`)
- `MODEL_REGISTRY_BRANCH` (default: `model-registry`)

Important: the API fetches model/data artifacts from the model registry branch
when local artifacts are not bundled. If the branch is private, artifact download
will fail without a public endpoint.

### 4) Deploy

- Trigger deployment from the Vercel dashboard
- After deploy, verify:

```bash
curl https://<your-vercel-domain>/api/health
```

Expected: `status: ok` when model and feature artifacts are available.

### 5) Inference endpoint usage

```bash
curl -X POST https://<your-vercel-domain>/api/predict/live \
    -H "Content-Type: application/json" \
    -d '{
        "circuit_label": "Balanced/Technical",
        "is_wet": 0,
        "track_temp": 25,
        "overrides": [
            {"driver_id": "hamilton", "grid_position": 3, "quali_position": 2},
            {"driver_id": "verstappen", "grid_position": 1, "quali_position": 1}
        ]
    }'
```

### API routes

- `GET /api`
- `GET /api/health`
- `GET /api/drivers`
- `POST /api/predict/live`

### Notes on architecture

- Vercel deployment targets the API, not Streamlit UI.
- Model/data artifacts are loaded from repository files if present; otherwise,
    fetched from model registry and cached in `/tmp` per serverless instance.
- This keeps deployments lightweight and resilient when large ML artifacts are
    not committed directly to the branch.

## GitHub Actions Automation (Weekly + Race-Week)

This repository includes a scheduled workflow at `.github/workflows/pipeline.yml` that keeps the model refreshed automatically.

### Automated schedule

- Runs daily at 02:15 UTC.
- Runs additional race-window refreshes every 6 hours from Friday through Monday (UTC).
- Uses `historical` ingestion mode from `START_YEAR=2022` through the current year.
- Rebuilds datasets, retrains rankers, and uploads artifacts.

### Streamlit keepalive automation

To reduce Streamlit inactivity pauses, the repository includes a keepalive
workflow at `.github/workflows/streamlit-keepalive.yml` that pings the deployed
app every 30 minutes.

- Health endpoint: `/_stcore/health`
- Root endpoint: `/`

Note: on free hosting tiers, keepalive is best-effort and may not fully bypass
provider inactivity policies.

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
