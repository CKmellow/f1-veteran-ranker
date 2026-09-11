# Vercel API Endpoints

This folder contains the Vercel deployment entrypoint for Python serverless inference.

## Routes

- `GET /api` - service metadata
- `GET /api/health` - artifact and model readiness
- `GET /api/drivers` - default driver setup and latest rolling features
- `POST /api/predict/live` - run live race prediction with optional grid/qualifying overrides

## Required artifacts

The API loads artifacts in this order:

1. Repository-local files (if present)
2. Download from `https://raw.githubusercontent.com/<MODEL_REGISTRY_REPO>/<MODEL_REGISTRY_BRANCH>/...`

Default values:

- `MODEL_REGISTRY_REPO=CKmellow/f1-veteran-ranker`
- `MODEL_REGISTRY_BRANCH=model-registry`

If your registry branch is private, expose artifacts publicly or switch to an authenticated artifact storage provider.
