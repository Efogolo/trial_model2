# Trial Conversion Model

Predicts, from a trial's first 3 days of behavior, whether the user will convert to a paid plan.

## Setup

1. Install dependencies:
```bash
   uv sync
```
2. Copy `.env.example` to `.env` and fill in your database credentials:
```bash
   cp .env.example .env
```

## Usage

Pull and process the latest data (hits the database, requires `.env`):
```bash
uv run python scripts/pull_data.py
```

Train the model on the processed data (no database connection needed):
```bash
uv run python scripts/train.py
```

## Project structure

- `src/trial_conversion_model_2/` — core package: data loading/cleaning, feature engineering, training
- `scripts/` — command-line entry points
- `notebooks/` — exploratory analysis
- `data/processed/` — generated, not tracked in git
- `models/` — trained model and metrics history, not tracked in git

## Metrics

Each training run appends a record to `models/metrics.json` with the AUC and timestamp.