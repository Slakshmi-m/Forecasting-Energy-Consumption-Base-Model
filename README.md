# Energy Forecasting for Critical Infrastructure

Safety-critical time-series forecasting pipeline for European energy load data (ENTSO-E),
implementing `spotforecast2-safe` - a Compliance-by-Design subset that embeds EU AI Act
(Reg. 2024/1689), IEC 61508, and Cyber Resilience Act requirements into API contracts,
persistence formats, and CI gates.

See the [spotforecast2-safe paper](https://arxiv.org/abs/2604.23859) and
[documentation](https://sequential-parameter-optimization.github.io/spotforecast2-safe/).

---

## Repository layout

```
.
├── pyproject.toml              # dependencies + tooling config
├── data/
│   ├── raw/                    # ENTSO-E load series landing zone
│   └── processed/              # preprocessed parquet outputs
├── model_cards/                # generated EU AI Act model cards
├── models/                     # serialised fitted forecasters (joblib)
├── submissions/                # competition submission files
├── src/energy_forecast/
│   ├── config.py               # all secrets loaded here — nowhere else
│   ├── exceptions.py           # typed exception hierarchy
│   ├── make_submission.py      # assembles competition submission
│   ├── train_model.py          # end-to-end training entry point
│   ├── data/
│   │   ├── fetch.py            # ENTSO-E API via entsoe-py
│   │   ├── preprocess.py       # deterministic preprocessing
│   │   ├── openmeteo.py        # Open-Meteo weather API client
│   │   └── weather.py          # ERA5/CDS weather data fetching
│   ├── models/
│   │   ├── baseline.py         # XGBoost/LightGBM/CatBoost + skforecast
│   │   └── persistence.py      # joblib save/load for fitted forecasters
│   ├── evaluation/
│   │   └── metrics.py          # MAPE, MAE, RMSE, walk-forward backtesting
│   └── compliance/
│       └── model_card.py       # EU AI Act model card generation
└── tests/
    ├── conftest.py
    ├── test_fetch.py
    ├── test_make_submission.py
    ├── test_model_card.py
    ├── test_models.py
    ├── test_openmeteo.py
    ├── test_persistence.py
    ├── test_preprocess.py
    ├── test_train_model.py
    └── test_weather.py
```

---

## Prerequisites

- Python 3.11+
- ENTSO-E API key - [register here](https://transparencyplatform.zendesk.com/hc/en-us/articles/12845911031188)

---

## Setup

```bash
# 1. Clone and enter the project
git clone <repo-url> && cd <repo>

# 2. Create virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -e ".[dev]"

# 4. Set up secrets — create a .env file with the following variables:
# ENTSOE_API_KEY=<your-key>          # required — see Prerequisites above
# ENTSOE_COUNTRY_CODE=DE_LU          # optional, default: DE_LU
# RANDOM_SEED=42                     # optional, default: 42
# RAW_DATA_DIR=data/raw              # optional, default: data/raw
# PROCESSED_DATA_DIR=data/processed  # optional, default: data/processed
# MODEL_CARD_DIR=model_cards         # optional, default: model_cards

# 5. Verify config loads
python -c "from energy_forecast.config import settings; print(settings.country_code)"

# 6. Run tests
pytest tests/

# 7. Fetch first data slice (2024-01-01 to 2024-02-01 for DE_LU)
python -m energy_forecast.data.fetch
```

---

## Four non-negotiable code rules

1. **Zero dead code** - no unused imports, variables, or commented-out blocks
2. **Deterministic processing** - all random ops use `settings.random_seed` explicitly
3. **Fail-safe handling** - every external call raises typed exceptions; errors never swallowed
4. **Minimal dependencies** - every dep justified in `pyproject.toml`

---

## Running tests

```bash
pytest tests/ --cov=src/energy_forecast --cov-report=term-missing
```

Minimum 80% line coverage required.
