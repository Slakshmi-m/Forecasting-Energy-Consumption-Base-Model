# Model/Method Card: energy-forecast (eigen_squad)

This card describes the day-ahead electricity load forecasting pipeline submitted by **team eigen_squad** to the DDMO SS 2026 load-forecasting challenge. It documents the pipeline's architecture, intended use, data requirements, and the conditions under which its results are valid. It follows the [Hugging Face Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook) taxonomy.

## 1. Model Details

| Field | Value |
| --- | --- |
| Name | energy-forecast |
| Version | 0.1.0 |
| Type | Deterministic recursive multi-step load forecasting pipeline combining ENTSO-E data ingestion, calendar and weather feature engineering, and a gradient-boosting regressor (CatBoost / XGBoost / LightGBM) wrapped in a `skforecast.ForecasterRecursive`. |
| Developed by | Team eigen_squad |
| Competition | DDMO SS 2026 Load-Forecasting Challenge |
| Repository | `Forecasting-Energy-Consumption-Base-Model/` |
| Language | Python 3.11 or newer |
| License | See repository root |
| Target | Day-ahead electricity load forecasting (DE_LU bidding zone, 24-hour horizon, 1-hour resolution) |

**Key technical components:**

| Component | Version | Purpose |
| --- | --- | --- |
| skforecast | >= 0.14.0 | scikit-learn-compatible recursive multi-step forecasting wrapper |
| CatBoost | >= 1.2 | Production gradient-boosting regressor (built-in categorical encoding) |
| XGBoost | >= 2.0 | Alternative gradient-boosting backend |
| LightGBM | >= 4.0 | Alternative gradient-boosting backend |
| entsoe-py | >= 0.8.0 | ENTSO-E Transparency Platform REST client |
| cdsapi / xarray | >= 0.7.7 / >= 2024.0 | Copernicus ERA5 reanalysis temperature data |
| pandas | >= 2.0 | Time-series structures and preprocessing |
| joblib | >= 1.3 | Model serialisation and persistence |

**Reproducibility:** The production forecaster is trained with a fixed random seed (`RANDOM_SEED=42`), single-threaded regressor mode (`thread_count=1`), and deterministic feature engineering (no random draws in feature construction). Given the same ENTSO-E data snapshot and seed, the output is identical on the same hardware. The seed and training timestamp are recorded in `models/metadata_<timestamp>.json`.

**Responsibilities:**

| Responsibility | Party |
| --- | --- |
| Pipeline development, training, and maintenance | Team Eigen_Squad |
| Daily submission generation | eigen_squad team |
| Real-world deployment, monitoring, and audit | System integrator (not applicable for this challenge entry) |

## 2. Intended Use and Scope

This pipeline forecasts the next 24 hours of electricity load for the **DE_LU bidding zone** (Germany + Luxembourg), using ENTSO-E historical actual load and ERA5/Open-Meteo weather data. Forecasts are produced once per day to support the challenge leaderboard submission.

**Primary use cases:**
- Challenge participation and leaderboard ranking (current intended use)
- Walk-forward backtesting and regressor comparison (CatBoost vs. XGBoost vs. LightGBM)
- Reproducible reference for time-series feature engineering on ENTSO-E load data
- Benchmark baseline for alternative forecasting methods

**Design constraints:**
- **No tuning inside this package.** Regressor hyperparameters use fixed defaults (500 estimators/iterations, seed 42). External tuning workflows can be applied and the retrained artifact dropped into `models/`.
- **Deterministic and stateless.** Given the same ENTSO-E data snapshot and seed, the output is always identical on the same hardware.
- **24-hour ahead forecasting only.** The pipeline produces exactly one 24-step forecast per target date, not rolling or probabilistic forecasts.
- **DE_LU only.** Calendar features (holidays, weekday structure) are hardcoded for Germany and Luxembourg; adaptation to other regions requires code changes.
- **Single-target regression.** Forecasts load only; weather is treated as an exogenous input.

**What the pipeline does:**
1. Fetches or reads cached ENTSO-E actual load data (DE_LU, hourly)
2. Fetches or reads cached ERA5 temperature reanalysis (Copernicus CDS, spatially averaged over DE_LU)
3. Audits gaps in the raw series before any modification (CR-1 auditability)
4. Fills gaps ≤ 2 consecutive hours via forward-fill; rejects longer gaps with an error
5. Constructs 13 calendar features (hour, weekday, month, holiday flags, interaction terms, lagged calendar values)
6. Constructs 3 weather features (temperature, 24-hour lag, 20-day rolling deviation)
7. Trains a `ForecasterRecursive(CatBoost, lags=168)` on the full cleaned history
8. Runs a walk-forward backtest (non-overlapping 24-h folds) and records MAPE, MAE, RMSE
9. Saves the fitted forecaster (joblib) and a JSON metadata sidecar
10. Generates a versioned EU AI Act Art. 13 model card
11. At submission time, applies a 7-day rolling bias correction (±4,000 MW cap) and writes the competition CSV

**What it does NOT do:**
- Plot or visualise results (no plotting backend is installed)
- Tune hyperparameters automatically (tuning is an external, separate workflow)
- Quantify forecast uncertainty (produces point forecasts only)
- Handle missing values by silent imputation (NaN / Inf raises `PreprocessingError`)
- Support grids other than DE_LU without code changes

## 3. How to Get Started

### Installation and setup

```bash
# Clone and install
git clone <repository-url>
cd Forecasting-Energy-Consumption-Base-Model
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure credentials
cp .env.example .env   # set ENTSOE_API_KEY to your 36-character UUID
```

### Train a model

```bash
# Full training run: fetches data, trains CatBoost, backtests, saves artifact + model card
python -m energy_forecast.train_model

# Train up to a specific historical cutoff (for backtesting or reproducibility)
python -m energy_forecast.train_model --cutoff 2026-06-30
```

### Generate a daily submission

```bash
# Generate forecast for the target date (uses the latest model in models/current.json)
python -m energy_forecast.make_submission --team eigen_squad --target-date 2026-07-02
```

### Compare regressor backends

```bash
# Compare CatBoost, XGBoost, and LightGBM on pre-processed data — no API calls needed
python -m energy_forecast.compare_models --lags 168 --steps 24
```

### Python API

```python
from energy_forecast.models.baseline import build_forecaster, train, backtest
from energy_forecast.data.preprocess import build_features
from energy_forecast.data.fetch import load_entsoe_series

# Load and preprocess
raw = load_entsoe_series()
clean, exog = build_features(raw)

# Train
forecaster = build_forecaster("catboost", lags=168)
train(forecaster, clean, exog=exog)

# Backtest
metrics = backtest(forecaster, clean, steps=24, initial_train_size=31524, exog=exog)
print(metrics)  # {"mape": 2.59, "mae": 1380.0, "rmse": 1920.0}
```

See `src/energy_forecast/train_model.py` for the full end-to-end workflow.

## 4. Technical Specification

### Task definition

Recursive 24-step-ahead forecasting of hourly electricity load (MW) for the DE_LU bidding zone. The pipeline produces one forecast per calendar day, submitted in the evening of day $T$ for the 24 hours of day $T+1$.

### Mathematical formulation

Given a historical load series $\{l_1, l_2, \ldots, l_T\}$ (hourly, MW) and a lag window of $w = 168$ hours (one week), the pipeline builds one feature row per target value:

$$X_{row,t} = [\,l_{t-168},\; l_{t-167},\; \ldots,\; l_{t-1},\; e_t\,] \;\rightarrow\; y_t = l_t$$

where $e_t$ is the exogenous feature vector at time $t$ (calendar and weather features). The target $l_t$ never appears in its own feature row, preventing look-ahead leakage by construction.

Recursive multi-step prediction for horizon $H = 24$:

$$\hat{l}_{T+h} = f\!\left([\,\hat{l}_{T+h-168},\;\ldots,\;\hat{l}_{T+h-1}\,],\; e_{T+h}\right), \quad h = 1, \ldots, 24$$

Predicted values from earlier steps fill the lag window as the horizon grows.

### Architecture

**Five-layer design:**

| Layer | Modules | Responsibility |
| --- | --- | --- |
| **Data I/O** | `data/fetch.py`, `data/weather.py`, `data/openmeteo.py` | ENTSO-E load fetch, ERA5 temperature, Open-Meteo forecast temperature |
| **Preprocessing** | `data/preprocess.py` | Gap audit, gap fill, calendar features, weather features |
| **Model** | `models/baseline.py`, `models/persistence.py` | Forecaster construction, training, backtest, save/load |
| **Evaluation** | `evaluation/metrics.py` | MAPE, MAE, RMSE |
| **Compliance** | `compliance/model_card.py` | EU AI Act Art. 13 model card generation |

**Pipeline flow:**

```
ENTSO-E API / Cache  →  ERA5 CDS / Cache  →  Open-Meteo API / Fallback
            ↓
      Gap audit (CR-1: log before any filling)
            ↓
      Forward-fill gaps ≤ 2 h; reject longer gaps
            ↓
      Calendar features (13) + weather features (3)
            ↓
      ForecasterRecursive(CatBoost, lags=168).fit(load, exog)
            ↓
      Walk-forward backtest (non-overlapping 24-h folds)
            ↓
      Compute MAPE, MAE, RMSE
            ↓
      Save .joblib + metadata.json + model_card.md
            ↓
      [Daily] 7-day rolling bias correction (±4,000 MW cap)
            ↓
      Validate: > 0 MW, within 5,000–150,000 MW
            ↓
      CSV: submissions/<team>/<YYYY-MM-DD>.csv
```

### Regressor options

| Regressor | Production | Hyperparameters |
| --- | --- | --- |
| **CatBoost** | Yes | 500 iterations, `thread_count=1`, `verbose=0`, `random_seed=42` |
| XGBoost | No (comparison) | 500 estimators, `n_jobs=1`, `random_state=42` |
| LightGBM | No (comparison) | 500 estimators, `n_jobs=1`, `verbose=-1`, `random_state=42` |

All three use single-threaded mode to eliminate floating-point reordering across threads and guarantee bit-for-bit reproducibility.

### Feature set

**Autoregressive lags:** 168 hourly lags (one full week of load history).

**Exogenous features — calendar (13):**

| Feature | Description |
| --- | --- |
| `hour` | Hour of day [0–23] |
| `weekday` | Day of week [0–6, Monday = 0] |
| `month` | Month of year [1–12] |
| `is_weekend` | 1 if Saturday or Sunday |
| `is_friday` | 1 if Friday |
| `is_saturday` | 1 if Saturday |
| `is_sunday` | 1 if Sunday |
| `is_holiday` | 1 if DE_LU public holiday |
| `hour_weekday_interaction` | Unique (hour, weekday) pair index [0–167] |
| `hour_weekend_interaction` | `hour × is_weekend` [0–23] |
| `hour_holiday_interaction` | `hour × is_holiday` [0–23] |
| `hour_lag_24` | Hour-of-day value 24 h ago |
| `weekday_lag_168` | Weekday value 168 h ago |

**Exogenous features — weather (3):**

| Feature | Description |
| --- | --- |
| `temp_c` | 2 m air temperature in °C (ERA5 or Open-Meteo) |
| `temp_lag_24` | Temperature 24 h ago |
| `temp_deviation` | Deviation from 20-day rolling mean temperature |

**Holiday calendar** covers all German federal public holidays plus Luxembourg national holidays, computed from an anonymous Gregorian Easter algorithm. Holidays included: New Year, Good Friday, Easter Monday, Labour Day, Ascension, Whit Monday, Corpus Christi, German Unity Day, All Saints' Day, Christmas Day, Boxing Day, Europe Day (LU), Luxembourg National Day (LU), Assumption of Mary (LU).

### Design principles

1. **Determinism:** Same input + same seed → same output bits. Fixed seeds, single-threaded regressors, no random draws in feature engineering.
2. **Leakage-free:** The target $l_t$ never appears in its own feature row.
3. **Fail-safe:** Invalid input raises an explicit typed exception (`PreprocessingError`) — never silent repair.
4. **Auditability (CR-1):** All gaps are logged before any filling; each training run produces a metadata sidecar and versioned model card.

## 5. Interfaces and Runtime

### Inputs

| Input | Format | Source | Required? |
| --- | --- | --- | --- |
| Load history | `pandas.Series`, UTC `DatetimeIndex`, 60-min, named `load_MW`, all positive | ENTSO-E Transparency / `data/raw/load_DE_LU_training_cache.csv` | Yes |
| Temperature history | `pandas.Series`, UTC `DatetimeIndex`, 60-min, Kelvin | Copernicus ERA5 / `data/raw/temperature_DE_LU_training_cache.csv` | Yes (for weather features) |
| Temperature forecast | `pandas.Series`, UTC `DatetimeIndex`, 60-min, Celsius | Open-Meteo API (up to +72 h); falls back to ERA5 rolling mean | Optional |
| Target date | ISO date string `YYYY-MM-DD` | CLI `--target-date` or current UTC date | Yes (for submission) |

All time indices must be regular 60-minute UTC grids. Any missing entry in the exogenous features raises a `ValueError` before prediction begins.

### Outputs

**Primary output — competition CSV:**

```
timestamp_utc,forecast_mw
2026-07-02T00:00:00Z,<MW>
2026-07-02T01:00:00Z,<MW>
...
2026-07-02T23:00:00Z,<MW>
```

Strictly 24 rows. Non-positive values or values outside 5,000–150,000 MW are caught before the file is written.

**Secondary outputs — training artifacts:**
- `models/forecaster_<timestamp>.joblib` — fitted `ForecasterRecursive` (compression level 3)
- `models/metadata_<timestamp>.json` — training provenance (dates, seed, n_samples, metrics)
- `models/current.json` — pointer to the latest model pair
- `model_cards/model_card_<timestamp>.md` — versioned EU AI Act Art. 13 card

### Runtime environment

| Property | Value |
| --- | --- |
| OS | macOS, Linux, Windows |
| Python | 3.11+ (tested on 3.13) |
| CPU | Single-threaded deterministic mode (`thread_count=1`); no GPU required or used |
| Memory | Peak ≈ series_length × lags × 4 bytes (32-bit feature matrix); ~500 MB for the full training set |
| Duration | CatBoost fit on ~39,000 hourly rows: seconds on one commodity CPU core |
| Network | Required for ENTSO-E and ERA5 fetch; offline operation possible with cached data |

### Serialisation

- **Forecaster:** `joblib.dump(forecaster, path, compress=3)`, `.joblib` extension
- **Forecast output:** `pandas.DataFrame` → CSV
- **Metadata:** JSON sidecar alongside each `.joblib`
- **Feature matrix:** cast to `float32` before fitting (memory efficiency)

**Reproducibility trade-off:**
- **Deterministic mode** (serial, `thread_count=1`, `RANDOM_SEED=42`): bit-identical on the same hardware, Python version, and dependency stack
- **Parallel mode** (if `n_jobs` / `thread_count` > 1): faster but NOT bit-reproducible across runs (floating-point scheduling variance)

## 6. Data and Operational Design Domain

### Data sources and provenance

| Source | Data | Frequency | Coverage | License |
| --- | --- | --- | --- | --- |
| ENTSO-E Transparency Platform | Actual Total Load, DE_LU bidding zone | 15-min resampled to 60-min mean | 2022-01-01 → present | CC0 (public domain) |
| Copernicus ERA5 Reanalysis | 2 m air temperature, DE_LU bounding box (47.3°N–55.1°N, 5.9°E–15.0°E), spatially averaged | Hourly, Kelvin | 2022-01-01 → ~T−5 days (ERA5T lag) | CC BY 4.0 |
| Open-Meteo | 2 m temperature forecast, centre of Germany (51.166°N, 10.452°E), Celsius | Hourly, up to +72 h | Live, no key required | CC BY 4.0 (non-commercial) |

**Cached snapshots (`data/raw/`):**
- `load_DE_LU_training_cache.csv` — incremental ENTSO-E cache (~39,000+ hourly rows); refreshed at each training run
- `temperature_DE_LU_training_cache.csv` — ERA5 temperature cache; ERA5T has ~5-day publication lag

### Data exclusions

| Period | Reason |
| --- | --- |
| Pre-2020 | Pre-energy-crisis demand level — structural shift in DE_LU baseline load |
| 2020 | COVID-19 lockdown demand collapse — anomalous, not expected to recur |
| 2021 | Partial-recovery year with mixed structural demand — excluded for consistency |

Training starts at **2022-01-01**, the first full year of post-COVID demand recovery.

### Operational Design Domain (ODD)

The pipeline is valid only within the following conditions. Outside them it raises an explicit error rather than returning an unreliable result.

| Condition | Valid range | Outside the range |
| --- | --- | --- |
| Target bidding zone | DE_LU (Germany + Luxembourg) | untested; code changes required for other zones |
| Forecast horizon | 1–24 hours, produced once per day | unreliable; not tested for longer horizons |
| Load series | UTC `DatetimeIndex`, regular 60-min intervals, named `load_MW`, all positive | `PreprocessingError` |
| Exogenous features | numeric, complete, aligned to load index | `ValueError` on any missing entry |
| Consecutive gap | ≤ 2 hours | `PreprocessingError` — pipeline halts |
| Minimum history | > 168 hours (one week) | forecaster cannot be called |
| Forecast values | > 0 MW | `PreprocessingError` — submission blocked |
| Plausibility range | 5,000–150,000 MW | warning logged — submission proceeds |
| Model age | ≤ 7 days since training | warning logged if older |
| Concept drift | Stable seasonal patterns, no structural regime change | accuracy degrades; retrain recommended |

### Coverage validation (pipeline guards)

The pipeline enforces three checks before any feature engineering:

1. **Series type and timezone:** Must be a `pandas.Series` with a UTC `DatetimeIndex` named `load_MW`; raises `PreprocessingError` otherwise.
2. **Gap audit:** All gaps are logged (total missing hours, gap windows, max consecutive gap, breakdown by hour-of-day / weekday / month) before any filling takes place.
3. **Gap fill limit:** Forward-fill applied only for gaps ≤ 2 consecutive hours; longer gaps raise `PreprocessingError` and halt the pipeline.

### Training / validation split

| Window | Date range | Samples | Purpose |
| --- | --- | --- | --- |
| Initial training | 2022-01-01 → early 2025 | 31,524 | Model fit (backtest initial window) |
| Backtest evaluation | Remaining samples | ~7,881 | Walk-forward 24-h folds |
| Full re-fit (production) | 2022-01-01 → 2026-06-30 | 39,405 | Production model artifact |

No future observations influence a past prediction: `skforecast` enforces time-ordered splits throughout.

### Known limitations and drift risk

1. **Holiday effects:** Public holidays are encoded, but irregular local events (plant shutdowns, large-scale events) are not in the training features and degrade accuracy.
2. **Extreme weather:** Cold snaps and heat waves cause load spikes that exceed the model's typical temperature–load correlation; forecasts are less accurate on such days.
3. **COVID-era patterns excluded:** If a structural demand collapse similar to 2020 recurs, the model will overestimate load.
4. **Seasonal transitions:** After summer / winter changeover, load profiles shift; expect transient error increase for 1–2 weeks until a retrain incorporates the new pattern.
5. **Grid topology:** The model assumes a stable network; transmission constraints, unplanned outages, or major new generation capacity are outside the training domain.
6. **ERA5 publication lag:** ERA5T data has a ~5-day lag; for the most recent days the pipeline falls back to a 2-day rolling mean from cache, introducing slight temperature approximation error.

## 7. Evaluation

### Accuracy metrics

The pipeline uses three metrics, all computed on the walk-forward backtest:

$$\text{MAPE} = \frac{1}{N}\sum_{t}\frac{|l_t - \hat{l}_t|}{l_t} \times 100 \qquad \text{MAE} = \frac{1}{N}\sum_{t}|l_t - \hat{l}_t| \qquad \text{RMSE} = \sqrt{\frac{1}{N}\sum_{t}(l_t - \hat{l}_t)^2}$$

MAPE requires $l_t > 0$ (enforced by preprocessing). MAE and RMSE are in MW.

### Current production model results

| Metric | Value | Description |
| --- | --- | --- |
| MAPE | **2.59%** | Scale-independent relative error |
| MAE | **1,380 MW** | Average absolute hourly deviation |
| RMSE | **1,920 MW** | Root mean squared error (penalises large errors) |

These metrics are from training run `20260630_211911` on 39,405 samples (2022-01-01 to 2026-06-30). Per-run metrics are recorded in `models/metadata_<timestamp>.json` and reproduced in `model_cards/model_card_<timestamp>.md`.

**Backtest configuration:**
- Initial training window: 31,524 samples
- Evaluation window: remaining samples in non-overlapping 24-h folds
- Refit between folds: No (fixed model weights)
- Steps per fold: 24

### Reproducibility verification

```bash
# Retrain and check that metrics reproduce
python -m energy_forecast.train_model --cutoff 2026-06-30
# MAPE should reproduce at ~2.59%, MAE ~1,380 MW, RMSE ~1,920 MW
# with RANDOM_SEED=42 on the same hardware and Python version
```

### Software quality checks

- Missing or infinite values in the input series raise `PreprocessingError` before any computation.
- All lag matrices are constructed by `skforecast`, which guarantees leakage-free time-ordered splits.
- New code must achieve ≥ **80% line coverage** (enforced by `pytest-cov`).
- Static type checking enforced by `mypy` (strict mode).
- Zero dead code enforced by `ruff` (rules F401, F811, F841).

## 8. Model Transparency

### Point vs. probabilistic forecasts

The pipeline produces **point forecasts only** — a single MW value per hour. It does not natively quantify uncertainty or produce prediction intervals. If uncertainty estimates are needed, they must be added externally (e.g., via conformal prediction wrappers or ensemble bootstrapping on the `ForecasterRecursive` output).

### White-box architecture

There are no compiled inference kernels or opaque weights. Every transformation can be read and audited in source:

1. **Feature engineering:** All 16 feature definitions are in `data/preprocess.py`. Lag indices are explicit (`lags=168`). Holiday dates are computed from a documented anonymous Gregorian Easter algorithm.
2. **Regressor:** CatBoost feature importances are accessible via `forecaster.regressor.get_feature_importance()`. XGBoost and LightGBM expose `feature_importances_` directly.
3. **Bias correction:** Applied in `make_submission.py`, logged, and capped at ±4,000 MW. If the cap is exceeded, a warning is emitted with the uncorrected and corrected values.
4. **Backtest:** Implemented by `skforecast.model_selection.backtesting_forecaster`; its source is publicly auditable.

### Feature importance

Post-hoc interpretability is available through each regressor's own mechanism:

```python
# CatBoost (production)
importances = forecaster.regressor.get_feature_importance()

# XGBoost or LightGBM
importances = forecaster.regressor.feature_importances_
```

No separate explainability backend (SHAP, LIME) ships with the package, consistent with the minimal-dependency policy.

### Audit trail

Every training run produces:
- **Gap audit log** to stdout: total missing hours, gap windows, max consecutive gap, breakdown by hour / weekday / month — all logged before any filling.
- **Metadata sidecar** (`models/metadata_<timestamp>.json`): training start/end dates, n_training_samples, regressor type, lags, seed, MAPE, MAE, RMSE.
- **Versioned model card** (`model_cards/model_card_<timestamp>.md`): EU AI Act Art. 13 fields, generated at training time, never overwritten.
- **Submission log:** corrected and uncorrected forecast values, bias correction delta, cap-exceeded warnings if applicable.

## 9. Operation: Monitoring and Response

### Monitoring checklist

If this pipeline is deployed operationally, the operator should monitor:

**1. Input data quality**
- ENTSO-E API uptime and freshness of the load cache
- Gaps or duplicates in the hourly load series
- Out-of-range load values (< 5,000 MW or > 150,000 MW)
- ERA5 publication lag exceeding 5 days (triggers rolling-mean fallback for temperature)

**2. Forecast quality**
- MAPE vs. a naive baseline (e.g., same-day-last-week, 7-day moving average)
- Systematic bias (mean forecast error): if consistently too high or too low, retrain
- Spike detection: if |error| > 2× typical, investigate data quality or weather event
- Seasonal drift: after summer / winter transition, expect transient error increase for 1–2 weeks

**3. Computational health**
- Model age: pipeline warns if the fitted model exceeds 7 days
- Bias correction cap: if the ±4,000 MW cap is triggered repeatedly, the model has drifted and should be retrained

### Response protocols

| Trigger | Action |
| --- | --- |
| Data gap > 2 consecutive hours | `PreprocessingError` raised; check ENTSO-E feed; do not submit until resolved |
| ENTSO-E API failure, cache gap ≤ 48 h | Cached data used automatically; warning logged; investigate API |
| ENTSO-E API failure, cache gap > 48 h | `DataFetchError` raised; manual intervention required |
| ERA5 API failure | Naive 2-day rolling mean fallback; warning logged |
| Open-Meteo timeout | ERA5 rolling mean fallback; warning logged |
| Forecast ≤ 0 MW | `PreprocessingError` raised; submission blocked; inspect feature pipeline |
| Forecast outside 5,000–150,000 MW | Warning logged; submission proceeds; human review recommended |
| Bias correction repeatedly hits ±4,000 MW cap | Trigger immediate retraining with fresh data |
| MAPE > 5% on recent days | Schedule retraining with latest history |
| Model older than 7 days | Warning logged; retrain recommended |

### Retraining schedule

| Action | Recommended cadence |
| --- | --- |
| Generate submission | Daily |
| Retrain model | Weekly (after each new week of ENTSO-E data) |
| Review backtest metrics | At each retraining |
| Full data re-fetch and audit | Monthly |
| Emergency retrain | After any structural break in load series (outage, policy change, etc.) |

### Offline fallback

If the ENTSO-E API is unavailable, the pipeline falls back to the most recent cache in `data/raw/` automatically (gap ≤ 48 h). If ERA5 or Open-Meteo is unavailable, the pipeline substitutes a 2-day rolling mean from the ERA5 cache — forecast quality is slightly reduced but submission proceeds.

## 10. Compliance and Challenge Submission

### Challenge submission requirements

Every forecast is validated before the CSV is written:

| Check | Details |
| --- | --- |
| **CSV schema** | Columns: `timestamp_utc`, `forecast_mw`; types: ISO datetime string, float |
| **Row count** | Exactly 24 rows (one per UTC hour of the target date) |
| **Values** | All positive (> 0 MW); range 5,000–150,000 MW recommended for DE_LU |
| **Filename** | `<YYYY-MM-DD>.csv` under `submissions/<team_id>/` |

Validation is enforced by `make_submission.py` before any file is written.

### Reproducibility record

Each training run is accompanied by:
- **Metadata sidecar** (`models/metadata_<timestamp>.json`): training dates, seed, n_samples, metrics
- **Versioned model card** (`model_cards/model_card_<timestamp>.md`): EU AI Act Art. 13 fields
- **Model pointer** (`models/current.json`): path and timestamp of the latest artifact

### Regulatory compliance

| Standard | Article / Clause | How this pipeline supports it |
| --- | --- | --- |
| EU AI Act 2024/1689 | Art. 13 (Transparency) | This model card; versioned per-run cards in `model_cards/` |
| EU AI Act 2024/1689 | Art. 9 (Risk management) | Gap audit (CR-1) + fail-safe output checks (CR-3) |
| EU AI Act 2024/1689 | Art. 10 (Data governance) | Missing / infinite values rejected by default; gap audit before filling |
| EU AI Act 2024/1689 | Art. 12 (Record-keeping) | Metadata sidecars and versioned model cards at every training run |
| EU AI Act 2024/1689 | Art. 15 (Accuracy and robustness) | Deterministic, reproducible transformations; fixed random seed |
| IEC 61508 SIL-1 | Functional safety | Fail-safe output validation — non-positive load raises `PreprocessingError` |
| ISA/IEC 62443 SL-1 | Cyber security | API key loaded only from `.env`; never hardcoded in source |
| Cyber Resilience Act | Dependency traceability | Pinned dependency versions in `pyproject.toml` |

### Licensing and attribution

| Component | License | Attribution |
| --- | --- | --- |
| energy-forecast (this package) | See repository root | Eigen Squad |
| skforecast | BSD-3-Clause | Joaquin Amat Rodrigo |
| CatBoost | Apache-2.0 | Yandex LLC |
| XGBoost | Apache-2.0 | DMLC; see https://github.com/dmlc/xgboost |
| LightGBM | MIT | Microsoft; see https://github.com/microsoft/LightGBM |
| ENTSO-E data | CC0 (Public Domain) | ENTSO-E Transparency Platform |
| ERA5 data | CC BY 4.0 | Copernicus Climate Change Service / ECMWF |
| Open-Meteo data | CC BY 4.0 | Open-Meteo.com |

All dependencies are declared in `pyproject.toml`.

## 11. Glossary

| Term | Meaning |
| --- | --- |
| **DE_LU** | ENTSO-E bidding zone code for Germany + Luxembourg |
| **ENTSO-E** | European Network of Transmission System Operators for Electricity |
| **ERA5** | ECMWF Reanalysis v5 — historical hourly weather reanalysis from Copernicus CDS |
| **ERA5T** | ERA5 near-real-time extension; has a ~5-day publication lag |
| **EU AI Act** | Regulation (EU) 2024/1689 on artificial intelligence, in force since 2024-08-01 |
| **IEC 61508** | International standard for functional safety of electrical / electronic safety-related systems |
| **ISA/IEC 62443** | Standard series for security of industrial automation and control systems |
| **MAPE** | Mean Absolute Percentage Error — `mean(|y_true − y_pred| / y_true) × 100` |
| **MAE** | Mean Absolute Error — `mean(|y_true − y_pred|)` in MW |
| **RMSE** | Root Mean Squared Error — `sqrt(mean((y_true − y_pred)²))` in MW |
| **ODD** | Operational Design Domain — conditions under which results are valid |
| **CR-1** | Compliance requirement 1 (spotforecast2-safe): auditability of data gaps before filling |
| **CR-3** | Compliance requirement 3 (spotforecast2-safe): fail-safe behaviour on invalid output |
| **ForecasterRecursive** | skforecast wrapper handling lag-matrix construction and the recursive prediction loop |
| **Recursive forecasting** | Multi-step prediction where each step uses previously predicted values as lag inputs for the next |

## 12. How to Audit

An auditor or reviewer can validate this pipeline as follows.

1. **Verify dependencies:**
   ```bash
   pip install -e ".[dev]"
   # Confirm prohibited libraries are absent:
   pip show torch tensorflow optuna matplotlib plotly 2>&1 | grep -c "not found"
   # Should print 5
   ```

2. **Run the full test suite and coverage check:**
   ```bash
   pytest tests/ --cov=src/energy_forecast --cov-report=term-missing --cov-fail-under=80
   # All tests must pass; line coverage must be ≥ 80%
   ```

3. **Check static types and dead-code rules:**
   ```bash
   mypy src/
   ruff check src/
   # Both must exit 0
   ```

4. **Verify gap-audit behaviour (fail-safe, CR-1):**
   - Read `src/energy_forecast/data/preprocess.py`: confirm `audit_gaps` is called before `fill_gaps` and that `fill_gaps` raises `PreprocessingError` for gaps exceeding 2 consecutive hours.

5. **Verify fail-safe output validation (CR-3):**
   - Read `src/energy_forecast/make_submission.py`: confirm non-positive forecast values raise `PreprocessingError` and that the bias correction is capped at ±4,000 MW.

6. **Inspect training provenance:**
   ```bash
   cat models/current.json                    # latest model timestamp
   cat models/metadata_<timestamp>.json       # training dates, seed, n_samples, metrics
   cat model_cards/model_card_<timestamp>.md  # Art. 13 fields
   ```

7. **Reproduce the production metrics:**
   ```bash
   python -m energy_forecast.train_model --cutoff 2026-06-30
   # MAPE should reproduce at ~2.59%, MAE ~1,380 MW, RMSE ~1,920 MW
   # with RANDOM_SEED=42 on the same hardware and Python version
   ```

8. **Validate a submission CSV:**
   ```bash
   python -m energy_forecast.evaluate_submission submissions/eigen_squad/<YYYY-MM-DD>.csv
   # Must complete without errors; checks schema, 24 rows, positive values
   ```

## 13. Citation and Contact

Maintainer: Team Eigen Squad 

```bibtex
@misc{eigenSquad2026,
  author       = {Team Eigen Squad},
  title        = {{energy-forecast}: Day-Ahead Load Forecasting Pipeline for DE\_LU},
  year         = {2026},
  note         = {DDMO SS 2026 Load-Forecasting Challenge — team eigen\_squad}
}
```

### Upstream references

This work builds on:

- **skforecast:** Amat Rodrigo, J. (2024). *skforecast: Time series forecasting with scikit-learn regressors*. https://github.com/JoaquinAmatRodrigo/skforecast
- **spotforecast2-safe compliance framework:** Bartz-Beielstein, T. (2026). *spotforecast2-safe: Safety-critical Subset of spotforecast2*. https://github.com/sequential-parameter-optimization/spotforecast2-safe
- **ENTSO-E data:** ENTSO-E Transparency Platform. https://transparency.entsoe.eu/

### Contact and support

- **Pipeline issues:** raise a GitHub issue in the repository
- **Challenge questions:** DDMO SS 2026 leaderboard and course materials

## 14. Disclaimer and Liability

**Limitation of liability.** This pipeline is provided as is, without warranty of any kind. The authors accept no liability for any direct or indirect damage, forecast error, system failure, or financial loss arising from its use.

### Specific disclaimers

- **Forecast accuracy is not guaranteed.** Electricity load is influenced by unpredictable factors (weather events, grid incidents, policy changes) that are outside the model's training domain.
- **For research and education only.** This is a course project and challenge entry, not a production-grade forecasting system. Do not use it for critical operational decisions without independent validation.
- **Data quality is as-is.** ENTSO-E and ERA5 data are used under their respective public data terms; errors in upstream data are propagated without correction beyond the documented fail-safe guards.
- **Not a replacement for domain expertise.** Grid operators and energy traders must always apply human judgment and domain knowledge alongside automated forecasts.
- **Reproducibility on non-reference platforms.** Bit-exactness is only claimed for the same hardware, Python version, and dependency stack. Different platforms may produce slightly different floating-point results.

It is the sole responsibility of the system integrator to perform full system-level safety validation before deploying this software in a production or safety-critical environment.
