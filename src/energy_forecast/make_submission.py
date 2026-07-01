"""Daily submission generator for the DDMO SS 2026 load-forecasting challenge.

Usage:
    python -m energy_forecast.make_submission \
        --team <team_id> \
        --target-date <YYYY-MM-DD>

Workflow:
    1. Load the saved CatBoost forecaster from models/ (trained separately)
    2. Warn if the model is stale (> 7 days old) - suggest retraining
    3. Refresh the data cache up to target_date (fast incremental fetch)
    4. Extract the last ``lags`` hours as the autoregressive context window
    5. Build calendar exog for the prediction horizon
    6. Predict with last_window — no retraining, weights are fixed
    7. Validate output (positive values, count, sanity range)
    8. Write submissions/<team_id>/<YYYY-MM-DD>.csv

Separation of concerns:
    train_model.py   - retrains on fresh history, saves artifact + model card
    make_submission.py - lightweight: load artifact → predict → write CSV

Run train_model.py weekly (or when demand patterns shift).
Run make_submission.py daily to produce the competition CSV.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from energy_forecast.data.fetch import load_or_refresh_cache
from energy_forecast.data.openmeteo import fetch_openmeteo_forecast
from energy_forecast.data.preprocess import (
    add_calendar_features,
    fill_gaps,
    validate_series,
)
from energy_forecast.data.weather import load_or_refresh_weather_cache
from energy_forecast.exceptions import PreprocessingError, WeatherFetchError
from energy_forecast.models.persistence import load_forecaster

_logger = logging.getLogger(__name__)

_HORIZON = 24  # hours (day-ahead forecast)
_MODELS_DIR = Path("models")


def _compute_naive_temp_forecast(cached_temp: pd.Series, horizon: int) -> pd.Series:
    """Compute naive temperature forecast using 2-day rolling mean.

    Forward-fills the rolling mean across the forecast horizon. Conservative but
    reproducible strategy for temperature that doesn't require external API.

    Args:
        cached_temp: Recent temperature series (Kelvin), at least 48 hours of data.
        horizon: Number of hourly steps to forecast.

    Returns:
        Series with naive forecast temperatures (Kelvin), indexed from next hour.

    Raises:
        PreprocessingError: If insufficient data or all NaNs.
    """
    if len(cached_temp) < 48:
        raise PreprocessingError(
            f"Need at least 48 hours of recent temperature data for naive forecast; "
            f"got {len(cached_temp)} hours."
        )

    recent = cached_temp.tail(48)
    if recent.isna().all():
        raise PreprocessingError("Recent temperature data contains only NaNs.")

    rolling_mean = recent.mean()
    last_ts = cached_temp.index[-1]
    future_index = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1), periods=horizon, freq="60min", tz="UTC"
    )
    return pd.Series(rolling_mean, index=future_index, name="temp_K")


def _build_exog_range(
    start: pd.Timestamp, periods: int, temp_forecast: pd.Series | None = None
) -> pd.DataFrame:
    """Build calendar + weather features for ``periods`` consecutive hours from ``start``.

    Args:
        start: First timestamp (UTC-aware).
        periods: Number of hourly steps to generate.
        temp_forecast: Optional temperature series (Kelvin) for the forecast horizon.

    Returns:
        DataFrame with columns hour/weekday/month/is_weekend and optionally
        temp_c/temp_lag_24/temp_deviation.
    """
    future_index = pd.date_range(start=start, periods=periods, freq="60min", tz="UTC")
    dummy = pd.Series(0.0, index=future_index, name="load_MW")
    exog = add_calendar_features(dummy)

    if temp_forecast is not None:
        temp_c = temp_forecast.values - 273.15
        exog["temp_c"] = temp_c
        exog["temp_lag_24"] = temp_c  # Naive: use forecast as both current and lag
        exog["temp_deviation"] = 0.0  # Neutral deviation estimate
        _logger.info(
            "Added naive temperature forecast: %.1f °C (2-day rolling mean)",
            temp_c.mean(),
        )

    return exog


def _compute_level_bias(
    forecaster: object,
    clean_recent: pd.Series,
    lags: int,
    raw_temp: pd.Series | None = None,
    retro_days: int = 7,
) -> float:
    """Additive level correction averaged over the last ``retro_days`` days.

    Runs a 24-step retro-prediction for each of the last ``retro_days`` days,
    computes mean(actual − predicted) per day, then returns the average across
    all days.  A multi-day window is more robust to structural level drift than
    a single-day snapshot.

    Args:
        forecaster: Fitted ForecasterRecursive loaded from disk.
        clean_recent: Recent actual load series (needs ≥ lags + retro_days*24).
        lags: Maximum lag order (= context window size).
        raw_temp: ERA5 temperature cache used to build retro exog features.
        retro_days: Number of past days to average the bias over (default 7).

    Returns:
        Additive MW correction capped at ±4 000 MW.  Returns 0.0 on any failure
        so the live forecast is never blocked by bias-correction errors.
    """
    min_needed = lags + retro_days * 24
    if len(clean_recent) < min_needed:
        _logger.warning(
            "Insufficient data for %d-day bias window (%d < %d); skipping.",
            retro_days,
            len(clean_recent),
            min_needed,
        )
        return 0.0

    first_pred_ts = clean_recent.index[-1] + pd.Timedelta(hours=1)
    daily_biases: list[float] = []

    for day_offset in range(retro_days, 0, -1):
        retro_pred_start = first_pred_ts - pd.Timedelta(hours=day_offset * 24)
        idx = clean_recent.index.get_indexer([retro_pred_start], method="pad")[0]
        if idx < lags or idx + 24 > len(clean_recent):
            continue

        retro_window = clean_recent.iloc[idx - lags : idx]
        retro_actuals = clean_recent.iloc[idx : idx + 24]

        retro_temp: pd.Series | None = None
        if raw_temp is not None:
            vals = raw_temp.reindex(retro_actuals.index).ffill().bfill()
            if not vals.isna().all():
                retro_temp = vals.rename("temp_K")

        retro_exog = _build_exog_range(
            retro_actuals.index[0], 24, temp_forecast=retro_temp
        )

        try:
            retro_preds = forecaster.predict(  # type: ignore[union-attr]
                steps=24,
                last_window=retro_window,
                exog=retro_exog,
            )
            daily_biases.append(
                float((retro_actuals.values - retro_preds.values).mean())
            )
        except Exception as exc:
            _logger.warning(
                "Retro-prediction for day -%d failed: %s. Skipping day.", day_offset, exc
            )

    if not daily_biases:
        _logger.warning("All retro-predictions failed; skipping bias correction.")
        return 0.0

    bias = float(np.mean(daily_biases))
    _BIAS_CAP = 4_000.0
    capped = max(-_BIAS_CAP, min(_BIAS_CAP, bias))
    if capped != bias:
        _logger.warning(
            "Raw bias %.0f MW exceeds ±%.0f MW cap; clamped to %+.0f MW.",
            bias,
            _BIAS_CAP,
            capped,
        )
    _logger.info(
        "Level bias correction (%d/%d days): %+.0f MW (raw: %+.0f MW)",
        len(daily_biases),
        retro_days,
        capped,
        bias,
    )
    return capped


def _validate_forecast(forecast: pd.Series) -> None:
    """Fail-safe checks on the raw forecast output (CR-3).

    Args:
        forecast: Predicted load series, 24 steps.

    Raises:
        PreprocessingError: If the forecast is physically impossible or
            the wrong length.
    """
    if len(forecast) != _HORIZON:
        raise PreprocessingError(
            f"Expected {_HORIZON} forecast steps, got {len(forecast)}."
        )
    if (forecast <= 0).any():
        n_bad = int((forecast <= 0).sum())
        raise PreprocessingError(
            f"Forecast contains {n_bad} non-positive value(s); "
            "physically impossible — check model or input data."
        )
    if forecast.min() < 5_000 or forecast.max() > 150_000:
        _logger.warning(
            "Forecast range %.0f–%.0f MW is outside the typical DE_LU bounds "
            "(5 000–150 000 MW). Inspect before submitting.",
            forecast.min(),
            forecast.max(),
        )


def make_submission(
    team_id: str, target_date: pd.Timestamp, cutoff: pd.Timestamp | None = None
) -> Path:
    """Run the submission pipeline for one target date.

    The forecaster is loaded from disk — training is not repeated here.
    Only the last ``lags`` hours of actual data are needed as the
    autoregressive context window; the full history is not loaded into memory
    for inference.

    Args:
        team_id: Challenge team identifier (used as sub-directory name).
        target_date: UTC midnight of the day to forecast (24 h ahead).

    Returns:
        Path to the written submission CSV.

    Raises:
        PreprocessingError: If no trained model exists, the series is
            incomplete, or the forecast is invalid.
        ENTSOEFetchError: If the live data fetch fails and the cache is stale.
    """
    # 1. Load saved forecaster — raises PreprocessingError if none exists.
    forecaster, metadata = load_forecaster(_MODELS_DIR)
    lags: int = int(forecaster.lags[-1])  # max lag = context window size

    data_cutoff = cutoff if cutoff is not None else target_date

    # 2. Refresh cache to get the most recent actual load values.
    raw_series = load_or_refresh_cache(data_cutoff)

    # 2b. Refresh temperature cache for naive forecast.
    try:
        raw_temp = load_or_refresh_weather_cache(data_cutoff)
        _logger.info("Temperature cache loaded: %d records", len(raw_temp))
    except Exception as exc:
        _logger.warning(
            "Temperature forecast failed: %s. Proceeding with calendar-only features.",
            exc,
        )
        raw_temp = None

    # 3. Validate and fill short gaps in the recent tail only.
    #    We need at least ``lags`` clean hours ending before target_date.
    tail_needed = lags * 4  # 4-day buffer for robustness
    recent_raw = raw_series.tail(tail_needed)
    clean_recent = fill_gaps(validate_series(recent_raw))

    # 4. Extract the last_window — exactly ``lags`` observations ending as
    #    close to target_date - 1h as the API allows.
    last_window = clean_recent.tail(lags)
    last_ts = last_window.index[-1]

    if last_ts >= target_date:
        raise PreprocessingError(
            f"Last known data timestamp {last_ts.isoformat()} is at or after "
            f"the target date {target_date.date()}. Cannot forecast the past."
        )

    # 5. Compute the full prediction span.
    #    skforecast requires exog to start exactly one step after last_window end.
    first_pred_ts = last_ts + pd.Timedelta(hours=1)
    last_target_ts = target_date + pd.Timedelta(hours=_HORIZON - 1)
    steps_needed = int((last_target_ts - last_ts).total_seconds() / 3600)
    _logger.info(
        "Predicting %d step(s) from %s to reach target day %s (last_window ends %s)",
        steps_needed,
        first_pred_ts.isoformat(),
        target_date.date(),
        last_ts.isoformat(),
    )

    # 6. Build exog covering the full prediction span.
    #    Primary: Open-Meteo day-ahead forecast (free API, no key required).
    #    Fallback: naive 2-day rolling mean from ERA5 cache.
    temp_forecast = None
    try:
        temp_forecast = fetch_openmeteo_forecast(target_date, steps_needed)
        _logger.info("Using Open-Meteo temperature forecast for prediction horizon.")
    except WeatherFetchError as exc:
        _logger.warning(
            "Open-Meteo forecast failed: %s. Falling back to naive ERA5 rolling mean.", exc
        )
        if raw_temp is not None:
            try:
                temp_forecast = _compute_naive_temp_forecast(raw_temp, steps_needed)
            except PreprocessingError as exc2:
                _logger.warning(
                    "Naive temperature forecast also failed: %s. Using calendar-only features.",
                    exc2,
                )

    future_exog = _build_exog_range(
        first_pred_ts, steps_needed, temp_forecast=temp_forecast
    )

    # 6b. Bias correction: average retro-prediction error over the last 7 days.
    bias_correction = _compute_level_bias(
        forecaster, clean_recent, lags, raw_temp=raw_temp
    )

    # 7. Predict using the saved model weights + current last_window.
    all_preds = forecaster.predict(
        steps=steps_needed,
        last_window=last_window,
        exog=future_exog,
    )
    raw_forecast = all_preds[all_preds.index >= target_date].head(_HORIZON)

    # 7b. Apply level bias correction before validation.
    if bias_correction != 0.0:
        raw_forecast = raw_forecast + bias_correction

    _validate_forecast(raw_forecast)

    # 8. Build submission DataFrame with the required column names.
    future_index = pd.date_range(
        start=target_date, periods=_HORIZON, freq="60min", tz="UTC"
    )
    submission = pd.DataFrame(
        {
            "timestamp_utc": future_index.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_mw": raw_forecast.to_numpy().round(2),
        }
    )

    # 9. Write to submissions/<team_id>/<YYYY-MM-DD>.csv (CR-1 persistence).
    out_dir = Path("submissions") / team_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{target_date.strftime('%Y-%m-%d')}.csv"
    submission.to_csv(out_path, index=False)
    _logger.info("Submission written to %s (%d rows)", out_path, len(submission))

    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a day-ahead load forecast submission CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python -m energy_forecast.make_submission "
            "--team team_alpha --target-date 2026-05-26\n\n"
            "Prerequisites:\n"
            "  python -m energy_forecast.train_model   (run first, then weekly)"
        ),
    )
    parser.add_argument(
        "--team",
        required=True,
        help="Challenge team identifier (e.g. 'team_alpha').",
    )
    parser.add_argument(
        "--target-date",
        required=True,
        dest="target_date",
        help="Date to forecast in YYYY-MM-DD format (UTC midnight).",
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help=(
            "Cap data fetching at this UTC timestamp (YYYY-MM-DDTHH:MM). "
            "Defaults to target-date midnight if omitted."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    args = _parse_args()
    try:
        target = pd.Timestamp(args.target_date, tz="UTC")
    except Exception as exc:
        raise SystemExit(f"Invalid --target-date '{args.target_date}': {exc}") from exc

    cutoff_ts: pd.Timestamp | None = None
    if args.cutoff:
        try:
            cutoff_ts = pd.Timestamp(args.cutoff, tz="UTC")
        except Exception as exc:
            raise SystemExit(f"Invalid --cutoff '{args.cutoff}': {exc}") from exc

    out = make_submission(team_id=args.team, target_date=target, cutoff=cutoff_ts)
    print(f"Done. Submission saved to: {out}")
