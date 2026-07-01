"""Deterministic preprocessing for ENTSO-E load series.

Public API:
    validate_series(series)          -> pd.Series    — schema + sanity checks
    audit_gaps(series)               -> dict          — gap pattern analysis before filling
    fill_gaps(series, limit)         -> pd.Series    — forward-fill short gaps
    add_calendar_features(series)    -> pd.DataFrame — hour/weekday/month/weekend
    build_features(series)           -> tuple[pd.Series, pd.DataFrame]

All operations are deterministic: no random imputation, no global state.
Missing value strategy: forward-fill with a hard cap (``limit`` hours).
Gaps longer than the cap raise PreprocessingError rather than propagating NaNs.
"""

from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from energy_forecast.config import settings
from energy_forecast.exceptions import PreprocessingError

_logger = logging.getLogger(__name__)

_EXPECTED_SERIES_NAME = "load_MW"
_EXPECTED_TZ = "UTC"
_FREQ = "60min"


def _easter(year: int) -> datetime.date:
    """Return Easter Sunday for the given year (anonymous Gregorian algorithm)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return datetime.date(year, month, day + 1)


def _de_lu_holidays(years: range) -> frozenset[datetime.date]:
    """Return DE_LU public holidays for the given year range.

    Includes all German federal holidays, widely observed German regional
    holidays (Fronleichnam/Corpus Christi covers NRW, Bayern, BW, Hessen,
    RLP, Saarland; Allerheiligen covers similar states), and Luxembourg
    national holidays.  All are treated equally since DE_LU load reflects
    the combined demand of both regions.
    """
    dates: set[datetime.date] = set()
    delta = datetime.timedelta
    for year in years:
        e = _easter(year)
        dates.update([
            # Easter-relative (DE + LU)
            e - delta(days=2),   # Karfreitag / Good Friday
            e + delta(days=1),   # Ostermontag / Easter Monday
            e + delta(days=39),  # Christi Himmelfahrt / Ascension
            e + delta(days=50),  # Pfingstmontag / Whit Monday
            e + delta(days=60),  # Fronleichnam / Corpus Christi (DE regional + LU)
            # Fixed German federal
            datetime.date(year, 1, 1),    # Neujahr / New Year
            datetime.date(year, 5, 1),    # Tag der Arbeit / Labour Day
            datetime.date(year, 10, 3),   # Tag der Deutschen Einheit
            datetime.date(year, 11, 1),   # Allerheiligen / All Saints (DE regional + LU)
            datetime.date(year, 12, 25),  # 1. Weihnachtstag / Christmas
            datetime.date(year, 12, 26),  # 2. Weihnachtstag / Boxing Day
            # Luxembourg-specific
            datetime.date(year, 5, 9),    # Europatag / Europe Day
            datetime.date(year, 6, 23),   # Nationalfeiertag / Luxembourg National Day
            datetime.date(year, 8, 15),   # Maria Himmelfahrt / Assumption
        ])
    return frozenset(dates)


def validate_series(series: pd.Series) -> pd.Series:
    """Validate the schema and basic sanity of a raw load series.

    Args:
        series: Load series to validate.

    Returns:
        The input series unchanged if all checks pass.

    Raises:
        PreprocessingError: If the series fails any check.
    """
    if not isinstance(series.index, pd.DatetimeIndex):
        raise PreprocessingError(
            f"Expected pd.DatetimeIndex, got {type(series.index).__name__}."
        )
    if series.index.tz is None:
        raise PreprocessingError("Series index must be timezone-aware (expected UTC).")
    if str(series.index.tz) != _EXPECTED_TZ:
        raise PreprocessingError(f"Expected UTC timezone, got '{series.index.tz}'.")
    if series.name != _EXPECTED_SERIES_NAME:
        raise PreprocessingError(
            f"Expected series name '{_EXPECTED_SERIES_NAME}', got '{series.name}'."
        )
    if (series <= 0).any():
        n_bad = int((series <= 0).sum())
        raise PreprocessingError(
            f"Load series contains {n_bad} non-positive value(s); "
            "physical load must be strictly positive."
        )
    _logger.info(
        "Series validated: %d records, %s – %s",
        len(series),
        series.index[0].isoformat(),
        series.index[-1].isoformat(),
    )
    return series


def audit_gaps(series: pd.Series) -> dict[str, Any]:
    """Analyse the pattern of missing values in a load series.

    Resamples to 60-min frequency (same as ``fill_gaps``) and then examines
    *where* and *how* gaps cluster — without modifying the data.

    Use this before ``fill_gaps`` to decide whether forward-fill is safe or
    whether the missing data follows a systematic pattern (e.g., always at
    07:00, always on Mondays) that should be investigated at source.

    Args:
        series: UTC-indexed load series (any sub-hourly or hourly frequency).

    Returns:
        Dict with keys:
            ``total_missing``  int — total NaN count after 60-min resampling.
            ``gap_windows``    list[tuple[str, str, int]] — each gap as
                               (start_utc_iso, end_utc_iso, length_hours).
            ``by_hour``        dict[int, int] — gap count per hour of day (0–23).
            ``by_weekday``     dict[int, int] — gap count per weekday (0=Mon).
            ``by_month``       dict[int, int] — gap count per month (1–12).
            ``max_run``        int — longest consecutive gap in hours.
    """
    resampled = series.resample(_FREQ).mean()
    missing_mask = resampled.isna()
    total_missing = int(missing_mask.sum())

    # Identify contiguous gap windows and their lengths.
    gap_windows: list[tuple[str, str, int]] = []
    if total_missing > 0:
        gap_starts = missing_mask.index[
            missing_mask & ~missing_mask.shift(1, fill_value=False)
        ]
        gap_ends = missing_mask.index[
            missing_mask & ~missing_mask.shift(-1, fill_value=False)
        ]
        for start_ts, end_ts in zip(gap_starts, gap_ends):
            length = int((end_ts - start_ts).total_seconds() / 3600) + 1
            gap_windows.append((start_ts.isoformat(), end_ts.isoformat(), length))

    missing_times = resampled.index[missing_mask]
    by_hour: dict[int, int] = missing_times.hour.value_counts().sort_index().to_dict()
    by_weekday: dict[int, int] = (
        missing_times.weekday.value_counts().sort_index().to_dict()
    )
    by_month: dict[int, int] = missing_times.month.value_counts().sort_index().to_dict()
    max_run = max((w[2] for w in gap_windows), default=0)

    _logger.info(
        "Gap audit: %d missing hour(s) across %d gap window(s), longest run = %d h",
        total_missing,
        len(gap_windows),
        max_run,
    )

    if total_missing == 0:
        _logger.info("No gaps found — series is complete at 60-min frequency.")
    else:
        _logger.info("By hour of day:  %s", by_hour)
        _logger.info("By weekday (0=Mon): %s", by_weekday)
        _logger.info("By month:        %s", by_month)

    return {
        "total_missing": total_missing,
        "gap_windows": gap_windows,
        "by_hour": by_hour,
        "by_weekday": by_weekday,
        "by_month": by_month,
        "max_run": max_run,
    }


def fill_gaps(series: pd.Series, *, limit: int = 2) -> pd.Series:
    """Regularise the series to 60-min frequency and forward-fill short gaps.

    Missing value strategy: forward-fill (deterministic, conservative).
    A gap is any period where no observed value exists after resampling.
    Gaps longer than ``limit`` consecutive hours are rejected because
    forward-filling more than a few hours would introduce misleading data.

    Args:
        series: UTC-indexed load series (any sub-hourly or hourly frequency).
        limit: Maximum consecutive NaN hours to fill. Defaults to 2.

    Returns:
        Gap-free series at exactly 60-min frequency.

    Raises:
        PreprocessingError: If any gap exceeds ``limit`` hours after filling.
    """
    regularised = series.resample(_FREQ).mean()
    n_gaps_before = int(regularised.isna().sum())

    if n_gaps_before == 0:
        return regularised

    _logger.warning(
        "Found %d gap(s); forward-filling up to %d hour(s).", n_gaps_before, limit
    )
    filled = regularised.ffill(limit=limit)

    remaining = int(filled.isna().sum())
    if remaining > 0:
        raise PreprocessingError(
            f"{remaining} gap(s) exceed the fill limit of {limit} consecutive hour(s). "
            "Inspect the raw series before proceeding."
        )
    return filled


def add_calendar_features(series: pd.Series) -> pd.DataFrame:
    """Derive calendar features from the UTC load series index.

    Features added:
        - ``hour``                     int [0, 23]   — hour of day
        - ``weekday``                  int [0, 6]    — 0 = Monday
        - ``month``                    int [1, 12]   — calendar month
        - ``is_weekend``               int {0, 1}    — 1 on Saturday and Sunday
        - ``is_friday``                int {0, 1}    — 1 on Friday; load drops from ~13h
        - ``is_saturday``              int {0, 1}    — 1 on Saturday; flat daytime, high late-night
        - ``is_sunday``                int {0, 1}    — 1 on Sunday; lowest overall level
        - ``is_holiday``               int {0, 1}    — 1 on DE_LU public holidays (Christmas,
                                                       Whit Monday, Corpus Christi, etc.)
        - ``hour_weekday_interaction`` int [0, 167]  — unique (hour, weekday) pair
        - ``hour_weekend_interaction`` int [0, 23]   — hour × is_weekend (0 on weekdays)
        - ``hour_holiday_interaction`` int [0, 23]   — hour × is_holiday (0 on non-holidays)
        - ``hour_lag_24``              int [0, 23]   — hour of day 24h ago
        - ``weekday_lag_168``          int [0, 6]    — weekday 168h ago (same time last week)

    Args:
        series: UTC-indexed load series with hourly frequency.

    Returns:
        DataFrame with thirteen integer feature columns, same index as ``series``.
    """
    idx = series.index

    years = range(int(idx.year.min()), int(idx.year.max()) + 1)
    holiday_set = _de_lu_holidays(years)
    is_holiday = pd.array(
        [int(d in holiday_set) for d in idx.date], dtype="int64"
    )

    df = pd.DataFrame(
        {
            "hour": idx.hour,
            "weekday": idx.weekday,
            "month": idx.month,
            "is_weekend": (idx.weekday >= 5).astype(int),
            "is_friday": (idx.weekday == 4).astype(int),
            "is_saturday": (idx.weekday == 5).astype(int),
            "is_sunday": (idx.weekday == 6).astype(int),
            "is_holiday": is_holiday,
        },
        index=idx,
    )

    # hour_weekday_interaction: encodes all 168 unique (hour, weekday) combinations
    df["hour_weekday_interaction"] = df["hour"] + (df["weekday"] * 24)

    # hour_weekend_interaction: captures the different intraday shape on weekends
    # (flat midday, elevated late-night) vs weekdays (pronounced midday peak)
    df["hour_weekend_interaction"] = df["hour"] * df["is_weekend"]

    # hour_holiday_interaction: captures the Sunday-like intraday shape on public holidays
    df["hour_holiday_interaction"] = df["hour"] * df["is_holiday"]

    # Lagged features: for rows without 24h/168h history, fill with current value
    df["hour_lag_24"] = df["hour"].shift(24)
    df["hour_lag_24"] = df["hour_lag_24"].fillna(df["hour"])

    df["weekday_lag_168"] = df["weekday"].shift(168)
    df["weekday_lag_168"] = df["weekday_lag_168"].fillna(df["weekday"])

    return df


def add_weather_features(
    load_series: pd.Series, temp_series: pd.Series
) -> pd.DataFrame:
    """Derive temperature-based features from ERA5 reanalysis.

    Features added:
        - ``temp_c``       float — temperature in Celsius (from Kelvin)
        - ``temp_lag_24``  float — temperature 24h ago (daily pattern capture)
        - ``temp_deviation`` float — deviation from 20-day rolling mean (anomaly detection)

    Args:
        load_series: UTC-indexed load series (used only for shape/index alignment check).
        temp_series: UTC-indexed temperature series (unit Kelvin, name ``temp_K``).

    Returns:
        DataFrame with three weather feature columns, same index as ``load_series``.

    Raises:
        PreprocessingError: If indices don't align after resampling to 60-min frequency.
    """
    # Ensure 60-min alignment then intersect on shared timestamps.
    # ERA5 and ENTSO-E series may cover slightly different date ranges depending
    # on API availability, so we join on the common index rather than requiring
    # identical lengths.
    temp_resampled = temp_series.resample("60min").mean()
    load_resampled = load_series.resample("60min").mean()

    shared_index = temp_resampled.index.intersection(load_resampled.index)
    if shared_index.empty:
        raise PreprocessingError(
            f"Temperature and load series have no overlapping timestamps after resampling. "
            f"Temp: {temp_resampled.index[0]} – {temp_resampled.index[-1]}, "
            f"Load: {load_resampled.index[0]} – {load_resampled.index[-1]}"
        )

    temp_aligned = temp_resampled.reindex(shared_index)
    load_aligned = load_resampled.reindex(shared_index)

    missing_temp = temp_aligned.isna().sum()
    if missing_temp > 0:
        _logger.warning(
            "Temperature series has %d missing hours after alignment; forward-filling.",
            missing_temp,
        )
        temp_aligned = temp_aligned.ffill()

    _logger.info(
        "Weather/load alignment: %d shared hours (temp=%d, load=%d)",
        len(shared_index),
        len(temp_resampled),
        len(load_resampled),
    )

    df = pd.DataFrame(
        {"temp_c": temp_aligned.values - 273.15},  # Kelvin to Celsius
        index=load_aligned.index,
    )

    # Lag 24: temperature from same hour yesterday
    df["temp_lag_24"] = df["temp_c"].shift(24)
    df["temp_lag_24"] = df["temp_lag_24"].fillna(df["temp_c"])

    # Deviation: captures unusual weather (heat waves, cold snaps)
    df["temp_deviation"] = (
        df["temp_c"] - df["temp_c"].rolling(window=480, center=True).mean()
    )
    df["temp_deviation"] = df["temp_deviation"].fillna(0)

    _logger.info(
        "Weather features added: temp range %.1f–%.1f °C",
        df["temp_c"].min(),
        df["temp_c"].max(),
    )

    return df


def build_features(
    series: pd.Series, temp_series: pd.Series | None = None
) -> tuple[pd.Series, pd.DataFrame]:
    """Run the full preprocessing pipeline on a raw load series.

    Steps: validate → fill_gaps → add_calendar_features → (optionally) add_weather_features.
    Saves the cleaned series and feature matrix to ``settings.processed_data_dir``.

    Args:
        series: Raw load series as returned by ``load_load_data``.
        temp_series: Optional temperature series (Kelvin) from ERA5. If provided,
            weather features are merged into exog DataFrame.

    Returns:
        Tuple of (cleaned_series, exog_df) ready for model training.

    Raises:
        PreprocessingError: If validation, gap-filling, or weather alignment fails.
    """
    clean = fill_gaps(validate_series(series))
    exog = add_calendar_features(clean)

    if temp_series is not None:
        weather = add_weather_features(clean, temp_series)
        exog = pd.concat([exog, weather], axis=1)

    _save_processed(clean, exog, settings.processed_data_dir)
    return clean, exog


def _save_processed(series: pd.Series, exog: pd.DataFrame, output_dir: Path) -> None:
    """Save the cleaned series and exog features to CSV.

    Args:
        series: Cleaned load series.
        exog: Calendar feature DataFrame.
        output_dir: Directory to write into (created if absent).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_str = series.index[0].strftime("%Y%m%d")
    end_str = series.index[-1].strftime("%Y%m%d")
    prefix = f"load_{settings.country_code}_{start_str}_{end_str}"

    series_path = output_dir / f"{prefix}_clean.csv"
    exog_path = output_dir / f"{prefix}_exog.csv"

    series.to_frame().to_csv(series_path)
    exog.to_csv(exog_path)
    _logger.info("Saved cleaned series to %s", series_path)
    _logger.info("Saved exog features to %s", exog_path)
