"""ENTSO-E load data fetching.

Public API:
    load_load_data(start, end)           -> pd.Series  — fetch hourly load in MW (UTC index)
    load_or_refresh_cache(cutoff)        -> pd.Series  — incremental cache (first run: full
                                                          history; subsequent: gap only)

Run as a script to pull one month of DE_LU data to data/raw/:
    python -m energy_forecast.data.fetch
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from entsoe import EntsoePandasClient

from energy_forecast.config import settings
from energy_forecast.exceptions import ENTSOEFetchError

_logger = logging.getLogger(__name__)


def load_load_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Fetch actual load from ENTSO-E Transparency Platform.

    Args:
        start: Query start — must be timezone-aware (UTC recommended).
        end: Query end — must be timezone-aware (UTC recommended).

    Returns:
        Hourly load series indexed by UTC timestamp, unit MW, name ``load_MW``.

    Raises:
        ENTSOEFetchError: If timestamps are tz-naive, the API call fails,
            or no data is returned.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ENTSOEFetchError(
            f"start and end must be timezone-aware pd.Timestamp; "
            f"got start.tzinfo={start.tzinfo}, end.tzinfo={end.tzinfo}."
        )

    client = EntsoePandasClient(api_key=settings.entsoe_api_key)
    try:
        result = client.query_load(settings.country_code, start=start, end=end)
    except Exception as exc:
        raise ENTSOEFetchError(
            f"ENTSO-E load query failed for {settings.country_code} "
            f"{start}–{end}: {exc}"
        ) from exc

    if result is None or (hasattr(result, "empty") and result.empty):
        raise ENTSOEFetchError(
            f"No load data returned for {settings.country_code} {start}–{end}."
        )

    # entsoe-py returns a DataFrame with an "Actual Load" column and a
    # Europe/Berlin index. Extract the column explicitly rather than using
    # .squeeze() so that an unexpected schema change raises KeyError immediately.
    if isinstance(result, pd.DataFrame):
        series: pd.Series = result["Actual Load"]
    else:
        series = result

    # Normalise timezone to UTC and resample to exactly 60-minute frequency.
    # DE_LU data may arrive at 15-min resolution in recent periods; resampling
    # with .mean() enforces the "hourly series" contract unconditionally.
    series = series.tz_convert("UTC").resample("60min").mean()

    return series.rename("load_MW")


_HISTORICAL_START = pd.Timestamp("2022-01-01", tz="UTC")
_CACHE_FILENAME = "load_DE_LU_training_cache.csv"


def load_or_refresh_cache(cutoff: pd.Timestamp) -> pd.Series:
    """Return historical load data from the on-disk cache, topping up any gap.

    On first run the full history from ``_HISTORICAL_START`` is fetched from
    ENTSO-E and saved to ``data/raw/load_DE_LU_training_cache.csv``.  On every
    subsequent run only the gap between the cache end and ``cutoff`` is fetched,
    concatenated, and the cache is updated in-place.

    ENTSO-E publishes actual load with a ~1-hour delay.  If the gap fetch fails
    and the cache is less than 48 hours old, a warning is logged and the cached
    data is returned as-is — it contains enough history for a quality forecast.

    Args:
        cutoff: Fetch data up to (not including) this UTC-aware timestamp.

    Returns:
        UTC-indexed hourly load series from ``_HISTORICAL_START`` to the last
        available observation before ``cutoff``.

    Raises:
        ENTSOEFetchError: If the API call fails and the cache is stale (> 48 h).
    """
    cache_path = settings.raw_data_dir / _CACHE_FILENAME

    if cache_path.exists():
        cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached = cached_df.squeeze().rename("load_MW")
        if cached.index.tz is None:
            cached.index = cached.index.tz_localize("UTC")
        else:
            cached.index = cached.index.tz_convert("UTC")

        cache_end = cached.index[-1]
        _logger.info(
            "Cache loaded: %d records, %s → %s",
            len(cached),
            cached.index[0].date(),
            cache_end.date(),
        )

        gap_start = cache_end + pd.Timedelta(hours=1)
        if gap_start < cutoff:
            _logger.info("Fetching gap: %s → %s", gap_start.isoformat(), cutoff.isoformat())
            try:
                gap_series = load_load_data(gap_start, cutoff)
                full_series = (
                    pd.concat([cached, gap_series])
                    .drop_duplicates()
                    .sort_index()
                    .rename("load_MW")
                )
                full_series.to_frame().to_csv(cache_path)
                _logger.info(
                    "Cache updated: %d records, now ends at %s",
                    len(full_series),
                    full_series.index[-1].isoformat(),
                )
            except ENTSOEFetchError as exc:
                gap_hours = (cutoff - cache_end).total_seconds() / 3600
                if gap_hours <= 48:
                    _logger.warning(
                        "Gap fetch failed (%.0f h gap): %s. "
                        "Cache is recent — using it as-is.",
                        gap_hours,
                        exc,
                    )
                    full_series = cached
                else:
                    raise
        else:
            full_series = cached
    else:
        _logger.info(
            "No cache found. Fetching full history %s → %s (first run only).",
            _HISTORICAL_START.date(),
            cutoff.date(),
        )
        full_series = load_load_data(_HISTORICAL_START, cutoff)
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        full_series.to_frame().to_csv(cache_path)
        _logger.info("Cache saved: %d records to %s", len(full_series), cache_path)

    return full_series[full_series.index < cutoff]


def _save_series(series: pd.Series, output_dir: Path) -> Path:
    """Persist a load series to a timestamped CSV in output_dir.

    Args:
        series: Load series with UTC DatetimeIndex, name ``load_MW``.
        output_dir: Directory to write into (created if absent).

    Returns:
        Path of the written CSV file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    start_str = series.index[0].strftime("%Y%m%d")
    end_str = series.index[-1].strftime("%Y%m%d")
    filename = f"load_{settings.country_code}_{start_str}_{end_str}.csv"
    path = output_dir / filename
    series.to_frame().to_csv(path)
    _logger.info("Saved %d load records to %s", len(series), path)
    return path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    _start = pd.Timestamp("2024-01-01", tz="UTC")
    _end = pd.Timestamp("2024-02-01", tz="UTC")
    _series = load_load_data(start=_start, end=_end)
    _output_path = _save_series(_series, settings.raw_data_dir)
    _logger.info("Fetch complete. Shape: %s", _series.shape)
