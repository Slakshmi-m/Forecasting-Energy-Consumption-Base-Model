"""ERA5 weather data fetching from Copernicus Climate Data Store.

Public API:
    fetch_era5_temperature(country, start, end)     -> pd.Series  — fetch hourly 2m temp (UTC index)
    load_or_refresh_weather_cache(cutoff)           -> pd.Series  — incremental cache (first run: full
                                                                    history; subsequent: gap only)
"""

from __future__ import annotations

import logging
import os
import tempfile

import cdsapi
import pandas as pd
import xarray as xr

from energy_forecast.config import settings
from energy_forecast.exceptions import WeatherFetchError

_logger = logging.getLogger(__name__)

# ERA5 country-to-CDS bounding box mapping. DE_LU covers Germany + Luxembourg.
# Format: (north, west, south, east) in degrees
_COUNTRY_BBOX = {
    "DE_LU": (55.1, 5.9, 47.3, 15.0),  # Germany + Luxembourg
}

_HISTORICAL_START = pd.Timestamp("2022-01-01", tz="UTC")
_CACHE_FILENAME = "temperature_DE_LU_training_cache.csv"


def _fetch_year_chunk(
    client: cdsapi.Client,
    year: int,
    months: list[str],
    north: float,
    west: float,
    south: float,
    east: float,
) -> pd.Series:
    """Fetch and parse ERA5 2m temperature for one calendar year chunk.

    Requests are split by year to stay within the CDS per-request size limit.
    Temperature is spatially averaged over all grid points in the bounding box.

    Args:
        client: Authenticated CDS API client.
        year: Calendar year to fetch.
        months: Zero-padded month strings (e.g. ["01", "02"]).
        north: Northern latitude bound (degrees).
        west: Western longitude bound (degrees).
        south: Southern latitude bound (degrees).
        east: Eastern longitude bound (degrees).

    Returns:
        UTC-indexed hourly temperature series (Kelvin) for the requested period.

    Raises:
        WeatherFetchError: If the CDS request or netCDF parsing fails.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".nc")
    os.close(fd)
    try:
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "2m_temperature",
                "year": [str(year)],
                "month": months,
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [north, west, south, east],
                "format": "netcdf",
            },
            tmp_path,
        )
        ds = xr.open_dataset(tmp_path)
        # CDS API v2 uses "valid_time"; v1 uses "time" — handle both
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        spatial_dims = [d for d in ds["t2m"].dims if d != time_dim]
        series = (
            ds["t2m"]
            .mean(dim=spatial_dims)
            .to_series()
            .rename("temp_K")
        )
        ds.close()
    except WeatherFetchError:
        raise
    except Exception as exc:
        raise WeatherFetchError(
            f"ERA5 chunk fetch failed for year={year} months={months}: {exc}"
        ) from exc
    finally:
        os.unlink(tmp_path)

    if series.index.tz is None:
        series.index = series.index.tz_localize("UTC")
    else:
        series.index = series.index.tz_convert("UTC")

    return series


def fetch_era5_temperature(
    country: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """Fetch 2m air temperature from ERA5 reanalysis via Copernicus CDS.

    Requests are chunked by calendar year to stay within the CDS per-request
    size limit (~120 000 fields). Temperature is spatially averaged over all
    ERA5 grid points within the country bounding box.

    Args:
        country: Country/region code (e.g., "DE_LU"). Must exist in _COUNTRY_BBOX.
        start: Query start — must be timezone-aware (UTC).
        end: Query end — must be timezone-aware (UTC).

    Returns:
        Hourly temperature series indexed by UTC timestamp, unit Kelvin, name ``temp_K``.

    Raises:
        WeatherFetchError: If timestamps are tz-naive, bounding box is unknown,
            the CDS API call fails, or no data is returned.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise WeatherFetchError(
            f"start and end must be timezone-aware pd.Timestamp; "
            f"got start.tzinfo={start.tzinfo}, end.tzinfo={end.tzinfo}."
        )

    if country not in _COUNTRY_BBOX:
        raise WeatherFetchError(
            f"Unknown country code '{country}'; supported: {list(_COUNTRY_BBOX.keys())}"
        )

    client = cdsapi.Client()
    north, west, south, east = _COUNTRY_BBOX[country]

    chunks: list[pd.Series] = []
    for year in range(start.year, end.year + 1):
        month_start = start.month if year == start.year else 1
        month_end = end.month if year == end.year else 12
        months = [f"{m:02d}" for m in range(month_start, month_end + 1)]
        _logger.info(
            "Querying ERA5 2m temperature for %s year=%d months=%s–%s",
            country,
            year,
            months[0],
            months[-1],
        )
        chunk = _fetch_year_chunk(client, year, months, north, west, south, east)
        chunks.append(chunk)

    if not chunks:
        raise WeatherFetchError(
            f"No temperature data returned for {country} {start}–{end}."
        )

    result = (
        pd.concat(chunks)
        .sort_index()
        .loc[start:end]
        .rename("temp_K")
    )

    if result.empty:
        raise WeatherFetchError(
            f"No temperature data returned for {country} {start}–{end}."
        )

    return result


def load_or_refresh_weather_cache(cutoff: pd.Timestamp) -> pd.Series:
    """Return historical temperature data from on-disk cache, topping up any gap.

    On first run the full history from ``_HISTORICAL_START`` is fetched from
    Copernicus CDS and saved to ``data/raw/temperature_DE_LU_training_cache.csv``.
    On every subsequent run only the gap between the cache end and ``cutoff`` is
    fetched, concatenated, and the cache is updated in-place.

    If the gap fetch fails and the cache is less than 48 hours old, a warning is
    logged and the cached data is returned as-is — it contains enough history
    for a quality forecast.

    Args:
        cutoff: Fetch data up to (not including) this UTC-aware timestamp.

    Returns:
        UTC-indexed hourly temperature series (Kelvin) from ``_HISTORICAL_START``
        to the last available observation before ``cutoff``.

    Raises:
        WeatherFetchError: If the API call fails and the cache is stale (> 48 h).
    """
    cache_path = settings.raw_data_dir / _CACHE_FILENAME

    if cache_path.exists():
        cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        cached = cached_df.squeeze().rename("temp_K")
        if cached.index.tz is None:
            cached.index = cached.index.tz_localize("UTC")
        else:
            cached.index = cached.index.tz_convert("UTC")

        cache_end = cached.index[-1]
        _logger.info(
            "Weather cache loaded: %d records, %s → %s",
            len(cached),
            cached.index[0].date(),
            cache_end.date(),
        )

        gap_start = cache_end + pd.Timedelta(hours=1)
        if gap_start < cutoff:
            _logger.info(
                "Fetching weather gap: %s → %s",
                gap_start.isoformat(),
                cutoff.isoformat(),
            )
            try:
                gap_series = fetch_era5_temperature("DE_LU", gap_start, cutoff)
                full_series = (
                    pd.concat([cached, gap_series])
                    .drop_duplicates()
                    .sort_index()
                    .rename("temp_K")
                )
                full_series.to_frame().to_csv(cache_path)
                _logger.info(
                    "Weather cache updated: %d records, now ends at %s",
                    len(full_series),
                    full_series.index[-1].isoformat(),
                )
            except WeatherFetchError as exc:
                gap_hours = (cutoff - cache_end).total_seconds() / 3600
                # ERA5T (near-real-time) has a ~5-day publication lag, so gaps
                # up to 168 h are expected near the training cutoff date.
                if gap_hours <= 168:
                    _logger.warning(
                        "Weather gap fetch failed (%.0f h gap): %s. "
                        "Cache is recent enough (ERA5T lag ≤7 days) — using it as-is.",
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
            "No weather cache found. Fetching full history %s → %s (first run only).",
            _HISTORICAL_START.date(),
            cutoff.date(),
        )
        full_series = fetch_era5_temperature("DE_LU", _HISTORICAL_START, cutoff)
        settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
        full_series.to_frame().to_csv(cache_path)
        _logger.info(
            "Weather cache saved: %d records to %s", len(full_series), cache_path
        )

    return full_series
