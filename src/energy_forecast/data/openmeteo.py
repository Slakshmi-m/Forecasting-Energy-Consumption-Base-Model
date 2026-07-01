"""Open-Meteo weather forecast API integration for day-ahead temperature prediction.

Uses the free Open-Meteo API (no API key required) to fetch hourly 2m temperature
forecasts for Germany. Returns Kelvin-unit series compatible with the ERA5 pipeline.

Public API:
    fetch_openmeteo_forecast(target_date, horizon) -> pd.Series
"""

from __future__ import annotations

import json
import logging
import urllib.request
from urllib.error import URLError

import pandas as pd

from energy_forecast.exceptions import WeatherFetchError

_logger = logging.getLogger(__name__)

# Geographic center of Germany — spatially representative for DE_LU load-weighted demand.
_DE_LAT = 51.165691
_DE_LON = 10.451526

_OPENMETEO_BASE = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT = 15  # seconds


def fetch_openmeteo_forecast(
    target_date: pd.Timestamp,
    horizon: int = 24,
) -> pd.Series:
    """Fetch hourly 2m temperature forecast from Open-Meteo for Germany.

    Queries the Open-Meteo free-tier API (no authentication required) at the
    geographic centre of Germany. Temperature is returned in Kelvin to remain
    compatible with the ERA5-based training pipeline.

    Args:
        target_date: UTC midnight of the forecast target day. The returned series
            starts at this timestamp.
        horizon: Number of consecutive hourly steps to return. Defaults to 24.

    Returns:
        UTC-indexed hourly temperature series (Kelvin), named ``temp_K``,
        with ``horizon`` values starting at ``target_date``.

    Raises:
        WeatherFetchError: If the API call fails, returns a malformed response,
            or does not cover the full requested horizon.
    """
    params = (
        f"latitude={_DE_LAT}"
        f"&longitude={_DE_LON}"
        f"&hourly=temperature_2m"
        f"&timezone=UTC"
        f"&forecast_days=3"
        f"&temperature_unit=celsius"
    )
    url = f"{_OPENMETEO_BASE}?{params}"

    _logger.info("Fetching Open-Meteo 2m temperature forecast: %s", url)

    try:
        with urllib.request.urlopen(url, timeout=_REQUEST_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except URLError as exc:
        raise WeatherFetchError(
            f"Open-Meteo API request failed (check network connectivity): {exc}"
        ) from exc
    except Exception as exc:
        raise WeatherFetchError(
            f"Unexpected error fetching Open-Meteo forecast: {exc}"
        ) from exc

    try:
        raw_times = payload["hourly"]["time"]
        temps_c: list[float] = payload["hourly"]["temperature_2m"]
    except (KeyError, TypeError) as exc:
        raise WeatherFetchError(
            f"Open-Meteo response missing expected fields ('hourly.time' / "
            f"'hourly.temperature_2m'): {exc}"
        ) from exc

    index = pd.to_datetime(raw_times, utc=True)
    series = pd.Series(
        [t + 273.15 for t in temps_c],
        index=index,
        name="temp_K",
        dtype=float,
    )

    forecast = series[series.index >= target_date].head(horizon)

    if len(forecast) < horizon:
        raise WeatherFetchError(
            f"Open-Meteo returned only {len(forecast)} hours for target "
            f"{target_date.date()}; expected {horizon}. "
            "Try a target_date closer to today."
        )

    _logger.info(
        "Open-Meteo forecast: %.1f–%.1f °C over %d hours from %s",
        forecast.min() - 273.15,
        forecast.max() - 273.15,
        horizon,
        target_date.isoformat(),
    )

    return forecast
