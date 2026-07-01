"""Unit tests for energy_forecast.data.openmeteo."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pandas as pd
import pytest

from energy_forecast.data.openmeteo import fetch_openmeteo_forecast
from energy_forecast.exceptions import WeatherFetchError


def _mock_payload(temps_c: list[float], start: str = "2026-06-01T00:00") -> bytes:
    """Build a minimal Open-Meteo-style JSON response."""
    times = [
        (pd.Timestamp(start, tz="UTC") + pd.Timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
        for i in range(len(temps_c))
    ]
    return json.dumps({"hourly": {"time": times, "temperature_2m": temps_c}}).encode()


def _patch_urlopen(response_bytes: bytes):
    """Context manager that replaces urllib.request.urlopen with a fake."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_bytes
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return patch(
        "energy_forecast.data.openmeteo.urllib.request.urlopen",
        return_value=mock_resp,
    )


class TestFetchOpenmeteoForecast:
    def test_returns_correct_length(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([20.0] * 72)):
            result = fetch_openmeteo_forecast(target, horizon=24)
        assert len(result) == 24

    def test_celsius_converted_to_kelvin(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([20.0] * 72)):
            result = fetch_openmeteo_forecast(target, horizon=24)
        assert abs(result.iloc[0] - 293.15) < 1e-6

    def test_series_name_is_temp_k(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([15.0] * 72)):
            result = fetch_openmeteo_forecast(target, horizon=24)
        assert result.name == "temp_K"

    def test_index_is_utc(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([15.0] * 72)):
            result = fetch_openmeteo_forecast(target, horizon=24)
        assert str(result.index.tz) == "UTC"

    def test_index_starts_at_target_date(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([15.0] * 72)):
            result = fetch_openmeteo_forecast(target, horizon=24)
        assert result.index[0] == target

    def test_custom_horizon_respected(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([10.0] * 72)):
            result = fetch_openmeteo_forecast(target, horizon=12)
        assert len(result) == 12

    def test_insufficient_data_raises(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([15.0] * 10)):
            with pytest.raises(WeatherFetchError, match="only 10 hours"):
                fetch_openmeteo_forecast(target, horizon=24)

    def test_network_error_raises_weather_fetch_error(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with patch(
            "energy_forecast.data.openmeteo.urllib.request.urlopen",
            side_effect=URLError("connection refused"),
        ):
            with pytest.raises(WeatherFetchError, match="API request failed"):
                fetch_openmeteo_forecast(target, horizon=24)

    def test_malformed_response_raises(self) -> None:
        target = pd.Timestamp("2026-06-01", tz="UTC")
        bad_payload = json.dumps({"unexpected": "shape"}).encode()
        with _patch_urlopen(bad_payload):
            with pytest.raises(WeatherFetchError, match="missing expected fields"):
                fetch_openmeteo_forecast(target, horizon=24)

    def test_target_date_filters_past_hours(self) -> None:
        # API response starting 12h before target — only target-date hours returned.
        start = "2026-05-31T12:00"
        target = pd.Timestamp("2026-06-01", tz="UTC")
        with _patch_urlopen(_mock_payload([5.0] * 72, start=start)):
            result = fetch_openmeteo_forecast(target, horizon=24)
        assert result.index[0] == target
        assert len(result) == 24
