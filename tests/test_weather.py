"""Tests for weather data integration and preprocessing."""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from energy_forecast.data.weather import fetch_era5_temperature
from energy_forecast.data.preprocess import add_weather_features, build_features
from energy_forecast.exceptions import PreprocessingError, WeatherFetchError


class TestWeatherFetch:
    """Test ERA5 temperature fetching and caching."""

    def test_fetch_era5_temperature_tz_naive_raises(self):
        """Fetching with tz-naive timestamps should raise WeatherFetchError."""
        start = pd.Timestamp("2022-01-01")
        end = pd.Timestamp("2022-01-02")
        with pytest.raises(WeatherFetchError, match="timezone-aware"):
            fetch_era5_temperature("DE_LU", start, end)

    def test_fetch_era5_temperature_unknown_country_raises(self):
        """Fetching unknown country code should raise WeatherFetchError."""
        start = pd.Timestamp("2022-01-01", tz="UTC")
        end = pd.Timestamp("2022-01-02", tz="UTC")
        with pytest.raises(WeatherFetchError, match="Unknown country code"):
            fetch_era5_temperature("XX", start, end)

    def test_add_weather_features_mismatched_lengths_uses_intersection(self):
        """Weather and load series with different lengths should align on intersection."""
        load = pd.Series(
            [1000, 1100, 1200],
            index=pd.date_range("2022-01-01", periods=3, freq="60min", tz="UTC"),
            name="load_MW"
        )
        temp = pd.Series(
            [280, 281, 282, 283],
            index=pd.date_range("2022-01-01", periods=4, freq="60min", tz="UTC"),
            name="temp_K"
        )
        result = add_weather_features(load, temp)
        assert len(result) == 3  # intersection of load (3) and temp (4)

    def test_add_weather_features_creates_correct_columns(self):
        """Weather features should include temp_c, temp_lag_24, temp_deviation."""
        load = pd.Series(
            list(range(100, 148)),
            index=pd.date_range("2022-01-01", periods=48, freq="60min", tz="UTC"),
            name="load_MW"
        )
        temp = pd.Series(
            [280] * 48,  # Constant 280 K ≈ 7°C
            index=load.index,
            name="temp_K"
        )
        result = add_weather_features(load, temp)

        assert "temp_c" in result.columns
        assert "temp_lag_24" in result.columns
        assert "temp_deviation" in result.columns
        assert len(result) == 48

    def test_add_weather_features_kelvin_to_celsius_conversion(self):
        """Temperature should be converted from Kelvin to Celsius."""
        load = pd.Series(
            [1000] * 48,
            index=pd.date_range("2022-01-01", periods=48, freq="60min", tz="UTC"),
            name="load_MW"
        )
        temp = pd.Series(
            [273.15] * 48,  # 0°C
            index=load.index,
            name="temp_K"
        )
        result = add_weather_features(load, temp)
        assert abs(result["temp_c"].iloc[0] - 0.0) < 0.01

    def test_add_weather_features_lag_24_shifts_correctly(self):
        """temp_lag_24 should represent temperature 24 hours prior."""
        load = pd.Series(
            [1000] * 48,
            index=pd.date_range("2022-01-01", periods=48, freq="60min", tz="UTC"),
            name="load_MW"
        )
        # Two distinct temperature blocks
        temp_values = [280.0] * 24 + [290.0] * 24
        temp = pd.Series(
            temp_values,
            index=load.index,
            name="temp_K"
        )
        result = add_weather_features(load, temp)
        # At hour 24, temp_lag_24 should be from hour 0 (280K ≈ 7°C)
        assert abs(result["temp_lag_24"].iloc[24] - 6.85) < 0.1

    def test_build_features_with_optional_temperature(self):
        """build_features should merge weather features when temp_series provided."""
        load = pd.Series(
            [1000] * 48,
            index=pd.date_range("2022-01-01", periods=48, freq="60min", tz="UTC"),
            name="load_MW"
        )
        temp = pd.Series(
            [280] * 48,
            index=load.index,
            name="temp_K"
        )
        clean, exog = build_features(load, temp)

        assert "temp_c" in exog.columns
        assert "temp_lag_24" in exog.columns
        assert "hour" in exog.columns  # Calendar features still present
        assert len(exog) == 48

    def test_build_features_without_temperature(self):
        """build_features should work without temperature (calendar-only)."""
        load = pd.Series(
            [1000] * 48,
            index=pd.date_range("2022-01-01", periods=48, freq="60min", tz="UTC"),
            name="load_MW"
        )
        clean, exog = build_features(load)

        assert "temp_c" not in exog.columns
        assert "hour" in exog.columns  # Calendar features present
        assert len(exog) == 48

    def test_add_weather_features_handles_nan_gracefully(self):
        """Weather features should forward-fill NaNs in derived features."""
        load = pd.Series(
            [1000] * 48,
            index=pd.date_range("2022-01-01", periods=48, freq="60min", tz="UTC"),
            name="load_MW"
        )
        temp = pd.Series(
            [280] * 48,
            index=load.index,
            name="temp_K"
        )
        result = add_weather_features(load, temp)

        # Should have no NaNs after forward-fill
        assert not result[["temp_c", "temp_lag_24"]].isna().any().any()
