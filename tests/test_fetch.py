"""Unit tests for energy_forecast.data.fetch.

All tests mock EntsoePandasClient so no real network calls are made.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from energy_forecast.data.fetch import _save_series, load_load_data, load_or_refresh_cache
from energy_forecast.exceptions import ENTSOEFetchError


def _make_berlin_df(freq: str = "15min") -> pd.DataFrame:
    """Build a synthetic DataFrame matching entsoe-py's query_load output."""
    index = pd.date_range(
        "2024-01-01",
        "2024-01-02",
        freq=freq,
        tz="Europe/Berlin",
        inclusive="left",
    )
    return pd.DataFrame({"Actual Load": 50000.0}, index=index)


class TestLoadLoadData:
    def test_success_returns_utc_hourly_series(self) -> None:
        with patch(
            "energy_forecast.data.fetch.EntsoePandasClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.query_load.return_value = _make_berlin_df(
                "15min"
            )
            result = load_load_data(
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-02", tz="UTC"),
            )

        assert result.name == "load_MW"
        assert str(result.index.tz) == "UTC"
        assert result.index.freq == pd.tseries.frequencies.to_offset("60min")
        assert not result.isna().any()

    def test_api_connection_error_raises_fetch_error(self) -> None:
        with patch(
            "energy_forecast.data.fetch.EntsoePandasClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.query_load.side_effect = ConnectionError(
                "timeout"
            )
            with pytest.raises(ENTSOEFetchError, match="DE_LU"):
                load_load_data(
                    start=pd.Timestamp("2024-01-01", tz="UTC"),
                    end=pd.Timestamp("2024-01-02", tz="UTC"),
                )

    def test_empty_dataframe_raises_fetch_error(self) -> None:
        with patch(
            "energy_forecast.data.fetch.EntsoePandasClient"
        ) as mock_client_cls:
            mock_client_cls.return_value.query_load.return_value = pd.DataFrame()
            with pytest.raises(ENTSOEFetchError, match="No load data"):
                load_load_data(
                    start=pd.Timestamp("2024-01-01", tz="UTC"),
                    end=pd.Timestamp("2024-01-02", tz="UTC"),
                )

    def test_tz_naive_start_raises_fetch_error(self) -> None:
        with pytest.raises(ENTSOEFetchError, match="timezone-aware"):
            load_load_data(
                start=pd.Timestamp("2024-01-01"),  # tz-naive
                end=pd.Timestamp("2024-01-02", tz="UTC"),
            )

    def test_tz_naive_end_raises_fetch_error(self) -> None:
        with pytest.raises(ENTSOEFetchError, match="timezone-aware"):
            load_load_data(
                start=pd.Timestamp("2024-01-01", tz="UTC"),
                end=pd.Timestamp("2024-01-02"),  # tz-naive
            )


class TestLoadOrRefreshCache:
    """Tests for load_or_refresh_cache — incremental CSV cache logic."""

    def _make_hourly_series(self, start: str, periods: int) -> pd.Series:
        idx = pd.date_range(start, periods=periods, freq="60min", tz="UTC")
        return pd.Series(50_000.0, index=idx, name="load_MW")

    def _mock_settings(self, tmp_path: Path) -> MagicMock:
        m = MagicMock()
        m.raw_data_dir = tmp_path
        m.country_code = "DE_LU"
        return m

    def test_cold_start_fetches_full_history_and_saves(self, tmp_path: Path) -> None:
        """No cache file → fetches from _HISTORICAL_START, writes CSV."""
        fetched = self._make_hourly_series("2022-01-01", 100)

        with (
            patch("energy_forecast.data.fetch.settings", self._mock_settings(tmp_path)),
            patch("energy_forecast.data.fetch.load_load_data", return_value=fetched),
        ):
            result = load_or_refresh_cache(pd.Timestamp("2022-01-05", tz="UTC"))

        assert (tmp_path / "load_DE_LU_training_cache.csv").exists()
        assert result.name == "load_MW"
        assert len(result) == 100

    def test_warm_start_uses_cache_when_fresh(self, tmp_path: Path) -> None:
        """Cache covers cutoff → no gap fetch needed."""
        cached = self._make_hourly_series("2022-01-01", 200)
        (tmp_path / "load_DE_LU_training_cache.csv").write_text(
            cached.to_frame().to_csv()
        )

        with (
            patch("energy_forecast.data.fetch.settings", self._mock_settings(tmp_path)),
            patch("energy_forecast.data.fetch.load_load_data") as mock_fetch,
        ):
            cutoff = cached.index[100]   # inside the cached window
            result = load_or_refresh_cache(cutoff)

        mock_fetch.assert_not_called()
        assert result.name == "load_MW"

    def test_gap_fetch_failure_uses_cache_when_recent(self, tmp_path: Path) -> None:
        """Gap fetch fails but cache is < 48 h old → returns cached data."""
        cache_end = pd.Timestamp.now("UTC").floor("h") - pd.Timedelta(hours=10)
        start = cache_end - pd.Timedelta(hours=99)
        cached = self._make_hourly_series(str(start), 100)
        (tmp_path / "load_DE_LU_training_cache.csv").write_text(
            cached.to_frame().to_csv()
        )
        cutoff = pd.Timestamp.now("UTC").floor("h")

        with (
            patch("energy_forecast.data.fetch.settings", self._mock_settings(tmp_path)),
            patch(
                "energy_forecast.data.fetch.load_load_data",
                side_effect=ENTSOEFetchError("API unavailable"),
            ),
        ):
            result = load_or_refresh_cache(cutoff)

        assert result.name == "load_MW"
        assert len(result) == 100


class TestSaveSeries:
    def test_creates_csv_with_correct_header(self, tmp_path: Path) -> None:
        index = pd.date_range(
            "2024-01-01", periods=24, freq="60min", tz="UTC"
        )
        series = pd.Series(50000.0, index=index, name="load_MW")

        output_path = _save_series(series, tmp_path)

        assert output_path.exists()
        df = pd.read_csv(output_path, index_col=0)
        assert "load_MW" in df.columns
        assert len(df) == 24
