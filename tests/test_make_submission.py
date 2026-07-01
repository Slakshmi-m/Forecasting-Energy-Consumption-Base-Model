"""Tests for make_submission.py — submission pipeline using a saved model."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor

from energy_forecast.exceptions import PreprocessingError
from energy_forecast.make_submission import _validate_forecast, make_submission


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_clean_series(periods: int = 200) -> pd.Series:
    idx = pd.date_range("2026-05-01", periods=periods, freq="60min", tz="UTC")
    return pd.Series(50_000.0 + np.arange(periods, dtype=float), index=idx, name="load_MW")


def _make_fitted_forecaster(lags: int = 3) -> ForecasterRecursive:
    f = ForecasterRecursive(
        estimator=XGBRegressor(n_estimators=3, random_state=0, n_jobs=1),
        lags=lags,
    )
    series = _make_clean_series(100)
    exog = pd.DataFrame({"hour": series.index.hour}, index=series.index)
    f.fit(y=series, exog=exog)
    return f


# ---------------------------------------------------------------------------
# _validate_forecast
# ---------------------------------------------------------------------------


class TestValidateForecast:
    def _series(self, values: list[float]) -> pd.Series:
        idx = pd.date_range("2026-05-26", periods=len(values), freq="60min", tz="UTC")
        return pd.Series(values, index=idx)

    def test_valid_forecast_passes(self) -> None:
        forecast = self._series([50_000.0] * 24)
        _validate_forecast(forecast)   # should not raise

    def test_wrong_length_raises(self) -> None:
        forecast = self._series([50_000.0] * 20)
        with pytest.raises(PreprocessingError, match="24"):
            _validate_forecast(forecast)

    def test_non_positive_value_raises(self) -> None:
        values = [50_000.0] * 24
        values[5] = -1.0
        with pytest.raises(PreprocessingError, match="non-positive"):
            _validate_forecast(self._series(values))

    def test_zero_value_raises(self) -> None:
        values = [50_000.0] * 24
        values[0] = 0.0
        with pytest.raises(PreprocessingError, match="non-positive"):
            _validate_forecast(self._series(values))

    def test_out_of_range_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        forecast = self._series([200_000.0] * 24)
        with caplog.at_level(logging.WARNING):
            _validate_forecast(forecast)
        assert any("outside" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# make_submission (integration, all I/O mocked)
# ---------------------------------------------------------------------------


class TestMakeSubmission:
    def _target_date(self) -> pd.Timestamp:
        return pd.Timestamp("2026-05-26", tz="UTC")

    def _mock_forecaster_and_cache(
        self, tmp_path: Path
    ) -> tuple[ForecasterRecursive, pd.Series]:
        forecaster = _make_fitted_forecaster(lags=3)
        # Cache ends 3 hours before target date so last_window fits
        target = self._target_date()
        cache = _make_clean_series(200)
        # Shift cache to end just before target date
        cache.index = pd.date_range(
            end=target - pd.Timedelta(hours=1),
            periods=200,
            freq="60min",
            tz="UTC",
        )
        cache.name = "load_MW"
        return forecaster, cache

    def test_creates_submission_csv(self, tmp_path: Path) -> None:
        forecaster, cache = self._mock_forecaster_and_cache(tmp_path)
        target = self._target_date()

        with (
            patch("energy_forecast.make_submission.load_forecaster") as mock_lf,
            patch("energy_forecast.make_submission.load_or_refresh_cache") as mock_cache,
            patch("energy_forecast.make_submission.Path") as mock_path_cls,
        ):
            mock_lf.return_value = (forecaster, {"trained_at_utc": "20260525_000000"})
            mock_cache.return_value = cache

            # Redirect output to tmp_path
            out_dir = tmp_path / "submissions" / "team_test"
            out_dir.mkdir(parents=True)

            out = make_submission.__wrapped__ if hasattr(make_submission, "__wrapped__") else None

            # Call directly with patched Path for output
            with patch("energy_forecast.make_submission.Path", side_effect=lambda x: tmp_path / x if x == "models" else Path(x)):
                result = make_submission("team_test", target)

        assert result.exists()
        assert result.suffix == ".csv"

    def test_submission_has_required_columns(self, tmp_path: Path) -> None:
        forecaster, cache = self._mock_forecaster_and_cache(tmp_path)
        target = self._target_date()

        with (
            patch("energy_forecast.make_submission.load_forecaster") as mock_lf,
            patch("energy_forecast.make_submission.load_or_refresh_cache") as mock_cache,
            patch("energy_forecast.make_submission.Path", side_effect=lambda x: tmp_path / x if x == "models" else Path(x)),
        ):
            mock_lf.return_value = (forecaster, {"trained_at_utc": "20260525_000000"})
            mock_cache.return_value = cache

            result = make_submission("team_test", target)

        df = pd.read_csv(result)
        assert "timestamp_utc" in df.columns
        assert "forecast_mw" in df.columns
        assert len(df) == 24

    def test_no_model_raises_preprocessing_error(self, tmp_path: Path) -> None:
        with patch(
            "energy_forecast.make_submission.load_forecaster",
            side_effect=PreprocessingError("No trained model found"),
        ):
            with pytest.raises(PreprocessingError, match="No trained model"):
                make_submission("team_test", self._target_date())
