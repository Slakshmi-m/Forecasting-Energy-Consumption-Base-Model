"""Unit and integration tests for baseline model and evaluation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor

from energy_forecast.evaluation.metrics import compute_all, mae, mape, rmse
from energy_forecast.exceptions import PreprocessingError
from energy_forecast.models.baseline import backtest, build_forecaster, train


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_series() -> pd.Series:
    """100 hourly samples with gentle upward trend — no zeros, clearly positive."""
    idx = pd.date_range("2024-01-01", periods=100, freq="60min", tz="UTC")
    values = 50_000.0 + np.arange(100, dtype=float)
    return pd.Series(values, index=idx, name="load_MW")


@pytest.fixture()
def tiny_forecaster() -> ForecasterRecursive:
    """Tiny XGBoost forecaster for fast tests (5 trees, 3 lags)."""
    return ForecasterRecursive(
        estimator=XGBRegressor(n_estimators=5, random_state=42, n_jobs=1),
        lags=3,
    )


# ---------------------------------------------------------------------------
# build_forecaster
# ---------------------------------------------------------------------------


class TestBuildForecaster:
    def test_returns_forecaster_recursive(self) -> None:
        f = build_forecaster("xgboost", lags=3)
        assert isinstance(f, ForecasterRecursive)

    @pytest.mark.parametrize("name", ["xgboost", "lightgbm", "catboost"])
    def test_all_regressors_build(self, name: str) -> None:
        f = build_forecaster(name, lags=3)
        assert isinstance(f, ForecasterRecursive)

    def test_invalid_name_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown regressor"):
            build_forecaster("random_forest")

    def test_lags_stored_on_forecaster(self) -> None:
        f = build_forecaster("xgboost", lags=12)
        assert f.lags.tolist() == list(range(1, 13))


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


class TestTrain:
    def test_train_returns_forecaster(
        self, tiny_forecaster: ForecasterRecursive, tiny_series: pd.Series
    ) -> None:
        result = train(tiny_forecaster, tiny_series)
        assert result is tiny_forecaster

    def test_forecaster_is_fitted_after_train(
        self, tiny_forecaster: ForecasterRecursive, tiny_series: pd.Series
    ) -> None:
        train(tiny_forecaster, tiny_series)
        assert tiny_forecaster.is_fitted

    def test_train_with_exog(
        self, tiny_forecaster: ForecasterRecursive, tiny_series: pd.Series
    ) -> None:
        exog = pd.DataFrame(
            {"hour": tiny_series.index.hour}, index=tiny_series.index
        )
        train(tiny_forecaster, tiny_series, exog=exog)
        assert tiny_forecaster.is_fitted


# ---------------------------------------------------------------------------
# backtest
# ---------------------------------------------------------------------------


class TestBacktest:
    def test_returns_metric_keys(
        self, tiny_forecaster: ForecasterRecursive, tiny_series: pd.Series
    ) -> None:
        result = backtest(
            tiny_forecaster, tiny_series, steps=5, initial_train_size=80
        )
        assert set(result.keys()) == {"mape", "mae", "rmse"}

    def test_metrics_are_non_negative_floats(
        self, tiny_forecaster: ForecasterRecursive, tiny_series: pd.Series
    ) -> None:
        result = backtest(
            tiny_forecaster, tiny_series, steps=5, initial_train_size=80
        )
        for key, value in result.items():
            assert isinstance(value, float), f"{key} is not float"
            assert value >= 0.0, f"{key} is negative"

    def test_default_initial_train_size_is_80_percent(
        self, tiny_forecaster: ForecasterRecursive, tiny_series: pd.Series
    ) -> None:
        # Should not raise — 80 % of 100 = 80 samples is a valid training size
        result = backtest(tiny_forecaster, tiny_series, steps=5)
        assert "mape" in result


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def _series(self, values: list[float]) -> pd.Series:
        return pd.Series(values, dtype=float)

    def test_mape_perfect_forecast_is_zero(self) -> None:
        y = self._series([100.0, 200.0, 300.0])
        assert mape(y, y) == pytest.approx(0.0)

    def test_mape_known_value(self) -> None:
        y_true = self._series([100.0, 100.0])
        y_pred = self._series([110.0, 90.0])
        # |10/100| = 10 %, average = 10 %
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_raises_on_zero_in_y_true(self) -> None:
        y_true = self._series([100.0, 0.0])
        y_pred = self._series([100.0, 50.0])
        with pytest.raises(ValueError, match="zero"):
            mape(y_true, y_pred)

    def test_mae_known_value(self) -> None:
        y_true = self._series([100.0, 200.0])
        y_pred = self._series([110.0, 180.0])
        assert mae(y_true, y_pred) == pytest.approx(15.0)

    def test_mae_perfect_forecast_is_zero(self) -> None:
        y = self._series([50_000.0, 60_000.0, 55_000.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_known_value(self) -> None:
        y_true = self._series([0.0, 0.0])
        y_pred = self._series([3.0, 4.0])
        # sqrt((9 + 16) / 2) = sqrt(12.5)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(12.5))

    def test_rmse_perfect_forecast_is_zero(self) -> None:
        y = self._series([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_compute_all_returns_all_keys(self) -> None:
        y_true = self._series([100.0, 200.0, 300.0])
        y_pred = self._series([110.0, 190.0, 310.0])
        result = compute_all(y_true, y_pred)
        assert set(result.keys()) == {"mape", "mae", "rmse"}
        assert all(isinstance(v, float) for v in result.values())
