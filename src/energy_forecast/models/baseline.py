"""Baseline gradient-boosting forecasting pipeline.

Uses skforecast.ForecasterRecursive with XGBoost, LightGBM, or CatBoost.
All regressors receive an explicit random seed — never rely on global state.

Public API:
    build_forecaster(regressor_name, lags) -> ForecasterRecursive
    train(forecaster, series, exog)        -> ForecasterRecursive
    backtest(forecaster, series, ...)      -> dict[str, float]
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor

from energy_forecast.config import settings
from energy_forecast.evaluation.metrics import compute_all

_logger = logging.getLogger(__name__)

# Regressor factories — lambdas so settings.random_seed is read at call time,
# not at module import time.
_REGRESSOR_FACTORIES: dict[str, Any] = {
    "xgboost": lambda: XGBRegressor(
        n_estimators=500,
        random_state=settings.random_seed,
        n_jobs=1,  # reproducibility across core counts
    ),
    "lightgbm": lambda: LGBMRegressor(
        n_estimators=500,
        random_state=settings.random_seed,
        n_jobs=1,
        verbose=-1,
    ),
    "catboost": lambda: CatBoostRegressor(
        iterations=500,
        random_seed=settings.random_seed,  # CatBoost uses random_seed, not random_state
        thread_count=1,
        verbose=0,
    ),
}


def build_forecaster(
    regressor_name: str = "xgboost",
    lags: int = 24,
) -> ForecasterRecursive:
    """Create a ForecasterRecursive with the named gradient-boosting regressor.

    Args:
        regressor_name: One of ``"xgboost"``, ``"lightgbm"``, ``"catboost"``.
        lags: Number of hourly lags to use as autoregressive features.
            Default 24 = one day of history.

    Returns:
        Untrained ForecasterRecursive instance.

    Raises:
        ValueError: If regressor_name is not in the supported set.
    """
    if regressor_name not in _REGRESSOR_FACTORIES:
        raise ValueError(
            f"Unknown regressor '{regressor_name}'. "
            f"Choose one of: {sorted(_REGRESSOR_FACTORIES)}"
        )
    regressor = _REGRESSOR_FACTORIES[regressor_name]()
    _logger.info("Building ForecasterRecursive with %s, lags=%d", regressor_name, lags)
    return ForecasterRecursive(estimator=regressor, lags=lags)


def train(
    forecaster: ForecasterRecursive,
    series: pd.Series,
    exog: pd.DataFrame | None = None,
) -> ForecasterRecursive:
    """Fit the forecaster on the full series.

    Args:
        forecaster: Untrained ForecasterRecursive.
        series: UTC-indexed hourly load series (``load_MW``).
        exog: Optional calendar feature DataFrame aligned with series.

    Returns:
        The same forecaster, fitted in-place.
    """
    _logger.info("Training on %d samples.", len(series))
    forecaster.fit(y=series, exog=exog)
    return forecaster


def backtest(
    forecaster: ForecasterRecursive,
    series: pd.Series,
    *,
    steps: int = 24,
    initial_train_size: int | None = None,
    exog: pd.DataFrame | None = None,
) -> dict[str, float]:
    """Walk-forward backtesting with MAPE, MAE, and RMSE metrics.

    A single initial training window (no refitting) is used: the forecaster
    is trained once on the first ``initial_train_size`` samples, then evaluated
    on the remainder in non-overlapping ``steps``-sized windows.

    Args:
        forecaster: ForecasterRecursive (does not need to be pre-trained).
        series: Full UTC-indexed hourly load series.
        steps: Forecast horizon in hours per fold. Default 24 (day-ahead).
        initial_train_size: Samples for the initial training window.
            Defaults to 80 % of the series length.
        exog: Calendar feature DataFrame aligned with series.

    Returns:
        Dict with keys ``mape``, ``mae``, ``rmse`` (float, MW or %).
    """
    if initial_train_size is None:
        initial_train_size = int(len(series) * 0.8)

    cv = TimeSeriesFold(
        steps=steps,
        initial_train_size=initial_train_size,
        refit=False,
    )
    _, predictions = backtesting_forecaster(
        forecaster=forecaster,
        y=series,
        cv=cv,
        metric="mean_absolute_error",
        exog=exog,
        verbose=False,
        show_progress=False,
    )

    y_true = series.loc[predictions.index]
    y_pred = predictions["pred"]
    metrics = compute_all(y_true, y_pred)

    _logger.info(
        "Backtest results — MAPE: %.2f %%, MAE: %.0f MW, RMSE: %.0f MW",
        metrics["mape"],
        metrics["mae"],
        metrics["rmse"],
    )
    return metrics
