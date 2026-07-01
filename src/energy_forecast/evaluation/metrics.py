"""Forecast evaluation metrics.

Public API:
    mape(y_true, y_pred)   -> float  — Mean Absolute Percentage Error (%)
    mae(y_true, y_pred)    -> float  — Mean Absolute Error (MW)
    rmse(y_true, y_pred)   -> float  — Root Mean Squared Error (MW)
    compute_all(y_true, y_pred) -> dict[str, float]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Percentage Error in percent.

    Args:
        y_true: Observed values. Must contain no zeros (MAPE is undefined there).
        y_pred: Predicted values, aligned with y_true.

    Returns:
        MAPE in percent (e.g., 3.5 means 3.5 %).

    Raises:
        ValueError: If y_true contains any zero values.
    """
    if (y_true == 0).any():
        raise ValueError(
            "y_true contains zero value(s); MAPE is undefined for zero observations."
        )
    return float(np.mean(np.abs((y_true.to_numpy() - y_pred.to_numpy()) / y_true.to_numpy())) * 100)


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Mean Absolute Error in the same unit as the input (MW).

    Args:
        y_true: Observed values.
        y_pred: Predicted values, aligned with y_true.

    Returns:
        MAE as a non-negative float.
    """
    return float(np.mean(np.abs(y_true.to_numpy() - y_pred.to_numpy())))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Root Mean Squared Error in the same unit as the input (MW).

    Args:
        y_true: Observed values.
        y_pred: Predicted values, aligned with y_true.

    Returns:
        RMSE as a non-negative float.
    """
    return float(np.sqrt(np.mean((y_true.to_numpy() - y_pred.to_numpy()) ** 2)))


def compute_all(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute MAPE, MAE, and RMSE in a single call.

    Args:
        y_true: Observed values (no zeros — required by MAPE).
        y_pred: Predicted values, aligned with y_true.

    Returns:
        Dict with keys ``mape``, ``mae``, ``rmse``.
    """
    return {
        "mape": mape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
