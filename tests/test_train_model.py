"""Tests for train_model.py — full training pipeline with all I/O mocked."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from energy_forecast.train_model import run_training


def _make_series(periods: int = 200) -> pd.Series:
    idx = pd.date_range("2022-01-01", periods=periods, freq="60min", tz="UTC")
    return pd.Series(50_000.0 + np.arange(periods, dtype=float), index=idx, name="load_MW")


class TestRunTraining:
    def test_returns_model_card_path(self, tmp_path: Path) -> None:
        series = _make_series(300)
        exog = pd.DataFrame({"hour": series.index.hour}, index=series.index)

        with (
            patch("energy_forecast.train_model.load_or_refresh_cache", return_value=series),
            patch("energy_forecast.train_model.audit_gaps", return_value={"total_missing": 0, "max_run": 0}),
            patch("energy_forecast.train_model.build_features", return_value=(series, exog)),
            patch("energy_forecast.train_model.build_forecaster") as mock_bf,
            patch("energy_forecast.train_model.train"),
            patch("energy_forecast.train_model.backtest", return_value={"mape": 3.5, "mae": 1500.0, "rmse": 2000.0}),
            patch("energy_forecast.train_model.save_forecaster"),
            patch("energy_forecast.train_model.generate_model_card", return_value=tmp_path / "model_card.md") as mock_card,
            patch("energy_forecast.train_model._MODELS_DIR", tmp_path),
            patch("energy_forecast.train_model.settings") as mock_settings,
        ):
            mock_settings.country_code = "DE_LU"
            mock_settings.random_seed = 42
            mock_settings.model_card_dir = tmp_path
            mock_bf.return_value = MagicMock(is_fitted=True)

            result = run_training(pd.Timestamp("2026-05-01", tz="UTC"))

        mock_card.assert_called_once()
        assert result == tmp_path / "model_card.md"

    def test_warns_on_gaps_in_input(self, tmp_path: Path) -> None:
        import logging

        series = _make_series(300)
        exog = pd.DataFrame({"hour": series.index.hour}, index=series.index)

        with (
            patch("energy_forecast.train_model.load_or_refresh_cache", return_value=series),
            patch("energy_forecast.train_model.audit_gaps", return_value={"total_missing": 5, "max_run": 2}),
            patch("energy_forecast.train_model.build_features", return_value=(series, exog)),
            patch("energy_forecast.train_model.build_forecaster", return_value=MagicMock()),
            patch("energy_forecast.train_model.train"),
            patch("energy_forecast.train_model.backtest", return_value={"mape": 3.5, "mae": 1500.0, "rmse": 2000.0}),
            patch("energy_forecast.train_model.save_forecaster"),
            patch("energy_forecast.train_model.generate_model_card", return_value=tmp_path / "card.md"),
            patch("energy_forecast.train_model._MODELS_DIR", tmp_path),
            patch("energy_forecast.train_model.settings") as mock_settings,
        ):
            mock_settings.country_code = "DE_LU"
            mock_settings.random_seed = 42
            mock_settings.model_card_dir = tmp_path

            result = run_training(pd.Timestamp("2026-05-01", tz="UTC"))

        assert result is not None
