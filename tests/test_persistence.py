"""Tests for models/persistence.py — save/load ForecasterRecursive."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from skforecast.recursive import ForecasterRecursive
from xgboost import XGBRegressor

from energy_forecast.exceptions import PreprocessingError
from energy_forecast.models.persistence import load_forecaster, save_forecaster


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_forecaster() -> ForecasterRecursive:
    f = ForecasterRecursive(
        estimator=XGBRegressor(n_estimators=3, random_state=0, n_jobs=1),
        lags=3,
    )
    idx = pd.date_range("2024-01-01", periods=50, freq="60min", tz="UTC")
    series = pd.Series(50_000.0 + np.arange(50, dtype=float), index=idx, name="load_MW")
    f.fit(y=series)
    return f


@pytest.fixture()
def sample_metadata() -> dict:
    return {
        "regressor": "xgboost",
        "lags": 3,
        "country_code": "DE_LU",
        "random_seed": 42,
        "training_start": "2024-01-01",
        "training_end": "2024-01-03",
        "n_training_samples": 50,
        "metrics": {"mape": 1.23, "mae": 500.0, "rmse": 700.0},
    }


# ---------------------------------------------------------------------------
# save_forecaster
# ---------------------------------------------------------------------------


class TestSaveForecaster:
    def test_creates_joblib_artifact(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        artifact = save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        assert artifact.exists()
        assert artifact.suffix == ".joblib"

    def test_creates_metadata_sidecar(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        meta_files = list(tmp_path.glob("metadata_*.json"))
        assert len(meta_files) == 1
        data = json.loads(meta_files[0].read_text())
        assert data["regressor"] == "xgboost"
        assert "trained_at_utc" in data
        assert "artifact_path" in data

    def test_updates_current_json(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        artifact = save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        pointer = json.loads((tmp_path / "current.json").read_text())
        assert pointer["artifact_path"] == str(artifact)

    def test_creates_models_dir_if_absent(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        new_dir = tmp_path / "nested" / "models"
        save_forecaster(tiny_forecaster, sample_metadata.copy(), new_dir)
        assert new_dir.exists()

    def test_second_save_updates_current_pointer(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        artifact2 = save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        pointer = json.loads((tmp_path / "current.json").read_text())
        assert pointer["artifact_path"] == str(artifact2)


# ---------------------------------------------------------------------------
# load_forecaster
# ---------------------------------------------------------------------------


class TestLoadForecaster:
    def test_round_trip_returns_forecaster(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        loaded, _ = load_forecaster(tmp_path)
        assert isinstance(loaded, ForecasterRecursive)
        assert loaded.is_fitted

    def test_round_trip_metadata_intact(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
    ) -> None:
        save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)
        _, meta = load_forecaster(tmp_path)
        assert meta["regressor"] == "xgboost"
        assert meta["metrics"]["mape"] == pytest.approx(1.23)

    def test_raises_if_no_model(self, tmp_path: Path) -> None:
        with pytest.raises(PreprocessingError, match="No trained model"):
            load_forecaster(tmp_path)

    def test_stale_model_logs_warning(
        self,
        tmp_path: Path,
        tiny_forecaster: ForecasterRecursive,
        sample_metadata: dict,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        save_forecaster(tiny_forecaster, sample_metadata.copy(), tmp_path)

        # Overwrite trained_at_utc in the saved metadata with a stale date.
        # (save_forecaster stamps the current time, so we patch after the fact.)
        meta_file = next(tmp_path.glob("metadata_*.json"))
        data = json.loads(meta_file.read_text())
        data["trained_at_utc"] = "20200101_000000"
        meta_file.write_text(json.dumps(data))

        with caplog.at_level(logging.WARNING, logger="energy_forecast.models.persistence"):
            load_forecaster(tmp_path)
        assert any("day(s) old" in r.getMessage() for r in caplog.records)
