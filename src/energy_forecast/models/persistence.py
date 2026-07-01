"""Save and load trained ForecasterRecursive artifacts.

Each save produces two files under ``models_dir``:

* ``forecaster_<YYYYMMDD_HHMMSS>.joblib`` — binary model artifact (gitignored)
* ``metadata_<YYYYMMDD_HHMMSS>.json``     — training provenance (committed)

A pointer file ``current.json`` always points at the latest pair so that
``load_forecaster`` never needs to scan the directory.

Public API:
    save_forecaster(forecaster, metadata, models_dir) -> Path
    load_forecaster(models_dir)                       -> tuple[ForecasterRecursive, dict]
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from skforecast.recursive import ForecasterRecursive

from energy_forecast.exceptions import PreprocessingError

_logger = logging.getLogger(__name__)

_STALE_DAYS = 7


def save_forecaster(
    forecaster: ForecasterRecursive,
    metadata: dict[str, Any],
    models_dir: Path,
) -> Path:
    """Serialise a fitted forecaster and write its metadata sidecar.

    Args:
        forecaster: Fitted ForecasterRecursive.
        metadata: Training provenance dict (metrics, config, data range).
        models_dir: Directory to write artifacts into (created if absent).

    Returns:
        Path to the saved ``.joblib`` artifact.
    """
    models_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now("UTC").strftime("%Y%m%d_%H%M%S")
    artifact_path = models_dir / f"forecaster_{ts}.joblib"
    meta_path = models_dir / f"metadata_{ts}.json"

    joblib.dump(forecaster, artifact_path)
    _logger.info("Forecaster saved to %s", artifact_path)

    metadata["artifact_path"] = str(artifact_path)
    metadata["trained_at_utc"] = ts
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    pointer: dict[str, str] = {
        "artifact_path": str(artifact_path),
        "metadata_path": str(meta_path),
    }
    (models_dir / "current.json").write_text(json.dumps(pointer, indent=2))
    _logger.info("current.json updated → %s", artifact_path.name)

    return artifact_path


def load_forecaster(models_dir: Path) -> tuple[ForecasterRecursive, dict[str, Any]]:
    """Load the current forecaster and its metadata from ``models_dir``.

    Also warns if the model artifact is older than ``_STALE_DAYS`` days so
    that operators know to retrain before the model drifts too far from recent
    demand patterns.

    Args:
        models_dir: Directory containing ``current.json``.

    Returns:
        Tuple of (fitted ForecasterRecursive, metadata dict).

    Raises:
        PreprocessingError: If ``current.json`` is missing (no model trained yet).
    """
    pointer_path = models_dir / "current.json"
    if not pointer_path.exists():
        raise PreprocessingError(
            f"No trained model found in '{models_dir}'. "
            "Run: python -m energy_forecast.train_model"
        )

    pointer: dict[str, str] = json.loads(pointer_path.read_text())
    forecaster: ForecasterRecursive = joblib.load(pointer["artifact_path"])
    metadata: dict[str, Any] = json.loads(
        Path(pointer["metadata_path"]).read_text()
    )

    trained_at_str = metadata.get("trained_at_utc", "")
    if trained_at_str:
        try:
            trained_at = pd.Timestamp(
                datetime.strptime(trained_at_str, "%Y%m%d_%H%M%S"), tz="UTC"
            )
            age_days = (pd.Timestamp.now("UTC") - trained_at).days
            if age_days > _STALE_DAYS:
                _logger.warning(
                    "Model is %d day(s) old (threshold: %d). "
                    "Consider retraining: python -m energy_forecast.train_model",
                    age_days,
                    _STALE_DAYS,
                )
            else:
                _logger.info(
                    "Loaded forecaster trained at %s UTC (%d day(s) old, MAPE %.2f %%)",
                    trained_at_str,
                    age_days,
                    metadata.get("metrics", {}).get("mape", float("nan")),
                )
        except ValueError:
            _logger.warning("Could not parse trained_at_utc='%s'", trained_at_str)

    return forecaster, metadata
