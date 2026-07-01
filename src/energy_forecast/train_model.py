"""Model training script for the DDMO SS 2026 load-forecasting challenge.

Usage:
    python -m energy_forecast.train_model
    python -m energy_forecast.train_model --cutoff 2026-05-01

Workflow:
    1. Load cached historical data (2022-01-01 onward), refreshing the gap
    2. Audit gaps in the raw series (CR-1 auditability)
    3. Preprocess with build_features()
    4. Train CatBoost forecaster on the full history
    5. Walk-forward backtest → MAPE / MAE / RMSE
    6. Save forecaster + metadata sidecar to models/
    7. Generate EU AI Act Art. 13 model card to model_cards/

Run this script weekly (or whenever demand patterns shift) to keep the model
current.  make_submission.py loads the saved artifact — it never retrains.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from energy_forecast.compliance.model_card import generate_model_card
from energy_forecast.config import settings
from energy_forecast.data.fetch import load_or_refresh_cache
from energy_forecast.data.preprocess import audit_gaps, build_features
from energy_forecast.data.weather import load_or_refresh_weather_cache
from energy_forecast.models.baseline import backtest, build_forecaster, train
from energy_forecast.models.persistence import save_forecaster

_logger = logging.getLogger(__name__)

_REGRESSOR = "catboost"
_LAGS = 168 #changed the window size to 1 week
_MODELS_DIR = Path("models")


def run_training(cutoff: pd.Timestamp) -> Path:
    """Execute the full training pipeline and return the model card path.

    Args:
        cutoff: Train on data up to (not including) this UTC-aware timestamp.

    Returns:
        Path to the generated model card Markdown file.

    Raises:
        ENTSOEFetchError: If the live data fetch fails and the cache is stale.
        PreprocessingError: If the series is incomplete after gap-filling.
    """
    # 1. Load full historical series (cache + live gap).
    raw_series = load_or_refresh_cache(cutoff)

    # 1b. Fetch historical temperature data (ERA5).
    temp_series = load_or_refresh_weather_cache(cutoff)

    # 2. Audit gaps before any filling — logged for traceability (CR-1).
    report = audit_gaps(raw_series)
    if report["total_missing"] > 0:
        _logger.warning(
            "Input data has %d missing hour(s) (max run: %d h). "
            "Review gap_windows before trusting this model.",
            report["total_missing"],
            report["max_run"],
        )

    # 3. Preprocess: validate → fill_gaps (hard cap 2 h) → calendar features + weather features.
    clean_series, train_exog = build_features(raw_series, temp_series)
    _logger.info(
        "Training on %d observations (%s → %s).",
        len(clean_series),
        clean_series.index[0].date(),
        clean_series.index[-1].date(),
    )

    # 4. Train CatBoost forecaster on the full historical window.
    forecaster = build_forecaster(_REGRESSOR, lags=_LAGS)
    train(forecaster, clean_series, exog=train_exog)

    # 5. Walk-forward backtest to measure generalisation.
    initial_train_size = int(len(clean_series) * 0.8)
    metrics = backtest(
        forecaster,
        clean_series,
        steps=24,
        initial_train_size=initial_train_size,
        exog=train_exog,
    )

    # 6. Assemble metadata and save forecaster + sidecar.
    metadata: dict = {
        "regressor": _REGRESSOR,
        "lags": _LAGS,
        "country_code": settings.country_code,
        "random_seed": settings.random_seed,
        "training_start": str(clean_series.index[0].date()),
        "training_end": str(clean_series.index[-1].date()),
        "n_training_samples": len(clean_series),
        "metrics": {
            "mape": round(float(metrics["mape"]), 4),
            "mae": round(float(metrics["mae"]), 2),
            "rmse": round(float(metrics["rmse"]), 2),
        },
        "backtest_config": {
            "steps": 24,
            "initial_train_size": initial_train_size,
            "refit": False,
        },
        "weather_features": {
            "source": "Copernicus ERA5 reanalysis",
            "temp_range_c": [
                round(float(train_exog["temp_c"].min()), 1),
                round(float(train_exog["temp_c"].max()), 1),
            ],
        },
    }
    save_forecaster(forecaster, metadata, _MODELS_DIR)

    # 7. Generate EU AI Act Art. 13 model card.
    card_path = generate_model_card(metadata, settings.model_card_dir)

    _logger.info(
        "Training complete — MAPE %.2f %%, MAE %.0f MW, RMSE %.0f MW. "
        "Model card: %s",
        metrics["mape"],
        metrics["mae"],
        metrics["rmse"],
        card_path,
    )
    return card_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DE_LU load forecaster and generate a model card.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m energy_forecast.train_model\n"
            "  python -m energy_forecast.train_model --cutoff 2026-05-01"
        ),
    )
    parser.add_argument(
        "--cutoff",
        default=None,
        help=(
            "Train on data before this timestamp (YYYY-MM-DD or YYYY-MM-DDTHH:MM, UTC). "
            "Defaults to the current UTC hour."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    args = _parse_args()
    if args.cutoff:
        try:
            cutoff_ts = pd.Timestamp(args.cutoff, tz="UTC")
        except Exception as exc:
            raise SystemExit(f"Invalid --cutoff '{args.cutoff}': {exc}") from exc
    else:
        cutoff_ts = pd.Timestamp.now("UTC").floor("h")

    card = run_training(cutoff_ts)
    print(f"Done. Model card: {card}")
