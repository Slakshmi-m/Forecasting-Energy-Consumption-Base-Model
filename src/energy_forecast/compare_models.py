"""Compare backtest errors across XGBoost, LightGBM, and CatBoost.

Loads pre-processed data from data/processed/ (no API calls required) and
runs identical walk-forward backtests for each regressor, then prints a
side-by-side comparison table.

Usage:
    python -m energy_forecast.compare_models
    python -m energy_forecast.compare_models --lags 168 --steps 24
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

from energy_forecast.models.baseline import backtest, build_forecaster

_logger = logging.getLogger(__name__)

_REGRESSORS = ["xgboost", "lightgbm", "catboost"]
_DEFAULT_LAGS = 168
_DEFAULT_STEPS = 24
_PROCESSED_DIR = Path("data/processed")


def _load_processed_data() -> tuple[pd.Series, pd.DataFrame]:
    """Load the most recent cleaned series and exog features from data/processed/."""
    clean_files = sorted(_PROCESSED_DIR.glob("*_clean.csv"))
    exog_files = sorted(_PROCESSED_DIR.glob("*_exog.csv"))

    if not clean_files:
        raise FileNotFoundError(
            f"No *_clean.csv found in {_PROCESSED_DIR}. "
            "Run train_model.py first to generate processed data."
        )

    clean_path = clean_files[-1]
    exog_path = exog_files[-1]
    _logger.info("Loading clean series from %s", clean_path)
    _logger.info("Loading exog features from %s", exog_path)

    series = (
        pd.read_csv(clean_path, index_col=0)
        .squeeze("columns")
        .rename("load_MW")
    )
    series.index = pd.to_datetime(series.index, utc=True)
    series.index.name = None
    series.index.freq = pd.tseries.frequencies.to_offset("h")

    exog = pd.read_csv(exog_path, index_col=0)
    exog.index = pd.to_datetime(exog.index, utc=True)
    exog.index.name = None
    exog.index.freq = pd.tseries.frequencies.to_offset("h")

    return series, exog


def run_comparison(lags: int = _DEFAULT_LAGS, steps: int = _DEFAULT_STEPS) -> pd.DataFrame:
    """Backtest all three regressors and return a metrics DataFrame.

    Args:
        lags: Number of hourly autoregressive lags.
        steps: Forecast horizon per fold (hours).

    Returns:
        DataFrame indexed by regressor name with columns mape, mae, rmse.
    """
    series, exog = _load_processed_data()
    initial_train_size = int(len(series) * 0.8)

    _logger.info(
        "Backtesting %d regressors — %d samples, 80/20 split, lags=%d, steps=%d",
        len(_REGRESSORS),
        len(series),
        lags,
        steps,
    )

    results: dict[str, dict[str, float]] = {}
    for name in _REGRESSORS:
        print(f"\n[{name}] Building and backtesting... ", end="", flush=True)
        t0 = time.perf_counter()

        forecaster = build_forecaster(name, lags=lags)
        metrics = backtest(
            forecaster,
            series,
            steps=steps,
            initial_train_size=initial_train_size,
            exog=exog,
        )

        elapsed = time.perf_counter() - t0
        print(f"done ({elapsed:.0f}s)")
        results[name] = metrics

    return pd.DataFrame(results).T[["mape", "mae", "rmse"]]


def _print_table(df: pd.DataFrame) -> None:
    """Print a formatted comparison table to stdout."""
    best_mape = df["mape"].idxmin()
    best_mae = df["mae"].idxmin()
    best_rmse = df["rmse"].idxmin()

    print("\n" + "=" * 56)
    print("  Backtest Comparison — XGBoost / LightGBM / CatBoost")
    print("=" * 56)
    print(f"  {'Model':<12}  {'MAPE (%)':>10}  {'MAE (MW)':>10}  {'RMSE (MW)':>10}")
    print("-" * 56)
    for name, row in df.iterrows():
        mape_tag = " *" if name == best_mape else "  "
        mae_tag  = " *" if name == best_mae  else "  "
        rmse_tag = " *" if name == best_rmse else "  "
        print(
            f"  {name:<12}  {row['mape']:>9.3f}{mape_tag}"
            f"  {row['mae']:>9.1f}{mae_tag}"
            f"  {row['rmse']:>9.1f}{rmse_tag}"
        )
    print("-" * 56)
    print("  * = best in column")
    print("=" * 56)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare backtest errors for XGBoost, LightGBM, and CatBoost.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m energy_forecast.compare_models\n"
            "  python -m energy_forecast.compare_models --lags 24 --steps 24"
        ),
    )
    parser.add_argument("--lags", type=int, default=_DEFAULT_LAGS,
                        help=f"Autoregressive lag count (default: {_DEFAULT_LAGS})")
    parser.add_argument("--steps", type=int, default=_DEFAULT_STEPS,
                        help=f"Forecast horizon per fold in hours (default: {_DEFAULT_STEPS})")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    args = _parse_args()
    results_df = run_comparison(lags=args.lags, steps=args.steps)
    _print_table(results_df)
