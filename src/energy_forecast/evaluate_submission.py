"""Evaluate a submission CSV against actual ENTSO-E load.

Usage:
    python -m energy_forecast.evaluate_submission \
        --submission submissions/eigen_squad/2026-06-13.csv \
        --date 2026-06-13
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from energy_forecast.data.fetch import load_load_data
from energy_forecast.evaluation.metrics import compute_all

_logger = logging.getLogger(__name__)


def evaluate(submission_path: str, date: str) -> dict[str, float]:
    target = pd.Timestamp(date, tz="UTC")
    start = target
    end = target + pd.Timedelta(hours=24)

    _logger.info("Fetching actual load for %s from ENTSO-E...", date)
    actual = load_load_data(start, end)
    actual = actual[actual.index < end].head(24)

    submission = pd.read_csv(submission_path, parse_dates=["timestamp_utc"])
    submission["timestamp_utc"] = pd.to_datetime(
        submission["timestamp_utc"], utc=True
    )
    forecast = submission.set_index("timestamp_utc")["forecast_mw"]

    common = actual.index.intersection(forecast.index)
    if len(common) != 24:
        _logger.warning(
            "Only %d matching hours found (expected 24). Proceeding with available data.",
            len(common),
        )
    actual_aligned = actual.loc[common]
    forecast_aligned = forecast.loc[common]

    metrics = compute_all(actual_aligned, forecast_aligned)

    print(f"\n=== Evaluation: {date} ===")
    print(f"  Hours compared : {len(common)}")
    print(f"  MAE            : {metrics['mae']:,.2f} MW")
    print(f"  RMSE           : {metrics['rmse']:,.2f} MW")
    print(f"  MAPE           : {metrics['mape']:.4f} %")

    print("\n  Hour-by-hour breakdown:")
    print(f"  {'Hour (UTC)':<22} {'Actual (MW)':>12} {'Forecast (MW)':>14} {'Error (MW)':>11}")
    print("  " + "-" * 62)
    for ts in common:
        a = actual_aligned[ts]
        f = forecast_aligned[ts]
        print(f"  {ts.strftime('%Y-%m-%dT%H:%M:%SZ'):<22} {a:>12,.2f} {f:>14,.2f} {f - a:>+11,.2f}")

    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a submission CSV against actual ENTSO-E load."
    )
    parser.add_argument(
        "--submission",
        required=True,
        help="Path to the submission CSV (timestamp_utc, forecast_mw).",
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Target date in YYYY-MM-DD format.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    args = _parse_args()
    evaluate(args.submission, args.date)
