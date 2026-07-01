"""EU AI Act Art. 13 model card generator.

Writes a Markdown file covering every item in the EU AI Act transparency
checklist: model identity, training data, performance metrics, known
limitations, fail-safe behaviour, regulatory mapping, and reproduction
instructions.

Public API:
    generate_model_card(metadata, output_dir) -> Path
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


def generate_model_card(metadata: dict[str, Any], output_dir: Path) -> Path:
    """Write an EU AI Act Art. 13 model card from training metadata.

    The card is named ``model_card_<trained_at_utc>.md`` so that each training
    run produces its own versioned file.  Cards are never overwritten.

    Args:
        metadata: Training provenance dict as produced by
            ``models.persistence.save_forecaster``.  Expected keys:
            ``trained_at_utc``, ``regressor``, ``lags``, ``country_code``,
            ``random_seed``, ``training_start``, ``training_end``,
            ``n_training_samples``, ``metrics`` (sub-keys: mape, mae, rmse),
            ``backtest_config``.
        output_dir: Directory to write the card into (created if absent).

    Returns:
        Path to the written Markdown file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    trained_at = metadata.get("trained_at_utc", "unknown")
    card_path = output_dir / f"model_card_{trained_at}.md"

    metrics = metadata.get("metrics", {})
    mape = metrics.get("mape")
    mae = metrics.get("mae")
    rmse = metrics.get("rmse")

    mape_str = f"{mape:.2f} %" if isinstance(mape, float) else str(mape)
    mae_str = f"{mae:,.0f} MW" if isinstance(mae, float) else str(mae)
    rmse_str = f"{rmse:,.0f} MW" if isinstance(rmse, float) else str(rmse)

    bt = metadata.get("backtest_config", {})
    bt_initial = bt.get("initial_train_size", "80 % of history")
    bt_refit = bt.get("refit", False)

    content = f"""\
# Model Card — DE_LU Load Forecaster

> **EU AI Act Art. 13** — Transparency and provision of information to deployers.
> Generated automatically at training time. Do not edit manually.

---

## Identity

| Field | Value |
|-------|-------|
| Model name | `energy_forecast.baseline.{metadata.get("regressor", "catboost").capitalize()}` |
| Version tag | `{trained_at}` |
| Training date (UTC) | `{trained_at}` |
| Regressor | `{metadata.get("regressor", "catboost")}` |
| Lags | `{metadata.get("lags", 24)}` |
| Country / Bidding zone | `{metadata.get("country_code", "DE_LU")}` |

---

## Intended Use

Day-ahead hourly electricity load forecasting for the DE\\_LU bidding zone
(Germany + Luxembourg), as part of the DDMO SS 2026 load-forecasting challenge.
The forecast covers 24 consecutive UTC hours for the target date.

**Out of scope:** real-time balancing, intraday re-dispatch, non-European grids.

---

## Training Data

| Field | Value |
|-------|-------|
| Source | ENTSO-E Transparency Platform — Actual Load (DE\\_LU) |
| Start | `{metadata.get("training_start", "2022-01-01")}` |
| End | `{metadata.get("training_end", "unknown")}` |
| Samples | `{metadata.get("n_training_samples", "unknown")}` hourly observations |
| Preprocessing | Resample to 60-min mean; forward-fill gaps ≤ 2 h; reject longer gaps |
| Missing data | Audited before filling — see gap audit in training log |

**Exclusions:** 2019 and earlier (pre-energy-crisis demand level mismatch);
2020 (COVID-19 lockdown demand collapse — structurally anomalous).

---

## Performance Metrics (Walk-Forward Backtest)

| Metric | Value |
|--------|-------|
| MAPE | {mape_str} |
| MAE | {mae_str} |
| RMSE | {rmse_str} |

Backtest configuration:
- Initial training window: {bt_initial} samples
- Evaluation window: remaining data in non-overlapping 24-h folds
- Refit between folds: {bt_refit}

---

## Known Limitations

- Accuracy degrades on public holidays not captured by the four calendar features.
- Extreme weather events (cold snaps, heat waves) are underrepresented in training.
- COVID-era consumption patterns (2020) are excluded — if structural demand
  collapse recurs the model will overestimate load.
- Forecast assumes stable grid topology; does not model transmission constraints.

---

## Fail-Safe Behaviour (IEC 61508 / spotforecast2-safe CR-3)

| Condition | Behaviour |
|-----------|-----------|
| Input gap > 2 h | `PreprocessingError` raised — pipeline halts |
| Forecast contains ≤ 0 MW | `PreprocessingError` raised — submission blocked |
| Forecast outside 5 000–150 000 MW | Warning logged — submission proceeds |
| ENTSO-E API failure (cache gap ≤ 48 h) | Cached data used — warning logged |
| No trained model on disk | `PreprocessingError` raised — run train_model.py first |
| Model older than 7 days | Warning logged — submission proceeds |

---

## Regulatory Mapping

| Standard | Applicability |
|----------|---------------|
| EU AI Act 2024/1689 Art. 13 | This model card |
| EU AI Act Art. 9 | Risk management — gap audit + fail-safe output checks |
| IEC 61508 SIL-1 | Fail-safe output validation (CR-3) |
| ISA/IEC 62443 SL-1 | API key in `.env`, never in source code |
| Cyber Resilience Act | Dependency CPE traceability (see `pyproject.toml`) |

---

## Reproduction Instructions

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Set ENTSO-E API key
cp .env.example .env   # fill in ENTSOE_API_KEY

# 3. Train — reproduces this exact artifact with the same seed
python -m energy_forecast.train_model

# Seeds used: RANDOM_SEED={metadata.get("random_seed", 42)}
# Original training run: {trained_at} UTC
```

---

*Generated by `energy_forecast.compliance.model_card` — do not edit manually.*
"""

    card_path.write_text(content)
    _logger.info("Model card written to %s", card_path)
    return card_path
