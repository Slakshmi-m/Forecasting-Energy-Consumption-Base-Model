"""Single source of truth for all runtime configuration.

All environment variables are loaded and type-checked here.
No other module may read os.environ or load .env directly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    entsoe_api_key: str
    country_code: str
    random_seed: int
    raw_data_dir: Path
    processed_data_dir: Path
    model_card_dir: Path


def _require(key: str) -> str:
    value = os.environ.get(key, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{key}' is not set. "
            "Copy .env.example to .env and fill in your values."
        )
    return value


def load_settings() -> Settings:
    """Load and validate all settings from environment variables.

    Returns:
        Populated Settings instance.

    Raises:
        EnvironmentError: If any required variable is missing or empty.
    """
    return Settings(
        entsoe_api_key=_require("ENTSOE_API_KEY"),
        country_code=os.environ.get("ENTSOE_COUNTRY_CODE", "DE_LU"),
        random_seed=int(os.environ.get("RANDOM_SEED", "42")),
        raw_data_dir=Path(os.environ.get("RAW_DATA_DIR", "data/raw")),
        processed_data_dir=Path(
            os.environ.get("PROCESSED_DATA_DIR", "data/processed")
        ),
        model_card_dir=Path(os.environ.get("MODEL_CARD_DIR", "model_cards")),
    )


# Module-level singleton 
settings = load_settings()
