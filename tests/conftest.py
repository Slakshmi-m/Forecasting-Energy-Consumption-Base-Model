"""Test suite configuration.

Environment variables are set at module level (before any test module is
collected) so that config.py can call load_settings() when first imported.
load_dotenv() in config.py does not override pre-existing env vars, so these
fake values take precedence over any .env file present on disk.
"""

import os

os.environ.setdefault("ENTSOE_API_KEY", "test-key-00000000-0000-0000-0000-000000000000")
os.environ.setdefault("ENTSOE_COUNTRY_CODE", "DE_LU")
os.environ.setdefault("RANDOM_SEED", "42")
os.environ.setdefault("RAW_DATA_DIR", "data/raw")
os.environ.setdefault("PROCESSED_DATA_DIR", "data/processed")
os.environ.setdefault("MODEL_CARD_DIR", "model_cards")
