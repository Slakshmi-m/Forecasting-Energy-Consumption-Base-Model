"""Tests for compliance/model_card.py — EU AI Act Art. 13 card generator."""

from __future__ import annotations

from pathlib import Path

import pytest

from energy_forecast.compliance.model_card import generate_model_card


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_metadata() -> dict:
    return {
        "trained_at_utc": "20260525_143022",
        "regressor": "catboost",
        "lags": 24,
        "country_code": "DE_LU",
        "random_seed": 42,
        "training_start": "2022-01-01",
        "training_end": "2026-05-24",
        "n_training_samples": 38000,
        "metrics": {"mape": 3.79, "mae": 1523.4, "rmse": 2104.2},
        "backtest_config": {
            "steps": 24,
            "initial_train_size": 30400,
            "refit": False,
        },
    }


# ---------------------------------------------------------------------------
# generate_model_card
# ---------------------------------------------------------------------------


class TestGenerateModelCard:
    def test_creates_markdown_file(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        card = generate_model_card(sample_metadata, tmp_path)
        assert card.exists()
        assert card.suffix == ".md"

    def test_filename_contains_trained_at(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        card = generate_model_card(sample_metadata, tmp_path)
        assert "20260525_143022" in card.name

    def test_card_contains_identity_section(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        card = generate_model_card(sample_metadata, tmp_path)
        content = card.read_text()
        assert "## Identity" in content
        assert "catboost" in content
        assert "DE_LU" in content

    def test_card_contains_metrics(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        card = generate_model_card(sample_metadata, tmp_path)
        content = card.read_text()
        assert "3.79 %" in content
        assert "1,523 MW" in content
        assert "2,104 MW" in content

    def test_card_contains_regulatory_mapping(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        card = generate_model_card(sample_metadata, tmp_path)
        content = card.read_text()
        assert "EU AI Act" in content
        assert "IEC 61508" in content

    def test_card_contains_reproduction_instructions(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        card = generate_model_card(sample_metadata, tmp_path)
        content = card.read_text()
        assert "train_model" in content
        assert "RANDOM_SEED=42" in content

    def test_creates_output_dir_if_absent(
        self, tmp_path: Path, sample_metadata: dict
    ) -> None:
        nested = tmp_path / "deep" / "model_cards"
        generate_model_card(sample_metadata, nested)
        assert nested.exists()

    def test_missing_metrics_handled_gracefully(
        self, tmp_path: Path
    ) -> None:
        meta: dict = {
            "trained_at_utc": "20260101_000000",
            "regressor": "catboost",
            "lags": 24,
        }
        card = generate_model_card(meta, tmp_path)
        assert card.exists()
