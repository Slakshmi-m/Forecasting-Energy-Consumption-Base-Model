"""Unit tests for energy_forecast.data.preprocess."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_forecast.data.preprocess import (
    add_calendar_features,
    audit_gaps,
    build_features,
    fill_gaps,
    validate_series,
)
from energy_forecast.exceptions import PreprocessingError


def _make_series(
    n: int = 24,
    freq: str = "60min",
    tz: str = "UTC",
    name: str = "load_MW",
    value: float = 50000.0,
) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=n, freq=freq, tz=tz)
    return pd.Series(value, index=index, name=name)


class TestAuditGaps:
    def test_no_gaps_returns_zero_missing(self) -> None:
        s = _make_series(n=48)
        report = audit_gaps(s)
        assert report["total_missing"] == 0
        assert report["gap_windows"] == []
        assert report["max_run"] == 0

    def test_single_gap_detected(self) -> None:
        s = _make_series(n=48)
        # Insert a 2-hour gap at index 10-11 by dropping those rows.
        s = s.drop(s.index[10:12])
        report = audit_gaps(s)
        assert report["total_missing"] == 2
        assert len(report["gap_windows"]) == 1
        assert report["gap_windows"][0][2] == 2  # length

    def test_max_run_equals_longest_gap(self) -> None:
        s = _make_series(n=72)
        s = s.drop(s.index[5:10])   # 5-hour gap
        s = s.drop(s.index[20:22])  # 2-hour gap (shifted index)
        report = audit_gaps(s)
        assert report["max_run"] == 5

    def test_by_hour_dict_populated(self) -> None:
        s = _make_series(n=48)
        # Drop hour 0 of day 2 (index 24)
        s = s.drop(s.index[24])
        report = audit_gaps(s)
        assert isinstance(report["by_hour"], dict)
        assert sum(report["by_hour"].values()) == 1

    def test_by_weekday_and_month_are_dicts(self) -> None:
        s = _make_series(n=168)
        s = s.drop(s.index[5])
        report = audit_gaps(s)
        assert isinstance(report["by_weekday"], dict)
        assert isinstance(report["by_month"], dict)

    def test_multiple_gaps_counted(self) -> None:
        s = _make_series(n=100)
        s = s.drop(s.index[5])
        s = s.drop(s.index[20])
        s = s.drop(s.index[40])
        report = audit_gaps(s)
        assert report["total_missing"] == 3
        assert len(report["gap_windows"]) == 3


class TestValidateSeries:
    def test_valid_series_passes(self) -> None:
        s = _make_series()
        result = validate_series(s)
        assert result is s

    def test_non_datetime_index_raises(self) -> None:
        s = pd.Series([1.0, 2.0], index=[0, 1], name="load_MW")
        with pytest.raises(PreprocessingError, match="DatetimeIndex"):
            validate_series(s)

    def test_tz_naive_index_raises(self) -> None:
        s = _make_series(tz=None)  # type: ignore[arg-type]
        with pytest.raises(PreprocessingError, match="timezone-aware"):
            validate_series(s)

    def test_wrong_timezone_raises(self) -> None:
        s = _make_series(tz="Europe/Berlin")
        with pytest.raises(PreprocessingError, match="UTC"):
            validate_series(s)

    def test_wrong_name_raises(self) -> None:
        s = _make_series(name="power")
        with pytest.raises(PreprocessingError, match="load_MW"):
            validate_series(s)

    def test_non_positive_values_raise(self) -> None:
        s = _make_series()
        s.iloc[3] = 0.0
        with pytest.raises(PreprocessingError, match="non-positive"):
            validate_series(s)


class TestFillGaps:
    def test_no_gaps_returns_unchanged(self) -> None:
        s = _make_series(n=24)
        result = fill_gaps(s)
        assert result.isna().sum() == 0
        assert len(result) == 24

    def test_short_gap_is_filled(self) -> None:
        s = _make_series(n=24)
        s.iloc[5] = float("nan")
        result = fill_gaps(s, limit=2)
        assert result.isna().sum() == 0

    def test_gap_exceeding_limit_raises(self) -> None:
        s = _make_series(n=24)
        s.iloc[5:9] = float("nan")  # 4-hour gap
        with pytest.raises(PreprocessingError, match="exceed"):
            fill_gaps(s, limit=2)

    def test_subhourly_input_resampled_to_60min(self) -> None:
        s = _make_series(n=96, freq="15min")
        result = fill_gaps(s)
        assert len(result) == 24


class TestAddCalendarFeatures:
    def test_returns_expected_columns(self) -> None:
        s = _make_series(n=24)
        df = add_calendar_features(s)
        expected_cols = {
            "hour", "weekday", "month", "is_weekend",
            "is_friday", "is_saturday", "is_sunday",
            "is_holiday",
            "hour_weekday_interaction", "hour_weekend_interaction",
            "hour_holiday_interaction",
            "hour_lag_24", "weekday_lag_168",
        }
        assert set(df.columns) == expected_cols

    def test_hour_range(self) -> None:
        s = _make_series(n=24)
        df = add_calendar_features(s)
        assert df["hour"].min() == 0
        assert df["hour"].max() == 23

    def test_is_weekend_binary(self) -> None:
        s = _make_series(n=168)  # one week
        df = add_calendar_features(s)
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_is_friday_set_on_friday(self) -> None:
        # 2024-01-05 is a Friday
        index = pd.date_range("2024-01-05", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["is_friday"] == 1).all()
        assert (df["is_saturday"] == 0).all()
        assert (df["is_sunday"] == 0).all()

    def test_is_saturday_sunday_set_on_weekend(self) -> None:
        # 2024-01-06 is Saturday, 2024-01-07 is Sunday
        index = pd.date_range("2024-01-06", periods=48, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        sat_rows = df[df["is_saturday"] == 1]
        sun_rows = df[df["is_sunday"] == 1]
        assert len(sat_rows) == 24
        assert len(sun_rows) == 24
        assert (sat_rows["is_weekend"] == 1).all()
        assert (sun_rows["is_weekend"] == 1).all()

    def test_hour_weekend_interaction_zero_on_weekday(self) -> None:
        # 2024-01-01 is Monday
        index = pd.date_range("2024-01-01", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["hour_weekend_interaction"] == 0).all()

    def test_hour_weekend_interaction_equals_hour_on_weekend(self) -> None:
        # 2024-01-06 is Saturday
        index = pd.date_range("2024-01-06", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["hour_weekend_interaction"] == df["hour"]).all()

    def test_index_matches_input(self) -> None:
        s = _make_series(n=24)
        df = add_calendar_features(s)
        assert df.index.equals(s.index)

    def test_is_holiday_set_on_whit_monday(self) -> None:
        # 2026-05-25 is Pfingstmontag (Whit Monday): Easter 2026 = April 5, +50 days
        index = pd.date_range("2026-05-25", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["is_holiday"] == 1).all()

    def test_is_holiday_set_on_corpus_christi(self) -> None:
        # 2026-06-04 is Fronleichnam (Corpus Christi): Easter 2026 = April 5, +60 days
        index = pd.date_range("2026-06-04", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["is_holiday"] == 1).all()

    def test_is_holiday_zero_on_regular_monday(self) -> None:
        # 2026-06-08 is a regular Monday (not a holiday)
        index = pd.date_range("2026-06-08", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["is_holiday"] == 0).all()

    def test_is_holiday_set_on_christmas(self) -> None:
        index = pd.date_range("2024-12-25", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["is_holiday"] == 1).all()

    def test_is_holiday_binary(self) -> None:
        s = _make_series(n=168)  # one week
        df = add_calendar_features(s)
        assert set(df["is_holiday"].unique()).issubset({0, 1})

    def test_hour_holiday_interaction_equals_hour_on_holiday(self) -> None:
        # 2026-05-25 is Whit Monday — all 24 hours should equal their hour value
        index = pd.date_range("2026-05-25", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["hour_holiday_interaction"] == df["hour"]).all()

    def test_hour_holiday_interaction_zero_on_non_holiday(self) -> None:
        # 2026-06-08 is a regular Monday
        index = pd.date_range("2026-06-08", periods=24, freq="60min", tz="UTC")
        s = pd.Series(50000.0, index=index, name="load_MW")
        df = add_calendar_features(s)
        assert (df["hour_holiday_interaction"] == 0).all()


class TestBuildFeatures:
    def test_returns_tuple_of_correct_types(self, tmp_path: pytest.TempPathFactory) -> None:
        s = _make_series(n=24)
        clean, exog = build_features(s)
        assert isinstance(clean, pd.Series)
        assert isinstance(exog, pd.DataFrame)

    def test_exog_has_same_length_as_series(self) -> None:
        s = _make_series(n=48)
        clean, exog = build_features(s)
        assert len(clean) == len(exog)

    def test_invalid_series_propagates_error(self) -> None:
        s = _make_series(name="wrong")
        with pytest.raises(PreprocessingError):
            build_features(s)
