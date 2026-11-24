"""Tests for the authoritative seasonal assignment module.

The season mapping is the single source of truth for every experiment, and the
season-year boundary is the property that keeps a winter from being split
across two years. Both are tested directly, together with the delegation of
the three legacy helpers.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aerosphinx.features.seasonal import (
    SEASON_MONTHS,
    SEASONS,
    assign_season,
    get_season_year,
    season_features,
    season_of_month,
    validate_season_assignment,
    validate_temporal_order,
)

pytestmark = pytest.mark.unit


def test_all_twelve_months_map_to_a_season():
    """Every month must belong to exactly one season."""
    seen = {}
    for month in range(1, 13):
        season = season_of_month(month)
        assert season in SEASONS
        seen[season] = seen.get(season, 0) + 1
    assert set(seen) == set(SEASONS)
    assert sum(seen.values()) == 12
    for season, months in SEASON_MONTHS.items():
        assert seen[season] == len(months), f"{season} month count"


def test_month_mapping_is_the_documented_north_indian_definition():
    """The mapping must match the provisional North-Indian definition."""
    assert SEASON_MONTHS["winter"] == (12, 1, 2)
    assert SEASON_MONTHS["pre_monsoon"] == (3, 4, 5)
    assert SEASON_MONTHS["monsoon"] == (6, 7, 8, 9)
    assert SEASON_MONTHS["post_monsoon"] == (10, 11)


def test_assign_season_covers_a_full_year():
    """A full-year range must contain all four seasons."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    labels = assign_season(dates)
    assert set(labels.unique()) == set(SEASONS)


def test_assign_season_preserves_the_input_index():
    """Labels must align with the caller's index, not reset it."""
    dates = pd.Series(
        pd.to_datetime(["2024-01-15", "2024-07-15", "2024-11-15"]),
        index=[10, 20, 30],
    )
    labels = assign_season(dates)
    assert list(labels.index) == [10, 20, 30]
    assert list(labels) == ["winter", "monsoon", "post_monsoon"]


def test_winter_crossing_new_year_shares_one_season_year():
    """Dec 2024, Jan 2025 and Feb 2025 must be one winter season-year."""
    dates = pd.to_datetime(["2024-12-15", "2025-01-15", "2025-02-15"])
    assert list(get_season_year(dates)) == [2024, 2024, 2024]
    assert set(assign_season(dates)) == {"winter"}


def test_non_winter_season_year_is_the_calendar_year():
    """The other three seasons do not cross a year boundary."""
    dates = pd.to_datetime(["2024-04-01", "2024-08-01", "2024-10-01"])
    assert list(get_season_year(dates)) == [2024, 2024, 2024]


def test_december_belongs_to_the_starting_year_not_the_next():
    """December is the first month of its winter, not the last of the old year."""
    assert int(get_season_year(pd.to_datetime(["2024-12-31"])).iloc[0]) == 2024


def test_season_features_are_calendar_only():
    """Every season feature is a deterministic function of the date."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    frame = season_features(dates)
    assert list(frame.columns) == ["month", "season", "season_year", "day_of_year"]
    assert frame["month"].tolist() == [1, 1, 1, 1, 1]
    assert frame["day_of_year"].tolist() == [1, 2, 3, 4, 5]


def test_validation_passes_for_the_authoritative_assignment():
    """The module's own output must satisfy its validator."""
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    result = validate_season_assignment(dates)
    assert result["passed"] is True
    assert result["n_failed"] == 0


def test_validation_detects_a_wrong_assignment():
    """A mislabelled month must be caught, not silently accepted."""
    dates = pd.date_range("2024-01-01", "2024-01-05", freq="D")
    wrong = ["monsoon"] * len(dates)
    result = validate_season_assignment(dates, assignments=wrong)
    assert result["passed"] is False
    assert "season_matches_month" in result["failed_checks"]


def test_unknown_region_is_refused():
    """The mapping is only asserted for the declared region."""
    with pytest.raises(ValueError, match="unsupported region"):
        assign_season(pd.date_range("2024-01-01", periods=2), region="south_india")


def test_invalid_month_is_refused():
    """A non-month number is an error, not a silent 'unknown'."""
    with pytest.raises(ValueError):
        season_of_month(13)
    with pytest.raises(ValueError):
        season_of_month(0)


def test_temporal_order_validation_accepts_a_sorted_frame():
    frame = pd.DataFrame({"Date": pd.date_range("2024-01-01", periods=10)})
    assert validate_temporal_order(frame)["passed"] is True


def test_temporal_order_validation_detects_duplicates():
    frame = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])}
    )
    result = validate_temporal_order(frame)
    assert result["passed"] is False
    assert "dates_are_unique" in result["failed_checks"]


def test_legacy_helpers_delegate_to_the_authoritative_module():
    """regime, track_a and cold_start must agree with the single authority."""
    from aerosphinx.tracks.cold_start import _season_of
    from aerosphinx.tracks.regime import assign_regime
    from aerosphinx.tracks.track_a import _season

    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    expected = assign_season(dates).tolist()

    assert assign_regime(dates).tolist() == expected
    assert _season(dates).tolist() == expected
    assert [_season_of(month) for month in range(1, 13)] == [
        season_of_month(month) for month in range(1, 13)
    ]


def test_legacy_season_helpers_handle_a_series_index():
    """The delegation must preserve index alignment for a Series input."""
    from aerosphinx.tracks.track_a import _season

    dates = pd.Series(pd.to_datetime(["2024-02-01", "2024-06-01"]), index=[7, 9])
    labels = _season(dates)
    assert list(labels.index) == [7, 9]
    assert labels.tolist() == ["winter", "monsoon"]
