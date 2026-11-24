"""Unit tests for CPCB ingestion.

Covers the four CPCB test categories required by ``AGENTS.md`` section 8:

* duplicate-hour test
* unequal-file-size aggregation test
* timestamp test
* missing-hour test
"""

from __future__ import annotations

import pandas as pd
import pytest

from aerosphinx.ingestion.cpcb import (
    CPCBRules,
    load_cpcb_daily,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Duplicate-hour test
# --------------------------------------------------------------------------


def test_duplicate_detection_and_exclusion(cpcb_factory, tmp_path, rules):
    """Duplicate hours are detected, counted, and exclude the affected day."""
    cpcb_factory("a.csv", [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(20)])
    cpcb_factory(
        "b.csv",
        [("2024-01-01 00:00:00", 60.0)]
        + [(f"2024-01-02 {h:02d}:00:00", 70.0) for h in range(24)],
    )

    directory = tmp_path / "cpcb"
    result = load_cpcb_daily(directory, rules)

    assert result.audit.duplicate_hours == 1
    assert result.audit.days_excluded_duplicate == 1
    assert result.audit.rejection_reasons.get("duplicate_hour") == 1

    # 2024-01-01 is excluded (duplicate); 2024-01-02 survives.
    dates = result.frame["Date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2024-01-02"]


# --------------------------------------------------------------------------
# Unequal-file-size aggregation test (the D-013 correction)
# --------------------------------------------------------------------------


def test_daily_mean_is_hour_weighted_across_files(cpcb_factory, tmp_path, rules):
    """The daily mean must weight every hour equally across all files.

    This is the correction for D-013. The legacy implementation averaged
    *file-level daily means*, so a date whose hours were split 20/4 between
    two files would have been weighted 50/50 between the files rather than
    20/4 between the hours.

    Here 20 hours carry a value of 10 and 4 hours carry 100. The correct
    hourly-weighted mean is 25; the mean-of-file-means would be 55.
    """
    cpcb_factory("a.csv", [(f"2024-01-01 {h:02d}:00:00", 10.0) for h in range(20)])
    cpcb_factory("b.csv", [(f"2024-01-01 {h:02d}:00:00", 100.0) for h in range(20, 24)])

    directory = tmp_path / "cpcb"
    result = load_cpcb_daily(directory, rules)

    assert len(result.frame) == 1
    row = result.frame.iloc[0]
    assert row["valid_hours"] == 24
    # (20 * 10 + 4 * 100) / 24 = 25.0
    assert row["PM25"] == pytest.approx(25.0)
    # The mean-of-file-means would have been (10 + 100) / 2 = 55.0
    assert row["PM25"] != pytest.approx(55.0)


def test_aggregation_is_independent_of_file_split(cpcb_factory, tmp_path, rules):
    """The same hours split differently across files must give one answer."""
    rows = [(f"2024-01-01 {h:02d}:00:00", 10.0 + h) for h in range(24)]

    # One file holding everything.
    cpcb_factory("all.csv", rows)
    single = load_cpcb_daily(tmp_path / "cpcb", rules)
    single_value = single.frame.iloc[0]["PM25"]
    single.frame.to_csv("/dev/null", index=False)

    # Same hours split across two files.

    split_dir = tmp_path / "cpcb_split"
    split_dir.mkdir()
    pd.DataFrame(rows[:16], columns=["Timestamp", "PM2.5 (µg/m³)"]).to_csv(
        split_dir / "part1.csv", index=False
    )
    pd.DataFrame(rows[16:], columns=["Timestamp", "PM2.5 (µg/m³)"]).to_csv(
        split_dir / "part2.csv", index=False
    )
    split = load_cpcb_daily(split_dir, rules)

    assert split.frame.iloc[0]["PM25"] == pytest.approx(single_value)
    assert split.frame.iloc[0]["valid_hours"] == 24


# --------------------------------------------------------------------------
# Timestamp tests
# --------------------------------------------------------------------------


def test_unparseable_timestamp_is_rejected_and_counted(cpcb_factory, tmp_path, rules):
    """A record with an unparseable timestamp cannot be placed on a calendar."""
    cpcb_factory(
        "a.csv",
        [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(24)]
        + [("not-a-timestamp", 99.0)],
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)

    assert result.audit.rows_unparseable_timestamp == 1
    assert result.audit.rejection_reasons.get("unparseable_timestamp") == 1
    assert len(result.frame) == 1
    assert len(result.rejected_records) == 1
    assert result.rejected_records.iloc[0]["reason"] == "unparseable_timestamp"


def test_off_grid_timestamp_is_rejected(cpcb_factory, tmp_path, rules):
    """A timestamp that is not on an exact hour is rejected."""
    cpcb_factory(
        "a.csv",
        [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(23)]
        + [("2024-01-01 23:30:00", 50.0)],
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)

    assert result.audit.rows_off_grid_hour == 1
    assert result.audit.rejection_reasons.get("off_grid_timestamp") == 1


def test_mixed_date_formats_are_normalised(cpcb_factory, tmp_path, rules):
    """Mixed but parseable timestamp formats must land on the same calendar."""
    cpcb_factory(
        "a.csv",
        [("2024-01-01 00:00:00", 50.0)]
        + [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(1, 24)],
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)
    assert len(result.frame) == 1
    assert result.frame.iloc[0]["valid_hours"] == 24


# --------------------------------------------------------------------------
# Missing-hour test
# --------------------------------------------------------------------------


def test_day_below_minimum_hours_is_excluded(cpcb_factory, tmp_path, rules):
    """A day with fewer than the minimum valid hours is excluded."""
    cpcb_factory("a.csv", [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(17)])
    cpcb_factory("b.csv", [(f"2024-01-02 {h:02d}:00:00", 60.0) for h in range(24)])

    result = load_cpcb_daily(tmp_path / "cpcb", rules)

    assert result.audit.days_excluded_insufficient_hours == 1
    assert result.audit.rejection_reasons.get("day_below_min_hours") == 1
    dates = result.frame["Date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2024-01-02"]


def test_day_whose_every_hour_is_rejected_is_counted(cpcb_factory, tmp_path, rules):
    """A day with no usable hour must be excluded *and counted*.

    Without the calendar reindex in ``aggregate_daily`` such a day vanishes
    from both the output and the audit, so the day counts silently fail to
    reconcile against the calendar span.
    """
    cpcb_factory(
        "a.csv",
        [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(24)]
        # Every hour of 2024-01-02 is out of range.
        + [(f"2024-01-02 {h:02d}:00:00", 900.0) for h in range(24)],
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)

    assert result.audit.days_excluded_all_missing == 1
    assert result.audit.rejection_reasons.get("day_no_valid_observations") == 1
    dates = result.frame["Date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2024-01-01"]


def test_day_counts_reconcile_against_calendar_span(cpcb_factory, tmp_path, rules):
    """Accepted plus excluded days must equal the number of calendar days.

    This is the reconciliation that catches silently dropped days.
    """
    cpcb_factory(
        "a.csv",
        [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(24)]
        + [(f"2024-01-02 {h:02d}:00:00", 900.0) for h in range(24)]  # all rejected
        + [(f"2024-01-03 {h:02d}:00:00", 60.0) for h in range(10)],  # too few hours
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)
    audit = result.audit

    calendar_days = 3
    accounted = (
        result.n_days
        + audit.days_excluded_all_missing
        + audit.days_excluded_insufficient_hours
    )
    assert accounted == calendar_days
    assert result.n_days == 1


def test_out_of_range_values_are_missing_not_clipped(cpcb_factory, tmp_path, rules):
    """An 800 ug/m3 reading must be dropped, not clipped to 500."""
    cpcb_factory(
        "a.csv",
        [(f"2024-01-01 {h:02d}:00:00", 800.0 if h == 0 else 50.0) for h in range(24)],
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)

    assert result.audit.rows_out_of_range == 1
    row = result.frame.iloc[0]
    assert row["valid_hours"] == 23
    # Clipping to 500 would give (500 + 23 * 50) / 24 = 68.75
    assert row["PM25"] == pytest.approx(50.0)


def test_negative_values_are_rejected(cpcb_factory, tmp_path, rules):
    """A negative concentration is a fault, not a value."""
    cpcb_factory(
        "a.csv",
        [(f"2024-01-01 {h:02d}:00:00", -5.0 if h == 0 else 50.0) for h in range(24)],
    )

    result = load_cpcb_daily(tmp_path / "cpcb", rules)
    assert result.audit.rows_out_of_range == 1
    assert result.frame.iloc[0]["PM25"] == pytest.approx(50.0)


def test_coverage_fraction_is_recorded(cpcb_factory, tmp_path, rules):
    """Hourly coverage per accepted day is recorded."""
    cpcb_factory("a.csv", [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(24)])

    result = load_cpcb_daily(tmp_path / "cpcb", rules)
    row = result.frame.iloc[0]
    assert row["expected_hours"] == 24
    assert row["coverage_fraction"] == pytest.approx(1.0)


def test_no_imputation_is_performed(cpcb_factory, tmp_path, rules):
    """A gap between days must not be filled by interpolation."""
    cpcb_factory("a.csv", [(f"2024-01-01 {h:02d}:00:00", 50.0) for h in range(24)])
    cpcb_factory("b.csv", [(f"2024-01-05 {h:02d}:00:00", 60.0) for h in range(24)])

    result = load_cpcb_daily(tmp_path / "cpcb", rules)

    dates = result.frame["Date"].dt.strftime("%Y-%m-%d").tolist()
    assert dates == ["2024-01-01", "2024-01-05"]
    # No synthetic rows for 2024-01-02 .. 2024-01-04.
    assert len(result.frame) == 2


# --------------------------------------------------------------------------
# Rule validation
# --------------------------------------------------------------------------


def test_rules_reject_impossible_configuration():
    """A rule set that cannot be satisfied must fail loudly."""
    with pytest.raises(ValueError):
        CPCBRules(pm25_valid_min=500.0, pm25_valid_max=0.0)
    with pytest.raises(ValueError):
        CPCBRules(min_hours_per_day=0)
    with pytest.raises(ValueError):
        CPCBRules(min_hours_per_day=25, expected_hours_per_day=24)
    with pytest.raises(ValueError):
        CPCBRules(duplicate_hour_policy="average")


def test_missing_directory_raises(tmp_path):
    """A missing input directory is an error, not an empty dataset."""
    with pytest.raises(FileNotFoundError):
        load_cpcb_daily(tmp_path / "does_not_exist")
