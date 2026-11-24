"""Integration tests for the Track A ingestion pipeline.

These tests run the real modules against the real local CPCB data. They
assert the properties the Phase 0 audit established and that Phase 1 is
required to preserve or correct.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from aerosphinx.features.lags import calendar_lag
from aerosphinx.ingestion.cpcb import load_cpcb_daily
from aerosphinx.quality.coverage import compute_coverage, detect_calendar_gaps

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
CPCB_DIR = ROOT / "data" / "cpcb"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not CPCB_DIR.exists(), reason="local CPCB data not present"),
]


@pytest.fixture(scope="module")
def loaded():
    return load_cpcb_daily(CPCB_DIR)


def test_daily_series_covers_the_expected_record(loaded):
    """The accepted daily series must span the full CPCB record."""
    assert loaded.n_days == 2979
    assert loaded.start.date().isoformat() == "2016-01-01"
    assert loaded.end.date().isoformat() == "2024-12-31"


def test_day_counts_reconcile_against_the_calendar(loaded):
    """Accepted plus excluded days must equal the calendar span.

    This is the reconciliation that catches silently dropped days - the
    defect found and fixed during Phase 1 development.
    """
    audit = loaded.audit
    span = (audit.raw_end - audit.raw_start).days + 1
    accounted = (
        loaded.n_days
        + audit.days_excluded_all_missing
        + audit.days_excluded_insufficient_hours
    )
    assert accounted == span == 3288


def test_no_duplicate_hours_in_the_real_record(loaded):
    """The real CPCB files are partitioned by year, so no hour repeats."""
    assert loaded.audit.duplicate_hours == 0


def test_rejections_are_all_attributed(loaded):
    """Every rejected record must be attributed to a named reason."""
    reasons = loaded.audit.as_dict()["rejections"]
    assert set(reasons) == {
        "non_numeric_pm25",
        "pm25_out_of_range",
        "day_below_min_hours",
        "day_no_valid_observations",
    }
    assert reasons["pm25_out_of_range"] == 736
    assert reasons["non_numeric_pm25"] == 4695


def test_no_out_of_range_value_survives(loaded):
    """Every accepted daily mean must lie inside the valid window."""
    assert loaded.frame["PM25"].between(0.0, 500.0).all()


def test_daily_means_are_hour_weighted(loaded):
    """A day's mean must equal the mean of its own hourly values.

    Recomputed from the raw files to prove the aggregation is genuinely
    hourly-weighted rather than a mean of file-level means.
    """
    raw = pd.read_csv(CPCB_DIR / "City_wise_raw_data_1Hr_2024_Faridabad_1Hr.csv")
    column = [c for c in raw.columns if "PM2.5" in c][0]
    values = pd.to_numeric(raw[column], errors="coerce")
    values = values.where((values >= 0) & (values <= 500))
    raw["Date"] = pd.to_datetime(raw["Timestamp"]).dt.normalize()
    raw["screened"] = values

    expected = raw.groupby("Date")["screened"].mean()
    actual = loaded.frame[loaded.frame["Date"].dt.year == 2024].set_index("Date")[
        "PM25"
    ]
    common = expected.index.intersection(actual.index)
    difference = (expected[common] - actual[common]).abs()
    assert difference.max() < 1e-9


def test_calendar_gaps_match_the_phase_0_audit(loaded):
    """The gap structure must match the Phase 0 findings."""
    gaps = detect_calendar_gaps(loaded.frame)
    assert len(gaps) == 123
    assert max(g["gap_days"] for g in gaps) == 24


def test_coverage_report_is_consistent(loaded):
    """The coverage report must agree with the daily frame."""
    coverage = compute_coverage(loaded.frame)
    assert coverage.n_days == loaded.n_days
    assert coverage.calendar_span_days == 3288
    assert coverage.missing_days == 309
    assert coverage.coverage_fraction == pytest.approx(2979 / 3288)


def test_calendar_aware_lag_rejects_stale_values(loaded):
    """The D-001 correction must hold on the real record.

    Every row whose previous calendar day is absent must have a NaN lag.
    There are 123 such rows following a gap, plus the first row of the
    series, which has no preceding calendar day either.
    """
    daily = loaded.frame
    lag1 = calendar_lag(daily, "PM25", 1)

    dates = pd.DatetimeIndex(daily["Date"])
    previous_day_exists = (dates - pd.Timedelta(days=1)).isin(set(dates))

    assert lag1.isna().equals(pd.Series(~previous_day_exists))
    assert int(lag1.isna().sum()) == 124
    # 123 rows follow a gap; the last is the first row of the series.
    assert int((~previous_day_exists).sum()) == 124


def test_row_based_lag_would_have_been_stale(loaded):
    """Quantify the defect the calendar-aware lag removes.

    This is the regression guard: it asserts that the legacy behaviour
    would have produced materially stale features on the real data.
    """
    daily = loaded.frame
    row_based = daily["PM25"].shift(1)
    calendar_aware = calendar_lag(daily, "PM25", 1)

    stale_rows = row_based.notna() & calendar_aware.isna()
    assert int(stale_rows.sum()) == 123

    dates = pd.DatetimeIndex(daily["Date"])
    previous_day_exists = (dates - pd.Timedelta(days=1)).isin(set(dates))
    ages = [
        (dates[i] - dates[previous_day_exists & (dates <= dates[i])][-1]).days
        for i in range(len(daily))
        if stale_rows.iloc[i]
    ]
    assert max(ages) == 25, "the worst stale gap must still be 25 days"


def test_no_imputation_was_performed(loaded):
    """The number of distinct dates must equal the number of rows."""
    assert loaded.frame["Date"].is_unique
    assert len(loaded.frame) == loaded.frame["Date"].nunique()
