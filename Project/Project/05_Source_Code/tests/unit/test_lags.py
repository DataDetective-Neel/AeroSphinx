"""Tests for calendar-aware temporal features.

These are the proof tests for the correction of **D-001**: the legacy
pipeline used row-based ``shift()``, so a missing calendar day caused a
stale observation to be presented as a recent one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aerosphinx.features.lags import (
    LagSpec,
    calendar_lag,
    calendar_lags,
    calendar_rolling,
    calendar_rolling_stats,
    observation_age_days,
)

pytestmark = pytest.mark.unit


def frame_with_gap() -> pd.DataFrame:
    """A daily series with a three-day gap in the middle.

    2024-01-05 and 2024-01-06 are absent. A row-based lag would map
    2024-01-07 onto 2024-01-05's value while calling it "yesterday".
    """
    dates = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-07",
            "2024-01-08",
        ]
    )
    return pd.DataFrame({"Date": dates, "PM25": [10.0, 20.0, 30.0, 40.0, 70.0, 80.0]})


def test_lag_across_gap_is_nan_not_stale():
    """A lag must be NaN when the exact previous calendar day is absent."""
    frame = frame_with_gap()
    lag1 = calendar_lag(frame, "PM25", 1)

    assert pd.isna(lag1.iloc[4]), "2024-01-07 must not borrow 2024-01-04's value"
    # The first row has no preceding calendar day at all.
    assert pd.isna(lag1.iloc[0])
    assert lag1.iloc[1] == pytest.approx(10.0)
    assert lag1.iloc[3] == pytest.approx(30.0)
    assert lag1.iloc[5] == pytest.approx(70.0)


def test_row_based_shift_would_have_been_wrong():
    """Demonstrate the defect the calendar-aware lag prevents.

    This test asserts the *legacy* behaviour is wrong, so the regression it
    guards against cannot come back unnoticed.
    """
    frame = frame_with_gap()
    row_based = frame["PM25"].shift(1)

    # The legacy shift maps 2024-01-07 onto 2024-01-04 (a three-day-old value).
    assert row_based.iloc[4] == pytest.approx(40.0)

    calendar_aware = calendar_lag(frame, "PM25", 1)
    assert pd.isna(calendar_aware.iloc[4])


def test_lag_two_across_gap_is_nan():
    """A two-day lag must also respect the calendar."""
    frame = frame_with_gap()
    lag2 = calendar_lag(frame, "PM25", 2)

    # 2024-01-08 minus two calendar days is 2024-01-06, which is absent.
    assert pd.isna(lag2.iloc[5])
    # 2024-01-07 minus two calendar days is 2024-01-05, which is absent.
    assert pd.isna(lag2.iloc[4])


def test_lag_respects_calendar_not_row_count():
    """Across a gap, the lag must look back the correct number of days."""
    frame = frame_with_gap()
    lag3 = calendar_lag(frame, "PM25", 3)

    # 2024-01-07 minus three calendar days is 2024-01-04, which exists.
    assert lag3.iloc[4] == pytest.approx(40.0)
    # 2024-01-08 minus three calendar days is 2024-01-05, which is absent.
    assert pd.isna(lag3.iloc[5])


def test_lag_result_has_no_forward_fill():
    """No value may be carried forward across a gap at any lag."""
    frame = frame_with_gap()
    for periods in (1, 2, 3, 7):
        lagged = calendar_lag(frame, "PM25", periods)
        assert lagged.notna().sum() <= frame["PM25"].notna().sum()


def test_rolling_uses_only_prior_days():
    """A rolling window must never include the target day's own value."""
    frame = frame_with_gap()
    rolled = calendar_rolling(frame, "PM25", 3)

    # For 2024-01-04 the prior three calendar days are 01-01, 01-02, 01-03.
    assert rolled.iloc[3] == pytest.approx((10.0 + 20.0 + 30.0) / 3)
    # The target value 40.0 must not appear in its own window.
    assert rolled.iloc[3] != pytest.approx(40.0)


def test_rolling_reports_observation_count():
    """The rolling helper must expose how many observations contributed."""
    frame = frame_with_gap()
    stats = calendar_rolling_stats(frame, "PM25", 3)

    assert stats["PM25_roll3_n"].iloc[3] == 3
    # 2024-01-07's prior window is 01-04, 01-05, 01-06 -> only 01-04 exists.
    assert stats["PM25_roll3_n"].iloc[4] == 1
    assert stats["PM25_roll3_mean"].iloc[4] == pytest.approx(40.0)


def test_rolling_never_uses_future_values():
    """Perturbing a future value must not change a past rolling value."""
    frame = frame_with_gap()
    baseline = calendar_rolling(frame, "PM25", 3)

    perturbed = frame.copy()
    perturbed.loc[perturbed["Date"] == pd.Timestamp("2024-01-08"), "PM25"] = 9999.0
    after = calendar_rolling(perturbed, "PM25", 3)

    pd.testing.assert_series_equal(baseline.iloc[:5], after.iloc[:5])


def test_observation_age_reveals_staleness():
    """The age diagnostic must expose the gap that a lag conceals.

    With the default one-day lookback, the value is 1 on a gap-free series
    and jumps on the row that follows a gap.
    """
    frame = frame_with_gap()
    age = observation_age_days(frame)

    # 2024-01-01 has no preceding observation.
    assert pd.isna(age.iloc[0])
    # 2024-01-02 .. 2024-01-04 use the immediately preceding day.
    assert age.iloc[1] == 1
    assert age.iloc[2] == 1
    assert age.iloc[3] == 1
    # 2024-01-07 follows a three-day gap: the most recent usable
    # observation is 2024-01-04, three days stale.
    assert age.iloc[4] == 3, "2024-01-07 must be reported as 3 days stale"
    # 2024-01-08 follows 2024-01-07, which is present.
    assert age.iloc[5] == 1


def test_observation_age_never_hides_a_gap():
    """The age series must not be constant across a gap."""
    frame = frame_with_gap()
    age = observation_age_days(frame)
    assert age.max() > age[age.notna()].min()


def test_multiple_lags_at_once():
    """The batch helper must agree with the single-lag helper."""
    frame = frame_with_gap()
    specs = [LagSpec("l1", 1), LagSpec("l2", 2), LagSpec("l7", 7)]
    lags = calendar_lags(frame, "PM25", specs)

    pd.testing.assert_series_equal(
        lags["l1"], calendar_lag(frame, "PM25", 1, "l1"), check_names=False
    )
    pd.testing.assert_series_equal(
        lags["l2"], calendar_lag(frame, "PM25", 2, "l2"), check_names=False
    )


def test_duplicate_dates_are_rejected():
    """A frame with two rows for one date cannot define a lag."""
    frame = pd.DataFrame(
        {"Date": pd.to_datetime(["2024-01-01", "2024-01-01"]), "PM25": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="duplicate dates"):
        calendar_lag(frame, "PM25", 1)


def test_invalid_period_is_rejected():
    """A zero or negative lag is meaningless."""
    frame = frame_with_gap()
    with pytest.raises(ValueError):
        calendar_lag(frame, "PM25", 0)


def test_missing_column_raises():
    """A missing input column is an error, not a silent NaN column."""
    frame = frame_with_gap()
    with pytest.raises(KeyError):
        calendar_lag(frame, "NOT_A_COLUMN", 1)
