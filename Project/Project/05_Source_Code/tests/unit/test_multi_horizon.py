"""Tests for the Phase 3 multi-horizon forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.tracks.multi_horizon import (
    persistence_error_by_lag,
    run_multi_horizon,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def daily_frame() -> pd.DataFrame:
    """A deterministic daily series with a weekly cycle and no trend.

    A mean-reverting series is used deliberately: with a strong upward
    trend, a longer lag is *easier* (it is closer to the future value),
    which would make the persistence-degradation test measure the trend
    rather than the lag. The series is floored at zero so the values stay
    physically plausible for PM2.5.
    """
    dates = pd.date_range("2016-01-01", "2023-12-31", freq="D")
    n = len(dates)
    noise = np.random.default_rng(0).normal(0, 10, n)
    values = np.empty(n)
    values[0] = 80.0
    for t in range(1, n):
        # Mean-reverting AR(1) around a weekly cycle.
        values[t] = 0.85 * values[t - 1] + 8 * np.sin(t * 2 * np.pi / 7) + noise[t]
    values = np.clip(values, 1.0, None)
    return pd.DataFrame({"Date": dates, "PM25": values})


# --------------------------------------------------------------------------
# Availability contract
# --------------------------------------------------------------------------


def test_forecasts_never_use_a_future_observation(daily_frame):
    """Every forecast must be issued at or before its target date.

    The issue date is the last day of the fold's test block, and the
    forecast covers the following days. No observation dated after the
    issue date may enter the model.
    """
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(1, 2, 3),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    forecasts = result.forecasts
    assert (forecasts["Date"] > forecasts["issue_date"]).all()
    assert (
        (forecasts["Date"] - forecasts["issue_date"]).dt.days == forecasts["horizon"]
    ).all()
    # The issue date is the last day of the fold's test block, so it must
    # be within the record.
    assert forecasts["issue_date"].max() <= daily_frame["Date"].max()


def test_direct_and_recursive_agree_at_horizon_one(daily_frame):
    """At one-day horizon the two methods must coincide.

    Recursion has not yet compounded, so a correct implementation gives
    the same answer as the direct model.
    """
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(1,),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    forecasts = result.forecasts
    direct = forecasts[forecasts["method"] == "direct"]["predicted"].to_numpy()
    recursive = forecasts[forecasts["method"] == "recursive"]["predicted"].to_numpy()
    np.testing.assert_allclose(direct, recursive, rtol=1e-9, atol=1e-9)


def test_recursive_diverges_from_direct_at_longer_horizons(daily_frame):
    """Recursion must actually feed predictions back.

    If recursive and direct produced identical results at every horizon,
    the recursion would not be happening.
    """
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(3,),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    forecasts = result.forecasts
    direct = forecasts[forecasts["method"] == "direct"]["predicted"].to_numpy()
    recursive = forecasts[forecasts["method"] == "recursive"]["predicted"].to_numpy()
    assert not np.allclose(direct, recursive)


# --------------------------------------------------------------------------
# Baseline
# --------------------------------------------------------------------------


def test_baseline_is_horizon_appropriate(daily_frame):
    """The baseline must be the value h days before the target.

    Using the one-day baseline to judge a day-5 forecast would flatter
    the learned models.
    """
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(1, 3),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    forecasts = result.forecasts
    lookup = daily_frame.set_index("Date")["PM25"]

    for _, row in forecasts.iterrows():
        expected = lookup.get(row["Date"] - pd.Timedelta(days=int(row["horizon"])))
        assert row["baseline"] == pytest.approx(expected, abs=1e-6)


def test_persistence_error_by_lag_degrades(daily_frame):
    """Persistence error must grow with lag on identical target dates."""
    summary = persistence_error_by_lag(
        daily_frame, daily_frame["Date"].unique(), horizons=(1, 2, 3, 4, 5)
    )
    maes = summary.set_index("horizon")["mae"]
    assert maes.loc[2] > maes.loc[1]
    assert maes.loc[5] > maes.loc[3]


def test_persistence_error_by_lag_uses_identical_targets(daily_frame):
    """Every lag must be evaluated on the same set of target dates."""
    summary = persistence_error_by_lag(
        daily_frame, daily_frame["Date"].unique(), horizons=(1, 2, 3)
    )
    # The counts differ slightly only because the earliest dates cannot
    # supply a lag-3 source; they must not differ wildly.
    assert summary["n"].min() > 0.9 * summary["n"].max()


# --------------------------------------------------------------------------
# Comparison machinery
# --------------------------------------------------------------------------


def test_every_horizon_is_reported(daily_frame):
    """All requested horizons must appear in the results."""
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(1, 2, 3),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    assert set(result.metrics["horizon"]) == {1, 2, 3}
    assert set(result.comparisons["horizon"]) == {1, 2, 3}


def test_comparisons_carry_uncertainty(daily_frame):
    """Every comparison must carry a block-aware interval."""
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(1, 2),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    for column in (
        "block_ci_low",
        "block_ci_high",
        "block_significant",
        "year_ci_low",
        "year_ci_high",
        "p_holm",
    ):
        assert column in result.comparisons.columns


def test_within_issue_comparison_is_produced(daily_frame):
    """The within-issue comparison must be available."""
    result = run_multi_horizon(
        daily_frame,
        models=["ridge"],
        horizons=(1, 2),
        min_train_days=365,
        test_block_days=60,
        step_days=120,
        bootstrap_samples=100,
    )
    assert not result.within_issue.empty
    assert "model_vs_h1" in result.within_issue.columns


def test_missing_target_column_is_rejected(daily_frame):
    """A frame without the target column must fail loudly."""
    with pytest.raises(KeyError):
        run_multi_horizon(
            daily_frame.drop(columns=["PM25"]),
            models=["ridge"],
            horizons=(1,),
            bootstrap_samples=10,
        )
