"""Calendar-aware temporal features for AeroSphinx.

This module implements the correction for **D-001**: the legacy pipeline
computed lags with ``Series.shift(k)`` on a row-ordered frame, so a missing
calendar day caused a two-day-old observation to be presented as
"yesterday's PM2.5". The worst case in the CPCB record was 25 days.

Here a lag is defined against the **calendar**: ``lag_k`` at date *t* is the
observation dated exactly *k* calendar days before *t*, or ``NaN`` when no
observation exists on that date. Nothing is carried forward, interpolated or
otherwise invented to fill the gap.

Rolling statistics obey the same rule and additionally report how many
observations contributed to each window, so a 3-day mean computed from a
single observation is distinguishable from one computed from three.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = [
    "LagSpec",
    "calendar_lag",
    "calendar_lags",
    "calendar_rolling",
    "calendar_rolling_stats",
    "observation_age_days",
]


@dataclass(frozen=True)
class LagSpec:
    """One calendar-aware lag.

    Attributes
    ----------
    name:
        Output column name.
    periods:
        Number of calendar days to look back.
    """

    name: str
    periods: int

    def __post_init__(self) -> None:
        if self.periods < 1:
            raise ValueError("lag periods must be at least 1")


def _require_date_index(frame: pd.DataFrame, column: str = "Date") -> pd.Series:
    """Return a DatetimeIndex-backed series of dates for ``frame``."""
    if column not in frame.columns:
        raise KeyError(f"frame is missing the {column!r} column")
    dates = pd.to_datetime(frame[column])
    if dates.isna().any():
        raise ValueError(f"{column!r} contains unparseable values")
    if dates.duplicated().any():
        raise ValueError(
            f"{column!r} contains duplicate dates; a calendar feature series "
            "must have one row per date"
        )
    return dates


def calendar_lag(
    frame: pd.DataFrame,
    value_column: str,
    periods: int,
    output_name: str | None = None,
    date_column: str = "Date",
) -> pd.Series:
    """Compute a calendar-aware lag of ``value_column``.

    The lag is the value observed exactly ``periods`` calendar days before
    each row's date. Where that date is absent from ``frame``, the result is
    ``NaN`` - never the most recent earlier observation.

    Parameters
    ----------
    frame:
        Rows with one entry per date. May be missing dates; the gaps are
        preserved.
    value_column:
        Column to lag.
    periods:
        Calendar days to look back.
    output_name:
        Name for the returned series. Defaults to ``f"{value_column}_lag_{periods}"``.
    date_column:
        Column holding the date.
    """
    if periods < 1:
        raise ValueError("periods must be at least 1")
    if value_column not in frame.columns:
        raise KeyError(f"frame is missing the {value_column!r} column")

    dates = _require_date_index(frame, date_column)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(
        values.to_numpy(), index=pd.DatetimeIndex(dates), name=value_column
    )

    # Reindex onto the full calendar spanned by the data so that a missing
    # date becomes an explicit NaN row rather than being skipped by a
    # row-position shift.
    calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
    reindexed = series.reindex(calendar)

    lagged = reindexed.shift(periods)
    name = output_name or f"{value_column}_lag_{periods}"
    # Map back onto the original rows only, preserving the caller's index so
    # the result can be concatenated directly onto the input frame.
    result = lagged.reindex(pd.DatetimeIndex(dates))
    result.index = frame.index
    result.name = name
    return result


def calendar_lags(
    frame: pd.DataFrame,
    value_column: str,
    specs: list[LagSpec],
    date_column: str = "Date",
) -> pd.DataFrame:
    """Compute several calendar-aware lags at once."""
    data = {
        spec.name: calendar_lag(
            frame, value_column, spec.periods, spec.name, date_column
        )
        for spec in specs
    }
    return pd.DataFrame(data, index=frame.index)


def calendar_rolling(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    min_periods: int = 1,
    date_column: str = "Date",
) -> pd.Series:
    """Rolling mean over the ``window`` calendar days strictly before each date.

    The window is shifted by one day first, so the value at date *t* uses
    only dates before *t*. Days with no observation inside the window are
    skipped, not carried forward.

    ``min_periods`` controls how many observations are required; the default
    of 1 keeps the legacy behaviour of averaging whatever is available.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    if min_periods < 1:
        raise ValueError("min_periods must be at least 1")

    dates = _require_date_index(frame, date_column)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates))

    calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
    reindexed = series.reindex(calendar)

    rolled = reindexed.shift(1).rolling(window=window, min_periods=min_periods).mean()
    result = rolled.reindex(pd.DatetimeIndex(dates))
    result.index = frame.index
    result.name = f"{value_column}_roll{window}"
    return result


def calendar_rolling_stats(
    frame: pd.DataFrame,
    value_column: str,
    window: int,
    min_periods: int = 1,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Rolling mean, standard deviation and observation count over prior days.

    The observation count is included so that a consumer can tell a mean
    built from one observation from one built from several - a distinction
    the legacy implementation could not make.
    """
    if window < 1:
        raise ValueError("window must be at least 1")

    dates = _require_date_index(frame, date_column)
    values = pd.to_numeric(frame[value_column], errors="coerce")
    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(dates))

    calendar = pd.date_range(series.index.min(), series.index.max(), freq="D")
    reindexed = series.reindex(calendar)

    shifted = reindexed.shift(1)
    rolling = shifted.rolling(window=window, min_periods=min_periods)
    result = pd.DataFrame(
        {
            f"{value_column}_roll{window}_mean": rolling.mean(),
            f"{value_column}_roll{window}_std": rolling.std(),
            f"{value_column}_roll{window}_n": rolling.count(),
        }
    )
    result = result.reindex(pd.DatetimeIndex(dates))
    result.index = frame.index
    return result


def observation_age_days(
    frame: pd.DataFrame,
    date_column: str = "Date",
    lookback: int = 1,
) -> pd.Series:
    """Age of the most recent usable observation for each target date.

    For a target date *t*, this reports how many calendar days old the most
    recent observation dated at or before *(t - lookback)* is, measured from
    *t*.

    With the default ``lookback=1`` this is the diagnostic counterpart to
    the 1-day lag: it answers "if I need yesterday's observation and
    yesterday is missing, how stale is what I would actually use?". On a
    gap-free series every value is 1 (the observation from *t-1* is one day
    old relative to *t*); on a series with a gap the value jumps, revealing
    exactly the staleness that a row-based shift conceals.

    Parameters
    ----------
    frame:
        Rows with one entry per date.
    date_column:
        Column holding the date.
    lookback:
        How many calendar days before the target to search for the most
        recent observation.

    Returns
    -------
    pandas.Series
        Age in days, ``NaN`` where no observation precedes the target.
    """
    if lookback < 1:
        raise ValueError("lookback must be at least 1")

    dates = _require_date_index(frame, date_column)
    observed_index = pd.DatetimeIndex(dates)

    # Candidate observation dates: those at or before (t - lookback).
    target = observed_index - pd.Timedelta(days=lookback)
    positions = observed_index.get_indexer(target, method="ffill")

    age = pd.Series(np.nan, index=observed_index, dtype="float64")
    valid = positions >= 0
    last_observed = observed_index[positions[valid]]
    age[valid] = (observed_index[valid] - last_observed).days

    result = age.reindex(observed_index)
    result.index = frame.index
    result.name = "observation_age_days"
    return result
