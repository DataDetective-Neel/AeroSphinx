"""Authoritative seasonal assignment for AeroSphinx.

This module is the **single** definition of the North-Indian meteorological
seasons used by every experiment. Before it existed the same month mapping was
duplicated in :mod:`aerosphinx.tracks.regime`, :mod:`aerosphinx.tracks.track_a`
and :mod:`aerosphinx.tracks.cold_start`; those helpers now delegate here.

Season definitions
------------------
=============  ==================
Season         Months
=============  ==================
``winter``      December, January, February
``pre_monsoon`` March, April, May
``monsoon``     June, July, August, September
``post_monsoon`` October, November
=============  ==================

These are a **provisional North-Indian** definition. The repository holds a
single station (Faridabad, Delhi-NCR); applying them to another region is an
assumption that must be declared by passing ``region``.

Season-year convention
----------------------
``get_season_year`` returns the **year in which the season starts**. A winter
therefore spans a calendar boundary: December 2024, January 2025 and February
2025 all belong to season-year **2024**. The other three seasons start and end
inside one calendar year, so their season-year is the calendar year itself.

Season assignment is a deterministic function of the calendar date alone. It
uses no observed quantity, so it cannot leak the target or any future value.
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "SEASONS",
    "SEASON_MONTHS",
    "SUPPORTED_REGIONS",
    "SEASON_YEAR_CONVENTION",
    "season_of_month",
    "assign_season",
    "get_season_year",
    "season_features",
    "validate_season_assignment",
    "validate_temporal_order",
]


#: Season name -> months, in seasonal order.
SEASON_MONTHS: dict[str, tuple[int, ...]] = {
    "winter": (12, 1, 2),
    "pre_monsoon": (3, 4, 5),
    "monsoon": (6, 7, 8, 9),
    "post_monsoon": (10, 11),
}

#: The season names in their canonical order.
SEASONS: tuple[str, ...] = tuple(SEASON_MONTHS)

#: Regions for which the month mapping is asserted to apply.
SUPPORTED_REGIONS: tuple[str, ...] = ("north_india",)

#: Documented season-year convention.
SEASON_YEAR_CONVENTION = "season_start_year"

_DEFAULT_REGION = "north_india"


def _check_region(region: str | None) -> None:
    if region is None:
        return
    if region not in SUPPORTED_REGIONS:
        raise ValueError(
            f"unsupported region {region!r}; the provisional season mapping "
            f"is only asserted for {list(SUPPORTED_REGIONS)}"
        )


def _as_datetime_series(dates) -> pd.Series:
    """Coerce dates to a datetime Series, preserving the caller's index."""
    if isinstance(dates, pd.Series):
        return pd.to_datetime(dates)
    if isinstance(dates, pd.DatetimeIndex):
        return pd.Series(dates, index=dates, dtype="datetime64[ns]")
    return pd.Series(pd.to_datetime(list(dates)))


def season_of_month(month: int) -> str:
    """Return the season for a month number (1-12).

    Raises
    ------
    ValueError
        If the month is not an integer in 1..12.
    """
    try:
        value = int(month)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"month must be an integer in 1..12, got {month!r}") from exc
    for season, months in SEASON_MONTHS.items():
        if value in months:
            return season
    raise ValueError(f"month must be in 1..12, got {month!r}")


def assign_season(dates, region: str | None = _DEFAULT_REGION) -> pd.Series:
    """Assign a season label to every date from the calendar alone.

    Parameters
    ----------
    dates:
        Dates as a Series, DatetimeIndex or sequence.
    region:
        Optional region assertion. Only ``"north_india"`` is supported; pass
        ``None`` to use the mapping without asserting a region.

    Returns
    -------
    pandas.Series
        Season labels, indexed like the input. Unparseable dates receive
        ``"unknown"`` and are caught by :func:`validate_season_assignment`.
    """
    _check_region(region)
    series = _as_datetime_series(dates)
    month = series.dt.month

    labels = pd.Series("unknown", index=series.index, dtype=object)
    for season, months in SEASON_MONTHS.items():
        labels[month.isin(months)] = season
    labels.name = "season"
    return labels


def get_season_year(dates, region: str | None = _DEFAULT_REGION) -> pd.Series:
    """Return the season-year (season starting year) for every date.

    December 2024, January 2025 and February 2025 all return **2024**, so a
    winter is never split across two season-years. The other seasons return
    their calendar year.
    """
    _check_region(region)
    series = _as_datetime_series(dates)
    year = series.dt.year
    month = series.dt.month
    result = year.where(~month.isin([1, 2]), year - 1)
    result.name = "season_year"
    return result


def season_features(dates, region: str | None = _DEFAULT_REGION) -> pd.DataFrame:
    """Calendar features derived from the authoritative season assignment.

    Returns ``month``, ``season``, ``season_year`` and ``day_of_year``. Every
    column is a deterministic function of the date, so the frame is known at
    forecast issue time.
    """
    _check_region(region)
    series = _as_datetime_series(dates)
    return pd.DataFrame(
        {
            "month": series.dt.month.to_numpy(),
            "season": assign_season(series, region).to_numpy(),
            "season_year": get_season_year(series, region).to_numpy(),
            "day_of_year": series.dt.day_of_year.to_numpy(),
        },
        index=series.index,
    )


def _check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def validate_season_assignment(
    dates,
    assignments=None,
    region: str | None = _DEFAULT_REGION,
) -> dict:
    """Audit a season assignment against the authoritative mapping.

    Parameters
    ----------
    dates:
        The dates the assignment describes.
    assignments:
        Optional labels to audit. When omitted the output of
        :func:`assign_season` is audited (useful as a self-check).
    region:
        Region assertion passed to the assignment.

    Returns
    -------
    dict
        ``{"passed", "n_checks", "n_failed", "failed_checks", "checks"}``.
    """
    _check_region(region)
    series = _as_datetime_series(dates).reset_index(drop=True)
    if assignments is None:
        labels = assign_season(series, region).reset_index(drop=True)
    else:
        labels = pd.Series(list(assignments)).reset_index(drop=True)

    checks: list[dict] = []
    if len(labels) != len(series):
        return {
            "passed": False,
            "n_checks": 1,
            "n_failed": 1,
            "failed_checks": ["assignment_length_matches_dates"],
            "checks": [
                {
                    "check": "assignment_length_matches_dates",
                    "passed": False,
                    "detail": f"{len(labels)} labels against {len(series)} dates",
                }
            ],
        }

    months = series.dt.month
    expected = months.map(
        lambda value: season_of_month(value) if pd.notna(value) else "unknown"
    )

    season_year = get_season_year(series, region)
    expected_year = series.dt.year.where(~months.isin([1, 2]), series.dt.year - 1)

    _check(
        checks,
        "dates_are_parseable",
        not bool(series.isna().any()),
        f"{int(series.isna().sum())} unparseable dates",
    )
    _check(
        checks,
        "every_date_is_assigned",
        bool(labels.ne("unknown").all() and labels.notna().all()),
        f"{int((labels == 'unknown').sum() + labels.isna().sum())} unassigned dates",
    )
    mismatched = labels.ne(expected) | expected.eq("unknown")
    _check(
        checks,
        "season_matches_month",
        not bool(mismatched.any()),
        f"{int(mismatched.sum())} dates disagree with the month mapping",
    )
    year_same = (season_year == expected_year) | (
        season_year.isna() & expected_year.isna()
    )
    _check(
        checks,
        "season_year_is_start_year",
        bool(year_same.all()),
        f"{int((~year_same).sum())} dates carry the wrong season-year",
    )

    failed = [entry["check"] for entry in checks if not entry["passed"]]
    return {
        "passed": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }


def validate_temporal_order(frame: pd.DataFrame, date_column: str = "Date") -> dict:
    """Check that a frame's dates are usable as a chronological series.

    Checks presence, parseability, uniqueness and monotonic order. A
    non-monotonic frame is reported but not reordered; ordering is the
    caller's responsibility.
    """
    checks: list[dict] = []
    if date_column not in frame.columns:
        return {
            "passed": False,
            "n_checks": 1,
            "n_failed": 1,
            "failed_checks": ["date_column_present"],
            "checks": [
                {
                    "check": "date_column_present",
                    "passed": False,
                    "detail": f"missing {date_column!r}",
                }
            ],
        }

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    _check(
        checks,
        "dates_are_parseable",
        not bool(dates.isna().any()),
        f"{int(dates.isna().sum())} unparseable dates",
    )
    _check(
        checks,
        "dates_are_unique",
        not bool(dates.duplicated().any()),
        f"{int(dates.duplicated().sum())} duplicate dates",
    )
    valid = dates.dropna()
    _check(
        checks,
        "dates_are_monotonic_increasing",
        bool(valid.is_monotonic_increasing),
        (
            "dates are in ascending order"
            if valid.is_monotonic_increasing
            else "dates are not in ascending order"
        ),
    )

    failed = [entry["check"] for entry in checks if not entry["passed"]]
    return {
        "passed": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
