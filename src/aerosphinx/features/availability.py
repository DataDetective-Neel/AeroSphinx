"""Multisource fusion with an explicit availability mask (Phase 6).

This module is the canonical fusion path for Track B. It replaces the
forward-fill that **D-003** identified as manufacturing environmental
observations.

The rule is simple and enforced structurally:

* The ground series defines the row calendar. An environmental source can
  never extend it.
* A source value appears on a row **only** when that source has a genuine
  observation for that exact date. Missing values stay ``NaN``. Nothing is
  forward-filled, interpolated or imputed.
* Every row carries a per-source ``<source>_observed`` flag, so a model can
  distinguish "no observation" from "observed as zero".
* Every row carries a per-source ``<source>_observation_age_days`` column.
  The age is a **diagnostic of staleness**, not an imputed value: it records
  how many calendar days have elapsed since the most recent genuine
  observation of that source. It never supplies a value for a missing day.
* AOD support counts (``aod_valid_pixels``, ``aod_overpasses``) are set to
  ``0`` when no AOD observation exists. They are counts of supporting
  pixels, not measurements, so a zero is the true support of an absent
  observation rather than a manufactured value.

Availability classes are inherited from :mod:`aerosphinx.features.groups`.
All three environmental sources are retrospective only: fusion here is
estimation, never an operational forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aerosphinx.features.groups import get_group

__all__ = [
    "FusionRules",
    "FusedDataset",
    "SOURCE_GROUPS",
    "value_columns",
    "support_columns",
    "build_fused_dataset",
    "validate_fused_dataset",
    "assert_fusion_alignment",
]

#: The environmental sources fused onto the ground series.
SOURCE_GROUPS: tuple[str, ...] = ("station_met", "merra", "aod")

#: Columns that describe the *support* of a source value rather than being a
#: measurement in their own right.
_SUPPORT_COLUMNS: dict[str, tuple[str, ...]] = {
    "aod": ("aod_valid_pixels", "aod_overpasses"),
}

#: The explicit policy recorded in every summary.
MISSING_VALUE_POLICY = (
    "Preserve missing environmental observations as NaN. No forward-fill, "
    "interpolation or imputation is permitted; per-source observation flags "
    "and observation ages make the missingness visible instead."
)


def value_columns(source: str) -> tuple[str, ...]:
    """The measurement columns contributed by a source."""
    support = set(_SUPPORT_COLUMNS.get(source, ()))
    return tuple(
        feature for feature in get_group(source).features if feature not in support
    )


def support_columns(source: str) -> tuple[str, ...]:
    """The support-metadata columns contributed by a source."""
    return tuple(_SUPPORT_COLUMNS.get(source, ()))


def _all_columns(source: str) -> tuple[str, ...]:
    return value_columns(source) + support_columns(source)


@dataclass(frozen=True)
class FusionRules:
    """Rules for the Phase 6 fused dataset."""

    date_column: str = "Date"
    target: str = "PM25"
    #: How many non-null measurement columns a source needs on a date for
    #: that date to count as observed. One is the minimum meaningful value.
    min_observed_values: int = 1

    def __post_init__(self) -> None:
        if self.min_observed_values < 1:
            raise ValueError("min_observed_values must be at least 1")


@dataclass
class FusedDataset:
    """The fused frame, its availability summary and its alignment audit."""

    frame: pd.DataFrame
    summary: dict
    rules: FusionRules
    requested_sources: tuple[str, ...]
    checks: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether every alignment check passed."""
        return bool(self.checks.get("passed", False))


def _normalise_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.normalize()


def _prepare_source(
    source: str, frame: pd.DataFrame | None, rules: FusionRules
) -> pd.DataFrame | None:
    """Validate and select a source frame, without filling anything."""
    if frame is None:
        return None

    date = rules.date_column
    required = (date, *_all_columns(source))
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"source {source!r} is missing required columns: {missing}")

    selected = frame[list(required)].copy()
    selected[date] = _normalise_dates(selected[date])
    if selected[date].isna().any():
        raise ValueError(f"source {source!r} contains an unparseable date")
    if selected[date].duplicated().any():
        raise ValueError(
            f"source {source!r} contains duplicate dates; a daily source "
            "must have exactly one row per date"
        )
    return selected.sort_values(date).reset_index(drop=True)


def build_fused_dataset(
    daily: pd.DataFrame,
    station_met: pd.DataFrame | None = None,
    merra: pd.DataFrame | None = None,
    aod: pd.DataFrame | None = None,
    rules: FusionRules | None = None,
) -> FusedDataset:
    """Fuse the environmental sources onto the ground series.

    Parameters
    ----------
    daily:
        The ground daily series. It defines the row calendar, so its dates
        and row count are preserved exactly.
    station_met, merra, aod:
        Daily source frames with canonical column names. A source may be
        omitted (``None``); its columns are then present and empty, so the
        schema is stable.
    rules:
        Fusion rules.

    Returns
    -------
    FusedDataset
        The fused frame with availability flags and ages, plus a summary.

    Raises
    ------
    ValueError
        If the ground series contains duplicate dates.
    KeyError
        If a supplied source frame lacks a required column.
    """
    rules = rules or FusionRules()
    date = rules.date_column

    if date not in daily.columns:
        raise KeyError(f"ground series is missing the date column {date!r}")

    base = daily.copy()
    base[date] = _normalise_dates(base[date])
    if base[date].isna().any():
        raise ValueError("ground series contains an unparseable date")
    if base[date].duplicated().any():
        raise ValueError("ground series contains duplicate dates")
    base = base.sort_values(date).reset_index(drop=True)

    supplied = {"station_met": station_met, "merra": merra, "aod": aod}
    requested = tuple(name for name in SOURCE_GROUPS if supplied[name] is not None)

    frame = base
    source_dates_outside_calendar: dict[str, int] = {}
    for source in SOURCE_GROUPS:
        prepared = _prepare_source(source, supplied[source], rules)
        if prepared is None:
            for column in value_columns(source):
                frame[column] = np.nan
            for column in support_columns(source):
                frame[column] = 0
            source_dates_outside_calendar[source] = 0
            continue

        source_dates_outside_calendar[source] = int(
            (~prepared[date].isin(set(frame[date]))).sum()
        )
        frame = frame.merge(prepared, on=date, how="left", validate="many_to_one")

    # AOD support counts are metadata: zero support for an absent
    # observation. They are not a measurement and are never imputed.
    for column in support_columns("aod"):
        frame[column] = frame[column].fillna(0)

    # --- availability mask -------------------------------------------------
    for source in SOURCE_GROUPS:
        columns = value_columns(source)
        has_value = (
            frame[list(columns)].notna().sum(axis=1) >= rules.min_observed_values
        )
        if source == "aod":
            # A present AOD value is only real when a valid pixel supports it.
            has_value = has_value & (frame["aod_valid_pixels"].to_numpy() > 0)
        frame[f"{source}_observed"] = has_value.to_numpy(dtype=bool)

    # --- observation age (diagnostic, never a substitute value) ------------
    for source in SOURCE_GROUPS:
        observed_dates = (
            frame.loc[frame[f"{source}_observed"], date].drop_duplicates().sort_values()
        )
        age_column = f"{source}_observation_age_days"
        if observed_dates.empty:
            frame[age_column] = np.nan
            continue

        lookup = pd.DataFrame(
            {
                date: observed_dates.to_numpy(),
                "_observed_date": observed_dates.to_numpy(),
            }
        )
        matched = pd.merge_asof(
            frame[[date]].sort_values(date),
            lookup,
            on=date,
            direction="backward",
            allow_exact_matches=True,
        )
        frame[age_column] = (frame[date] - matched["_observed_date"]).dt.days.to_numpy(
            dtype=float
        )

    # --- combined availability --------------------------------------------
    if requested:
        mask_columns = [f"{source}_observed" for source in requested]
        frame["n_sources_observed"] = frame[mask_columns].sum(axis=1).astype(int)
        frame["all_sources_observed"] = frame["n_sources_observed"] == len(requested)
    else:
        frame["n_sources_observed"] = 0
        frame["all_sources_observed"] = False

    summary = _summary(frame, base, requested, source_dates_outside_calendar, rules)
    return FusedDataset(
        frame=frame,
        summary=summary,
        rules=rules,
        requested_sources=requested,
    )


def _target_missingness(frame: pd.DataFrame, mask: pd.Series, target: str) -> dict:
    """Compare the target on observed and unobserved dates.

    The Phase 4 analysis found that AOD is least available when pollution is
    worst. Reporting the same contrast for every source, before any fusion
    experiment, keeps the availability structure visible.
    """
    if target not in frame.columns:
        return {}

    values = pd.to_numeric(frame[target], errors="coerce")
    observed = values[mask.to_numpy() & values.notna().to_numpy()]
    unobserved = values[(~mask.to_numpy()) & values.notna().to_numpy()]
    detail = {
        "target_n_observed": int(observed.size),
        "target_n_unobserved": int(unobserved.size),
        "target_mean_when_observed": (
            float(observed.mean()) if observed.size else float("nan")
        ),
        "target_mean_when_unobserved": (
            float(unobserved.mean()) if unobserved.size else float("nan")
        ),
    }
    if observed.size and unobserved.size:
        detail["target_mean_difference"] = (
            detail["target_mean_when_observed"] - detail["target_mean_when_unobserved"]
        )
    else:
        detail["target_mean_difference"] = float("nan")

    detail["mann_whitney_p"] = None
    if observed.size >= 10 and unobserved.size >= 10:
        try:
            from scipy.stats import mannwhitneyu

            detail["mann_whitney_p"] = float(
                mannwhitneyu(
                    observed.to_numpy(), unobserved.to_numpy(), alternative="two-sided"
                ).pvalue
            )
        except ImportError:  # pragma: no cover - scipy is a declared dependency
            pass
    return detail


def _summary(
    frame: pd.DataFrame,
    base: pd.DataFrame,
    requested: tuple[str, ...],
    source_dates_outside_calendar: dict[str, int],
    rules: FusionRules,
) -> dict:
    date = rules.date_column
    rows = len(frame)
    per_source: dict[str, dict] = {}

    for source in SOURCE_GROUPS:
        mask = frame[f"{source}_observed"]
        ages = frame[f"{source}_observation_age_days"]
        observed_dates = frame.loc[mask, date]
        detail = {
            "availability_class": get_group(source).availability.value,
            "operational_eligible": get_group(source).operational_eligible,
            "requested": source in requested,
            "value_columns": list(value_columns(source)),
            "support_columns": list(support_columns(source)),
            "fused_rows_observed": int(mask.sum()),
            "fused_rows_observed_fraction": float(mask.mean()) if rows else 0.0,
            "first_observed_date": (
                observed_dates.min().date().isoformat()
                if not observed_dates.empty
                else None
            ),
            "last_observed_date": (
                observed_dates.max().date().isoformat()
                if not observed_dates.empty
                else None
            ),
            "median_observation_age_days": (
                float(ages.median()) if ages.notna().any() else float("nan")
            ),
            "max_observation_age_days": (
                float(ages.max()) if ages.notna().any() else float("nan")
            ),
            "source_dates_outside_ground_calendar": int(
                source_dates_outside_calendar.get(source, 0)
            ),
        }
        detail.update(_target_missingness(frame, mask, rules.target))
        per_source[source] = detail

    complete = frame["all_sources_observed"]
    any_source = frame["n_sources_observed"] > 0
    return {
        "rows": int(rows),
        "date_range": (
            [
                frame[date].min().date().isoformat(),
                frame[date].max().date().isoformat(),
            ]
            if rows
            else [None, None]
        ),
        "requested_sources": list(requested),
        "missing_value_policy": MISSING_VALUE_POLICY,
        "per_source": per_source,
        "rows_with_all_sources_observed": int(complete.sum()),
        "fraction_with_all_sources_observed": (float(complete.mean()) if rows else 0.0),
        "rows_with_any_source_observed": int(any_source.sum()),
        "rows_with_no_source_observed": int((~any_source).sum()),
        "ground_rows": int(len(base)),
        "ground_calendar_preserved": bool(
            len(base) == rows and set(base[date]) == set(frame[date])
        ),
    }


def _check(checks: list[dict], name: str, passed: bool, detail: str) -> None:
    checks.append({"check": name, "passed": bool(passed), "detail": detail})


def validate_fused_dataset(
    dataset: FusedDataset,
    daily: pd.DataFrame | None = None,
    station_met: pd.DataFrame | None = None,
    merra: pd.DataFrame | None = None,
    aod: pd.DataFrame | None = None,
) -> dict:
    """Audit a fused frame for alignment and availability integrity.

    The checks are the Phase 6 gate: a fused dataset that fails any of them
    must not be used for a Track B experiment.

    Returns
    -------
    dict
        ``{"passed": bool, "checks": [...], "n_failed": int}``.
    """
    rules = dataset.rules
    date = rules.date_column
    frame = dataset.frame
    checks: list[dict] = []

    # --- calendar integrity ------------------------------------------------
    _check(
        checks,
        "dates_are_unique",
        not bool(frame[date].duplicated().any()),
        f"{int(frame[date].duplicated().sum())} duplicate dates",
    )
    _check(
        checks,
        "dates_are_chronological",
        bool(frame[date].is_monotonic_increasing),
        "the fused frame is sorted by date",
    )

    if daily is not None:
        ground_dates = set(_normalise_dates(daily[date]))
        fused_dates = set(frame[date])
        _check(
            checks,
            "row_count_matches_ground_series",
            len(frame) == len(daily),
            f"{len(frame)} fused rows against {len(daily)} ground rows",
        )
        _check(
            checks,
            "ground_calendar_preserved",
            fused_dates == ground_dates,
            f"{len(fused_dates - ground_dates)} extended dates, "
            f"{len(ground_dates - fused_dates)} dropped dates",
        )

        if rules.target in daily.columns and rules.target in frame.columns:
            joined = frame[[date, rules.target]].merge(
                daily[[date, rules.target]],
                on=date,
                how="left",
                suffixes=("", "_ground"),
            )
            close = np.isclose(
                joined[rules.target].to_numpy(dtype=float),
                joined[f"{rules.target}_ground"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-9,
                equal_nan=True,
            )
            _check(
                checks,
                "target_preserved",
                bool(close.all()),
                f"{int((~close).sum())} target values changed by the fusion",
            )

    # --- mask consistency --------------------------------------------------
    for source in SOURCE_GROUPS:
        columns = list(value_columns(source))
        observed = frame[f"{source}_observed"].to_numpy(dtype=bool)
        has_value = frame[columns].notna().any(axis=1).to_numpy(dtype=bool)
        if source == "aod":
            has_value = has_value & (
                frame["aod_valid_pixels"].to_numpy(dtype=float) > 0
            )

        _check(
            checks,
            f"{source}:value_without_observation_flag",
            not bool((has_value & ~observed).any()),
            f"{int((has_value & ~observed).sum())} rows carry a value "
            "without being flagged observed",
        )
        _check(
            checks,
            f"{source}:observation_flag_without_value",
            not bool((observed & ~has_value).any()),
            f"{int((observed & ~has_value).sum())} rows are flagged "
            "observed without a value",
        )

        age = frame[f"{source}_observation_age_days"]
        age_zero = age.eq(0).to_numpy()
        age_missing = age.isna().to_numpy()
        consistent = bool(
            (observed & age_zero).sum() == observed.sum()
            and not (age_zero & ~observed).any()
            and not (observed & age_missing).any()
        )
        _check(
            checks,
            f"{source}:observation_age_is_consistent",
            consistent,
            "the observation age is 0 exactly on observed rows and is "
            "never used to supply a value",
        )

    # --- AOD support metadata ---------------------------------------------
    pixels = frame["aod_valid_pixels"].to_numpy(dtype=float)
    overpasses = frame["aod_overpasses"].to_numpy(dtype=float)
    _check(
        checks,
        "aod:support_counts_are_non_negative_integers",
        bool(
            np.isfinite(pixels).all()
            and np.isfinite(overpasses).all()
            and (pixels >= 0).all()
            and (overpasses >= 0).all()
            and np.allclose(pixels, np.round(pixels))
            and np.allclose(overpasses, np.round(overpasses))
        ),
        "pixel and overpass counts are non-negative integers",
    )
    _check(
        checks,
        "aod:value_requires_a_valid_pixel",
        not bool((frame["AOD"].notna().to_numpy() & (pixels < 1)).any()),
        "every AOD value is supported by at least one valid pixel",
    )

    # --- same-day provenance ----------------------------------------------
    supplied = {"station_met": station_met, "merra": merra, "aod": aod}
    for source in SOURCE_GROUPS:
        prepared = _prepare_source(source, supplied[source], rules)
        if prepared is None:
            continue

        merged = frame[[date, f"{source}_observed", *_all_columns(source)]].merge(
            prepared,
            on=date,
            how="left",
            suffixes=("", "_source"),
            validate="one_to_one",
        )
        for column in _all_columns(source):
            left = merged[column].to_numpy(dtype=float)
            right = merged[f"{column}_source"].to_numpy(dtype=float)
            if column in support_columns(source):
                # Support counts are metadata: they are 0 when the source has
                # no row for the date, and must match exactly where it does.
                has_row = ~np.isnan(right)
                same = np.isclose(
                    left[has_row],
                    right[has_row],
                    rtol=0.0,
                    atol=1e-9,
                    equal_nan=True,
                )
                zero_where_absent = bool((left[~has_row] == 0).all())
                passed = bool(same.all()) and zero_where_absent
                detail = (
                    f"{int((~same).sum())} source rows differ; "
                    f"{int((left[~has_row] != 0).sum())} absent rows carry "
                    "non-zero support"
                )
            else:
                same = np.isclose(left, right, rtol=0.0, atol=1e-9, equal_nan=True)
                passed = bool(same.all())
                detail = (
                    f"{int((~same).sum())} rows differ from the source "
                    "record on the same date"
                )
            _check(
                checks,
                f"{source}:same_day_provenance:{column}",
                passed,
                detail,
            )

    failed = [entry["check"] for entry in checks if not entry["passed"]]
    return {
        "passed": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }


def assert_fusion_alignment(validation: dict) -> None:
    """Raise if a fused dataset failed any alignment check.

    Raises
    ------
    ValueError
        Listing every failed check.
    """
    if validation.get("passed"):
        return None
    failed = validation.get("failed_checks", [])
    raise ValueError(
        "fused dataset failed the Phase 6 alignment gate: " + ", ".join(failed)
    )
