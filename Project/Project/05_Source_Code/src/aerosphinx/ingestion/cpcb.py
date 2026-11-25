"""CPCB ground-observation ingestion for AeroSphinx Track A.

This module is the Phase 1 replacement for the CPCB loading logic that
previously lived inside ``aerosphinx_pipeline.load_cpcb_daily``. It corrects
two defects recorded in ``docs/reference/DEFECT_REGISTER.md``:

* **D-013** - the legacy loader averaged *file-level daily means*, which is
  only correct when every file contributes the same number of hours to a
  day. This loader aggregates at the hourly level across all files and then
  computes one hourly-weighted daily mean per date.
* **D-001 (ingestion half)** - the loader records, for every day, which
  calendar date each contributing observation came from, so downstream lag
  construction can be calendar-aware rather than row-based.

Design rules
------------
1. Raw files are opened read-only. Nothing in ``data/`` is ever written.
2. Every rejected record is counted and attributed to a reason.
3. The returned daily frame carries a complete calendar so that a missing
   day is represented as an absent row rather than a silently compressed
   one.
4. No imputation, no interpolation, no forward filling. A day that fails a
   quality rule is absent from the output and counted in the audit.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd

__all__ = [
    "CPCBRules",
    "CPCBAudit",
    "CPCBDaily",
    "load_cpcb_daily",
    "read_cpcb_hourly",
    "aggregate_daily",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CPCBRules:
    """Data-quality rules applied during CPCB ingestion.

    Defaults mirror ``config/data_quality.yaml``. Every rule is explicit so
    that the audit output can attribute each rejected record to a cause.
    """

    pm25_valid_min: float = 0.0
    pm25_valid_max: float = 500.0
    min_hours_per_day: int = 18
    expected_hours_per_day: int = 24
    duplicate_hour_policy: str = "exclude_day"
    require_whole_hour: bool = True
    timestamp_column: str = "Timestamp"
    pm25_column_candidates: tuple[str, ...] = (
        "PM2.5 (µg/m³)",
        "PM2.5 (ug/m3)",
        "PM25",
    )

    def __post_init__(self) -> None:
        if self.pm25_valid_min >= self.pm25_valid_max:
            raise ValueError("pm25_valid_min must be below pm25_valid_max")
        if self.min_hours_per_day < 1:
            raise ValueError("min_hours_per_day must be at least 1")
        if self.min_hours_per_day > self.expected_hours_per_day:
            raise ValueError("min_hours_per_day cannot exceed expected_hours_per_day")
        if self.duplicate_hour_policy not in {"exclude_day"}:
            raise ValueError(
                "duplicate_hour_policy must be 'exclude_day'; silently "
                "de-duplicating would fabricate an observation"
            )


# --------------------------------------------------------------------------
# Audit records
# --------------------------------------------------------------------------


@dataclass
class CPCBAudit:
    """Counts and reasons for everything the loader rejected."""

    files_read: int = 0
    hourly_rows_read: int = 0
    hourly_rows_used: int = 0
    rows_unparseable_timestamp: int = 0
    rows_non_numeric_pm25: int = 0
    rows_out_of_range: int = 0
    rows_off_grid_hour: int = 0
    duplicate_hours: int = 0
    days_excluded_duplicate: int = 0
    days_excluded_insufficient_hours: int = 0
    days_excluded_all_missing: int = 0
    # Calendar span of the *raw* record, including days whose every hour
    # was rejected. Without this, such days fall outside the span used for
    # reconciliation and are never counted.
    raw_start: pd.Timestamp | None = None
    raw_end: pd.Timestamp | None = None
    file_hashes: dict[str, str] = field(default_factory=dict)
    rejection_reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str, count: int = 1) -> None:
        self.rejection_reasons[reason] = self.rejection_reasons.get(reason, 0) + count

    def as_dict(self) -> dict:
        return {
            "files_read": self.files_read,
            "hourly_rows_read": self.hourly_rows_read,
            "hourly_rows_used": self.hourly_rows_used,
            "rejections": dict(sorted(self.rejection_reasons.items())),
            "days_excluded_duplicate": self.days_excluded_duplicate,
            "days_excluded_insufficient_hours": self.days_excluded_insufficient_hours,
            "days_excluded_all_missing": self.days_excluded_all_missing,
            "raw_start": (
                self.raw_start.date().isoformat()
                if self.raw_start is not None
                else None
            ),
            "raw_end": (
                self.raw_end.date().isoformat() if self.raw_end is not None else None
            ),
            "file_sha256": dict(sorted(self.file_hashes.items())),
        }


@dataclass
class CPCBDaily:
    """The validated CPCB daily series plus its audit trail."""

    frame: pd.DataFrame
    audit: CPCBAudit
    rejected_records: pd.DataFrame

    @property
    def n_days(self) -> int:
        return len(self.frame)

    @property
    def start(self) -> pd.Timestamp:
        return self.frame["Date"].min()

    @property
    def end(self) -> pd.Timestamp:
        return self.frame["Date"].max()


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_pm25_column(columns: Iterable[str], rules: CPCBRules) -> str:
    """Locate the PM2.5 column by name, tolerating encoding differences.

    The CPCB export has been seen with several spellings of the same header
    (``PM2.5 (µg/m³)``, ``PM2.5 (ug/m3)``, ``PM25``). Position is never
    assumed.
    """
    columns = list(columns)
    for candidate in rules.pm25_column_candidates:
        if candidate in columns:
            return candidate
    for column in columns:
        normalised = column.strip().upper().replace(" ", "")
        if "PM2.5" in column or normalised.startswith("PM25"):
            return column
    raise ValueError(f"No PM2.5 column found among: {columns}")


def _resolve_timestamp_column(columns: Iterable[str], rules: CPCBRules) -> str:
    columns = list(columns)
    if rules.timestamp_column in columns:
        return rules.timestamp_column
    for column in columns:
        if column.strip().lower() == rules.timestamp_column.lower():
            return column
    raise ValueError(f"No timestamp column found among: {columns}")


# --------------------------------------------------------------------------
# Stage 1 - read raw hourly records
# --------------------------------------------------------------------------


def read_cpcb_hourly(
    cpcb_dir: Path, rules: CPCBRules | None = None
) -> tuple[pd.DataFrame, CPCBAudit, pd.DataFrame]:
    """Read every CPCB CSV into one hourly frame, rejecting invalid records.

    Returns ``(hourly, audit, rejected)`` where ``hourly`` holds only
    usable observations and ``rejected`` holds every discarded record with
    the reason it was discarded.
    """
    rules = rules or CPCBRules()
    audit = CPCBAudit()
    rejected: list[dict] = []

    files = sorted(Path(cpcb_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CPCB CSV files found in {cpcb_dir}")

    frames: list[pd.DataFrame] = []
    for path in files:
        raw = pd.read_csv(path)
        ts_col = _resolve_timestamp_column(raw.columns, rules)
        pm_col = _find_pm25_column(raw.columns, rules)

        audit.files_read += 1
        audit.file_hashes[path.name] = _sha256(path)
        audit.hourly_rows_read += len(raw)

        timestamps = pd.to_datetime(raw[ts_col], errors="coerce")
        values = pd.to_numeric(raw[pm_col], errors="coerce")

        # Track the calendar span of the *raw* record, including rows that
        # are about to be rejected.
        raw_dates = timestamps.dropna()
        if not raw_dates.empty:
            file_start = raw_dates.min().normalize()
            file_end = raw_dates.max().normalize()
            if audit.raw_start is None or file_start < audit.raw_start:
                audit.raw_start = file_start
            if audit.raw_end is None or file_end > audit.raw_end:
                audit.raw_end = file_end

        # Unparseable timestamps: the record cannot be placed on a calendar.
        unparseable = timestamps.isna()
        if unparseable.any():
            count = int(unparseable.sum())
            audit.rows_unparseable_timestamp += count
            audit.reject("unparseable_timestamp", count)
            for index in raw.index[unparseable]:
                rejected.append(
                    {
                        "file": path.name,
                        "row_index": int(index),
                        "timestamp": str(raw.at[index, ts_col]),
                        "pm25": raw.at[index, pm_col],
                        "reason": "unparseable_timestamp",
                    }
                )

        # Values that are present but not numeric.
        non_numeric = values.isna() & ~unparseable
        if non_numeric.any():
            count = int(non_numeric.sum())
            audit.rows_non_numeric_pm25 += count
            audit.reject("non_numeric_pm25", count)
            for index in raw.index[non_numeric]:
                rejected.append(
                    {
                        "file": path.name,
                        "row_index": int(index),
                        "timestamp": str(raw.at[index, ts_col]),
                        "pm25": raw.at[index, pm_col],
                        "reason": "non_numeric_pm25",
                    }
                )

        # Values outside the physically valid window: set to missing, never
        # clipped. A clipped 800 ug/m3 reading would bias the daily mean.
        out_of_range = values.notna() & (
            (values < rules.pm25_valid_min) | (values > rules.pm25_valid_max)
        )
        if out_of_range.any():
            count = int(out_of_range.sum())
            audit.rows_out_of_range += count
            audit.reject("pm25_out_of_range", count)
            for index in raw.index[out_of_range]:
                rejected.append(
                    {
                        "file": path.name,
                        "row_index": int(index),
                        "timestamp": str(raw.at[index, ts_col]),
                        "pm25": float(values.at[index]),
                        "reason": "pm25_out_of_range",
                    }
                )

        # Off-grid timestamps (not on an exact hour).
        off_grid = timestamps.notna() & (
            (timestamps.dt.minute != 0) | (timestamps.dt.second != 0)
        )
        if off_grid.any():
            count = int(off_grid.sum())
            audit.rows_off_grid_hour += count
            if rules.require_whole_hour:
                audit.reject("off_grid_timestamp", count)
                for index in raw.index[off_grid]:
                    rejected.append(
                        {
                            "file": path.name,
                            "row_index": int(index),
                            "timestamp": str(raw.at[index, ts_col]),
                            "pm25": values.at[index],
                            "reason": "off_grid_timestamp",
                        }
                    )
                off_grid = pd.Series(True, index=raw.index)

        keep = ~(unparseable | non_numeric | out_of_range)
        if rules.require_whole_hour:
            keep = keep & ~off_grid

        frames.append(
            pd.DataFrame(
                {
                    "Timestamp": timestamps[keep],
                    "PM25": values[keep],
                    "source_file": path.name,
                }
            )
        )

    hourly = pd.concat(frames, ignore_index=True)
    hourly = hourly.dropna(subset=["Timestamp"])
    audit.hourly_rows_used = len(hourly)

    rejected_frame = pd.DataFrame(
        rejected,
        columns=["file", "row_index", "timestamp", "pm25", "reason"],
    )
    return hourly, audit, rejected_frame


# --------------------------------------------------------------------------
# Stage 2 - aggregate to daily, hour-weighted
# --------------------------------------------------------------------------


def aggregate_daily(
    hourly: pd.DataFrame,
    rules: CPCBRules | None = None,
    calendar_start: pd.Timestamp | None = None,
    calendar_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, CPCBAudit]:
    """Aggregate unique hourly observations into one daily mean per date.

    The aggregation is **hour-weighted across all files**: every unique
    hourly observation of a date contributes equally to that date's mean.
    This is the correction for D-013; the legacy implementation averaged
    per-file daily means, which silently mis-weights a date whose hours are
    split unevenly between files.

    Days carrying a duplicated hour are excluded entirely rather than
    de-duplicated, because choosing one of two conflicting values for the
    same hour would fabricate an observation.
    """
    rules = rules or CPCBRules()
    audit = CPCBAudit()

    if hourly.empty:
        return (
            pd.DataFrame(
                columns=[
                    "Date",
                    "PM25",
                    "valid_hours",
                    "expected_hours",
                    "coverage_fraction",
                ]
            ),
            audit,
        )

    frame = hourly.copy()
    frame["Date"] = frame["Timestamp"].dt.normalize()

    # Detect duplicated hours across the whole record.
    duplicated = frame.duplicated(subset=["Timestamp"], keep=False)
    n_duplicate = int(frame.loc[duplicated, "Timestamp"].nunique())
    if n_duplicate:
        audit.duplicate_hours = n_duplicate
        audit.reject("duplicate_hour", n_duplicate)
        duplicate_dates = set(frame.loc[duplicated, "Date"])

    # Daily aggregation over unique hours only.
    unique = frame.drop_duplicates(subset=["Timestamp"], keep="first")
    grouped = unique.groupby("Date", sort=True)
    daily = grouped.agg(
        PM25=("PM25", "mean"),
        valid_hours=("PM25", "count"),
        n_observations=("PM25", "size"),
    ).reset_index()

    # Reindex onto the full calendar spanned by the *raw* record so that a
    # day whose every hour was rejected is still represented, and counted.
    # Without this, such days vanish silently from both the output and the
    # audit. ``calendar_start``/``calendar_end`` come from the raw record,
    # not from the surviving rows, so a day rejected at the hourly stage
    # still lies inside the span.
    if calendar_start is None:
        calendar_start = frame["Date"].min()
    if calendar_end is None:
        calendar_end = frame["Date"].max()
    calendar = pd.date_range(
        pd.Timestamp(calendar_start), pd.Timestamp(calendar_end), freq="D"
    )
    daily = daily.set_index("Date").reindex(calendar).rename_axis("Date").reset_index()
    daily["valid_hours"] = daily["valid_hours"].fillna(0).astype(int)
    daily["n_observations"] = daily["n_observations"].fillna(0).astype(int)

    # A day with no valid observation at all.
    all_missing = daily["valid_hours"] == 0
    if all_missing.any():
        count = int(all_missing.sum())
        audit.days_excluded_all_missing = count
        audit.reject("day_no_valid_observations", count)
        daily = daily[~all_missing]

    # Days carrying a duplicated hour are excluded (see docstring).
    if n_duplicate:
        excluded = daily["Date"].isin(duplicate_dates)
        if excluded.any():
            count = int(excluded.sum())
            audit.days_excluded_duplicate = count
            audit.reject("day_contains_duplicate_hour", count)
            daily = daily[~excluded]

    # Days with insufficient hourly coverage.
    insufficient = daily["valid_hours"] < rules.min_hours_per_day
    if insufficient.any():
        count = int(insufficient.sum())
        audit.days_excluded_insufficient_hours = count
        audit.reject("day_below_min_hours", count)
        daily = daily[~insufficient]

    daily["expected_hours"] = rules.expected_hours_per_day
    daily["coverage_fraction"] = daily["valid_hours"] / rules.expected_hours_per_day
    daily = daily.sort_values("Date").reset_index(drop=True)

    return (
        daily[["Date", "PM25", "valid_hours", "expected_hours", "coverage_fraction"]],
        audit,
    )


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def load_cpcb_daily(
    cpcb_dir: Path | str,
    rules: CPCBRules | None = None,
) -> CPCBDaily:
    """Load the CPCB hourly exports into a validated daily series.

    Parameters
    ----------
    cpcb_dir:
        Directory of CPCB CSV exports. Read-only.
    rules:
        Quality rules; defaults to :class:`CPCBRules`.

    Returns
    -------
    CPCBDaily
        The daily frame, the audit counts, and the rejected records.
    """
    rules = rules or CPCBRules()
    cpcb_dir = Path(cpcb_dir)

    hourly, audit, rejected = read_cpcb_hourly(cpcb_dir, rules)
    daily, aggregate_audit = aggregate_daily(
        hourly,
        rules,
        calendar_start=audit.raw_start,
        calendar_end=audit.raw_end,
    )

    # Merge the two audit halves.
    for key, value in aggregate_audit.rejection_reasons.items():
        audit.reject(key, value)
    audit.days_excluded_duplicate = aggregate_audit.days_excluded_duplicate
    audit.days_excluded_insufficient_hours = (
        aggregate_audit.days_excluded_insufficient_hours
    )
    audit.days_excluded_all_missing = aggregate_audit.days_excluded_all_missing
    audit.duplicate_hours = aggregate_audit.duplicate_hours

    return CPCBDaily(frame=daily, audit=audit, rejected_records=rejected)
