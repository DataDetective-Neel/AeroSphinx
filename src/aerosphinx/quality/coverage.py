"""Coverage analysis for the validated daily series.

Coverage is treated as a first-class measurement rather than a diagnostic
sideline. The Phase 0 audit found that 27 of 169 INSAT granules contribute
zero valid pixels (D-004) and that 123 calendar gaps exist in the CPCB
record (D-001) - both are coverage facts that a downstream model must be
able to see.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["CoverageReport", "compute_coverage", "detect_calendar_gaps"]


@dataclass
class CoverageReport:
    """Summary of temporal coverage for a daily series."""

    n_days: int
    start: pd.Timestamp
    end: pd.Timestamp
    calendar_span_days: int
    missing_days: int
    coverage_fraction: float
    gaps: list[dict] = field(default_factory=list)
    per_year: dict[str, dict] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_days": self.n_days,
            "start": self.start.date().isoformat(),
            "end": self.end.date().isoformat(),
            "calendar_span_days": self.calendar_span_days,
            "missing_days": self.missing_days,
            "coverage_fraction": round(self.coverage_fraction, 6),
            "n_gaps": len(self.gaps),
            "longest_gap_days": max((g["gap_days"] for g in self.gaps), default=0),
            "per_year": self.per_year,
        }


def detect_calendar_gaps(frame: pd.DataFrame, date_column: str = "Date") -> list[dict]:
    """Find every gap between consecutive observed dates.

    A gap is a stretch of calendar days with no observation. Gaps matter
    because a row-based lag silently bridges them, presenting a stale value
    as a recent one.
    """
    if date_column not in frame.columns:
        raise KeyError(f"frame is missing the {date_column!r} column")
    dates = pd.to_datetime(frame[date_column]).sort_values().reset_index(drop=True)
    if dates.empty:
        return []

    deltas = dates.diff().dt.days
    gaps: list[dict] = []
    for position in range(1, len(dates)):
        if deltas.iloc[position] > 1:
            previous = dates.iloc[position - 1]
            current = dates.iloc[position]
            gaps.append(
                {
                    "gap_start": (previous + pd.Timedelta(days=1)).date().isoformat(),
                    "gap_end": (current - pd.Timedelta(days=1)).date().isoformat(),
                    "gap_days": int(deltas.iloc[position] - 1),
                    "preceding_date": previous.date().isoformat(),
                    "following_date": current.date().isoformat(),
                }
            )
    return gaps


def compute_coverage(frame: pd.DataFrame, date_column: str = "Date") -> CoverageReport:
    """Compute temporal coverage statistics for a daily frame."""
    if frame.empty:
        return CoverageReport(
            n_days=0,
            start=pd.Timestamp("1970-01-01"),
            end=pd.Timestamp("1970-01-01"),
            calendar_span_days=0,
            missing_days=0,
            coverage_fraction=0.0,
        )

    dates = pd.to_datetime(frame[date_column]).sort_values().reset_index(drop=True)
    span = int((dates.max() - dates.min()).days) + 1
    gaps = detect_calendar_gaps(frame, date_column)

    per_year: dict[str, dict] = {}
    years = dates.dt.year
    for year in sorted(years.unique()):
        year_dates = dates[years == year]
        year_span = int((year_dates.max() - year_dates.min()).days) + 1
        per_year[str(year)] = {
            "observed_days": int(len(year_dates)),
            "span_days": year_span,
            "coverage_fraction": round(len(year_dates) / year_span, 6),
        }

    return CoverageReport(
        n_days=int(len(dates)),
        start=dates.min(),
        end=dates.max(),
        calendar_span_days=span,
        missing_days=span - int(len(dates)),
        coverage_fraction=len(dates) / span,
        gaps=gaps,
        per_year=per_year,
    )
