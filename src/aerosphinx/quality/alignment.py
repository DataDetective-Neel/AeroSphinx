"""Daily multisource alignment artefacts.

The fused frame already carries per-source ``<source>_observed`` flags. This
module turns those flags into the named artefacts the multisource
specification asks for:

* **source coverage** - how many ground-calendar rows each source observes,
  with the first and last observed date;
* **source missingness** - the informative-missingness contrast: the target
  mean on observed versus unobserved dates;
* **alignment report** - row counts, the distribution of how many sources are
  observed per row, and the all-source overlap per year.

Everything is computed from a frame that is already on the ground calendar, so
no source can extend the calendar. Nothing is filled, imputed or invented.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "DEFAULT_SOURCES",
    "DailyAlignmentReport",
    "source_coverage",
    "source_missingness",
    "source_overlap",
    "build_alignment_report",
    "validate_alignment_report",
]

#: The environmental sources fused onto the ground series.
DEFAULT_SOURCES: tuple[str, ...] = ("station_met", "merra", "aod")


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce an observation flag column to bool.

    Handles the CSV round-trip, where a boolean column can come back as the
    strings ``"True"``/``"False"``.
    """
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(bool)
    # pandas 3.x can return a dedicated ``str`` dtype from ``Series.map``, so
    # both object and any string dtype must be handled. Checking only
    # ``object`` would send "False" strings through ``astype(bool)``, which is
    # true for every non-empty string.
    if pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(
        series.dtype
    ):
        normalised = series.astype(str).str.strip().str.lower()
        truthy = normalised.isin({"true", "1", "yes", "y", "t"})
        falsy = normalised.isin({"false", "0", "no", "n", "f"})
        if bool((truthy | falsy).all()):
            return truthy
    return series.astype(bool)


def _require_columns(frame: pd.DataFrame, sources, date_column: str) -> None:
    if date_column not in frame.columns:
        raise KeyError(f"frame is missing the date column {date_column!r}")
    for source in sources:
        flag = f"{source}_observed"
        if flag not in frame.columns:
            raise KeyError(
                f"frame is missing the availability flag {flag!r} for source "
                f"{source!r}"
            )


def source_coverage(
    frame: pd.DataFrame,
    sources=DEFAULT_SOURCES,
    date_column: str = "Date",
) -> pd.DataFrame:
    """Observed and missing row counts per source.

    Returns
    -------
    pandas.DataFrame
        One row per source with ``observed_rows``, ``missing_rows``,
        ``observed_fraction``, the first/last observed date and the years in
        which the source is observed.
    """
    sources = tuple(sources)
    _require_columns(frame, sources, date_column)
    dates = pd.to_datetime(frame[date_column])
    rows: list[dict] = []

    for source in sources:
        observed = _as_bool(frame[f"{source}_observed"])
        observed_dates = dates[observed]
        years = sorted(int(year) for year in observed_dates.dt.year.unique())
        rows.append(
            {
                "source": source,
                "observed_rows": int(observed.sum()),
                "missing_rows": int((~observed).sum()),
                "observed_fraction": (float(observed.mean()) if len(frame) else 0.0),
                "first_observed": (
                    observed_dates.min().date().isoformat()
                    if not observed_dates.empty
                    else None
                ),
                "last_observed": (
                    observed_dates.max().date().isoformat()
                    if not observed_dates.empty
                    else None
                ),
                "observed_years": ",".join(str(year) for year in years),
            }
        )

    return pd.DataFrame(rows)


def source_missingness(
    frame: pd.DataFrame,
    sources=DEFAULT_SOURCES,
    date_column: str = "Date",
    target: str = "PM25",
) -> pd.DataFrame:
    """Target mean on observed versus unobserved dates, per source.

    The Phase 4 analysis found that AOD is least available when pollution is
    worst. Reporting the same contrast for every source keeps informative
    missingness visible before any experiment.
    """
    sources = tuple(sources)
    _require_columns(frame, sources, date_column)
    values = (
        pd.to_numeric(frame[target], errors="coerce")
        if target in frame.columns
        else None
    )

    rows: list[dict] = []
    for source in sources:
        observed = _as_bool(frame[f"{source}_observed"])
        record = {
            "source": source,
            "n_observed": int(observed.sum()),
            "n_unobserved": int((~observed).sum()),
            "target_mean_when_observed": float("nan"),
            "target_mean_when_unobserved": float("nan"),
            "target_mean_difference": float("nan"),
        }
        if values is not None:
            on = values[observed.to_numpy() & values.notna().to_numpy()]
            off = values[(~observed).to_numpy() & values.notna().to_numpy()]
            record["target_mean_when_observed"] = (
                float(on.mean()) if len(on) else float("nan")
            )
            record["target_mean_when_unobserved"] = (
                float(off.mean()) if len(off) else float("nan")
            )
            record["target_mean_difference"] = (
                record["target_mean_when_observed"]
                - record["target_mean_when_unobserved"]
            )
        rows.append(record)

    return pd.DataFrame(rows)


def source_overlap(
    frame: pd.DataFrame,
    sources=DEFAULT_SOURCES,
    date_column: str = "Date",
) -> dict:
    """How many rows carry each number of simultaneously observed sources."""
    sources = tuple(sources)
    _require_columns(frame, sources, date_column)
    masks = pd.concat(
        [_as_bool(frame[f"{source}_observed"]).rename(source) for source in sources],
        axis=1,
    )
    counts = masks.sum(axis=1)
    total = int(len(frame))
    all_mask = counts.eq(len(sources))

    years = pd.to_datetime(frame[date_column]).dt.year
    per_year: dict[str, dict] = {}
    for year in sorted(int(year) for year in years.unique()):
        selected = years.eq(year)
        rows = int(selected.sum())
        all_rows = int((all_mask & selected).sum())
        per_year[str(year)] = {
            "rows": rows,
            "all_sources_observed": all_rows,
            "fraction_all_sources": (all_rows / rows) if rows else 0.0,
        }

    return {
        "n_sources": len(sources),
        "rows": total,
        "n_sources_distribution": {
            str(count): int(counts.eq(count).sum()) for count in range(len(sources) + 1)
        },
        "rows_with_all_sources_observed": int(all_mask.sum()),
        "fraction_with_all_sources_observed": (
            float(all_mask.mean()) if total else 0.0
        ),
        "rows_with_no_source_observed": int(counts.eq(0).sum()),
        "per_year": per_year,
    }


@dataclass
class DailyAlignmentReport:
    """The three named alignment artefacts plus their source list."""

    source_coverage: pd.DataFrame
    source_missingness: pd.DataFrame
    report: dict
    sources: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "sources": list(self.sources),
            "coverage": self.source_coverage.to_dict(orient="records"),
            "missingness": self.source_missingness.to_dict(orient="records"),
            **self.report,
        }


def build_alignment_report(
    frame: pd.DataFrame,
    sources=DEFAULT_SOURCES,
    date_column: str = "Date",
    target: str = "PM25",
) -> DailyAlignmentReport:
    """Build the coverage, missingness and overlap artefacts in one call."""
    sources = tuple(sources)
    coverage = source_coverage(frame, sources, date_column)
    missingness = source_missingness(frame, sources, date_column, target)
    overlap = source_overlap(frame, sources, date_column)

    dates = pd.to_datetime(frame[date_column])
    report = {
        "rows": int(len(frame)),
        "date_range": [
            dates.min().date().isoformat() if len(dates) else None,
            dates.max().date().isoformat() if len(dates) else None,
        ],
        **overlap,
    }
    return DailyAlignmentReport(
        source_coverage=coverage,
        source_missingness=missingness,
        report=report,
        sources=sources,
    )


def validate_alignment_report(report: DailyAlignmentReport) -> dict:
    """Audit the alignment artefacts for internal consistency.

    Checks each source's observed and missing counts sum to the row count and
    that every fraction lies in [0, 1].
    """
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    rows = int(report.report.get("rows", 0))
    for _, record in report.source_coverage.iterrows():
        source = record["source"]
        total = int(record["observed_rows"]) + int(record["missing_rows"])
        check(
            f"{source}:counts_sum_to_rows",
            total == rows,
            f"{record['observed_rows']} observed + {record['missing_rows']} "
            f"missing against {rows} rows",
        )
        fraction = float(record["observed_fraction"])
        check(
            f"{source}:fraction_in_unit_interval",
            0.0 <= fraction <= 1.0,
            f"observed fraction {fraction}",
        )

    for _, record in report.source_missingness.iterrows():
        source = record["source"]
        total = int(record["n_observed"]) + int(record["n_unobserved"])
        check(
            f"{source}:missingness_counts_sum_to_rows",
            total == rows,
            f"{record['n_observed']} observed + {record['n_unobserved']} "
            f"unobserved against {rows} rows",
        )

    failed = [entry["check"] for entry in checks if not entry["passed"]]
    return {
        "passed": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "failed_checks": failed,
        "checks": checks,
    }
