"""Per-season, per-source data sufficiency.

A seasonal specialist or a source-ablation comparison is only meaningful when
the season has enough observed rows to estimate and to compare. This module
assigns every ``(season, source)`` pair one of four statuses and records the
reason, so an experiment can refuse to train rather than fit noise.

Criteria (declared, not hidden)
-------------------------------
* ``NOT_EVALUABLE`` - the season has no ground rows at all, or the source is
  observed on zero rows in that season. There is nothing to compare.
* ``INSUFFICIENT`` - the source is observed on fewer than
  ``min_rows_for_inference`` rows. The default of **30** is the paired
  comparison threshold already documented in ``config/fusion.yaml``
  (``min_matched_rows_for_inference``); below it a comparison is not attempted.
* ``WARNING`` - at least 30 rows but fewer than ``min_rows_for_sufficient``.
  Enough for a comparison, not enough to treat a season-specific estimate as
  stable.
* ``SUFFICIENT`` - at least ``min_rows_for_sufficient`` observed rows. The
  default of **90** is one full meteorological season, the smallest span over
  which a season-specific estimate is not a single-quarter artefact.

These thresholds are declared here and echoed in every output report; they are
not tuned on evaluation data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aerosphinx.features.seasonal import SEASONS, assign_season

__all__ = [
    "SUFFICIENCY_STATUSES",
    "SufficiencyCriteria",
    "classify_sufficiency",
    "season_source_sufficiency",
    "summarise_sufficiency",
]

#: The four sufficiency statuses, strongest first.
SUFFICIENCY_STATUSES: tuple[str, ...] = (
    "SUFFICIENT",
    "WARNING",
    "INSUFFICIENT",
    "NOT_EVALUABLE",
)


@dataclass(frozen=True)
class SufficiencyCriteria:
    """Thresholds for a sufficiency call.

    Attributes
    ----------
    min_rows_for_inference:
        Minimum observed rows before a comparison is attempted. Defaults to
        the documented 30-row paired-comparison threshold.
    min_rows_for_sufficient:
        Observed rows needed for a stable season-specific estimate. Defaults
        to one full season (90 days).
    """

    min_rows_for_inference: int = 30
    min_rows_for_sufficient: int = 90

    def __post_init__(self) -> None:
        if self.min_rows_for_inference < 1:
            raise ValueError("min_rows_for_inference must be at least 1")
        if self.min_rows_for_sufficient < self.min_rows_for_inference:
            raise ValueError(
                "min_rows_for_sufficient cannot be below min_rows_for_inference"
            )

    def as_dict(self) -> dict:
        return {
            "min_rows_for_inference": self.min_rows_for_inference,
            "min_rows_for_sufficient": self.min_rows_for_sufficient,
            "min_rows_for_inference_rationale": (
                "config/fusion.yaml min_matched_rows_for_inference; below it a "
                "paired comparison is not attempted"
            ),
            "min_rows_for_sufficient_rationale": (
                "one full meteorological season (90 days); the smallest span "
                "over which a season-specific estimate is not a "
                "single-quarter artefact"
            ),
        }


def classify_sufficiency(
    n_observed: int,
    n_season_rows: int,
    criteria: SufficiencyCriteria | None = None,
) -> tuple[str, str]:
    """Return ``(status, reason)`` for one season/source cell."""
    criteria = criteria or SufficiencyCriteria()
    if n_season_rows <= 0:
        return "NOT_EVALUABLE", "the season has no ground rows"
    if n_observed <= 0:
        return "NOT_EVALUABLE", "the source is never observed in this season"
    if n_observed < criteria.min_rows_for_inference:
        return (
            "INSUFFICIENT",
            f"{n_observed} observed rows is below the "
            f"{criteria.min_rows_for_inference}-row inference threshold",
        )
    if n_observed < criteria.min_rows_for_sufficient:
        return (
            "WARNING",
            f"{n_observed} observed rows supports a comparison but is below "
            f"the {criteria.min_rows_for_sufficient}-row stability threshold",
        )
    return (
        "SUFFICIENT",
        f"{n_observed} observed rows meets the "
        f"{criteria.min_rows_for_sufficient}-row stability threshold",
    )


def season_source_sufficiency(
    frame: pd.DataFrame,
    sources=("station_met", "merra", "aod"),
    date_column: str = "Date",
    criteria: SufficiencyCriteria | None = None,
    target: str = "PM25",
) -> pd.DataFrame:
    """One row per ``(season, source)`` with status and supporting counts.

    Every season is reported, including seasons with no rows, so a missing
    season is visible rather than silently absent.
    """
    criteria = criteria or SufficiencyCriteria()
    sources = tuple(sources)

    if date_column not in frame.columns:
        raise KeyError(f"frame is missing the date column {date_column!r}")
    for source in sources:
        if f"{source}_observed" not in frame.columns:
            raise KeyError(
                f"frame is missing the availability flag "
                f"'{source}_observed' for source {source!r}"
            )

    working = frame.copy()
    working["_season"] = assign_season(working[date_column]).to_numpy()
    dates = pd.to_datetime(working[date_column])

    rows: list[dict] = []
    for season in SEASONS:
        in_season = working["_season"].eq(season)
        subset = working[in_season]
        season_dates = dates[in_season]
        n_season = int(len(subset))

        for source in sources:
            flag = subset[f"{source}_observed"].astype(bool)
            n_observed = int(flag.sum())
            status, reason = classify_sufficiency(n_observed, n_season, criteria)
            observed_dates = season_dates[flag]
            record = {
                "season": season,
                "source": source,
                "n_season_rows": n_season,
                "n_observed": n_observed,
                "observed_fraction": (n_observed / n_season if n_season else 0.0),
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
                "status": status,
                "reason": reason,
            }
            if target in subset.columns:
                values = pd.to_numeric(subset[target], errors="coerce")
                on = values[flag.to_numpy() & values.notna().to_numpy()]
                record["target_mean_when_observed"] = (
                    float(on.mean()) if len(on) else float("nan")
                )
            rows.append(record)

    return pd.DataFrame(rows)


def summarise_sufficiency(
    table: pd.DataFrame,
    criteria: SufficiencyCriteria | None = None,
) -> dict:
    """Summarise a sufficiency table by status and by season."""
    criteria = criteria or SufficiencyCriteria()
    if table.empty:
        return {
            "criteria": criteria.as_dict(),
            "counts": {status: 0 for status in SUFFICIENCY_STATUSES},
            "by_season": {},
            "not_evaluable": [],
            "insufficient": [],
        }

    counts = {
        status: int(table["status"].eq(status).sum()) for status in SUFFICIENCY_STATUSES
    }
    by_season: dict[str, dict] = {}
    for season, group in table.groupby("season", sort=False):
        by_season[season] = {
            status: int(group["status"].eq(status).sum())
            for status in SUFFICIENCY_STATUSES
        }

    return {
        "criteria": criteria.as_dict(),
        "counts": counts,
        "by_season": by_season,
        "not_evaluable": table.loc[
            table["status"].eq("NOT_EVALUABLE"), ["season", "source"]
        ].to_dict(orient="records"),
        "insufficient": table.loc[
            table["status"].eq("INSUFFICIENT"), ["season", "source"]
        ].to_dict(orient="records"),
    }
