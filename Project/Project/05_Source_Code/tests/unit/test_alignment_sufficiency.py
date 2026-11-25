"""Tests for the daily alignment artefacts and per-season sufficiency.

The alignment artefacts must sum to the ground row count and never let a source
extend the calendar. The sufficiency statuses must follow the declared
thresholds and must mark an absent source as NOT_EVALUABLE rather than zero.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aerosphinx.evaluation.sufficiency import (
    SUFFICIENCY_STATUSES,
    SufficiencyCriteria,
    classify_sufficiency,
    season_source_sufficiency,
    summarise_sufficiency,
)
from aerosphinx.quality.alignment import (
    build_alignment_report,
    source_coverage,
    source_missingness,
    source_overlap,
    validate_alignment_report,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    """A 120-row frame with known, different availability per source."""
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    n = len(dates)
    return pd.DataFrame(
        {
            "Date": dates,
            "PM25": 50.0 + pd.Series(range(n), dtype=float),
            # station_met always observed; merra on the first 60; aod on 40.
            "station_met_observed": True,
            "merra_observed": [True] * 60 + [False] * (n - 60),
            "aod_observed": [True] * 40 + [False] * (n - 40),
        }
    )


# --------------------------------------------------------------------------
# Alignment artefacts
# --------------------------------------------------------------------------


def test_source_coverage_counts_are_exact(synthetic_frame):
    coverage = source_coverage(synthetic_frame).set_index("source")
    assert coverage.loc["station_met", "observed_rows"] == 120
    assert coverage.loc["merra", "observed_rows"] == 60
    assert coverage.loc["aod", "observed_rows"] == 40
    assert coverage.loc["aod", "missing_rows"] == 80
    assert coverage.loc["aod", "observed_fraction"] == pytest.approx(40 / 120)


def test_coverage_does_not_extend_the_calendar(synthetic_frame):
    """Every source's first/last observed date lies inside the ground range."""
    coverage = source_coverage(synthetic_frame)
    start = synthetic_frame["Date"].min().date().isoformat()
    end = synthetic_frame["Date"].max().date().isoformat()
    for record in coverage.to_dict(orient="records"):
        assert start <= record["first_observed"] <= end
        assert start <= record["last_observed"] <= end


def test_missingness_reports_the_informative_contrast(synthetic_frame):
    missing = source_missingness(synthetic_frame).set_index("source")
    # PM25 increases over time, so observed (early) dates are less polluted.
    assert missing.loc["merra", "target_mean_when_observed"] < (
        missing.loc["merra", "target_mean_when_unobserved"]
    )
    assert missing.loc["station_met", "n_unobserved"] == 0


def test_overlap_distribution_sums_to_rows(synthetic_frame):
    overlap = source_overlap(synthetic_frame)
    assert sum(overlap["n_sources_distribution"].values()) == 120
    # Rows where all three are observed: first 40.
    assert overlap["rows_with_all_sources_observed"] == 40
    assert overlap["fraction_with_all_sources_observed"] == pytest.approx(40 / 120)


def test_overlap_is_computed_per_year(synthetic_frame):
    overlap = source_overlap(synthetic_frame)
    assert "2024" in overlap["per_year"]
    assert overlap["per_year"]["2024"]["rows"] == 120


def test_alignment_report_validates_cleanly(synthetic_frame):
    report = build_alignment_report(synthetic_frame)
    validation = validate_alignment_report(report)
    assert validation["passed"] is True
    assert validation["n_failed"] == 0


def test_alignment_report_rejects_an_inconsistent_source(synthetic_frame):
    """A hand-built report whose counts do not sum must fail the audit."""
    report = build_alignment_report(synthetic_frame)
    report.source_coverage.loc[0, "observed_rows"] = 999
    validation = validate_alignment_report(report)
    assert validation["passed"] is False


def test_missing_flag_column_raises(synthetic_frame):
    with pytest.raises(KeyError, match="aod_observed"):
        source_coverage(synthetic_frame.drop(columns=["aod_observed"]))


def test_string_boolean_flags_are_parsed(synthetic_frame):
    """A CSV round-trip that yields strings must not be misread."""
    frame = synthetic_frame.copy()
    frame["aod_observed"] = frame["aod_observed"].map({True: "True", False: "False"})
    coverage = source_coverage(frame).set_index("source")
    assert coverage.loc["aod", "observed_rows"] == 40


# --------------------------------------------------------------------------
# Sufficiency
# --------------------------------------------------------------------------


def test_status_thresholds_are_exact():
    criteria = SufficiencyCriteria(
        min_rows_for_inference=30, min_rows_for_sufficient=90
    )
    assert classify_sufficiency(0, 100, criteria)[0] == "NOT_EVALUABLE"
    assert classify_sufficiency(10, 100, criteria)[0] == "INSUFFICIENT"
    assert classify_sufficiency(30, 100, criteria)[0] == "WARNING"
    assert classify_sufficiency(89, 100, criteria)[0] == "WARNING"
    assert classify_sufficiency(90, 100, criteria)[0] == "SUFFICIENT"
    assert classify_sufficiency(50, 0, criteria)[0] == "NOT_EVALUABLE"


def test_criteria_reject_an_inverted_threshold():
    with pytest.raises(ValueError):
        SufficiencyCriteria(min_rows_for_inference=100, min_rows_for_sufficient=10)


def test_sufficiency_table_covers_every_season_and_source(synthetic_frame):
    table = season_source_sufficiency(synthetic_frame)
    assert len(table) == 4 * 3
    assert set(table["season"]) == {"winter", "pre_monsoon", "monsoon", "post_monsoon"}
    assert set(table["status"]).issubset(set(SUFFICIENCY_STATUSES))


def test_absent_source_is_not_evaluable_not_zero(synthetic_frame):
    """A source never observed in a season must be NOT_EVALUABLE."""
    table = season_source_sufficiency(synthetic_frame).set_index(["season", "source"])
    # AOD is observed only in January (winter), never later in the frame.
    assert table.loc[("monsoon", "aod"), "n_observed"] == 0
    assert table.loc[("monsoon", "aod"), "status"] == "NOT_EVALUABLE"


def test_sufficiency_summary_counts_match_the_table(synthetic_frame):
    table = season_source_sufficiency(synthetic_frame)
    summary = summarise_sufficiency(table)
    assert sum(summary["counts"].values()) == len(table)
    assert summary["criteria"]["min_rows_for_inference"] == 30
    assert summary["criteria"]["min_rows_for_sufficient"] == 90


def test_summary_lists_not_evaluable_cells(synthetic_frame):
    table = season_source_sufficiency(synthetic_frame)
    summary = summarise_sufficiency(table)
    assert ("monsoon", "aod") in {
        (record["season"], record["source"]) for record in summary["not_evaluable"]
    }


def test_missing_flag_column_raises_in_sufficiency(synthetic_frame):
    with pytest.raises(KeyError, match="merra_observed"):
        season_source_sufficiency(synthetic_frame.drop(columns=["merra_observed"]))
