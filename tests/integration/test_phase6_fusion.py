"""Integration tests for the Phase 6 fused dataset and alignment gate.

These run on the real local record: the Track A daily series, the CPCB
station-local meteorology, INSAT-3D AOD and MERRA-2. The test asserts the
D-003 property directly on the published fused frame: an environmental value
exists only where a same-day source record exists, and nothing is carried
forward.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from aerosphinx.features.availability import (
    SOURCE_GROUPS,
    FusionRules,
    build_fused_dataset,
    validate_fused_dataset,
    value_columns,
)
from aerosphinx.ingestion.aod import AODRules, load_aod_daily
from aerosphinx.ingestion.merra import load_merra_daily
from aerosphinx.ingestion.station_met import load_station_met_daily

ROOT = Path(__file__).resolve().parents[2]
FEATURES_CSV = ROOT / "data" / "processed" / "track_a_features.csv"

pytestmark = [
    pytest.mark.integration,
    pytest.mark.scientific,
    pytest.mark.skipif(
        not FEATURES_CSV.exists(),
        reason="Track A features not built; run scripts/build_track_a_dataset.py",
    ),
]


@pytest.fixture(scope="module")
def fusion():
    """Build and validate the real fused dataset once."""
    common = yaml.safe_load((ROOT / "config" / "common.yaml").read_text())
    quality = yaml.safe_load((ROOT / "config" / "data_quality.yaml").read_text())
    fusion_config = yaml.safe_load((ROOT / "config" / "fusion.yaml").read_text())[
        "fusion"
    ]
    station = common["project"]["station"]
    paths = common["paths"]

    daily = pd.read_csv(FEATURES_CSV, parse_dates=["Date"])
    station_met = load_station_met_daily(
        ROOT / paths["cpcb_dir"],
        valid_ranges={
            key: tuple(value)
            for key, value in quality["cpcb"]["station_met_valid_ranges"].items()
        },
        excluded_columns=tuple(quality["cpcb"]["station_met_excluded_columns"]),
    )
    merra = load_merra_daily(ROOT / paths["merra_dir"])
    insat = quality["insat_aod"]
    aod = load_aod_daily(
        ROOT / paths["aod_dir"],
        AODRules(
            fill_value=insat["fill_value"],
            aod_valid_min=insat["aod_valid_min"],
            aod_valid_max=insat["aod_valid_max"],
            window_half_width_deg=insat["collocation_window_deg"],
        ),
        station_lat=station["latitude"],
        station_lon=station["longitude"],
    )

    rules = FusionRules(
        date_column=fusion_config["date_column"],
        target=fusion_config["target"],
        min_observed_values=fusion_config["min_observed_values_per_source"],
    )
    fused = build_fused_dataset(
        daily,
        station_met=station_met.frame,
        merra=merra.frame,
        aod=aod.frame,
        rules=rules,
    )
    validation = validate_fused_dataset(
        fused,
        daily=daily,
        station_met=station_met.frame,
        merra=merra.frame,
        aod=aod.frame,
    )
    return {
        "fused": fused,
        "validation": validation,
        "daily": daily,
        "station_met": station_met.frame,
        "merra": merra.frame,
        "aod": aod.frame,
    }


def test_phase6_alignment_gate_passes(fusion):
    """The real multisource dataset must pass the Phase 6 gate."""
    validation = fusion["validation"]
    assert validation["passed"], validation["failed_checks"]
    assert validation["n_failed"] == 0


def test_fused_rows_are_the_ground_rows(fusion):
    """Fusion must not add or remove a single ground day."""
    fused = fusion["fused"].frame
    daily = fusion["daily"]
    assert len(fused) == len(daily)
    assert set(fused["Date"]) == set(daily["Date"])


def test_no_environmental_value_without_a_same_day_observation(fusion):
    """The D-003 property, checked on every source and every row."""
    fused = fusion["fused"].frame
    for source in SOURCE_GROUPS:
        unobserved = fused[~fused[f"{source}_observed"]]
        assert (
            unobserved[list(value_columns(source))].isna().all().all()
        ), f"{source} carries a value on an unobserved row"


def test_no_source_value_is_forward_filled(fusion):
    """Every present source value must equal the source's same-day record."""
    fused = fusion["fused"].frame
    source_frames = {
        "station_met": fusion["station_met"],
        "merra": fusion["merra"],
        "aod": fusion["aod"],
    }
    for source, frame in source_frames.items():
        merged = fused[["Date", *value_columns(source)]].merge(
            frame, on="Date", how="left", suffixes=("", "_source")
        )
        for column in value_columns(source):
            left = merged[column]
            right = merged[f"{column}_source"]
            both = left.notna() | right.notna()
            assert (
                left[both] == right[both]
            ).all(), f"{column} does not match the same-day source record"


def test_aod_never_appears_without_pixel_support(fusion):
    """An AOD value must always be backed by at least one valid pixel."""
    fused = fusion["fused"].frame
    supported = fused["AOD"].notna()
    assert supported.any()
    assert (fused.loc[supported, "aod_valid_pixels"] >= 1).all()
    # And the mask agrees with pixel support exactly.
    assert (fused["aod_observed"] == supported).all()


def test_merra_observed_rows_match_its_calendar_coverage(fusion):
    """MERRA-2 covers 2024 only; the mask must reflect exactly that."""
    fused = fusion["fused"].frame
    merra_dates = set(pd.to_datetime(fusion["merra"]["Date"]))
    ground_dates = set(fused["Date"])
    expected = len(merra_dates & ground_dates)
    assert int(fused["merra_observed"].sum()) == expected
    assert fused["merra_observed"].sum() < len(fused)


def test_availability_summary_reports_every_source(fusion):
    """The summary must describe coverage and informative missingness."""
    summary = fusion["fused"].summary
    assert summary["requested_sources"] == list(SOURCE_GROUPS)
    for source in SOURCE_GROUPS:
        detail = summary["per_source"][source]
        assert detail["fused_rows_observed"] >= 1
        assert detail["operational_eligible"] is False
        assert "target_mean_when_observed" in detail
        assert "target_mean_when_unobserved" in detail
    assert summary["rows_with_all_sources_observed"] >= 1
    # Station meteorology covers the whole ground record, so every ground
    # row has at least one environmental source (MERRA-2 and AOD do not).
    assert summary["rows_with_no_source_observed"] == 0
    assert summary["per_source"]["station_met"]["fused_rows_observed"] == len(
        fusion["daily"]
    )
    assert summary["ground_calendar_preserved"] is True
