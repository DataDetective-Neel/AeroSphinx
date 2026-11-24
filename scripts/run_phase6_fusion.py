#!/usr/bin/env python
"""Phase 6 - multisource fusion and availability.

Builds the canonical Track B fused dataset: the CPCB ground series with
INSAT-3D AOD, MERRA-2 and the station-local meteorology joined on the
ground calendar, carrying an explicit per-source availability mask.

This is the D-003 correction. Missing environmental observations are
preserved as NaN and flagged; nothing is forward-filled, interpolated or
imputed. The script then runs the Phase 6 alignment gate and refuses to
publish a dataset that fails it.

Outputs:

* ``data/processed/fused_dataset.csv`` - the fused frame with mask columns
* ``results/runs/<run_id>/availability_mask.csv`` - the mask alone
* ``results/runs/<run_id>/availability_summary.json`` - coverage and
  informative-missingness statistics
* ``results/runs/<run_id>/alignment_audit.json`` - the gate checks
* ``results/runs/<run_id>/manifest.json`` - Git SHA, environment, rules

Usage:
    python scripts/run_phase6_fusion.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.features.availability import (  # noqa: E402
    SOURCE_GROUPS,
    FusionRules,
    assert_fusion_alignment,
    build_fused_dataset,
    validate_fused_dataset,
)
from aerosphinx.features.groups import describe_groups  # noqa: E402
from aerosphinx.ingestion.aod import AODRules, load_aod_daily  # noqa: E402
from aerosphinx.ingestion.merra import load_merra_daily  # noqa: E402
from aerosphinx.ingestion.station_met import load_station_met_daily  # noqa: E402
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-failure",
        action="store_true",
        help="Write artefacts even when the alignment gate fails.",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    common = yaml.safe_load((root / "config" / "common.yaml").read_text())
    quality = yaml.safe_load((root / "config" / "data_quality.yaml").read_text())
    fusion_config = yaml.safe_load((root / "config" / "fusion.yaml").read_text())[
        "fusion"
    ]
    station = common["project"]["station"]
    paths = common["paths"]

    rules = FusionRules(
        date_column=fusion_config["date_column"],
        target=fusion_config["target"],
        min_observed_values=fusion_config["min_observed_values_per_source"],
    )

    print("[1/5] reading the ground series and the three environmental sources")
    features_csv = root / paths["processed_dir"] / "track_a_features.csv"
    if not features_csv.exists():
        print(f"      ERROR: {features_csv} is missing; run make build-track-a")
        return 1
    daily = pd.read_csv(features_csv, parse_dates=["Date"])
    print(
        f"      ground: {len(daily)} rows, {daily['Date'].min().date()} -> "
        f"{daily['Date'].max().date()}"
    )

    station_met = load_station_met_daily(
        root / paths["cpcb_dir"],
        valid_ranges={
            key: tuple(value)
            for key, value in quality["cpcb"]["station_met_valid_ranges"].items()
        },
        excluded_columns=tuple(quality["cpcb"]["station_met_excluded_columns"]),
    )
    merra = load_merra_daily(root / paths["merra_dir"])
    insat = quality["insat_aod"]
    aod = load_aod_daily(
        root / paths["aod_dir"],
        AODRules(
            fill_value=insat["fill_value"],
            aod_valid_min=insat["aod_valid_min"],
            aod_valid_max=insat["aod_valid_max"],
            window_half_width_deg=insat["collocation_window_deg"],
        ),
        station_lat=station["latitude"],
        station_lon=station["longitude"],
    )
    print(f"      station_met: {len(station_met.frame)} accepted days")
    print(f"      merra:       {len(merra.frame)} days")
    print(
        f"      aod:         {len(aod.frame)} days "
        f"({int((aod.frame['aod_valid_pixels'] == 0).sum())} with no valid pixel)"
    )

    print("\n[2/5] fusing on the ground calendar (no fill, no imputation)")
    fused = build_fused_dataset(
        daily,
        station_met=station_met.frame,
        merra=merra.frame,
        aod=aod.frame,
        rules=rules,
    )
    print(f"      {len(fused.frame)} rows, {len(fused.frame.columns)} columns")
    print(f"      policy: {fusion_config['missing_value_note'].strip()}")

    print("\n[3/5] running the Phase 6 alignment gate")
    validation = validate_fused_dataset(
        fused,
        daily=daily,
        station_met=station_met.frame,
        merra=merra.frame,
        aod=aod.frame,
    )
    for entry in validation["checks"]:
        print(
            f"      {'PASS' if entry['passed'] else 'FAIL'}  {entry['check']}: "
            f"{entry['detail']}"
        )
    print(f"      {validation['n_checks']} checks, " f"{validation['n_failed']} failed")

    fused.checks = validation
    if not validation["passed"]:
        print("\n      Phase 6 gate: FAIL")
        if not args.allow_failure:
            assert_fusion_alignment(validation)
    else:
        print("      Phase 6 gate: PASS")

    print("\n[4/5] availability by source")
    for source in SOURCE_GROUPS:
        detail = fused.summary["per_source"][source]
        print(
            f"      {source:12s} observed {detail['fused_rows_observed']:4d} rows "
            f"({detail['fused_rows_observed_fraction']:.1%}), "
            f"target mean observed "
            f"{detail['target_mean_when_observed']:.2f} vs unobserved "
            f"{detail['target_mean_when_unobserved']:.2f}, "
            f"max age {detail['max_observation_age_days']}"
        )
    print(
        f"      rows with all {len(fused.requested_sources)} sources: "
        f"{fused.summary['rows_with_all_sources_observed']} "
        f"({fused.summary['fraction_with_all_sources_observed']:.1%})"
    )
    print(
        f"      rows with no environmental source: "
        f"{fused.summary['rows_with_no_source_observed']}"
    )

    run_id = args.run_id or (
        f"F-fusion-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / paths["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = root / paths["processed_dir"]
    processed.mkdir(parents=True, exist_ok=True)

    fused_csv = processed / "fused_dataset.csv"
    fused.frame.to_csv(fused_csv, index=False)

    mask_columns = [rules.date_column, rules.target]
    for source in SOURCE_GROUPS:
        mask_columns += [
            f"{source}_observed",
            f"{source}_observation_age_days",
        ]
    mask_columns += ["n_sources_observed", "all_sources_observed"]
    fused.frame[mask_columns].to_csv(out_dir / "availability_mask.csv", index=False)

    (out_dir / "availability_summary.json").write_text(
        json.dumps(fused.summary, indent=2, default=str) + "\n"
    )
    (out_dir / "alignment_audit.json").write_text(
        json.dumps(validation, indent=2, default=str) + "\n"
    )

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "features_csv": str(features_csv.relative_to(root)),
            "cpcb_dir": str((root / paths["cpcb_dir"]).relative_to(root)),
            "aod_dir": str((root / paths["aod_dir"]).relative_to(root)),
            "merra_dir": str((root / paths["merra_dir"]).relative_to(root)),
        },
        dataset={
            "rows": int(len(fused.frame)),
            "date_range": fused.summary["date_range"],
            "ground_rows": int(len(daily)),
            "per_source_observed_rows": {
                source: fused.summary["per_source"][source]["fused_rows_observed"]
                for source in SOURCE_GROUPS
            },
            "rows_with_all_sources_observed": fused.summary[
                "rows_with_all_sources_observed"
            ],
            "rows_with_no_source_observed": fused.summary[
                "rows_with_no_source_observed"
            ],
        },
        rules={
            "fusion": fusion_config,
            "source_groups": describe_groups(list(SOURCE_GROUPS)),
            "availability_contract": {
                "ground_only": "operationally eligible",
                "station_met": "retrospective only - availability timestamp UNKNOWN",
                "merra": "retrospective only - reanalysis is never operational",
                "aod": "retrospective only - no satellite forecast product",
            },
            "phase": "6 - multisource fusion and availability",
        },
        outputs={
            "fused_dataset": str(fused_csv.relative_to(root)),
            "run_dir": _relative(out_dir, root),
            "files": sorted(path.name for path in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"\n[5/5] wrote {_relative(fused_csv, root)}")
    print(f"      wrote {_relative(out_dir, root)}/")
    return 0 if validation["passed"] else 1


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
