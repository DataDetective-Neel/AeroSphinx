#!/usr/bin/env python
"""Phase 4 - INSAT-3D AOD validation.

Validates the INSAT-3D L2G AOD extraction against the real granules and
produces the evidence required by the Phase 4 gate: dimension and
orientation verification, fill-value and QA verification, spatial-window
verification, duplicate-granule verification, and valid-pixel coverage per
date.

Outputs, under ``results/runs/<run_id>/``:

* ``aod_structural_audit.json`` - the audit of every granule
* ``aod_orientation_check.json`` - the physical orientation cross-check
* ``aod_daily.csv``            - the extracted daily series with coverage
* ``aod_coverage.csv``         - valid-pixel coverage for every date
* ``aod_window_sensitivity.csv`` - sensitivity to the collocation window
* ``manifest.json``            - Git SHA, environment, configuration

Usage:
    python scripts/run_aod_validation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.ingestion.aod import (  # noqa: E402
    AODRules,
    load_aod_daily,
    orientation_check,
    window_sensitivity,
)
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    common = yaml.safe_load((root / "config" / "common.yaml").read_text())
    quality = yaml.safe_load((root / "config" / "data_quality.yaml").read_text())
    insat = quality["insat_aod"]
    station = common["project"]["station"]

    rules = AODRules(
        fill_value=insat["fill_value"],
        aod_valid_min=insat["aod_valid_min"],
        aod_valid_max=insat["aod_valid_max"],
        window_half_width_deg=insat["collocation_window_deg"],
    )

    aod_dir = root / common["paths"]["aod_dir"]

    print(f"[1/4] validating INSAT-3D granules in {aod_dir}")
    result = load_aod_daily(
        aod_dir,
        rules,
        station_lat=station["latitude"],
        station_lon=station["longitude"],
    )
    audit = result.audit.as_dict()
    print(f"      {audit['n_granules']} granules, {audit['distinct_dates']} dates")
    print(f"      shapes: {audit['shapes']}")
    print(f"      dimension-scale bindings: {audit['dimension_scale_bindings']}")
    print(f"      fill values: {audit['fill_values']}")
    print(
        f"      duplicate (date, overpass) keys: {audit['n_duplicate_date_overpass_keys']}"
    )
    print(
        f"      granules with zero valid pixels: {audit['granules_with_zero_valid_pixels']}"
    )
    print(f"      QA dataset present: {audit['has_qa_dataset']}")

    print("\n[2/4] physical orientation cross-check")
    orientation = orientation_check(
        aod_dir,
        station_lat=station["latitude"],
        station_lon=station["longitude"],
    )
    for hypothesis, detail in orientation["hypotheses"].items():
        print(
            f"      {hypothesis:16s} ocean {detail['ocean_mean']:.4f}  "
            f"land {detail['land_mean']:.4f}  contrast {detail['ocean_land_contrast']:+.4f}  "
            f"plausible {detail['physically_plausible']}"
        )
    print(
        f"      dimension-scale orientation corroborated: "
        f"{orientation['dimension_scale_orientation_corroborated']}"
    )

    print("\n[3/4] collocation-window sensitivity")
    sensitivity = window_sensitivity(
        aod_dir,
        rules=rules,
        station_lat=station["latitude"],
        station_lon=station["longitude"],
    )
    print(sensitivity.round(4).to_string(index=False))

    run_id = (
        args.run_id or f"D-aod-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "aod_structural_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n"
    )
    (out_dir / "aod_orientation_check.json").write_text(
        json.dumps(orientation, indent=2, default=str) + "\n"
    )
    result.frame.to_csv(out_dir / "aod_daily.csv", index=False)

    coverage = result.frame.copy()
    coverage["coverage_fraction_of_window"] = coverage["aod_valid_pixels"] / 25.0
    coverage.to_csv(out_dir / "aod_coverage.csv", index=False)
    sensitivity.to_csv(out_dir / "aod_window_sensitivity.csv", index=False)

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={"aod_dir": str(aod_dir.relative_to(root))},
        dataset={
            "n_granules": audit["n_granules"],
            "n_days": int(result.n_days),
            "date_range": [
                result.frame["Date"].min().date().isoformat(),
                result.frame["Date"].max().date().isoformat(),
            ],
            "days_with_zero_valid_pixels": int(
                (result.frame["aod_valid_pixels"] == 0).sum()
            ),
            "mean_valid_pixels": float(result.frame["aod_valid_pixels"].mean()),
        },
        rules={
            "insat_aod": insat,
            "station": station,
            "workstream": "Phase 4 - INSAT-3D AOD validation",
            "orientation_corroborated": orientation[
                "dimension_scale_orientation_corroborated"
            ],
        },
        outputs={
            "run_dir": _relative(out_dir, root),
            "files": sorted(p.name for p in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"\n[4/4] wrote {_relative(out_dir, root)}/")
    return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
