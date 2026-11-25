#!/usr/bin/env python
"""Phase 5 - MERRA-2 validation.

A short validation: structure, units, grid and temporal coverage, plus the
physical cross-validation against the CPCB station. No feature-group
infrastructure is built, because the group cannot be evaluated on the
current record.

Outputs, under ``results/runs/<run_id>/``:

* ``merra_structural_audit.json``    - the audit of every granule
* ``merra_station_crosscheck.csv``   - the hour-matched comparison
* ``manifest.json``                  - Git SHA, environment, configuration

Usage:
    python scripts/run_merra_validation.py
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

from aerosphinx.quality.merra_validation import (  # noqa: E402
    station_cross_check,
    validate_merra,
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
    station = common["project"]["station"]

    merra_dir = root / common["paths"]["merra_dir"]
    cpcb_dir = root / common["paths"]["cpcb_dir"]

    print(f"[1/3] validating MERRA-2 granules in {merra_dir}")
    audit = validate_merra(merra_dir).as_dict()
    print(f"      {audit['n_granules']} granules")
    print(f"      dimensions: {audit['dimension_signatures']}")
    print(f"      unit mismatches: {len(audit['unit_mismatches'])}")
    print(f"      out-of-range values: {len(audit['out_of_range_values'])}")
    print(f"      missing values: {audit['n_missing_values']}")
    print(f"      grid cells: {audit['grid_cells']}")
    print(f"      time signatures: {audit['time_signatures']}")
    print(
        f"      coverage: {audit['date_start']} to {audit['date_end']}  "
        f"{audit['n_dates']} dates, {audit['n_missing_dates']} missing"
    )
    print(f"      local hours at the station (IST): {audit['local_hours_ist']}")
    print(f"      subsetting boxes: {audit['subsetting_boxes']}")

    print("\n[2/3] physical cross-check against the CPCB station")
    cross = station_cross_check(
        merra_dir,
        cpcb_dir,
        station_lat=station["latitude"],
        station_lon=station["longitude"],
    )
    print(f"      matched hour pairs: {len(cross)}")
    if not cross.empty:
        print(
            f"      T2M vs station AT : corr {cross.T2M_C.corr(cross.station_AT_C):+.4f}  "
            f"mean diff {(cross.T2M_C - cross.station_AT_C).mean():+.2f} C"
        )
        print(
            f"      QV2M vs station q : corr {cross.QV2M.corr(cross.station_q_implied):+.4f}  "
            f"mean diff {(cross.QV2M - cross.station_q_implied).mean() * 1000:+.3f} g/kg"
        )
        print(
            f"      TS  vs T2M        : mean {(cross.TS_C - cross.T2M_C).mean():+.2f} C"
        )
        print(
            f"      SLP minus PS      : {cross.SLP_minus_PS_hPa.mean():+.2f} hPa "
            f"(implied elevation {cross.SLP_minus_PS_hPa.mean() / 0.1263:.0f} m)"
        )
        print(
            f"      SLP vs station BP : corr {cross.SLP_Pa.corr(cross.station_BP_hPa):+.4f} "
            f"<- NEGATIVE is impossible; see docs/reports/PHASE5_REPORT.md section 6"
        )

    run_id = (
        args.run_id
        or f"D-merra-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "merra_structural_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n"
    )
    if not cross.empty:
        cross.to_csv(out_dir / "merra_station_crosscheck.csv", index=False)

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "merra_dir": str(merra_dir.relative_to(root)),
            "cpcb_dir": str(cpcb_dir.relative_to(root)),
        },
        dataset={
            "n_granules": audit["n_granules"],
            "n_dates": audit["n_dates"],
            "date_start": audit["date_start"],
            "date_end": audit["date_end"],
            "n_missing_dates": audit["n_missing_dates"],
            "local_hours_ist": audit["local_hours_ist"],
            "cross_check_pairs": int(len(cross)),
        },
        rules={
            "workstream": "Phase 5 - MERRA-2 validation (short validation only)",
            "station": station,
            "feature_group_built": False,
            "reason": (
                "MERRA-2 covers 172 days of a 3,288-day record; the group "
                "cannot be evaluated and is therefore left disabled."
            ),
        },
        outputs={
            "run_dir": _relative(out_dir, root),
            "files": sorted(p.name for p in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"\n[3/3] wrote {_relative(out_dir, root)}/")
    return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
