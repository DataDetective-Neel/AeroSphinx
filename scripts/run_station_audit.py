#!/usr/bin/env python
"""Phase 3 Workstream D - CPCB station-local meteorological variable audit.

Audits every candidate station-local meteorological column in the CPCB
export and produces the variable record required by the Phase 3
authorization.

Outputs, under ``results/runs/<run_id>/``:

* ``station_variable_audit.csv``  - one row per variable
* ``station_variable_audit.json`` - the same, with full evidence
* ``manifest.json``                - Git SHA, environment, configuration

Usage:
    python scripts/run_station_audit.py
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

from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)
from aerosphinx.quality.station_audit import audit_station_columns  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    common = yaml.safe_load((root / "config" / "common.yaml").read_text())

    cpcb_dir = root / common["paths"]["cpcb_dir"]
    merra_dir = root / common["paths"]["merra_dir"]
    station = common["project"]["station"]["name"]

    print(f"[1/3] auditing station-local columns in {cpcb_dir}")
    records = audit_station_columns(cpcb_dir, station_name=station, merra_dir=merra_dir)
    print(f"      {len(records)} candidate variables audited")

    table = pd.DataFrame([record.as_dict() for record in records])
    display_columns = [
        "variable",
        "source_column",
        "unit",
        "station_or_source",
        "missing_percentage",
        "quality_flag",
        "forecast_usable",
    ]
    print("\n[2/3] audit summary:")
    print(table[display_columns].to_string(index=False))

    run_id = (
        args.run_id
        or f"D-station-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir = args.out_dir or (root / common["paths"]["results_dir"] / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    table.to_csv(out_dir / "station_variable_audit.csv", index=False)
    (out_dir / "station_variable_audit.json").write_text(
        json.dumps([record.as_dict() for record in records], indent=2, default=str)
        + "\n"
    )

    usable = [r.variable for r in records if r.forecast_usable]
    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "cpcb_dir": str(cpcb_dir.relative_to(root)),
            "merra_dir": str(merra_dir.relative_to(root)),
        },
        dataset={
            "n_variables_audited": len(records),
            "n_forecast_usable": len(usable),
            "forecast_usable_variables": usable,
            "variables_blocked_by_unknown_availability": [
                r.variable for r in records if r.availability_timestamp == "UNKNOWN"
            ],
        },
        rules={
            "workstream": "D - CPCB station-local meteorology audit",
            "audit_fields": [
                "variable",
                "source_column",
                "physical_meaning",
                "unit",
                "station_or_source",
                "measurement_timestamp",
                "availability_timestamp",
                "missing_percentage",
                "valid_range",
                "quality_flag",
                "forecast_usable",
            ],
        },
        outputs={
            "run_dir": _relative(out_dir, root),
            "files": sorted(p.name for p in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"[3/3] wrote {_relative(out_dir, root)}/")
    print(f"      forecast-usable variables: {len(usable)} of {len(records)}")
    if usable:
        print(f"        {', '.join(usable)}")
    else:
        print("        none - the availability timestamp cannot be established")
        print("        from the local files, so no column is forecast-usable yet")
    return 0


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
