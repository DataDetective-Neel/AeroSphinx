#!/usr/bin/env python
"""Emit the named daily multisource alignment artefacts.

Reads the canonical fused frame (``data/processed/fused_dataset.csv``) and
writes, without modifying any input:

* ``daily_source_coverage.csv``      observed/missing rows per source
* ``daily_source_missingness.csv``   target mean on observed vs unobserved dates
* ``daily_alignment_report.json``    row counts, overlap distribution, per-year
* ``daily_dataset_manifest.json``    provenance and validation result

Usage:
    python scripts/run_daily_alignment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.quality.alignment import (  # noqa: E402
    DEFAULT_SOURCES,
    build_alignment_report,
    validate_alignment_report,
)
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Fused frame CSV (default: <root>/data/processed/fused_dataset.csv)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Artefact directory (default: <root>/results/alignment)",
    )
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    input_csv = args.input or (root / "data" / "processed" / "fused_dataset.csv")
    out_dir = args.out_dir or (root / "results" / "alignment")

    if not input_csv.exists():
        print(f"ERROR: {input_csv} is missing; run `make run-phase6` first")
        return 1

    run_id = args.run_id or (
        f"alignment-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_csv, parse_dates=["Date"])
    report = build_alignment_report(frame, DEFAULT_SOURCES)
    validation = validate_alignment_report(report)

    report.source_coverage.to_csv(out_dir / "daily_source_coverage.csv", index=False)
    report.source_missingness.to_csv(
        out_dir / "daily_source_missingness.csv", index=False
    )
    (out_dir / "daily_alignment_report.json").write_text(
        json.dumps(
            {**report.as_dict(), "validation": validation},
            indent=2,
            default=str,
        )
        + "\n"
    )

    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs=(
            {"fused_dataset": str(input_csv.relative_to(root))}
            if input_csv.is_relative_to(root)
            else {"fused_dataset": str(input_csv)}
        ),
        dataset={
            "rows": int(len(frame)),
            "date_range": report.report["date_range"],
            "per_source_observed_rows": {
                record["source"]: int(record["observed_rows"])
                for record in report.source_coverage.to_dict(orient="records")
            },
            "rows_with_all_sources_observed": report.report[
                "rows_with_all_sources_observed"
            ],
            "rows_with_no_source_observed": report.report[
                "rows_with_no_source_observed"
            ],
        },
        rules={
            "phase": "1 - daily multisource alignment",
            "sources": list(DEFAULT_SOURCES),
            "missing_value_policy": (
                "no fill, interpolation or imputation; flags are read as given"
            ),
        },
        outputs={
            "run_dir": (
                str(out_dir.relative_to(root))
                if out_dir.is_relative_to(root)
                else str(out_dir)
            ),
            "files": sorted(path.name for path in out_dir.glob("*")),
        },
    )
    write_provenance(provenance, out_dir / "daily_dataset_manifest.json")

    print(
        f"rows: {len(frame)}  range: {report.report['date_range'][0]} -> "
        f"{report.report['date_range'][1]}"
    )
    for record in report.source_coverage.to_dict(orient="records"):
        print(
            f"  {record['source']:12s} observed {record['observed_rows']:4d} "
            f"({record['observed_fraction']:.1%})  "
            f"{record['first_observed']} -> {record['last_observed']}"
        )
    print(
        f"  all sources observed: "
        f"{report.report['rows_with_all_sources_observed']} rows "
        f"({report.report['fraction_with_all_sources_observed']:.1%})"
    )
    print(
        f"validation: {'PASS' if validation['passed'] else 'FAIL'} "
        f"({validation['n_checks']} checks, {validation['n_failed']} failed)"
    )
    print(f"wrote {out_dir}/")
    return 0 if validation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
