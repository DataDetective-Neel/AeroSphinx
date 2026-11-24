#!/usr/bin/env python
"""Emit per-season, per-source data sufficiency.

Reads the canonical fused frame and writes:

* ``season_source_sufficiency.csv``  one row per (season, source)
* ``season_sufficiency_report.json`` status counts, criteria and exclusions
* ``manifest.json``                  provenance

Usage:
    python scripts/run_season_sufficiency.py
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

from aerosphinx.evaluation.sufficiency import (  # noqa: E402
    SufficiencyCriteria,
    season_source_sufficiency,
    summarise_sufficiency,
)
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--min-rows-inference", type=int, default=30)
    parser.add_argument("--min-rows-sufficient", type=int, default=90)
    args = parser.parse_args()

    root = args.root.resolve()
    input_csv = args.input or (root / "data" / "processed" / "fused_dataset.csv")
    out_dir = args.out_dir or (root / "results" / "sufficiency")
    if not input_csv.exists():
        print(f"ERROR: {input_csv} is missing; run `make run-phase6` first")
        return 1

    criteria = SufficiencyCriteria(
        min_rows_for_inference=args.min_rows_inference,
        min_rows_for_sufficient=args.min_rows_sufficient,
    )
    run_id = args.run_id or (
        f"sufficiency-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(input_csv, parse_dates=["Date"])
    table = season_source_sufficiency(frame, criteria=criteria)
    summary = summarise_sufficiency(table, criteria)

    table.to_csv(out_dir / "season_source_sufficiency.csv", index=False)
    (out_dir / "season_sufficiency_report.json").write_text(
        json.dumps(
            {
                "rows": int(len(frame)),
                "date_range": [
                    frame["Date"].min().date().isoformat(),
                    frame["Date"].max().date().isoformat(),
                ],
                "summary": summary,
                "table": table.to_dict(orient="records"),
            },
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
            "seasons": sorted(table["season"].unique().tolist()),
            "sources": sorted(table["source"].unique().tolist()),
        },
        rules={
            "phase": "5 - data sufficiency",
            "criteria": criteria.as_dict(),
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
    write_provenance(provenance, out_dir / "manifest.json")

    print(f"rows: {len(frame)}")
    print(f"{'season':14s} {'source':12s} {'obs':>5s} {'rows':>5s}  status")
    for record in table.to_dict(orient="records"):
        print(
            f"{record['season']:14s} {record['source']:12s} "
            f"{record['n_observed']:5d} {record['n_season_rows']:5d}  "
            f"{record['status']}"
        )
    print("\ncounts:", summary["counts"])
    print(f"wrote {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
