#!/usr/bin/env python
"""Track A dataset builder - Phase 1.

Reads the CPCB hourly exports, produces the validated daily series with
calendar-aware temporal features, and writes:

* ``data/processed/track_a_daily.csv``  - the daily series
* ``data/processed/track_a_features.csv`` - the daily series with features
* ``data/manifests/track_a_ingestion.json`` - the run manifest

The script is read-only with respect to ``data/cpcb/``.

Usage:
    python scripts/build_track_a_dataset.py [--root .] [--config config/track_a.yaml]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerosphinx.features.lags import (  # noqa: E402
    LagSpec,
    calendar_lags,
    calendar_rolling_stats,
)
from aerosphinx.ingestion.cpcb import CPCBRules, load_cpcb_daily  # noqa: E402
from aerosphinx.quality.coverage import compute_coverage  # noqa: E402
from aerosphinx.quality.provenance import (  # noqa: E402
    build_provenance,
    write_provenance,
)


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def build_features(frame: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Attach calendar-aware temporal features to the daily series.

    Every feature here is calendar-aware: a lag of *k* is the observation
    dated exactly *k* calendar days earlier, or missing. Nothing is carried
    forward across a gap.
    """
    out = frame.copy()

    specs = [LagSpec(f"PM25_Lag_{lag}", lag) for lag in config["features"]["pm25_lags"]]
    lags = calendar_lags(out, "PM25", specs)
    for column in lags.columns:
        out[column] = lags[column].to_numpy()

    for window in config["features"]["rolling_windows"]:
        stats = calendar_rolling_stats(
            out, "PM25", window, min_periods=config["features"]["rolling_min_periods"]
        )
        out[f"PM25_Roll{window}"] = stats[f"PM25_roll{window}_mean"].to_numpy()
        out[f"PM25_Roll{window}_n"] = stats[f"PM25_roll{window}_n"].to_numpy()

    dates = pd.to_datetime(out["Date"])
    day_of_week = dates.dt.day_of_week
    out["day_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    out["day_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)
    month = dates.dt.month
    out["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--min-hours", type=int, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    common = load_config(root / "config" / "common.yaml")
    quality = load_config(root / "config" / "data_quality.yaml")
    track_a = load_config(args.config or (root / "config" / "track_a.yaml"))

    cpcb_rules = quality["cpcb"]
    rules = CPCBRules(
        pm25_valid_min=cpcb_rules["pm25_valid_min"],
        pm25_valid_max=cpcb_rules["pm25_valid_max"],
        min_hours_per_day=(
            args.min_hours
            if args.min_hours is not None
            else cpcb_rules["min_hours_per_day"]
        ),
        expected_hours_per_day=cpcb_rules["expected_hours_per_day"],
        duplicate_hour_policy=cpcb_rules["duplicate_hour_policy"],
        require_whole_hour=cpcb_rules["require_whole_hour"],
    )

    cpcb_dir = root / common["paths"]["cpcb_dir"]
    print(f"[1/4] reading {cpcb_dir}")
    result = load_cpcb_daily(cpcb_dir, rules)
    print(
        f"      {result.n_days} days accepted "
        f"({result.start.date()} -> {result.end.date()})"
    )

    audit = result.audit
    coverage = compute_coverage(result.frame)
    print(
        f"[2/4] coverage {coverage.coverage_fraction:.1%} of the calendar, "
        f"{len(coverage.gaps)} gaps, longest "
        f"{max((g['gap_days'] for g in coverage.gaps), default=0)} days"
    )

    features = build_features(result.frame, track_a)
    n_complete = int(features[["PM25", "PM25_Lag_1"]].notna().all(axis=1).sum())
    print(
        f"[3/4] features: {n_complete} rows with a complete 1-day lag "
        f"(the rest have no previous calendar day)"
    )

    processed = root / common["paths"]["processed_dir"]
    manifests = root / common["paths"]["manifests_dir"]
    processed.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    daily_path = processed / "track_a_daily.csv"
    feature_path = processed / "track_a_features.csv"
    result.frame.to_csv(daily_path, index=False)
    features.to_csv(feature_path, index=False)

    run_id = f"A-ingest-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    provenance = build_provenance(
        run_id=run_id,
        root=root,
        inputs={
            "cpcb_dir": str(cpcb_dir.relative_to(root)),
            "file_sha256": audit.file_hashes,
        },
        dataset={
            "daily_rows": result.n_days,
            "date_range": [
                result.start.date().isoformat(),
                result.end.date().isoformat(),
            ],
            "calendar_span_days": coverage.calendar_span_days,
            "missing_days": coverage.missing_days,
            "n_gaps": len(coverage.gaps),
            "longest_gap_days": max((g["gap_days"] for g in coverage.gaps), default=0),
            "coverage_fraction": round(coverage.coverage_fraction, 6),
            "rows_with_complete_lag1": n_complete,
        },
        rules={
            "cpcb": {
                "pm25_valid_min": rules.pm25_valid_min,
                "pm25_valid_max": rules.pm25_valid_max,
                "min_hours_per_day": rules.min_hours_per_day,
                "expected_hours_per_day": rules.expected_hours_per_day,
                "duplicate_hour_policy": rules.duplicate_hour_policy,
                "require_whole_hour": rules.require_whole_hour,
            },
            "features": track_a["features"],
        },
        outputs={
            "daily_csv": str(daily_path.relative_to(root)),
            "features_csv": str(feature_path.relative_to(root)),
        },
    )
    provenance.dataset["audit"] = audit.as_dict()
    manifest_path = write_provenance(provenance, manifests / "track_a_ingestion.json")

    print(f"[4/4] wrote {daily_path.relative_to(root)}")
    print(f"      wrote {feature_path.relative_to(root)}")
    print(f"      wrote {manifest_path.relative_to(root)}")
    print()
    print("audit rejections:", json.dumps(audit.as_dict()["rejections"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
