#!/usr/bin/env python
"""AeroSphinx Phase 0 - data-layer audit.

Inspects the three raw sources (CPCB CSV, INSAT-3D HDF5, MERRA-2 NetCDF) and
writes a machine-readable evidence file describing what is actually present.

This script is READ-ONLY with respect to raw data and to the pipeline. It
never writes into data/, results/, docs/figures/ or models/ except its own
audit output under results/baseline/.

Usage:
    python scripts/audit_data.py [--root .] [--out results/baseline/data_audit.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Constants mirrored from legacy/aerosphinx_pipeline.py so the audit measures the
# same behaviour the pipeline actually exhibits.
TARGET = "PM25"
PM25_VALID_MIN = 0.0
PM25_VALID_MAX = 500.0
MIN_HOURS_PER_DAY = 18
FORWARD_FILL_LIMIT_DAYS = 3
STATION_LAT = 28.4089
STATION_LON = 77.3178
COLLOCATION_WINDOW_DEG = 0.25

AOD_FILL_VALUE = -999.0
AOD_VALID_MIN = 0.0
AOD_VALID_MAX = 5.0

MERRA_REQUESTED = ["T2M", "TS", "QV2M", "U10M", "V10M", "SLP", "PBLH"]
MERRA_FALLBACKS = {"PBLH": ["PBLTOP"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_pm25_column(columns) -> str:
    for col in columns:
        upper = col.strip().upper().replace(" ", "")
        if "PM2.5" in col or upper.startswith("PM25"):
            return col
    raise ValueError(f"No PM2.5 column found among: {list(columns)}")


# ---------------------------------------------------------------------------
# CPCB
# ---------------------------------------------------------------------------


def audit_cpcb(root: Path) -> dict:
    cpcb_dir = root / "data" / "cpcb"
    files = sorted(cpcb_dir.glob("*.csv"))
    out: dict = {
        "directory": str(cpcb_dir.relative_to(root)),
        "file_count": len(files),
        "files": [],
        "hourly_rows_total": 0,
        "daily_rows_accepted": 0,
        "daily_rows_rejected_min_hours": 0,
        "calendar_gaps_over_1_day": 0,
        "max_gap_days": 0,
        "duplicate_hour_rows_across_files": 0,
        "mean_of_means_vs_hour_weighted_mismatch_days": 0,
        "per_year": {},
    }
    if not files:
        out["status"] = "MISSING"
        return out

    hourly_frames = []
    daily_frames = []
    for path in files:
        raw = pd.read_csv(path)
        pm_col = _find_pm25_column(raw.columns)
        hourly = pd.DataFrame(
            {
                "Timestamp": pd.to_datetime(raw["Timestamp"], errors="coerce"),
                TARGET: pd.to_numeric(raw[pm_col], errors="coerce"),
            }
        ).dropna(subset=["Timestamp"])
        invalid = (hourly[TARGET] < PM25_VALID_MIN) | (hourly[TARGET] > PM25_VALID_MAX)
        hourly.loc[invalid, TARGET] = np.nan
        hourly["source_file"] = path.name
        hourly_frames.append(hourly)

        daily = (
            hourly.set_index("Timestamp")
            .sort_index()[TARGET]
            .resample("D")
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={"Timestamp": "Date", "mean": TARGET, "count": "valid_hours"}
            )
        )
        daily_frames.append(daily[["Date", TARGET, "valid_hours"]])

        out["files"].append(
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "hourly_rows": int(len(hourly)),
                "pm25_invalid_screened": int(invalid.sum()),
            }
        )
        out["hourly_rows_total"] += int(len(hourly))

    combined = pd.concat(daily_frames, ignore_index=True)
    grouped = combined.groupby("Date", as_index=False).agg(
        {TARGET: "mean", "valid_hours": "sum"}
    )
    accepted = (
        grouped[grouped["valid_hours"] >= MIN_HOURS_PER_DAY]
        .dropna(subset=[TARGET])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    out["daily_rows_accepted"] = int(len(accepted))
    out["daily_rows_rejected_min_hours"] = int(len(grouped) - len(accepted))
    out["daily_date_range"] = [
        accepted["Date"].min().date().isoformat(),
        accepted["Date"].max().date().isoformat(),
    ]

    gaps = accepted["Date"].diff().dt.days
    out["calendar_gaps_over_1_day"] = int((gaps > 1).sum())
    out["max_gap_days"] = int(np.nanmax(gaps.values)) if len(gaps) else 0

    all_hourly = pd.concat(hourly_frames, ignore_index=True)
    out["duplicate_hour_rows_across_files"] = int(
        all_hourly.duplicated(subset=["Timestamp"], keep=False).sum()
    )

    # Weighting sensitivity: mean-of-file-means vs hour-weighted daily mean.
    valid_hourly = all_hourly[all_hourly[TARGET].notna()]
    weighted = (
        valid_hourly.set_index("Timestamp").sort_index()[TARGET].resample("D").mean()
    )
    mean_of_means = accepted.set_index("Date")[TARGET]
    compare = pd.DataFrame(
        {"weighted": weighted, "mean_of_means": mean_of_means}
    ).dropna()
    delta = (compare["weighted"] - compare["mean_of_means"]).abs()
    out["mean_of_means_vs_hour_weighted_mismatch_days"] = int((delta > 1e-9).sum())
    out["mean_of_means_vs_hour_weighted_max_abs_diff"] = float(delta.max())

    accepted["year"] = accepted["Date"].dt.year
    out["per_year"] = {
        str(year): int(count) for year, count in accepted.groupby("year").size().items()
    }

    # Row-based lag defect: does PM25_Lag_1 really mean "previous calendar day"?
    lag1 = accepted[TARGET].shift(1)
    age = (accepted["Date"] - accepted["Date"].shift(1)).dt.days
    out["rows_where_lag1_is_not_previous_calendar_day"] = int((age > 1).sum())
    out["max_lag1_calendar_age_days"] = int(np.nanmax(age.values)) if len(age) else 0
    out["lag1_stale_examples"] = [
        {
            "date": accepted["Date"].iloc[i].date().isoformat(),
            "lag1_value": float(lag1.iloc[i]),
            "calendar_age_days": int(age.iloc[i]),
        }
        for i in range(min(8, len(accepted)))
        if age.iloc[i] > 1
    ]
    out["status"] = "OK"
    return out


# ---------------------------------------------------------------------------
# INSAT-3D AOD
# ---------------------------------------------------------------------------


def audit_aod(root: Path) -> dict:
    aod_dir = root / "data" / "aod"
    files = sorted(aod_dir.glob("*.h5"))
    out: dict = {
        "directory": str(aod_dir.relative_to(root)),
        "file_count": len(files),
        "status": "MISSING",
    }
    if not files:
        return out

    import h5py

    dates = []
    overpass_times: dict[str, int] = {}
    bboxes = set()
    datasets_seen: set[str] = set()
    valid_pixels: list[int] = []
    granule_valid_pixels: list[int] = []
    fill_seen = False
    attrs_of_record: dict = {}
    dim_scale_of_record: dict = {}
    over_values: list[float] = []

    for path in files:
        with h5py.File(path, "r") as handle:
            attrs = {
                k: (v.decode("utf-8", "ignore") if isinstance(v, bytes) else v)
                for k, v in handle.attrs.items()
            }
            datasets_seen.update(handle.keys())
            if path == files[0]:
                attrs_of_record = {k: str(v) for k, v in attrs.items()}
                for name in handle.keys():
                    labels = []
                    for dim in handle[name].dims:
                        for key in dim.keys():
                            labels.append(getattr(dim[key], "name", "?"))
                    dim_scale_of_record[name] = labels

            acquisition = attrs.get("Acquisition_Date")
            if isinstance(acquisition, bytes):
                acquisition = acquisition.decode("utf-8", "ignore")
            parsed = pd.to_datetime(
                str(acquisition).strip(), format="%d%b%Y", errors="coerce"
            )
            if pd.notna(parsed):
                dates.append(parsed.date().isoformat())

            time_token = re.search(r"_(\d{4})_", path.name)
            if time_token:
                overpass_times[time_token.group(1)] = (
                    overpass_times.get(time_token.group(1), 0) + 1
                )

            grid = np.asarray(handle["AOD"][0], dtype=float)
            lats = np.asarray(handle["latitude"][:], dtype=float)
            lons = np.asarray(handle["longitude"][:], dtype=float)
            bboxes.add(
                (
                    round(float(lats.min()), 3),
                    round(float(lats.max()), 3),
                    round(float(lons.min()), 3),
                    round(float(lons.max()), 3),
                )
            )
            if not fill_seen and "_FillValue" in handle["AOD"].attrs:
                fill_seen = True

            lat_mask = (lats >= STATION_LAT - COLLOCATION_WINDOW_DEG) & (
                lats <= STATION_LAT + COLLOCATION_WINDOW_DEG
            )
            lon_mask = (lons >= STATION_LON - COLLOCATION_WINDOW_DEG) & (
                lons <= STATION_LON + COLLOCATION_WINDOW_DEG
            )
            window = grid[np.ix_(lat_mask, lon_mask)]
            valid = window[(window >= AOD_VALID_MIN) & (window <= AOD_VALID_MAX)]
            granule_valid_pixels.append(int(valid.size))
            if valid.size:
                valid_pixels.append(int(valid.size))
                over_values.extend(valid.tolist())

    out.update(
        {
            "status": "OK",
            "datasets_present": sorted(datasets_seen),
            "has_qa_flag_dataset": any(
                d.lower() in {"qa", "quality", "cloud", "flags"} for d in datasets_seen
            ),
            "distinct_dates": len(set(dates)),
            "date_range": [min(dates), max(dates)] if dates else None,
            "overpass_time_utc_counts": dict(sorted(overpass_times.items())),
            "distinct_bounding_boxes": [list(b) for b in sorted(bboxes)],
            "hdf5_dimension_scales": dim_scale_of_record,
            "fill_value": AOD_FILL_VALUE,
            "station_window_pixels_max": 25,
            "granules_with_zero_valid_pixels": int(
                sum(1 for c in granule_valid_pixels if c == 0)
            ),
            "station_window_valid_pixels": {
                "count": len(valid_pixels),
                "min": int(np.min(valid_pixels)) if valid_pixels else None,
                "median": float(np.median(valid_pixels)) if valid_pixels else None,
                "mean": (
                    round(float(np.mean(valid_pixels)), 3) if valid_pixels else None
                ),
                "max": int(np.max(valid_pixels)) if valid_pixels else None,
            },
            "aod_value_range_observed": (
                [
                    round(float(np.min(over_values)), 6),
                    round(float(np.max(over_values)), 6),
                ]
                if over_values
                else None
            ),
            "sample_file_attributes": attrs_of_record,
        }
    )
    return out


# ---------------------------------------------------------------------------
# MERRA-2
# ---------------------------------------------------------------------------


def audit_merra(root: Path) -> dict:
    merra_dir = root / "data" / "merra2"
    files = sorted(merra_dir.glob("*.nc"))
    out: dict = {
        "directory": str(merra_dir.relative_to(root)),
        "file_count": len(files),
        "status": "MISSING",
    }
    if not files:
        return out

    import xarray as xr

    missing_counts = {var: 0 for var in MERRA_REQUESTED}
    grids = set()
    time_signatures = set()
    var_units: dict = {}
    dates = []
    for path in files:
        stem = re.search(r"Nx\.(\d{8})", path.name)
        if stem:
            dates.append(datetime.strptime(stem.group(1), "%Y%m%d").date().isoformat())
        with xr.open_dataset(path) as ds:
            grids.add((float(ds.lat.values[0]), float(ds.lon.values[0])))
            time_signatures.add(
                tuple(pd.to_datetime(ds["time"].values).strftime("%H:%M").tolist())
            )
            for var in MERRA_REQUESTED:
                if var not in ds:
                    missing_counts[var] += 1
                    continue
                if var not in var_units:
                    attrs = ds[var].attrs
                    var_units[var] = {
                        "units": attrs.get("units"),
                        "long_name": attrs.get("long_name"),
                        "standard_name": attrs.get("standard_name"),
                    }
            fallback_hits = {
                var: fb
                for var, fbs in MERRA_FALLBACKS.items()
                for fb in fbs
                if fb in ds
            }

    out.update(
        {
            "status": "OK",
            "date_range": [min(dates), max(dates)] if dates else None,
            "distinct_dates": len(set(dates)),
            "grid_points": [list(g) for g in sorted(grids)],
            "time_signatures_utc": [list(t) for t in sorted(time_signatures)],
            "timesteps_per_day": (
                len(next(iter(time_signatures))) if time_signatures else None
            ),
            "variables_present_in_any_granule": sorted(
                set(var_units) | {"PBLH"} - {"PBLH"}
            ),
            "requested_variable_availability": missing_counts,
            "requested_variables_missing_in_all_granules": [
                var for var, count in missing_counts.items() if count == len(files)
            ],
            "fallback_substitutions_actually_triggered": fallback_hits or None,
            "variable_metadata": var_units,
            "product": "MERRA2_400.tavg1_2d_slv_Nx (2-daily, 1-hourly, single-level)",
            "is_reanalysis": True,
            "is_forecast_available_at_issue_time": False,
        }
    )
    return out


# ---------------------------------------------------------------------------
# Cross-source alignment
# ---------------------------------------------------------------------------


def audit_alignment(root: Path) -> dict:
    """Check how much of the fused modelling frame rests on real observations.

    Mirrors the chapter pipeline's ``harmonise`` (including its 3-day forward
    fill) so the audit reports the behaviour of the shipped baseline, which
    lives in ``legacy/``.
    """
    import sys

    sys.path.insert(0, str(root / "legacy"))
    import aerosphinx_pipeline as ap

    cpcb = ap.load_cpcb_daily(root / "data" / "cpcb")
    aod = ap.load_aod_daily([root / "data" / "aod"])
    merra, _ = ap.load_merra_daily(root / "data" / "merra2")
    combined = ap.engineer_features(ap.harmonise(cpcb, aod, merra))
    features, unavailable = ap.select_available_features(
        combined, ap.CONFIGURATIONS[ap.DEFAULT_CONFIG]
    )
    model_df = (
        combined.dropna(subset=features + [ap.TARGET])
        .sort_values("Date")
        .reset_index(drop=True)
    )

    real_aod = set(pd.to_datetime(aod["Date"]))
    real_merra = set(pd.to_datetime(merra["Date"]))
    aod_obs = model_df["Date"].isin(real_aod)
    merra_obs = model_df["Date"].isin(real_merra)

    return {
        "modelling_rows": int(len(model_df)),
        "modelling_date_range": [
            model_df["Date"].min().date().isoformat(),
            model_df["Date"].max().date().isoformat(),
        ],
        "features_used": features,
        "features_unavailable": unavailable,
        "rows_with_genuine_aod": int(aod_obs.sum()),
        "rows_with_forward_filled_aod": int((~aod_obs).sum()),
        "forward_filled_aod_fraction": round(float((~aod_obs).mean()), 4),
        "rows_with_genuine_merra": int(merra_obs.sum()),
        "rows_with_forward_filled_merra": int((~merra_obs).sum()),
        "forward_fill_limit_days": FORWARD_FILL_LIMIT_DAYS,
        "cpcb_daily_rows_total": int(len(cpcb)),
        "aod_daily_rows": int(len(aod)),
        "merra_daily_rows": int(len(merra)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_path = args.out or (root / "results" / "baseline" / "data_audit.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "root": str(root),
        "cpcb": audit_cpcb(root),
        "insat_aod": audit_aod(root),
        "merra2": audit_merra(root),
    }
    try:
        report["alignment"] = audit_alignment(root)
    except Exception as exc:  # pragma: no cover - audit must never crash the run
        report["alignment"] = {"status": "ERROR", "error": repr(exc)}

    out_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    print(f"data audit written -> {out_path}")
    for section in ("cpcb", "insat_aod", "merra2", "alignment"):
        print(f"  {section}: {report[section].get('status', 'OK')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
