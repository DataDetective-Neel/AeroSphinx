"""MERRA-2 reanalysis validation (Phase 5).

Phase 5 is a **short validation**, not a build. Its purpose is to establish
whether the distributed MERRA-2 granules are what they claim to be, and to
quantify the coverage limit that makes the group untestable on the current
record. No feature-group infrastructure is added.

What the validation establishes
-------------------------------
**Structure.** All 172 granules have dimensions ``lat=1, lon=1, time=3`` and
the same twelve variables, with uniform CF-1 units and no missing values.

**Units.** Every variable's declared unit matches the expected physical
unit, and every value lies in a physically plausible range.

**Grid.** A single 1x1 cell at 28.5 N, 77.5 E — the MERRA-2 0.5 x 0.625
degree grid. The offset from the station is about 10 km north and 18 km east.

**Temporal coverage.** 2024-01-01 to 2024-06-20, **172 consecutive days
with no missing day**, three hourly timesteps per day at 11:30, 12:30 and
13:30 UTC.

**The timezone correction that matters.** Those three UTC timesteps are
**17:00, 18:00 and 19:00 IST**. The value is an *evening* window at the
station, not the Indian midday when boundary-layer mixing is strongest and
when the daily PM2.5 maximum usually occurs. This is the concrete form of
D-010 and it is worse than "not a daily mean": the sample hours are
systematically offset from the diurnal peak.

**Physical cross-validation.** The MERRA-2 fields are compared against the
CPCB station's own measurements at the same instants. This is the MERRA
analogue of the ocean-versus-land test that proved the INSAT-3D orientation:
it confirms that the variables are the physical quantities they name, and
it quantifies the grid-versus-station difference that any use must account
for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "MERRA_EXPECTED_UNITS",
    "MERRA_PLAUSIBLE_RANGES",
    "MerraAudit",
    "validate_merra",
    "station_cross_check",
]

#: The declared CF unit each variable is expected to carry.
MERRA_EXPECTED_UNITS = {
    "T2M": "K",
    "T10M": "K",
    "TS": "K",
    "T2MDEW": "K",
    "QV2M": "kg kg-1",
    "QV10M": "kg kg-1",
    "U10M": "m s-1",
    "V10M": "m s-1",
    "SLP": "Pa",
    "PS": "Pa",
    "TQL": "kg m-2",
    "TQI": "kg m-2",
}

#: Physically plausible ranges, used as a value-level sanity screen.
MERRA_PLAUSIBLE_RANGES = {
    "T2M": (180.0, 340.0),
    "T10M": (180.0, 340.0),
    "TS": (180.0, 350.0),
    "T2MDEW": (180.0, 320.0),
    "QV2M": (0.0001, 0.060),
    "QV10M": (0.0001, 0.060),
    "U10M": (-80.0, 80.0),
    "V10M": (-80.0, 80.0),
    "SLP": (80000.0, 110000.0),
    "PS": (80000.0, 110000.0),
    "TQL": (0.0, 80.0),
    "TQI": (0.0, 80.0),
}

#: The station's UTC offset, used to align MERRA UTC timesteps with the
#: local-time CPCB observations.
STATION_UTC_OFFSET_HOURS = 5.5


@dataclass
class MerraAudit:
    """Structural and coverage audit of the MERRA-2 granules."""

    n_granules: int = 0
    dimension_signatures: dict = field(default_factory=dict)
    variable_presence: dict = field(default_factory=dict)
    declared_units: dict = field(default_factory=dict)
    unit_mismatches: dict = field(default_factory=dict)
    out_of_range_values: dict = field(default_factory=dict)
    grid_cells: dict = field(default_factory=dict)
    time_signatures: dict = field(default_factory=dict)
    days_per_granule: dict = field(default_factory=dict)
    n_missing_values: int = 0
    date_start: str | None = None
    date_end: str | None = None
    n_dates: int = 0
    n_missing_dates: int = 0
    missing_dates: list = field(default_factory=list)
    subsetting_boxes: dict = field(default_factory=dict)
    local_hours: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_granules": self.n_granules,
            "dimension_signatures": {
                str(k): v for k, v in self.dimension_signatures.items()
            },
            "variable_presence": dict(sorted(self.variable_presence.items())),
            "declared_units": {
                k: sorted(v) for k, v in sorted(self.declared_units.items())
            },
            "unit_mismatches": dict(sorted(self.unit_mismatches.items())),
            "out_of_range_values": dict(sorted(self.out_of_range_values.items())),
            "grid_cells": {str(k): v for k, v in sorted(self.grid_cells.items())},
            "time_signatures": {
                str(k): v for k, v in sorted(self.time_signatures.items())
            },
            "days_per_granule": dict(sorted(self.days_per_granule.items())),
            "n_missing_values": self.n_missing_values,
            "date_start": self.date_start,
            "date_end": self.date_end,
            "n_dates": self.n_dates,
            "n_missing_dates": self.n_missing_dates,
            "missing_dates": self.missing_dates,
            "subsetting_boxes": dict(sorted(self.subsetting_boxes.items())),
            "local_hours_ist": self.local_hours,
        }


def _parse_subset_box(history: str) -> str:
    """Extract the spatial subset box from the L34RS provenance string."""
    marker = "Spatial:"
    if marker not in history:
        return "unknown"
    remainder = history.split(marker, 1)[1]
    return remainder.split("Variables:", 1)[0].strip()


def validate_merra(merra_dir: Path | str) -> MerraAudit:
    """Validate the structure, units, grid and temporal coverage of the granules.

    Every granule is inspected. The audit records what is uniform and what
    varies, and screens every value against a physically plausible range.
    """
    import xarray as xr

    merra_dir = Path(merra_dir)
    files = sorted(merra_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No MERRA-2 granules found in {merra_dir}")

    audit = MerraAudit(n_granules=len(files))
    dates: set[pd.Timestamp] = set()
    local_hours: set[str] = set()

    for path in files:
        with xr.open_dataset(path) as dataset:
            signature = tuple(
                (name, int(size)) for name, size in sorted(dataset.sizes.items())
            )
            audit.dimension_signatures[signature] = (
                audit.dimension_signatures.get(signature, 0) + 1
            )

            for name in dataset.data_vars:
                audit.variable_presence[name] = audit.variable_presence.get(name, 0) + 1
                unit = str(dataset[name].attrs.get("units"))
                audit.declared_units.setdefault(name, set()).add(unit)
                expected = MERRA_EXPECTED_UNITS.get(name)
                if expected is not None and unit != expected:
                    audit.unit_mismatches.setdefault(name, []).append(
                        {"file": path.name, "declared": unit, "expected": expected}
                    )

                values = np.asarray(dataset[name].values, dtype=float)
                if np.isnan(values).any():
                    audit.n_missing_values += int(np.isnan(values).sum())
                plausible = MERRA_PLAUSIBLE_RANGES.get(name)
                if plausible is not None:
                    low, high = plausible
                    outside = int(np.sum((values < low) | (values > high)))
                    if outside:
                        audit.out_of_range_values[name] = (
                            audit.out_of_range_values.get(name, 0) + outside
                        )

            grid = (
                float(dataset["lat"].values.ravel()[0]),
                float(dataset["lon"].values.ravel()[0]),
            )
            audit.grid_cells[grid] = audit.grid_cells.get(grid, 0) + 1

            times = pd.to_datetime(dataset["time"].values)
            signature_time = tuple(times.strftime("%H:%M").tolist())
            audit.time_signatures[signature_time] = (
                audit.time_signatures.get(signature_time, 0) + 1
            )
            unique_days = len(set(times.normalize()))
            audit.days_per_granule[unique_days] = (
                audit.days_per_granule.get(unique_days, 0) + 1
            )
            for timestamp in times:
                local = timestamp + pd.Timedelta(hours=STATION_UTC_OFFSET_HOURS)
                local_hours.add(local.strftime("%H:%M"))
                dates.add(timestamp.normalize())

            box = _parse_subset_box(str(dataset.attrs.get("history_L34RS", "")))
            audit.subsetting_boxes[box] = audit.subsetting_boxes.get(box, 0) + 1

    ordered = sorted(dates)
    audit.date_start = ordered[0].date().isoformat()
    audit.date_end = ordered[-1].date().isoformat()
    audit.n_dates = len(ordered)

    full = pd.date_range(ordered[0], ordered[-1], freq="D")
    missing = [d for d in full if d not in set(ordered)]
    audit.n_missing_dates = len(missing)
    audit.missing_dates = [d.date().isoformat() for d in missing]
    audit.local_hours = sorted(local_hours)

    return audit


def station_cross_check(
    merra_dir: Path | str,
    cpcb_dir: Path | str,
    station_lat: float = 28.4089,
    station_lon: float = 77.3178,
) -> pd.DataFrame:
    """Compare the MERRA-2 fields against the CPCB station at the same instants.

    This is the physical cross-validation. The MERRA-2 fields and the
    station's own measurements describe the same atmosphere at the same
    instants, so they must be strongly correlated. The differences are the
    grid-versus-station offset and are reported rather than hidden.

    The time alignment is explicit: MERRA-2 timestamps are UTC and the CPCB
    timestamps are local IST, so the station value is matched at
    ``utc_time + 5h30``.
    """
    import xarray as xr

    # The station's own measurements.
    frames = []
    for path in sorted(Path(cpcb_dir).glob("*.csv")):
        raw = pd.read_csv(path)
        raw["_timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
        frames.append(raw[["_timestamp", "AT (°C)", "RH (%)", "BP (mmHg)"]])
    station = pd.concat(frames, ignore_index=True)
    station = station.dropna(subset=["_timestamp"])
    station = station.rename(
        columns={
            "AT (°C)": "station_AT_C",
            "RH (%)": "station_RH",
            "BP (mmHg)": "station_BP_mmHg",
        }
    )

    rows = []
    for path in sorted(Path(merra_dir).glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            times = pd.to_datetime(dataset["time"].values)
            for index, timestamp in enumerate(times):
                local = timestamp + pd.Timedelta(hours=STATION_UTC_OFFSET_HOURS)
                match = station[station["_timestamp"] == local]
                if match.empty:
                    continue
                station_row = match.iloc[0]
                record = {
                    "utc_time": timestamp,
                    "local_time": local,
                    "T2M_K": float(np.asarray(dataset["T2M"].values).ravel()[index]),
                    "TS_K": float(np.asarray(dataset["TS"].values).ravel()[index]),
                    "QV2M": float(np.asarray(dataset["QV2M"].values).ravel()[index]),
                    "SLP_Pa": float(np.asarray(dataset["SLP"].values).ravel()[index]),
                    "PS_Pa": float(np.asarray(dataset["PS"].values).ravel()[index]),
                    "station_AT_C": float(station_row["station_AT_C"]),
                    "station_RH": float(station_row["station_RH"]),
                    "station_BP_mmHg": float(station_row["station_BP_mmHg"]),
                }
                # The station's specific humidity implied by its RH, the
                # MERRA-2 temperature and the MERRA-2 surface pressure:
                #     q = 0.622 * e / P   (e in hPa, P in hPa)
                # The Magnus formula takes temperature in **Celsius**.
                temperature_c = record["T2M_K"] - 273.15
                saturation = 6.112 * np.exp(
                    (17.67 * temperature_c) / (temperature_c + 243.5)
                )
                vapour = (record["station_RH"] / 100.0) * saturation
                record["station_q_implied"] = 0.622 * vapour / (record["PS_Pa"] / 100.0)
                # Sea-level minus surface pressure is the elevation term.
                record["SLP_minus_PS_hPa"] = (
                    record["SLP_Pa"] - record["PS_Pa"]
                ) / 100.0
                # The station's barometric pressure in hectopascals.
                record["station_BP_hPa"] = record["station_BP_mmHg"] * 1.33322
                rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["T2M_C"] = frame["T2M_K"] - 273.15
    frame["TS_C"] = frame["TS_K"] - 273.15
    frame["T_minus_station_C"] = frame["T2M_C"] - frame["station_AT_C"]
    frame["SLP_minus_station_hPa"] = frame["SLP_Pa"] / 100.0 - frame["station_BP_hPa"]
    return frame
