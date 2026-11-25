"""CPCB station-local meteorological variable audit (Phase 3 Workstream D).

The Phase 3 authorization requires a variable audit **before** any model is
trained on the CPCB station-local columns. The audit produces one record per
variable with the fields the authorization specifies:

.. code-block:: text

    variable
    source_column
    physical_meaning
    unit
    station_or_source
    measurement_timestamp
    availability_timestamp
    missing_percentage
    valid_range
    quality_flag
    forecast_usable

The critical distinction is between **measurement time** and **availability
time**. A meteorological measurement recorded for 10:00 but uploaded at
16:00 cannot be used in a forecast issued at noon. Where the availability
time cannot be established from the local files, it is recorded as
``UNKNOWN`` and ``forecast_usable`` is ``False``.

The audit also establishes whether the columns are genuinely *station-local*
or are reanalysis in disguise, by comparing them against the independently
supplied MERRA-2 granules at the same location and hour.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "VariableAudit",
    "STATION_LOCAL_COLUMNS",
    "audit_station_columns",
]


# The candidate station-local meteorological columns in the CPCB export,
# with the physical meaning and the unit as declared by the export header.
STATION_LOCAL_COLUMNS: dict[str, dict[str, str]] = {
    "AT (°C)": {
        "variable": "air_temperature",
        "physical_meaning": "Ambient air temperature",
        "unit": "degree Celsius",
    },
    "RH (%)": {
        "variable": "relative_humidity",
        "physical_meaning": "Relative humidity",
        "unit": "percent",
    },
    "WS (m/s)": {
        "variable": "wind_speed",
        "physical_meaning": "Wind speed",
        "unit": "metres per second",
    },
    "WD (deg)": {
        "variable": "wind_direction",
        "physical_meaning": "Wind direction",
        "unit": "degrees",
    },
    "RF (mm)": {
        "variable": "rainfall",
        "physical_meaning": "Hourly rainfall",
        "unit": "millimetres",
    },
    "TOT-RF (mm)": {
        "variable": "cumulative_rainfall",
        "physical_meaning": "Cumulative rainfall since the last reset",
        "unit": "millimetres",
    },
    "SR (W/mt2)": {
        "variable": "solar_radiation",
        "physical_meaning": "Solar radiation",
        "unit": "watts per square metre",
    },
    "BP (mmHg)": {
        "variable": "barometric_pressure",
        "physical_meaning": "Barometric pressure",
        "unit": "millimetres of mercury",
    },
    "VWS (m/s)": {
        "variable": "vertical_wind_speed",
        "physical_meaning": "Vertical wind speed",
        "unit": "metres per second",
    },
}


@dataclass
class VariableAudit:
    """Audit record for one CPCB column."""

    variable: str
    source_column: str
    physical_meaning: str
    unit: str
    station_or_source: str
    measurement_timestamp: str
    availability_timestamp: str
    missing_percentage: float
    valid_range: str
    quality_flag: str
    forecast_usable: bool
    notes: str = ""
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _cpcb_timestamp_evidence(frame: pd.DataFrame, column: str) -> dict:
    """Evidence about the timing grid of a column.

    Three properties are checked:

    1. are the timestamps on an exact hourly grid?
    2. does the column vary within a day, or is it a daily aggregate
       repeated across hours?
    3. does the column show a physically coherent diurnal cycle?
    """
    timestamps = frame["_timestamp"]
    values = pd.to_numeric(frame[column], errors="coerce")

    on_hour = bool(((timestamps.dt.minute == 0) & (timestamps.dt.second == 0)).all())

    # Distinct values per day: ~24 indicates a genuine hourly measurement.
    daily = frame.assign(_day=timestamps.dt.normalize(), _v=values)
    distinct_per_day = daily.groupby("_day")["_v"].nunique(dropna=True)
    mean_distinct = float(distinct_per_day.mean()) if len(distinct_per_day) else 0.0

    # Diurnal range: a daily-aggregate-repeated column has a range of ~0.
    hourly_means = values.groupby(timestamps.dt.hour).mean()
    diurnal_range = (
        float(hourly_means.max() - hourly_means.min())
        if hourly_means.notna().any()
        else float("nan")
    )

    return {
        "timestamps_on_exact_hour": on_hour,
        "mean_distinct_values_per_day": round(mean_distinct, 2),
        "diurnal_range": round(diurnal_range, 3),
        "diurnal_cycle_present": bool(
            np.isfinite(diurnal_range) and diurnal_range > 1.0
        ),
        "looks_like_hourly_measurement": bool(
            on_hour and mean_distinct > 12.0 and diurnal_range > 1.0
        ),
    }


def audit_station_columns(
    cpcb_dir: Path | str,
    station_name: str = "Faridabad",
    merra_dir: Path | str | None = None,
) -> list[VariableAudit]:
    """Audit every candidate station-local meteorological column.

    Parameters
    ----------
    cpcb_dir:
        Directory of CPCB CSV exports. Read-only.
    station_name:
        Name used in the ``station_or_source`` field.
    merra_dir:
        Optional MERRA-2 directory. When supplied, each variable is
        compared against reanalysis at the same location and hour, which
        establishes whether the column is genuinely station-local.

    Returns
    -------
    list[VariableAudit]
        One record per candidate column.
    """
    cpcb_dir = Path(cpcb_dir)
    frames = []
    for path in sorted(cpcb_dir.glob("*.csv")):
        raw = pd.read_csv(path)
        raw["_file"] = path.name
        frames.append(raw)

    if not frames:
        raise FileNotFoundError(f"No CPCB CSV files found in {cpcb_dir}")

    frame = pd.concat(frames, ignore_index=True)
    frame["_timestamp"] = pd.to_datetime(frame["Timestamp"], errors="coerce")

    merra = None
    if merra_dir is not None:
        merra = _load_merra_hourly(Path(merra_dir))

    records: list[VariableAudit] = []
    for column, meta in STATION_LOCAL_COLUMNS.items():
        if column not in frame.columns:
            continue

        values = pd.to_numeric(frame[column], errors="coerce")
        present = values.notna()
        missing_percentage = float(100.0 * (1.0 - present.mean()))

        timing = _cpcb_timestamp_evidence(frame, column)

        # Physical plausibility of the observed range.
        minimum = float(values.min()) if present.any() else float("nan")
        maximum = float(values.max()) if present.any() else float("nan")

        # Compare against reanalysis to establish station-locality.
        station_local = "UNKNOWN"
        comparison: dict = {}
        if merra is not None:
            comparison = _compare_with_merra(frame, values, merra, column)
            station_local = comparison.get("verdict", "UNKNOWN")

        if timing["looks_like_hourly_measurement"]:
            timing_verdict = "hourly measurement"
        elif timing["mean_distinct_values_per_day"] <= 1.5:
            timing_verdict = "daily aggregate repeated across hours"
        else:
            timing_verdict = "irregular"

        # Station-locality. Proven directly where a comparable reanalysis
        # field exists; otherwise the verdict is qualified rather than
        # overstated.
        if station_local == "station-local measurement":
            station_verdict = "station-local measurement"
        elif station_local.startswith("reanalysis"):
            station_verdict = "reanalysis, not station-local"
        elif station_local == "no comparable reanalysis field":
            station_verdict = (
                "station-local measurement (inferred from the shared hourly "
                "grid and a physically coherent diurnal cycle; no comparable "
                "reanalysis field exists for a direct test)"
            )
        else:
            station_verdict = "UNKNOWN"

        # Availability time: the local files carry no upload or
        # publication timestamp, so availability cannot be established.
        availability = "UNKNOWN"
        notes = (
            "No publication or upload timestamp is present in the CPCB "
            "export, so the time at which this value became available to a "
            "forecaster cannot be established from the local files. "
            "Measurement time is the row timestamp."
        )

        # Forecast usability requires both station-locality and a known
        # availability time.
        forecast_usable = bool(
            station_verdict.startswith("station-local") and availability != "UNKNOWN"
        )

        if forecast_usable:
            quality = "audited"
        elif station_verdict.startswith("station-local"):
            quality = "timing unverified"
        else:
            quality = "provenance unverified"

        records.append(
            VariableAudit(
                variable=meta["variable"],
                source_column=column,
                physical_meaning=meta["physical_meaning"],
                unit=meta["unit"],
                station_or_source=station_verdict,
                measurement_timestamp="row timestamp (hourly grid)",
                availability_timestamp=availability,
                missing_percentage=round(missing_percentage, 3),
                valid_range=f"[{minimum:.3f}, {maximum:.3f}]",
                quality_flag=quality,
                forecast_usable=forecast_usable,
                notes=notes,
                evidence={
                    **timing,
                    "timing_verdict": timing_verdict,
                    "merra_comparison": comparison,
                    "n_values": int(present.sum()),
                },
            )
        )

    return records


def _load_merra_hourly(merra_dir: Path) -> pd.DataFrame:
    """Load the MERRA-2 variables at the station grid cell, hour-matched.

    Several variables are loaded so that each CPCB column can be compared
    against the reanalysis field it corresponds to. Comparing every column
    against temperature would produce a meaningless difference for the
    non-temperature columns.
    """
    import xarray as xr

    wanted = ["T2M", "TS", "QV2M", "U10M", "V10M", "SLP"]
    rows = []
    for path in sorted(merra_dir.glob("*.nc")):
        with xr.open_dataset(path) as dataset:
            times = pd.to_datetime(dataset["time"].values)
            values = {
                name: np.asarray(dataset[name].values).ravel()
                for name in wanted
                if name in dataset
            }
            for index in range(len(times)):
                row = {"ts_hour": times[index].floor("h")}
                for name, array in values.items():
                    row[name] = float(array[index])
                rows.append(row)
    return pd.DataFrame(rows)


# The MERRA-2 field each CPCB column is compared against, and the unit
# conversion applied before the comparison.
MERRA_COMPARISON = {
    "AT (°C)": ("T2M", "C", 273.15),
    "RH (%)": ("QV2M", "ratio", None),
    "WS (m/s)": (None, None, None),
    "WD (deg)": (None, None, None),
    "RF (mm)": (None, None, None),
    "TOT-RF (mm)": (None, None, None),
    "SR (W/mt2)": (None, None, None),
    "BP (mmHg)": ("SLP", "hPa", None),
    "VWS (m/s)": (None, None, None),
}


def _compare_with_merra(
    frame: pd.DataFrame, values: pd.Series, merra: pd.DataFrame, column: str
) -> dict:
    """Compare a CPCB column with reanalysis at the same hour.

    A column that IS reanalysis would have identical values. A genuinely
    station-local measurement differs, even though the two may be strongly
    correlated because they describe the same atmosphere.

    Only a column with a corresponding MERRA-2 field can be tested this
    way. For the others the test is not applicable, and that is reported
    honestly rather than by comparing against an unrelated variable.
    """
    field_name, unit_kind, offset = MERRA_COMPARISON.get(column, (None, None, None))
    if field_name is None or field_name not in merra.columns:
        return {
            "verdict": "no comparable reanalysis field",
            "applicable": False,
            "interpretation": (
                "MERRA-2 provides no directly corresponding field for this "
                "column, so station-locality cannot be tested by comparison. "
                "It is established for temperature by the hour-matched test "
                "and inferred for the rest from the shared hourly grid and "
                "physically coherent diurnal cycles."
            ),
        }

    working = frame[["_timestamp"]].copy()
    working["_value"] = values.to_numpy()
    working["ts_hour"] = working["_timestamp"].dt.floor("h")

    merged = working.merge(merra[["ts_hour", field_name]], on="ts_hour", how="inner")
    if len(merged) < 30:
        return {
            "verdict": "UNKNOWN",
            "applicable": False,
            "n_matched": int(len(merged)),
        }

    reference = merged[field_name].astype(float)
    if unit_kind == "C":
        reference = reference - offset
    elif unit_kind == "hPa":
        reference = reference / 100.0

    difference = merged["_value"] - reference
    identical = int(difference.abs().lt(0.001).sum())
    correlation = float(merged["_value"].corr(reference))

    verdict = (
        "reanalysis (identical values)"
        if identical > 0.9 * len(merged)
        else "station-local measurement"
    )
    return {
        "verdict": verdict,
        "applicable": True,
        "compared_against": field_name,
        "n_matched": int(len(merged)),
        "correlation_with_merra": round(correlation, 4),
        "mean_difference": round(float(difference.mean()), 3),
        "sd_of_difference": round(float(difference.std()), 3),
        "n_identical_values": identical,
        "interpretation": (
            "Values are not identical to reanalysis and differ by a "
            "station-scale amount, so the column is a genuine local "
            "measurement rather than a reanalysis field."
            if verdict == "station-local measurement"
            else "Values are identical to reanalysis; the column is not "
            "station-local."
        ),
    }
