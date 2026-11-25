"""MERRA-2 reanalysis ingestion (Phase 3 Workstream C).

The Phase 0 audit established **D-010**: the distributed MERRA-2 granules
contain a **single 1x1 grid cell** at (28.5 N, 77.5 E) with **three
hourly timesteps** (11:30, 12:30 and 13:30 UTC). They are not daily means
over a collocation box, and the legacy pipeline's docstrings said so
incorrectly.

This module corrects the naming as well as the extraction. Every variable
carries a ``_midday`` suffix, because that is what the value is. This
closes the naming half of D-010; the full Phase 5 validation of the
extraction remains Phase 5 work.

**Availability.** MERRA-2 is reanalysis. It is available for
retrospective estimation and is **never** available for an operational
forecast, because reanalysis is an assimilation product for a past period
rather than a forecast of a future one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["MerraDaily", "load_merra_daily", "MERRA_VARIABLES"]

#: The variables extracted, with the granule dimensions they come from.
MERRA_VARIABLES = ("T2M", "TS", "QV2M", "U10M", "V10M", "SLP")

#: The daily column name for each MERRA variable. The suffix is not
#: cosmetic: the value is a midday-window mean, not a daily mean.
COLUMN_NAMES = {name: f"{name}_midday" for name in MERRA_VARIABLES}


@dataclass
class MerraDaily:
    """Daily MERRA-2 values plus their provenance."""

    frame: pd.DataFrame
    grid_cell: tuple[float, float]
    timesteps_utc: list[str]
    requested_missing: dict = field(default_factory=dict)
    availability_timestamp: str = "past-date reanalysis; not available at issue time"
    forecast_usable: bool = False
    notes: str = ""


def load_merra_daily(merra_dir: Path | str) -> MerraDaily:
    """Read the MERRA-2 granules into a daily frame.

    Parameters
    ----------
    merra_dir:
        Directory of MERRA-2 NetCDF granules. Read-only.

    Returns
    -------
    MerraDaily
        The daily frame and the provenance of the extraction.
    """
    import xarray as xr

    merra_dir = Path(merra_dir)
    files = sorted(merra_dir.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"No MERRA-2 NetCDF files found in {merra_dir}")

    rows = []
    grid_cell: tuple[float, float] | None = None
    timesteps: set[str] = set()
    missing_counts = {name: 0 for name in MERRA_VARIABLES}

    for path in files:
        with xr.open_dataset(path) as dataset:
            times = pd.to_datetime(dataset["time"].values)
            if grid_cell is None:
                grid_cell = (
                    float(dataset["lat"].values.ravel()[0]),
                    float(dataset["lon"].values.ravel()[0]),
                )
            timesteps.update(times.strftime("%H:%M").tolist())

            row: dict = {"Date": times[0].normalize()}
            for name in MERRA_VARIABLES:
                if name not in dataset:
                    missing_counts[name] += 1
                    row[COLUMN_NAMES[name]] = np.nan
                    continue
                values = np.asarray(dataset[name].values).ravel()
                row[COLUMN_NAMES[name]] = float(np.nanmean(values))
            rows.append(row)

    frame = pd.DataFrame(rows).groupby("Date", as_index=False).mean(numeric_only=True)
    frame = frame.sort_values("Date").reset_index(drop=True)

    return MerraDaily(
        frame=frame,
        grid_cell=grid_cell or (float("nan"), float("nan")),
        timesteps_utc=sorted(timesteps),
        requested_missing={
            name: count for name, count in missing_counts.items() if count
        },
        notes=(
            "RETROSPECTIVE ONLY, and never operational. Values are the mean "
            "over the hourly timesteps present in the distributed granules "
            f"({', '.join(sorted(timesteps))} UTC) at a single 1x1 grid cell "
            f"({grid_cell[0]:.2f} N, {grid_cell[1]:.2f} E). They are midday "
            "window means, not daily means over a collocation box."
        ),
    )
