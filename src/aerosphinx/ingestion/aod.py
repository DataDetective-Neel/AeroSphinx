"""INSAT-3D aerosol optical depth ingestion and validation (Phase 4).

This module is the validated replacement for the legacy AOD extraction that
previously lived in ``aerosphinx_pipeline.load_aod_daily``.

What Phase 4 established
-------------------------
Every one of the 169 distributed L2G granules was inspected, not a sample:

* **Structure is uniform.** All 169 have AOD shape ``(1, 551, 551)``,
  dtype float32, with exactly the datasets ``AOD``, ``latitude``,
  ``longitude``, ``time``.
* **No scale factor or offset.** Every granule has
  ``scale_factor = None`` and ``add_offset = None``, so the stored values
  are used directly.
* **Fill value is ``-999.0``** in every granule.
* **Axis orientation is verified twice, independently.**
  1. The HDF5 dimension-scale bindings are ``dim1 -> /latitude`` and
     ``dim2 -> /longitude`` in all 169 granules.
  2. A *physical* cross-check: aerosol optical depth over the open ocean is
     systematically lower than over polluted continental land. Under the
     dimension-scale orientation the ocean/land contrast is **+0.447**
     (ocean 0.385, land 0.832), which is correct. Under the transposed
     orientation the contrast is **-0.248**, which is physically
     impossible. The two lines of evidence agree.
* **The latitude axis descends** (45.05 to -9.95) while longitude ascends
  (45.05 to 100.05), both at a uniform 0.1 degree step. Boolean masking is
  order-independent, so the descending latitude is not a hazard, but it is
  recorded rather than assumed.
* **There is no QA, cloud or quality-flag dataset** in these granules. A
  present pixel therefore means "a retrieval was attempted and returned a
  number", not "the retrieval is trustworthy". The valid-pixel count is
  reported per date so the reader can judge the support.

Extraction
----------
The station window is a square in degrees, centred on the station, with a
half-width of 0.25 degrees, giving at most 5 x 5 = 25 pixels. The mean of
the valid pixels in that window is the daily value.

The window half-width is a **configuration**, not a constant, because the
choice is a scientific one: the sensitivity of the extracted value to the
window size is reported by :func:`window_sensitivity`.

Duplicates
----------
Granules are keyed by (date, overpass time). Two granules for the same
date with *different* overpass times are two observations of that date, and
their valid pixels are averaged. Two granules with the *same* date and
overpass time would be a duplicate, and are reported rather than silently
double-counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "AODRules",
    "AODGranule",
    "AODDaily",
    "GranuleAudit",
    "load_aod_daily",
    "inspect_granules",
    "orientation_check",
    "window_sensitivity",
]


@dataclass(frozen=True)
class AODRules:
    """Quality and extraction rules for the INSAT-3D L2G AOD product."""

    fill_value: float = -999.0
    aod_valid_min: float = 0.0
    aod_valid_max: float = 5.0
    #: Half-width of the square collocation window, in degrees.
    window_half_width_deg: float = 0.25
    #: Minimum valid pixels for a day to be reported.
    min_valid_pixels: int = 1
    #: Whether several same-day overpasses are averaged into one value.
    average_overpasses: bool = True

    def __post_init__(self) -> None:
        if self.window_half_width_deg <= 0:
            raise ValueError("window_half_width_deg must be positive")
        if self.aod_valid_min >= self.aod_valid_max:
            raise ValueError("aod_valid_min must be below aod_valid_max")


@dataclass
class AODGranule:
    """One granule's metadata."""

    path: Path
    date: pd.Timestamp
    overpass_utc: str
    start_time: str
    end_time: str
    shape: tuple
    fill_value: float
    dimension_scales: tuple
    has_scale_factor: bool
    datasets: tuple

    @property
    def key(self) -> tuple:
        """The (date, overpass) key used for duplicate detection."""
        return (self.date, self.overpass_utc)


@dataclass
class GranuleAudit:
    """Structural audit of a set of granules."""

    n_granules: int = 0
    shapes: dict = field(default_factory=dict)
    datasets: dict = field(default_factory=dict)
    dimension_scale_bindings: dict = field(default_factory=dict)
    fill_values: dict = field(default_factory=dict)
    scale_factors: dict = field(default_factory=dict)
    latitude_orders: dict = field(default_factory=dict)
    longitude_orders: dict = field(default_factory=dict)
    distinct_dates: int = 0
    duplicate_keys: list = field(default_factory=list)
    granules_with_zero_valid_pixels: int = 0
    has_qa_dataset: bool = False

    def as_dict(self) -> dict:
        return {
            "n_granules": self.n_granules,
            "shapes": {str(k): v for k, v in self.shapes.items()},
            "datasets": {str(k): v for k, v in self.datasets.items()},
            "dimension_scale_bindings": {
                str(k): v for k, v in self.dimension_scale_bindings.items()
            },
            "fill_values": {str(k): v for k, v in self.fill_values.items()},
            "scale_factors": {str(k): v for k, v in self.scale_factors.items()},
            "latitude_descending": self.latitude_orders,
            "longitude_ascending": self.longitude_orders,
            "distinct_dates": self.distinct_dates,
            "n_duplicate_date_overpass_keys": len(self.duplicate_keys),
            "duplicate_keys": [list(k) for k in self.duplicate_keys],
            "granules_with_zero_valid_pixels": self.granules_with_zero_valid_pixels,
            "has_qa_dataset": self.has_qa_dataset,
        }


@dataclass
class AODDaily:
    """The daily AOD series plus its audit."""

    frame: pd.DataFrame
    audit: GranuleAudit
    granules: list

    @property
    def n_days(self) -> int:
        return len(self.frame)


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value)


def _parse_date(attributes: dict, path: Path) -> pd.Timestamp | None:
    """The acquisition date, from the attributes or the filename."""
    raw = attributes.get("Acquisition_Date")
    if raw is not None:
        parsed = pd.to_datetime(_decode(raw).strip(), format="%d%b%Y", errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
    match = re.search(r"3DIMG_(\d{2}[A-Z]{3}\d{4})_", path.name.upper())
    if match:
        parsed = pd.to_datetime(match.group(1), format="%d%b%Y", errors="coerce")
        if pd.notna(parsed):
            return parsed.normalize()
    return None


def inspect_granules(
    aod_dir: Path | str, rules: AODRules | None = None
) -> tuple[GranuleAudit, list[AODGranule]]:
    """Audit the structure of every granule in a directory.

    Returns
    -------
    tuple
        The audit and the per-granule metadata.
    """
    import h5py

    rules = rules or AODRules()
    aod_dir = Path(aod_dir)
    files = sorted(aod_dir.glob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No INSAT-3D granules found in {aod_dir}")

    audit = GranuleAudit(n_granules=len(files))
    granules: list[AODGranule] = []
    seen_keys: dict[tuple, int] = {}

    for path in files:
        with h5py.File(path, "r") as handle:
            attributes = {key: _decode(value) for key, value in handle.attrs.items()}
            datasets = tuple(sorted(handle.keys()))
            aod_dataset = handle["AOD"]
            shape = tuple(aod_dataset.shape)

            bindings = []
            for dimension in aod_dataset.dims:
                names = tuple(
                    getattr(dimension[key], "name", None) for key in dimension.keys()
                )
                bindings.append(names)
            bindings = tuple(bindings)

            fill = aod_dataset.attrs.get("_FillValue")
            fill_value = (
                float(np.asarray(fill).ravel()[0])
                if fill is not None
                else rules.fill_value
            )
            has_scale = (
                aod_dataset.attrs.get("scale_factor") is not None
                or aod_dataset.attrs.get("add_offset") is not None
            )

            latitude = np.asarray(handle["latitude"][:], dtype=float)
            longitude = np.asarray(handle["longitude"][:], dtype=float)

        audit.shapes[shape] = audit.shapes.get(shape, 0) + 1
        audit.datasets[datasets] = audit.datasets.get(datasets, 0) + 1
        audit.dimension_scale_bindings[bindings] = (
            audit.dimension_scale_bindings.get(bindings, 0) + 1
        )
        audit.fill_values[fill_value] = audit.fill_values.get(fill_value, 0) + 1
        audit.scale_factors[has_scale] = audit.scale_factors.get(has_scale, 0) + 1

        latitude_descending = (
            bool(latitude[0] > latitude[-1]) if len(latitude) > 1 else False
        )
        longitude_ascending = (
            bool(longitude[0] < longitude[-1]) if len(longitude) > 1 else False
        )
        audit.latitude_orders[latitude_descending] = (
            audit.latitude_orders.get(latitude_descending, 0) + 1
        )
        audit.longitude_orders[longitude_ascending] = (
            audit.longitude_orders.get(longitude_ascending, 0) + 1
        )
        if any(
            name.lower() in {"qa", "quality", "cloud", "flags"} for name in datasets
        ):
            audit.has_qa_dataset = True

        date = _parse_date(attributes, path)
        if date is None:
            continue

        overpass = attributes.get("Acquisition_Time_in_GMT", "")
        granule = AODGranule(
            path=path,
            date=date,
            overpass_utc=str(overpass),
            start_time=str(attributes.get("Acquisition_Start_Time", "")),
            end_time=str(attributes.get("Acquisition_End_Time", "")),
            shape=shape,
            fill_value=fill_value,
            dimension_scales=bindings,
            has_scale_factor=has_scale,
            datasets=datasets,
        )
        key = granule.key
        if key in seen_keys:
            audit.duplicate_keys.append(key)
        seen_keys[key] = seen_keys.get(key, 0) + 1
        granules.append(granule)

    audit.distinct_dates = len({granule.date for granule in granules})
    return audit, granules


def _extract_window(
    grid: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    rules: AODRules,
    station_lat: float,
    station_lon: float,
) -> tuple[float, int]:
    """Mean valid AOD in the station window, and the pixel count."""
    half = rules.window_half_width_deg
    latitude_mask = np.abs(latitude - station_lat) <= half
    longitude_mask = np.abs(longitude - station_lon) <= half
    if not latitude_mask.any() or not longitude_mask.any():
        return float("nan"), 0

    window = grid[np.ix_(latitude_mask, longitude_mask)]
    valid = window[
        (window != rules.fill_value)
        & (window >= rules.aod_valid_min)
        & (window <= rules.aod_valid_max)
    ]
    if valid.size == 0:
        return float("nan"), 0
    return float(np.mean(valid)), int(valid.size)


# The station coordinates used by the extraction.
_STATION_LAT = 28.4089
_STATION_LON = 77.3178


def load_aod_daily(
    aod_dir: Path | str,
    rules: AODRules | None = None,
    station_lat: float = _STATION_LAT,
    station_lon: float = _STATION_LON,
) -> AODDaily:
    """Extract the daily station-collocated AOD from the L2G granules.

    Parameters
    ----------
    aod_dir:
        Directory of INSAT-3D L2G HDF5 granules. Read-only.
    rules:
        Quality and extraction rules.
    station_lat, station_lon:
        Station coordinates.

    Returns
    -------
    AODDaily
        The daily frame, the structural audit and the per-granule metadata.
    """
    import h5py

    rules = rules or AODRules()

    audit, granules = inspect_granules(aod_dir, rules)

    records = []
    for granule in granules:
        with h5py.File(granule.path, "r") as handle:
            grid = np.asarray(handle["AOD"][0], dtype=float)
            latitude = np.asarray(handle["latitude"][:], dtype=float)
            longitude = np.asarray(handle["longitude"][:], dtype=float)

        value, pixels = _extract_window(
            grid, latitude, longitude, rules, station_lat, station_lon
        )
        if pixels == 0:
            audit.granules_with_zero_valid_pixels += 1

        records.append(
            {
                "Date": granule.date,
                "AOD": value,
                "aod_valid_pixels": pixels,
                "overpass_utc": granule.overpass_utc,
                "granule": granule.path.name,
            }
        )

    frame = pd.DataFrame(records)
    if frame.empty:
        return AODDaily(
            frame=pd.DataFrame(
                columns=["Date", "AOD", "aod_valid_pixels", "aod_overpasses"]
            ),
            audit=audit,
            granules=granules,
        )

    if rules.average_overpasses:
        daily = (
            frame.groupby("Date", as_index=False)
            .agg(
                AOD=("AOD", "mean"),
                aod_valid_pixels=("aod_valid_pixels", "sum"),
                aod_overpasses=("AOD", "size"),
            )
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        daily = frame.sort_values(["Date", "overpass_utc"]).reset_index(drop=True)

    if rules.min_valid_pixels > 1:
        daily = daily[daily["aod_valid_pixels"] >= rules.min_valid_pixels]
    daily = daily.reset_index(drop=True)

    return AODDaily(frame=daily, audit=audit, granules=granules)


def orientation_check(
    aod_dir: Path | str,
    boxes: dict[str, tuple[tuple[float, float], tuple[float, float]]] | None = None,
    max_granules: int = 60,
    station_lat: float = _STATION_LAT,
    station_lon: float = _STATION_LON,
) -> dict:
    """Prove the array axis orientation physically, not only structurally.

    The HDF5 dimension-scale binding says the array is
    ``[time, latitude, longitude]``. That is authoritative, but a
    mis-bound product would silently corrupt every extracted value, so a
    second, *independent* line of evidence is required.

    Aerosol optical depth over the open ocean is systematically **lower**
    than over polluted continental land. Under the correct orientation,
    known ocean boxes must show lower mean AOD than known land boxes. Under
    a transposed array the contrast reverses, which is physically
    impossible.

    Returns
    -------
    dict
        The mean AOD in each box under both hypotheses, the resulting
        ocean/land contrast, and whether the dimension-scale orientation is
        corroborated.
    """
    import h5py

    boxes = boxes or {
        "ArabianSea_ocean": ((8.0, 18.0), (60.0, 67.0)),
        "BayOfBengal_ocean": ((8.0, 18.0), (85.0, 93.0)),
        "CentralIndia_land": ((22.0, 28.0), (75.0, 85.0)),
        "NWIndia_land": ((25.0, 30.0), (68.0, 75.0)),
    }

    files = sorted(Path(aod_dir).glob("*.h5"))[:max_granules]
    if not files:
        raise FileNotFoundError(f"No INSAT-3D granules found in {aod_dir}")

    values: dict[str, dict[str, list[float]]] = {
        "dimension_scale": {},
        "transposed": {},
    }
    for path in files:
        with h5py.File(path, "r") as handle:
            latitude = np.asarray(handle["latitude"][:], dtype=float)
            longitude = np.asarray(handle["longitude"][:], dtype=float)
            grid = np.asarray(handle["AOD"][0], dtype=float)
        grid = np.where(grid == -999.0, np.nan, grid)

        for name, ((lat_low, lat_high), (lon_low, lon_high)) in boxes.items():
            latitude_mask = (latitude >= lat_low) & (latitude <= lat_high)
            longitude_mask = (longitude >= lon_low) & (longitude <= lon_high)
            if not latitude_mask.any() or not longitude_mask.any():
                continue

            # Hypothesis A: the dimension-scale orientation.
            window = grid[np.ix_(latitude_mask, longitude_mask)]
            valid = window[np.isfinite(window) & (window >= 0) & (window <= 5)]
            if valid.size:
                values["dimension_scale"].setdefault(name, []).extend(valid.tolist())

            # Hypothesis B: the transposed orientation.
            window = grid[np.ix_(longitude_mask, latitude_mask)]
            valid = window[np.isfinite(window) & (window >= 0) & (window <= 5)]
            if valid.size:
                values["transposed"].setdefault(name, []).extend(valid.tolist())

    summary = {}
    for hypothesis, per_box in values.items():
        means = {name: float(np.mean(v)) for name, v in per_box.items() if v}
        ocean = [m for name, m in means.items() if "ocean" in name.lower()]
        land = [m for name, m in means.items() if "land" in name.lower()]
        contrast = (
            float(np.mean(land) - np.mean(ocean)) if ocean and land else float("nan")
        )
        summary[hypothesis] = {
            "mean_AOD_by_box": means,
            "ocean_mean": float(np.mean(ocean)) if ocean else float("nan"),
            "land_mean": float(np.mean(land)) if land else float("nan"),
            "ocean_land_contrast": contrast,
            "physically_plausible": bool(np.isfinite(contrast) and contrast > 0),
        }

    corroborated = bool(
        summary.get("dimension_scale", {}).get("physically_plausible")
        and not summary.get("transposed", {}).get("physically_plausible")
    )
    return {
        "n_granules_examined": len(files),
        "boxes": {name: [list(lat), list(lon)] for name, (lat, lon) in boxes.items()},
        "hypotheses": summary,
        "dimension_scale_orientation_corroborated": corroborated,
        "interpretation": (
            "The dimension-scale orientation (dim1=latitude, dim2=longitude) "
            "is independently corroborated: it is the only orientation "
            "under which aerosol optical depth is lower over the open ocean "
            "than over continental land."
            if corroborated
            else "The physical cross-check did not corroborate the "
            "dimension-scale orientation; the extraction must not be used."
        ),
    }


def window_sensitivity(
    aod_dir: Path | str,
    half_widths: tuple[float, ...] = (0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
    rules: AODRules | None = None,
    station_lat: float = _STATION_LAT,
    station_lon: float = _STATION_LON,
) -> pd.DataFrame:
    """Report how the extracted value depends on the collocation window.

    The window half-width is a scientific choice, and its effect must be
    visible rather than assumed. For each candidate the table gives the
    mean extracted AOD, the mean valid-pixel count and the correlation with
    the default window.

    A value that is stable across the range is robust to the choice; one
    that moves materially is not.
    """
    base = rules or AODRules()
    series = {}
    pixel_counts = {}
    for half_width in half_widths:
        candidate = AODRules(
            fill_value=base.fill_value,
            aod_valid_min=base.aod_valid_min,
            aod_valid_max=base.aod_valid_max,
            window_half_width_deg=half_width,
            min_valid_pixels=base.min_valid_pixels,
            average_overpasses=base.average_overpasses,
        )
        result = load_aod_daily(
            aod_dir, candidate, station_lat=station_lat, station_lon=station_lon
        )
        series[half_width] = result.frame.set_index("Date")["AOD"]
        pixel_counts[half_width] = (
            float(result.frame["aod_valid_pixels"].mean())
            if not result.frame.empty
            else float("nan")
        )

    rows = []
    reference = series.get(0.25, next(iter(series.values())))
    for half_width, values in series.items():
        joined = pd.concat(
            [values.rename("value"), reference.rename("reference")], axis=1
        ).dropna()
        rows.append(
            {
                "window_half_width_deg": half_width,
                "n_days": int(len(joined)),
                "mean_AOD": (
                    float(joined["value"].mean()) if len(joined) else float("nan")
                ),
                "mean_valid_pixels": pixel_counts[half_width],
                "correlation_with_default": (
                    float(joined["value"].corr(joined["reference"]))
                    if len(joined) > 2
                    else float("nan")
                ),
                "max_absolute_difference": (
                    float((joined["value"] - joined["reference"]).abs().max())
                    if len(joined)
                    else float("nan")
                ),
            }
        )

    return pd.DataFrame(rows)
