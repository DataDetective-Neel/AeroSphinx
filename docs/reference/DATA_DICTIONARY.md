# AeroSphinx — Data Dictionary

**Phase 0.** Field-level provenance, semantics and quality rules for every
input the pipeline consumes, and every field it derives.

Machine-readable evidence: `results/baseline/data_audit.json`.

---

## 1. Source summary

| Stream | Provider | Product | Local coverage | Files |
|---|---|---|---|---|
| Ground PM2.5 | CPCB | City-wise hourly monitoring export | 2016-01-01 to 2024-12-31 | 9 CSV |
| Aerosol optical depth | ISRO / SAC | INSAT-3D Imager L2G AOD V02R00 | 2024-01-01 to 2024-06-18 | 169 HDF5 |
| Meteorology | NASA GMAO | MERRA-2 `tavg1_2d_slv_Nx` (2-daily, 1-hourly) | 2024-01-01 to 2024-06-20 | 172 NetCDF4 |

Station: **Faridabad, Sector 16A**, Haryana, Delhi-NCR.
Reference coordinates **28.4089 N, 77.3178 E**; the CPCB export names the
station "Faridabad". The MERRA-2 grid cell is (28.5, 77.5) — an offset of
about 0.09 deg latitude and 0.18 deg longitude from the station, i.e. roughly
10 km and 17 km. See D-010 for why this offset is currently unquantified.

---

## 2. CPCB — ground PM2.5

### 2.1 File structure

One CSV per calendar year, 2016 through 2024. Consistent 25-column header
across all nine files. 8,760 rows for non-leap years and 8,784 for leap years
(2016, 2020 and 2024).

| Column | Unit | Used by pipeline |
|---|---|---|
| `Timestamp` | local naive datetime | **yes** — parsed as `pd.to_datetime` |
| `PM2.5 (µg/m³)` | µg/m³ | **yes** — the target |
| `PM10 (µg/m³)` | µg/m³ | no |
| `NO (µg/m³)` | µg/m³ | no |
| `NO2 (µg/m³)` | µg/m³ | no |
| `NOx (ppb)` | ppb | no |
| `NH3 (µg/m³)` | µg/m³ | no |
| `SO2 (µg/m³)` | µg/m³ | no |
| `CO (mg/m³)` | mg/m³ | no |
| `Ozone (µg/m³)` | µg/m³ | no |
| `Benzene`, `Toluene`, `Xylene`, `O Xylene`, `Eth-Benzene`, `MP-Xylene` | µg/m³ | no |
| `AT (°C)` | °C | no |
| `RH (%)` | % | no |
| `WS (m/s)`, `WD (deg)` | m/s, deg | no |
| `RF (mm)`, `TOT-RF (mm)` | mm | no |
| `SR (W/mt2)` | W/m² | no |
| `BP (mmHg)` | mmHg | no |
| `VWS (m/s)` | m/s | no |

**Note:** the repository holds substantial unused meteorological and
precipitation data from the same station. These are potential features for
Track A at **zero additional download cost** and are currently discarded.
This is a missed opportunity, recorded in the audit's prioritised findings.

The column containing PM2.5 is located by name rather than position
(`_find_pm25_column`), so the loader tolerates reordering, but it does not
tolerate a renamed column.

### 2.2 Time base

Timestamps are **local (IST, UTC+05:30) naive datetimes** with a regular
hourly grid, e.g. `2016-01-01 00:00:00`. No timezone is declared in the file.
No DST adjustment applies in India.

### 2.3 Quality rules applied

| Rule | Value | Rationale |
|---|---|---|
| Invalid PM2.5 range | outside **[0, 500] µg/m³** treated as missing | values above 500 are implausible for a 1-hour mean and are treated as instrument faults |
| Daily coverage | at least **18** valid hours required for a day to be accepted | 75% coverage |
| Invalid readings discarded | 407 in 2016, decreasing to 1 in 2022 | recorded per file in the audit manifest |
| Days failing the coverage rule | **309** rejected | |

### 2.4 Derived daily series

`load_cpcb_daily` produces one row per accepted date with:

| Field | Meaning |
|---|---|
| `Date` | calendar date (normalised to midnight) |
| `PM25` | daily mean concentration, µg/m³ |
| `valid_hours` | count of valid hourly observations contributing to the day |

**Accepted: 2,979 daily rows, 2016-01-01 to 2024-12-31.**
Per year: 2016: 261, 2017: 254, 2018: 340, 2019: 310, 2020: 359, 2021: 362,
2022: 363, 2023: 364, 2024: 366.

### 2.5 Known quality concerns

**Calendar gaps.** The accepted series contains **123 gaps** between
consecutive accepted dates. Longest gap: **25 days** (2017-10-09 to
2017-11-03). Other notable gaps: 13 days (2016-10-31 to 2016-11-13), 12 days
(2016-12-02 to 2016-12-14), 10 days (2017-09-21 to 2017-10-01).

**Year-to-year change in record character.** The hourly series is *not*
homogeneous across the nine years:

| Year | Rows | Mean abs. successive-hour difference | Integer-valued readings |
|---|---|---|---|
| 2016 | 8,784 | 19.26 | 96.1% |
| 2017 | 8,760 | 18.27 | 58.6% |
| 2018 | 8,760 | 19.18 | 92.1% |
| 2019 | 8,760 | 14.46 | 40.8% |
| 2020 | 8,784 | 10.88 | 7.4% |
| 2021 | 8,760 | 12.27 | 1.7% |
| 2022 | 8,760 | 13.51 | 0.6% |
| 2023 | 8,760 | 9.92 | 0.3% |
| 2024 | 8,784 | 10.91 | 1.2% |

Earlier years are raw hourly instrument values; later years are progressively
smoothed or pre-aggregated. **INFERRED:** upstream CPCB export resolution or
averaging changed over this period. **UNKNOWN:** the cause, and whether the
smoothing is a temporal average (which would blunt sub-daily peaks) or a
different measurement mode. The local files cannot settle this. **BLOCKED**
on upstream provenance.

This matters for Track A: a model trained on 2016-2019 raw hours and applied
to 2023-2024 smoothed values is being trained across a change in measurement
character, and no test in the current pipeline would detect it.

**Line endings.** The files under `data/cpcb/` use CRLF; the tracked
originals under `data/` used LF. Content is identical after normalisation for
all nine files. Recorded so future hash comparisons are not misread as data
changes. (D-015)

**Duplicates.** Zero duplicated hourly timestamps across the nine files.
Seven files nonetheless exist twice in git history (`data/` and
`ML model/CPCB Data/`) with identical SHA-256 digests. (D-012)

---

## 3. INSAT-3D — aerosol optical depth

### 3.1 Granule structure

Product `3DIMG_<DDMMMYYYY>_<HHMM>_L2G_AOD_V02R00.h5`, L2G gridded geophysical
product. Institute: BES, SAC/ISRO, Ahmedabad. Conventions: CF-1.6.

```
/AOD       Dataset {shape: (1, 551, 551)}   float32
              long_name: 'Aerosol Optical Depth'
              standard_name: 'Aerosol Optical Depth'
              units: '-'                       (dimensionless)
              _FillValue: -999.0
/latitude  Dataset {shape: (551,)}            float64
              units: 'degrees_north'
              standard_name: 'latitude'
/longitude Dataset {shape: (551,)}            float64
              units: 'degrees_east'
              standard_name: 'longitude'
/time      Dataset {shape: (1,)}              float64
              units: 'minutes since 2000-01-01 00:00:00'
```

### 3.2 Geolocation — orientation VERIFIED

The HDF5 dimension scales bind `AOD.dim1 -> /latitude` and
`AOD.dim2 -> /longitude`. This is authoritative for the array layout and
confirms the pipeline's `grid[np.ix_(lat_mask, lon_mask)]` indexing is
correct. `latitude` runs **descending** from 45.05 to -9.95; boolean masking
handles this correctly.

Bounding box, identical for all 169 granules:

| Bound | Value |
|---|---|
| `lower_latitude` / `upper_latitude` | -9.95 / 45.05 |
| `left_longitude` / `right_longitude` | 45.05 / 100.05 |
| `lat_interval` / `lon_interval` | 0.1 / 0.1 degrees |

### 3.3 Root attributes of record

| Attribute | Value |
|---|---|
| `Acquisition_Date` | `DDMMMYYYY`, matches the filename for all 169 granules |
| `Acquisition_Start_Time` / `_End_Time` | ISO-8601 |
| `Acquisition_Time_in_GMT` | e.g. `0530` |
| `Satellite_Name` / `Sensor_Name` | `INSAT-3D` / `IMAGER` |
| `Processing_Level` | `L2G` |
| `Product_Type` | `GEOPHY GRIDDED` |
| `Nominal_Altitude(km)` | 36000 |

### 3.4 Acquisition pattern

| Overpass (UTC) | Granules |
|---|---|
| 05:30 | 154 |
| 06:00 | 7 |
| 06:30 | 4 |
| 07:00 | 3 |
| 07:30 | 1 |

169 granules, **166 distinct dates**, 2024-01-01 to 2024-06-18. No duplicate
(date, time) pairs. Exactly one date carries two overpasses, which the
pipeline averages.

### 3.5 Extraction and quality rules

| Rule | Value |
|---|---|
| Collocation window | +/- **0.25 degrees** around 28.4089 N, 77.3178 E (about 27 km in latitude) |
| Maximum pixels per granule | 5 x 5 = **25** |
| Fill value | **-999.0** |
| Valid AOD range accepted | **[0.0, 5.0]** |
| Multiple same-day overpasses | averaged into one daily value |

Observed AOD values span [2.6e-06, 3.0]; the acceptance window discards
nothing.

### 3.6 Coverage and support

| Statistic | Value |
|---|---|
| Granules with **zero** valid pixels in the station window | **27** |
| Valid pixels per granule — min | 1 |
| Valid pixels per granule — median | **12** |
| Valid pixels per granule — mean | 14.1 |
| Valid pixels per granule — max | 25 |
| Days with no AOD at all | 27 (of 166) |

**No QA, cloud or quality-flag dataset exists in these granules.** A pixel is
either present or fill; "valid pixel" therefore cannot be distinguished from
"clear-sky retrieval", and a single-pixel daily AOD is not a
station-representative measurement. (D-004)

### 3.7 Derived fields

| Field | Meaning |
|---|---|
| `AOD` | mean valid AOD across the station window, dimensionless |
| `aod_valid_pixels` | count of valid pixels contributing to the daily mean |
| `aod_overpasses` | number of granules averaged into the day |

`aod_valid_pixels` is computed but **not used as a feature, a filter, or a
reported confidence measure** — a gap the availability-aware framework in the
plan is specifically designed to close.

---

## 4. MERRA-2 — meteorology

### 4.1 Granule structure

Subsetted from `MERRA2_400.tavg1_2d_slv_Nx` — *2-daily, 1-hourly,
time-averaged, single-level, assimilation*. NASA GMAO, VersionID 5.12.4,
DOI 10.5067/VJAFPLI1CSIV. Subset performed with CDO 1.9.8 / L34RS v1.4.4.

```
Dimensions: time = 3, lat = 1, lon = 1
Coordinates: time = 2024-01-01T11:30, 12:30, 13:30
             lat = 28.5, lon = 77.5
```

The granule's own `history_L34RS` attribute records the subsetting:
`Spatial: 77.2 77.4 28.3 28.5. Timesteps: 12 13 14`.

**Two consequences that the pipeline's docstrings currently misstate:**

1. Values are **not** spatially averaged over a collocation box — the granule
   holds a **single grid cell**.
2. Values are **not** daily means — only **three hourly timesteps
   (11:30, 12:30, 13:30 UTC)** are present, and the pipeline averages them.

The features are therefore **midday-window, single-grid-cell values**. (D-010)

### 4.2 Variables present

| Variable | Units | `long_name` | Used |
|---|---|---|---|
| `T2M` | K | 2-meter air temperature | **yes** |
| `TS` | K | surface skin temperature | **yes** |
| `QV2M` | kg/kg | 2-meter specific humidity | **yes** |
| `U10M` | m/s | 10-meter eastward wind | **yes** |
| `V10M` | m/s | 10-meter northward wind | **yes** |
| `SLP` | Pa | sea-level pressure | **yes** |
| `PS` | Pa | surface pressure | no |
| `QV10M` | kg/kg | 10-meter specific humidity | no |
| `T10M` | K | 10-meter air temperature | no |
| `T2MDEW` | K | dew-point temperature at 2 m | no |
| `TQI` | kg/m² | total precipitable ice water | no |
| `TQL` | kg/m² | total precipitable liquid water | no |
| `PBLH` | m | planetary boundary layer height | **requested — absent in all 172 files** |
| `PBLTOP` | Pa | pressure at top of boundary layer | **absent in all 172 files** |

`V10M` is missing from 2 of the 172 granules.

`T2MDEW` is available and is the physically correct basis for a relative
humidity or dew-point depression feature; neither is currently used.

### 4.3 Coverage

172 granules, 2024-01-01 to 2024-06-20, one per day with no internal gaps.
The series simply stops after 2024-06-20 — the 10 remaining June days were
never distributed.

### 4.4 Derived fields

`load_merra_daily` returns one row per date with the requested variables
averaged over the three available timesteps, plus a `missing_counts` map
reporting which requested variables no granule provided.

---

## 5. Fused modelling frame

### 5.1 Construction

1. Three daily series merged on `Date` with a **left join from CPCB**.
2. `AOD` and every MERRA-2 variable forward-filled with `limit=3`.
3. Features engineered (Section 6).
4. Rows with any missing feature or missing target dropped.

**Result: 166 rows, 2024-01-07 to 2024-06-21**, despite the CPCB record
spanning nine years. The binding constraint is the intersection of the three
coverage windows.

| Provenance of the fused rows | Count |
|---|---|
| Rows with a **genuine** same-day AOD observation | 159 |
| Rows whose AOD is **forward-filled** | **7** |
| Rows with a **genuine** same-day MERRA-2 observation | 165 |
| Rows whose MERRA-2 is **forward-filled** | **1** |

The forward-filled rows are indistinguishable from observed ones in the
modelling frame — no presence indicator, no observation age. (D-003)

### 5.2 Evaluation split

| Partition | Dates | Rows |
|---|---|---|
| Train | 2024-01-07 to 2024-05-16 | 130 |
| Validation (configuration selection) | 2024-04-10 to 2024-05-16 | — |
| Test | 2024-05-17 to 2024-06-18 | **33** |

The split is chronological and the validation window ends exactly where the
test window begins, so no test observation influences any modelling choice.
This separation is correct and was verified.

### 5.3 Phase 6 fused frame (current research path)

**Supersedes the forward-filled frame in 5.1 for all research work.** The
legacy `aerosphinx_pipeline.py` frame is retained unchanged as the Phase 0
baseline but is not a Track B modelling input.

Construction (`src/aerosphinx/features/availability.py`):

1. The ground daily series defines the row calendar; it is preserved exactly
   (**2,979 rows, 2016-01-01 to 2024-12-31**).
2. Each environmental source is left-joined on `Date` at one row per date.
   A source cannot add or remove a row.
3. Missing values are preserved as `NaN`. **No forward-fill, interpolation
   or imputation.** (D-003)
4. Per-source availability flags and observation ages are attached.
5. AOD support counts are set to `0` where no AOD observation exists; they
   are counts, not measurements.

Output: `data/processed/fused_dataset.csv` — 42 columns.

| Column | Meaning |
|---|---|
| `<source>_observed` | True only when that source has a genuine same-day observation |
| `<source>_observation_age_days` | Calendar days since the last genuine observation; a diagnostic, never a value |
| `n_sources_observed` | Count of requested sources observed on the row |
| `all_sources_observed` | Whether all requested sources are observed |

`<source>` is `station_met`, `merra` or `aod`.

| Source | Observed rows | Fraction | Window |
|---|---|---|---|
| `station_met` | 2,979 | 100.0% | 2016-01-01 to 2024-12-31 |
| `merra` | 172 | 5.8% | 2024-01-01 to 2024-06-20 |
| `aod` | 139 | 4.7% | 2024-01-07 to 2024-06-18 |
| **All three** | **139** | **4.7%** | 2024 only |

The 33-check alignment gate is documented in `docs/reports/PHASE6_REPORT.md`.

---

## 6. Derived feature dictionary

`BASE_FEATURE_ORDER` in `aerosphinx_pipeline.py:128`.

| Feature | Definition | Units | Group |
|---|---|---|---|
| `AOD` | mean valid INSAT-3D AOD in the station window | dimensionless | satellite |
| `T2M` | MERRA-2 2 m air temperature | K | meteorology |
| `TS` | MERRA-2 surface skin temperature | K | meteorology |
| `QV2M` | MERRA-2 2 m specific humidity | kg/kg | meteorology |
| `U10M` | MERRA-2 10 m zonal wind | m/s | meteorology |
| `V10M` | MERRA-2 10 m meridional wind | m/s | meteorology |
| `SLP` | MERRA-2 sea-level pressure | Pa | meteorology |
| `PBLH` | planetary boundary layer height — **unavailable**, dropped | m | meteorology |
| `Temp_Diff` | `TS - T2M` | K | meteorology (stability proxy) |
| `Humidity_Ratio` | `QV2M / (T2M + EPS)` | kg/(kg·K) | **mis-named** — see D-009 |
| `PM25_Lag_1` | `PM25` shifted by 1 **row** | µg/m³ | history — see D-001 |
| `PM25_Lag_2` | shifted by 2 rows | µg/m³ | history — see D-001 |
| `PM25_Lag_3` | shifted by 3 rows | µg/m³ | history — see D-001 |
| `PM25_Roll3` | mean of `PM25` over 3 rows, shifted by 1 | µg/m³ | history — see D-001 |
| `PM25_Roll7` | mean of `PM25` over 7 rows, shifted by 1 | µg/m³ | history — see D-001 |
| `day_sin` | `sin(2*pi*day_of_week/7)` | dimensionless | calendar |
| `day_cos` | `cos(2*pi*day_of_week/7)` | dimensionless | calendar |
| `month_sin` | `sin(2*pi*month/12)` | dimensionless | calendar |
| `month_cos` | `cos(2*pi*month/12)` | dimensionless | calendar |

Rolling features are computed **only from prior rows** (`shift(1)` before
rolling), so no future target value enters the feature vector. This part of
the implementation is correct. The defect is that "prior rows" is not the
same as "prior calendar days" (D-001).

`PM25_Roll3` and `PM25_Roll7` are configuration-dependent: present only under
the `tuned` configuration.

### 6.1 Feature importance (shipped baseline)

| Feature | Share |
|---|---|
| `month_sin` | **37.93%** |
| `PM25_Roll3` | 11.22% |
| `month_cos` | 6.29% |
| `PM25_Lag_1` | 6.19% |
| `PM25_Roll7` | 5.85% |
| `V10M` | 4.52% |
| `U10M` | 4.29% |
| `Temp_Diff` | 4.11% |
| `T2M` | 3.28% |
| `TS` | 2.64% |

---

## 7. Target and AQI banding

**Target:** `PM25`, daily mean concentration in µg/m³.

The AQI category is derived from the **PM2.5 sub-index breakpoints only**,
using `np.digitize` against `[30, 60, 90, 120, 250]`. The digitize semantics
were verified to match the CPCB National AQI PM2.5 bands exactly:

| Concentration | Category |
|---|---|
| 0 – 30 | Good |
| > 30 – 60 | Satisfactory |
| > 60 – 90 | Moderate |
| > 90 – 120 | Poor |
| > 120 – 250 | Very Poor |
| > 250 | Severe |

**Important scope limit, per the plan:** this is a **PM2.5 concentration
category**, not the official overall AQI, which is a sub-index computed across
multiple pollutants. The pipeline's column names `Actual_AQI_Class` and
`Predicted_AQI_Class` should not be read as official AQI. The plan requires
these to be distinguished explicitly.

---

## 8. Raw-input integrity

SHA-256 digests for all 350 raw input files (9 CPCB CSV, 169 INSAT HDF5,
172 MERRA-2 NetCDF) are recorded in `results/baseline/manifest.json` under
`dataset_hashes`, and are regenerated by `scripts/audit_repository.py`.

Digests for the relocated HDF5 and NetCDF files are **identical** to their
pre-relocation tracked blobs, confirming the reorganisation was content-
preserving.
