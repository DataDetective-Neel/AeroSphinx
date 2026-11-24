# AeroSphinx — Scientific Assumptions

**Phase 0.** Every scientific assumption the current pipeline relies on,
whether it holds, and what would falsify it.

Status legend: **VERIFIED** (tested against local evidence), **INFERRED**
(reasoned but not directly testable here), **UNSUPPORTED** (asserted without
evidence — must not be relied upon), **BLOCKED** (cannot be settled without
data or permission not available).

---

## A. Station and representativeness

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| A1 | The CPCB export is a single monitoring station at 28.4089 N, 77.3178 E (Faridabad Sector 16A, Haryana) | **VERIFIED** | Consistent across all 9 files; matches the pipeline constant `STATION_LAT`/`STATION_LON`. Falsified by a station-metadata file naming a different site. |
| A2 | The station's hourly series is representative of Faridabad / Delhi-NCR as a whole | **UNSUPPORTED** | One station cannot establish spatial generalisation. No second station is present locally. Falsified by comparison against any independent station. |
| A3 | The station record is homogeneous across 2016–2024 | **UNSUPPORTED — actively contradicted** | Integer-valued hourly readings fall from 96.1% (2016) to 0.3% (2023) and mean absolute successive-hour difference falls from 19.3 to 9.9 µg/m³. The record's *character* changes. See D-014 in the audit, §6.1. |

A3 is the most consequential unverified assumption in the repository. A
Track A model trained across the full record is trained across a change in
measurement character, and no existing test detects it.

---

## B. Ground observations

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| B1 | Timestamps are local (IST) naive datetimes on a regular hourly grid | **VERIFIED** | All 9 files parse to a regular hourly index with no DST discontinuity. Falsified by a non-uniform or duplicated hourly index. |
| B2 | PM2.5 outside [0, 500] µg/m³ indicates instrument fault, not real concentration | **INFERRED** | The pipeline's stated convention. Plausible for a 1-hour mean but **not verified against CPCB documentation**, and the threshold is a project choice rather than a published rule. 407 readings are screened out in 2016 alone. Falsified by CPCB's own QA protocol stating a different range. |
| B3 | A daily mean requires at least 18 valid hourly observations | **INFERRED** | A 75% coverage convention, not a CPCB standard. 309 days are rejected on this basis. Falsified by a different stated convention. |
| B4 | Hourly readings are equally weighted within a daily mean | **INFERRED** | Standard practice. Note the raw data shows a strong diurnal cycle (2024: 71 µg/m³ at 16:00 vs 100 at 23:00), so an unweighted mean over a partial day is sensitive to which hours are present. |

---

## C. Satellite observations

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| C1 | `AOD[0]` is indexed `[latitude, longitude]` | **VERIFIED** | HDF5 dimension scales bind `dim1 -> /latitude`, `dim2 -> /longitude`. Falsified only by a granule with different scale bindings. |
| C2 | Latitude descends (45.05 -> -9.95) | **VERIFIED** | Read from the coordinate array; boolean masking is order-independent. |
| C3 | A +/-0.25° square window is a scientifically defensible station collocation box | **INFERRED** | About 27 km in latitude. Reasonable for AOD, but the choice is undocumented in any cited source. Falsified by a sensitivity analysis over window sizes — **which has not been performed** and is required by the plan's "sensitivity to temporal and spatial extraction choices". |
| C4 | AOD is valid within [0, 5] | **INFERRED** | Physically sensible; discards nothing in the local data. Not a published INSAT validity range. |
| C5 | A valid pixel represents a usable retrieval | **UNSUPPORTED** | The granules contain **no QA, cloud or quality-flag dataset**. A present pixel cannot be distinguished from a clear-sky retrieval. Falsified by obtaining the QA dataset. |
| C6 | A daily AOD averaged over multiple overpasses is comparable to a single-overpass value | **INFERRED** | Only one date has two overpasses, so this is essentially untested locally. |
| C7 | Column AOD is informative about surface PM2.5 | **INFERRED — well established in principle** | Requires aerosol vertical distribution, humidity and boundary-layer information to convert. The pipeline does not attempt a conversion, which is correct; but it also does not include humidity or PBLH in a way that would let the model learn one. |
| C8 | The 27 zero-pixel granules indicate cloud or no-data rather than an extraction bug | **INFERRED** | `aod_valid_pixels` is 0 for these; the surrounding granules are well-formed. Falsified by a visual/geographic check of the fill pattern, **which has not been performed**. |

---

## D. Meteorological observations

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| D1 | MERRA-2 `tavg1_2d_slv_Nx` is reanalysis, not a forecast | **VERIFIED** | Product name contains `Nx` (assimilation); `LongName` states "Assimilation". |
| D2 | Retrospective MERRA-2 reanalysis may be used to *understand* historical atmospheric conditions | **VERIFIED (methodologically sound)** | This is the plan's Track B position. |
| D3 | Retrospective MERRA-2 reanalysis may be used as an *operational future* forecast input | **UNSUPPORTED — must not be done** | Reanalysis is not available at forecast issue time. This is the central correction in the plan's §4, and the shipped pipeline currently violates it (D-002). |
| D4 | MERRA-2 values are "spatially averaged over the Faridabad bounding box" and are daily means | **UNSUPPORTED — contradicted** | Granules hold a **1×1 grid cell** at (28.5, 77.5) with **3 hourly timesteps** (11:30, 12:30, 13:30 UTC), per the granule's own `history_L34RS` attribute. The pipeline's docstring misstates this. (D-010) |
| D5 | The single MERRA-2 grid cell is representative of the station | **INFERRED** | The cell centre is ~10 km north and ~17 km east of the station. MERRA-2's native resolution is 0.5° × 0.625° (~55 × 60 km), so the cell is coarse relative to urban gradients. Falsified by a cell-vs-point sensitivity check, **not yet performed**. |
| D6 | A midday window (11:30–13:30 UTC) represents the daily mean | **UNSUPPORTED** | Diurnal boundary-layer behaviour is strong; the window covers the period of peak mixing and is unrepresentative of the night-time accumulation that dominates winter PM2.5. Falsified by re-subsetting with full 24-hour coverage. |
| D7 | `TS - T2M` is a usable near-surface stability proxy | **INFERRED** | Physically motivated (skin-to-air temperature difference relates to radiative cooling and inversion strength) and used in the literature, but no validation is performed. |
| D8 | `QV2M / T2M` is a "humidity ratio" | **UNSUPPORTED — mis-defined** | Specific humidity (kg/kg) divided by temperature (K) has units of kg/(kg·K) and is not a humidity ratio in any conventional sense. (D-009) |
| D9 | `PBLTOP` can stand in for `PBLH` | **UNSUPPORTED** | Pressure at the top of the boundary layer is not a height. Neither variable exists in the local granules, so the fallback never fires — but it would silently activate on a future granule containing `PBLTOP` and produce a physically invalid feature. (D-008) |

---

## E. Feature engineering

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| E1 | A lag of 1 row is the previous calendar day | **UNSUPPORTED — contradicted** | 123 rows carry a `PM25_Lag_1` that is not the previous calendar day; worst case 25 days. (D-001) |
| E2 | Rolling statistics over prior rows do not use future information | **VERIFIED** | `shift(1)` precedes `.rolling()`; max discrepancy against a calendar-correct reference is 0.0 for `PM25_Roll3`. The *window membership* is wrong (E1); the *direction* is right. |
| E3 | Sine/cosine encoding preserves the circular topology of weekly and annual cycles | **VERIFIED** | Standard and correctly implemented for day-of-week and month. |
| E4 | Month is a sufficient seasonal encoding | **INFERRED** | `month_sin` carries 37.9% of importance, so the model leans on it heavily. A day-of-year cyclic encoding would represent the seasonal cycle more smoothly, and a monsoon-break discontinuity is not representable by a smooth annual cycle. Falsified by an alternative-encoding comparison, **not yet performed**. |
| E5 | Calendar features are known at forecast issue time and so are legitimate forecast inputs | **VERIFIED** | Deterministic functions of the target date. |
| E6 | Missing environmental observations are represented by an explicit mask and are never filled or imputed | **VERIFIED (Phase 6)** | The fused frame preserves 2,979 ground rows with 0 manufactured environmental values; the 33-check alignment gate verifies same-day provenance for every column. Falsified by any unobserved row carrying a value. See `docs/reports/PHASE6_REPORT.md`. |

---

## F. Evaluation

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| F1 | A chronological split with the most recent window held out estimates generalisation | **VERIFIED** | Train precedes test strictly; no test observation enters any modelling choice. |
| F2 | The validation window used for configuration selection is disjoint from the test window | **VERIFIED** | Validation ends 2024-05-16; test begins 2024-05-17. `select_configuration` raises if the windows overlap. |
| F3 | A single 33-day test window is sufficient to estimate performance | **UNSUPPORTED** | 33 days cannot characterise seasonal behaviour. The observed SD is 23.10 µg/m³ and the test period covers only late May to mid-June — one season, one episode type. Falsified by rolling-origin backtesting, **not yet performed**. |
| F4 | The ensemble beats persistence | **UNSUPPORTED — statistically** | Paired Wilcoxon p = 0.1495; bootstrap 95% CI of the MAE difference [-6.47, +1.39] spans zero. (D-006) |
| F5 | Adding satellite and meteorological features improves on ground history alone | **UNSUPPORTED — contradicted** | Ground-only MAE 16.617 beats the full ensemble's 17.298 on identical rows. (D-007) |
| F6 | The model learns physical aerosol relationships | **UNSUPPORTED** | `month_sin` alone carries 37.9% of importance, exceeding all three PM2.5 history features combined. The model is substantially a seasonal climatology with a persistence anchor. (D-011) |
| F7 | The five-day recursive forecast demonstrates operational forecasting skill | **UNSUPPORTED** | The forecast consumes future observed AOD and reanalysis. It demonstrates retrospective estimation. (D-002) |
| F8 | The persistence anchor value of 0.4 is appropriate at every forecast horizon | **UNSUPPORTED** | It was selected on a validation window for one-step prediction, then applied unchanged across a five-day recursion, where it compounds upward drift (96.6 -> 138.5 µg/m³). (D-016) |
| F9 | The AQI category computed here is the official AQI | **UNSUPPORTED — mislabelled** | It is a PM2.5 concentration category from the PM2.5 sub-index breakpoints only. The official AQI is a multi-pollutant index. The `np.digitize` band boundaries were verified to match the CPCB PM2.5 bands exactly, but the *scope* claim is wrong. |
| F10 | An observed-versus-unobserved target contrast computed over the full record measures informative missingness | **UNSUPPORTED** | The contrast is confounded by coverage: AOD and MERRA-2 exist only in 2024, so the "unobserved" group is dominated by 2016–2023. Only within-window contrasts (such as the Phase 4 AOD comparison on 2024 data) are interpretable. Phase 6 reports the full-record contrast for coverage description, not as evidence. |

---

## G. Reproducibility

| # | Assumption | Status | Basis / falsifier |
|---|---|---|---|
| G1 | Repeated runs on the same files give identical numbers | **VERIFIED** | 81 of 81 numeric metrics reproduce exactly. Two CSVs differ only at ~1e-13 float-representation level. |
| G2 | `random_state=42` with fixed hyperparameters removes run-to-run variance | **VERIFIED** | Verified by repeated execution. |
| G3 | Tracked model artefacts correspond to the tracked results | **UNSUPPORTED** | `models/*.joblib` are tracked binaries, modified in the working tree, with no recorded content hash linking them to a specific dataset, code version or metrics file. (Audit §7) |
| G4 | The environment described by `requirements.txt` reproduces the results | **PARTIAL** | Installed versions are all newer than the floors. The stated Python floor of 3.9 is wrong — the code uses PEP 604 unions and needs 3.10+. |
| G5 | Raw data integrity can be verified | **VERIFIED (now)** | SHA-256 for all 350 raw inputs recorded in `results/baseline/manifest.json`. |

---

## H. Assumptions the plan requires but that are not yet testable

| # | Assumption required | Why blocked |
|---|---|---|
| H1 | Future meteorological forecasts are obtainable from NCMRWF, ECMWF-IFS or GFS at issue time | No forecast product is present locally; requires network access and a documented issue-time schedule. |
| H2 | INSAT-3D granules exist for 2016–2023 at the required cadence | Not present locally; requires a data request. Without it, Track B stays confined to 2024. |
| H3 | A MERRA-2 subset with `PBLH` and full 24-hour coverage can be produced | Requires re-subsetting with the NASA GES DISC subsetting service. |
| H4 | The station's local meteorological columns (`AT`, `RH`, `WS`, `WD`, `RF`, `SR`, `BP`, `VWS`) are reliable enough to use as features | The columns exist in the CPCB exports but their provenance and QA status are not documented locally. They are currently discarded — a potential zero-cost feature source that needs provenance confirmation first. |

---

## Standing rule

An assumption may only be relied upon in a research claim when it is
**VERIFIED**. **INFERRED** assumptions must be stated as such in any write-up.
**UNSUPPORTED** assumptions may not be used to support a claim at all, and
where the shipped pipeline depends on one, that dependency must be removed or
the dependent claim withdrawn.

---

## I. Revision 2026-11-23 — assumptions re-tested on the canonical matrix

The Phase 0 rows above describe the legacy 33-day evaluation. Where a row's
verdict changes on the canonical Phase 7 matrix, the new evidence is recorded
here rather than by rewriting the original entry.

| Row | Original verdict | On the matrix | Consequence |
|---|---|---|---|
| F3 (a single 33-day window is enough) | UNSUPPORTED | Falsified, as expected: 97 folds and 2,615 matched predictions replace it | The rolling-origin protocol is the only evaluation quoted in the README |
| F4 (the ensemble beats persistence) | UNSUPPORTED, p = 0.1495 | Track A ridge −0.65, day CI [−1.14, −0.17], year CI [−1.20, +0.06] | Small gain; robust to day resampling, not to year resampling |
| F5 (fusion beats ground-only) | UNSUPPORTED — contradicted | No Track B ablation separates from zero (AOD +0.21 [−2.19, +2.61], fusion +4.70 [+0.30, +10.41], Holm p 0.34) | "Underpowered" replaces "contradicted": the ordering of point estimates is not a measurement |
| F7 (the five-day forecast shows skill) | UNSUPPORTED | Unchanged — D-002 is still open | No operational forecast claim anywhere in the README |

Two assumptions are now *testable* and were tested:

* **C3** required a sensitivity analysis over the AOD collocation window;
  `results/runs/D-aod-*/aod_window_sensitivity.csv` records it.
* **G1** ("repeated runs give identical numbers") is now enforced by
  `aerosphinx check-reproducibility`, which regenerates the matrix into a
  temporary directory and compares it field by field.
