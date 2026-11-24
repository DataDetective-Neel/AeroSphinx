# AeroSphinx — Feature Availability Contract

This document is the written form of the contract encoded in
`src/aerosphinx/features/groups.py`. It answers one question for every
modelled feature: **may a forecaster legitimately claim to have this value at
the moment the forecast is issued?**

The code is authoritative. This document is checked against it by
`tests/unit/test_feature_contract_doc.py`, so a feature cannot be added,
removed or reclassified without this file changing too.

---

## 1. Availability classes

| Class | Meaning | Eligible at issue time |
|---|---|---|
| `observed_ground_history` | PM2.5 observed on or before the issue date | **Yes** |
| `calendar` | Deterministic functions of the target date | **Yes** |
| `observed_satellite` | INSAT-3D AOD observed on a past date | No — retrospective |
| `reanalysis` | MERRA-2 assimilation product | **Never** |
| `station_local_unknown_availability` | CPCB station-local meteorology with no publication timestamp | No — retrospective |

Only `observed_ground_history` and `calendar` are operationally eligible.
`assert_operational_contract()` raises `ValueError` if an operational feature
set contains any other class; `partition_by_availability()` splits a group list
into the eligible and retrospective parts.

The classification is deliberately conservative. "The measurement time is
known" is not the same claim as "the value was published before the forecast
was issued", and the repository has evidence for the former only.

---

## 2. Per-feature table

`Observation` is the time the value describes. `Publication` is when it became
available to a forecaster. `Unknown` means the source files carry no
publication or upload timestamp, and no lag is assumed.

| Feature | Group | Class | Operational | Observation | Publication | Quality flag | Exclusion / reason |
|---|---|---|---|---|---|---|---|
| `PM25_Lag_1` | `pm25_history` | `observed_ground_history` | Yes | Calendar day *t−1* | Same day | Daily mean kept only with ≥ 18 valid hours | — |
| `PM25_Lag_2` | `pm25_history` | `observed_ground_history` | Yes | Calendar day *t−2* | Same day | as above | — |
| `PM25_Lag_3` | `pm25_history` | `observed_ground_history` | Yes | Calendar day *t−3* | Same day | as above | — |
| `PM25_Lag_7` | `pm25_history` | `observed_ground_history` | Yes | Calendar day *t−7* | Same day | as above | — |
| `PM25_Roll3` | `pm25_history` | `observed_ground_history` | Yes | Days *t−3 … t−1* | Same day | Count in `PM25_Roll3_n` | — |
| `PM25_Roll7` | `pm25_history` | `observed_ground_history` | Yes | Days *t−7 … t−1* | Same day | Count in `PM25_Roll7_n` | — |
| `day_sin` | `calendar` | `calendar` | Yes | Target date | Known in advance | — | Deterministic |
| `day_cos` | `calendar` | `calendar` | Yes | Target date | Known in advance | — | Deterministic |
| `month_sin` | `calendar` | `calendar` | Yes | Target date | Known in advance | — | Deterministic |
| `month_cos` | `calendar` | `calendar` | Yes | Target date | Known in advance | — | Deterministic |
| `AT` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | Daily mean from hourly export | No publication timestamp (Workstream D) |
| `RH` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `WS` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `WD` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `RF` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `TOT_RF` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `SR` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `VWS` | `station_met` | `station_local_unknown_availability` | No | Same day, hourly mean | Unknown | as above | as above |
| `T2M_midday` | `merra` | `reanalysis` | No | 11:30–13:30 UTC, same day | After the period ends | None in granules | Reanalysis is never a forecast input |
| `TS_midday` | `merra` | `reanalysis` | No | 11:30–13:30 UTC, same day | After the period ends | None in granules | as above |
| `QV2M_midday` | `merra` | `reanalysis` | No | 11:30–13:30 UTC, same day | After the period ends | None in granules | as above |
| `U10M_midday` | `merra` | `reanalysis` | No | 11:30–13:30 UTC, same day | After the period ends | None in granules | as above |
| `V10M_midday` | `merra` | `reanalysis` | No | 11:30–13:30 UTC, same day | After the period ends | None in granules | as above |
| `SLP_midday` | `merra` | `reanalysis` | No | 11:30–13:30 UTC, same day | After the period ends | None in granules | as above |
| `AOD` | `aod` | `observed_satellite` | No | Satellite overpass, same day | Unknown | Valid-pixel count only | Publication latency not established |
| `aod_valid_pixels` | `aod` | `observed_satellite` | No | Same overpass | Unknown | Pixel support | No QA/cloud flag in granules |
| `aod_overpasses` | `aod` | `observed_satellite` | No | Same day | Unknown | Number averaged | as above |

The target `PM25` is a label, not a feature, and never enters a feature set.

---

## 3. Exclusions that are part of the contract

| Excluded | Where | Why |
|---|---|---|
| `PBLH` / `PBLTOP` | MERRA-2 subset | Neither variable is present in the distributed granules; substituting `PBLTOP` would assert an equivalence the evidence does not support (D-008) |
| `BP` | CPCB station meteorology | Fails physical cross-validation: daily means up to 1062 mmHg, anti-correlated with MERRA-2 surface pressure, diurnal maximum at 10h (D-020) |
| Forward-filled environmental values | Fusion | A filled value is indistinguishable from an observed one; the availability mask replaces it (D-003) |
| Future observed AOD / meteorology | Operational forecasts | Not available when the forecast is issued |

Each excluded value remains visible as a `null` plus an availability column
(`station_met_observed`, `merra_observed`, `aod_observed`) and an observation
age in days. Missingness is data, not something to smooth away.

---

## 4. What the fused frame records

`data/processed/fused_dataset.csv` carries, for every date:

- one `*_observed` boolean per environmental source;
- one `*_observation_age_days` column, which is null when unobserved;
- `n_sources_observed` and `all_sources_observed`.

The coverage consequence of this contract is written to
`results/alignment/daily_alignment_report.json`: of 2,979 days,
`station_met` is observed on 2,979, `merra` on 172 (5.8%, 2024-01-01 to
2024-06-20) and `aod` on 139 (4.7%, 2024-01-07 to 2024-06-18). All three are
observed together on 139 days (4.7%), all in 2024 H1.

---

## 5. Enforcement and verification

```bash
python -m pytest tests/unit/test_availability_and_regime.py \
                 tests/unit/test_feature_contract_doc.py -q
```

`assert_operational_contract(["pm25_history", "calendar", "aod"])` raises,
which is the behaviour the forecast endpoint relies on: Track B estimation may
use AOD; an operational forecast may not.

This contract is a statement about availability, not about usefulness. A
retrospective-only group can still be valuable for estimation — proving that
is the purpose of the Track B experiments, and the current answer is negative.
