# AeroSphinx — Multisource Seasonal Forecast Audit (Phase 0)

**Gate:** Phase 0 of the Multisource Seasonal Daily Forecasting specification.
**Audit date:** 2026-11-22.
**Git:** branch `main`, HEAD `f080d8f`.
**Method:** direct inspection of the worktree and data files. Every claim below
is marked **VERIFIED** (inspected) or **INFERRED** (derived but not directly
proven). Nothing here is copied from earlier documentation without a check.

> This audit precedes implementation. The specification forbids large-scale
> implementation before the audit is complete, so this document establishes
> what exists, what is missing, and the smallest correct next step.

---

## 1. Executive summary

AeroSphinx is a **mature research repository**, not a blank project. Phases
0–6 of its own engineering contract are complete, and Phase 7 source ablations
have been run. The React application, the Python research package, the
availability contract, the fused dataset, and the eligibility-gated forecast
endpoint all exist.

The multisource seasonal specification is therefore an **extension**, and much
of its scientific substrate is already present:

| Spec requirement | Already present? | Evidence |
|---|---|---|
| Four North-Indian seasons, correct month mapping | **YES (in code, duplicated)** | `tracks/regime.py:57-61`, `tracks/track_a.py:84-97`, `tracks/cold_start.py:44-52` |
| Calendar-aware PM2.5 lags / rolling stats | **YES** | `features/lags.py` |
| Availability contract (operational vs retrospective) | **YES (in code)** | `features/groups.py` |
| Fusion without forward-fill | **YES** | `features/availability.py`; `fused_dataset.csv` |
| Rolling-origin chronological evaluation | **YES** | `evaluation/splits.py` |
| Persistence / seasonal-naive baselines | **YES** | `src/aerosphinx/models/baselines.py` |
| Ridge / RandomForest / XGBoost | **YES** | `src/aerosphinx/models/supervised.py` (`MODEL_REGISTRY`) |
| Global / seasonal / interaction / routing architectures | **YES** | `tracks/regime.py` |
| Source ablations (ground, +AOD, +MERRA, +all) | **YES (Phase 7 run)** | `results/F-phase7-20261113T102206Z/` |
| Eligibility-gated forecast (no hardcoded value) | **YES** | `api/server.py:forecast_endpoint` |
| Single authoritative `assign_season()` | **NO** | three duplicated private helpers |
| Season-year (DJF crossing a year boundary) | **NO** | no implementation found |
| M0–M8 experiment matrix as specified | **PARTIAL** | regimes map to M5–M8, not one-to-one |
| Operational (issue-time-valid) forecast | **NO** | Track B is retrospective only (D-002) |
| Seven specification documents | **NO** | only this audit exists |

**Verdict:** the foundation is sound and reusable. The work is to consolidate
season handling into one contract, formalise the M0–M8 matrix and per-season
sufficiency, and document the feature-availability and methodology contracts —
**not** to rebuild the pipeline.

---

## 2. Repository state (VERIFIED)

```
main @ f080d8f
Working tree: 4 modified files (api/server.py, Reports.tsx, ResearchLab.tsx,
              vite.config.ts), plus untracked results/ and screenshot artefacts.
```

| Component | Path | State |
|---|---|---|
| Research package | `src/aerosphinx/` | ingestion, features, evaluation, models, tracks, quality |
| API | `api/server.py` | std-lib HTTP server, 8 routes |
| Frontend | `ui/` | React 19, Vite 8, React Router 7, React Query 5, ECharts 6, Tailwind 4 |
| Config | `config/{common,data_quality,track_a,fusion}.yaml` | scientific rules externalised |
| Data | `data/{cpcb,aod,merra2,processed,manifests}/` | read-only raw + derived |
| Results | `results/runs/<RUN_ID>/`, `results/F-phase7-*/` | experiment artefacts |
| Tests | `tests/unit`, `tests/integration` | 15 test modules |
| Legacy | `legacy/` | preserved pre-Phase-0 record |

The API routes (VERIFIED, `api/server.py:303-313`): `/api/health`,
`/api/status`, `/api/dataset`, `/api/experiments`, `/api/coverage`,
`/api/series`, `/api/manifests`, `POST /api/forecast`.

---

## 3. Data source registry (VERIFIED)

| Stream | Product | Local files | Coverage | Frequency |
|---|---|---|---|---|
| CPCB ground PM2.5 | CPCB hourly export, Faridabad | 9 CSV | 2016-01-01 → 2024-12-31 | hourly |
| INSAT-3D AOD | ISRO/SAC L2G AOD V02R00 | 169 HDF5 | 2024-01-01 → 2024-06-18 | 1–2 overpasses/day |
| MERRA-2 | NASA GMAO `tavg1_2d_slv_Nx` | 172 NetCDF | 2024-01-01 → 2024-06-20 | 12/13/14 UTC |

Station: Faridabad (Sector 16A), 28.4089 N, 77.3178 E. Single station — **no
citywide spatial claim is supportable** (spec §2 item 13).

Timezone: CPCB timestamps are naive **local IST (UTC+05:30)**, no DST
(`config/common.yaml`). INSAT overpasses are UTC. MERRA midday window is UTC.

---

## 4. Canonical daily dataset (VERIFIED)

`data/processed/fused_dataset.csv` — **2,979 rows × 42 columns**,
2016-01-01 → 2024-12-31.

Availability flags (sum over rows):

| Source | Observed rows | Fraction |
|---|---|---|
| `station_met` | 2,979 | 100.0% |
| `merra` | 172 | 5.8% |
| `aod` | 139 | 4.7% |
| **all three** | **139** | **4.7%** |

Target `PM25`: n=2,979, mean 93.43, min 6.01, max 406.86 µg/m³.

Feature columns present: calendar-aware lags (`PM25_Lag_1/2/3/7`),
rolling (`PM25_Roll3/7` + counts), cyclic calendar (`day_sin/cos`,
`month_sin/cos`), station meteorology (`AT, RH, WS, WD, RF, TOT_RF, SR, VWS`),
MERRA midday (`T2M, TS, QV2M, U10M, V10M, SLP` `_midday`), AOD
(`AOD, aod_valid_pixels, aod_overpasses`), and per-source `_observed` and
`_observation_age_days` columns.

`data/processed/track_a_daily.csv` and `track_a_features.csv` each hold the
same 2,979 ground rows (ground-only features).

---

## 5. Daily aggregation rules (VERIFIED from code + config)

**CPCB** (`ingestion/cpcb.py`): hourly rows aggregated across all files at the
hour level, one hour-weighted daily mean. A day is accepted only with
**≥ 18 valid hours** (of 24); PM2.5 outside **[0, 500]** is set missing, never
clipped. Duplicate hours exclude the whole day. `coverage_fraction` is
retained, so a partial day is distinguishable from a full one. Missing day ≠
low-pollution day (the day is absent, not zero).

**INSAT AOD** (`ingestion/aod.py`): ±0.25° collocation window (max 25 px),
fill −999 removed, valid range [0, 5], same-day overpasses averaged,
`aod_valid_pixels` and `aod_overpasses` retained. No QA/cloud flag exists in
the granules. No AOD is fabricated for dates with no retrieval.

**MERRA-2** (`ingestion/merra.py`): single 1×1 cell at (28.5, 77.5), averaged
over the 12/13/14 UTC timesteps the granules contain. These are **midday
single-cell values, not daily means** (D-010). `PBLH`/`PBLTOP` absent (D-008).

---

## 6. Feature availability contract (VERIFIED — in code)

`src/aerosphinx/features/groups.py` defines groups with availability classes:

| Group | Class | Operational eligible |
|---|---|---|
| `pm25_history` | `observed_ground_history` | **yes** |
| `calendar` | `calendar` | **yes** |
| `station_met` | `station_local_unknown_availability` | no |
| `merra` | `reanalysis` | no (never) |
| `aod` | `observed_satellite` | no (retrospective) |

`assert_operational_contract` refuses an operational feature set containing an
ineligible group; `partition_by_availability` splits groups accordingly.

**Gap vs spec:** the contract exists as **code**, but the specification's
`docs/reference/FEATURE_AVAILABILITY_CONTRACT.md` deliverable does not. The per-feature
table (observation / ingestion / publication / availability timestamps,
quality flags, exclusion reason) is not yet written out.

---

## 7. Season handling (VERIFIED — duplicated, no season-year)

Seasons are already defined correctly and identically in three places:

```
winter        = (12, 1, 2)
pre_monsoon   = (3, 4, 5)
monsoon       = (6, 7, 8, 9)
post_monsoon  = (10, 11)
```

- `tracks/regime.py:57-61` (REGIMES mapping)
- `tracks/track_a.py:84-97` (`_season`)
- `tracks/cold_start.py:44-52` (`_season_of`)

**Gaps vs spec Phase 4:**
1. There is **no single authoritative `assign_season()`**; three private
   helpers implement the same rule. The spec requires one canonical function.
2. There is **no season-year (`get_season_year`)**: a DJF winter spanning
   Dec 2024 → Feb 2025 is not grouped as one season-year anywhere. This is
   required by spec Phase 4 and is not implemented.
3. No `validate_season_assignment` / `validate_temporal_order` /
   `validate_feature_availability` public validators.

Season assignment is deterministic on calendar month and is already asserted
not to use the target value (`tracks/regime.py` docstring + tests).

---

## 8. Existing models and architectures (VERIFIED)

`src/aerosphinx/models/baselines.py`: perseverance (`persistence_forecast`),
`SeasonalNaiveBaseline` (day-of-year climatology), `ClimatologyBuilder`.
`src/aerosphinx/models/supervised.py`: `MODEL_REGISTRY` with ridge, random_forest, xgboost.
All report training-only preprocessing, fixed seed (42).

`tracks/regime.py` implements architecture `kind ∈ {global, seasonal,
interaction, routing}` with a `min_specialist_days` threshold and fallback to
the global model. This maps loosely onto the spec's M5 (global+season), M6
(interactions), M7 (independent seasonal), M8 (seasonal with global fallback).

**Gap vs spec:** the M0–M8 matrix is not named or run as one shared experiment
runner with an identical eligible-date intersection across all eight. Phase 7
(`results/F-phase7-…`) ran **group-set** ablations (ground_only,
ground_station_met, ground_aod, ground_merra, ground_all), which is the
source-ablation half of the matrix, but not the full M0–M8 seasonal ladder.

---

## 9. Existing experiments (VERIFIED)

`results/runs/` contains:

| Run | Meaning |
|---|---|
| `A-roll…` (×2) | Track A rolling-origin backtest |
| `A-horizon…` | Track A multi-horizon (days 1–5) |
| `B-coldstart…` | cold-start / extrapolation diagnostics |
| `C-regime…` (×2) | regime architectures, ablations, by-season/horizon |
| `D-aod…`, `D-merra…`, `D-station…` | per-source validation |
| `F-fusion…` (×3) | fusion + availability gate |
| `F-phase7-…` | Phase 7 source ablations (5 group sets, matched rows) |

Each run writes `manifest.json` with input counts, fusion rules, source
coverage, thresholds and (where applicable) metrics/predictions.

**Note:** `AGENTS.md` states "Phase 7 not started", but a Phase 7 ablation run
exists (`results/F-phase7-20261113T102206Z/`). Documentation lags the artefacts.
This inconsistency must be reconciled in the next phase, not assumed away.

---

## 10. Forecast endpoint (VERIFIED — eligibility-gated)

`POST /api/forecast` was previously a stub returning a hardcoded `79.0` with
`status: VALIDATED`. It now performs eligibility checks and returns one of
`UNSUPPORTED`, `DEGRADED`, `ABSTAINED`, never a fabricated number:

| Gate | Status |
|---|---|
| malformed / missing payload | `UNSUPPORTED` |
| unknown model | `UNSUPPORTED` |
| horizon ≥ 3 | `DEGRADED` |
| required feature not observed | `UNSUPPORTED` |
| persistence (D-006) | `DEGRADED` |
| fusion (D-007) | `DEGRADED` |
| all checks pass, no inference endpoint | `ABSTAINED` |

There is **no inference endpoint** in the repository: forecasts are produced by
offline scripts. The endpoint correctly refuses to invent a value.

---

## 11. Known defects that bind this work (VERIFIED references)

| ID | Severity | Status | Bearing |
|---|---|---|---|
| D-002 | Critical | OPEN (Phase 8) | Track B is retrospective; no operational forecast exists |
| D-007 | High | Resolved as finding | Fusion does not beat ground-only (MAE 17.298 vs 16.617) |
| D-006 | High | Addressed | No model significantly beats persistence (p=0.15) |
| D-003 | High | Fixed | Forward-fill removed; availability mask replaces it |
| D-010 | Medium | Resolved/document | MERRA features are midday single-cell |
| D-011 | Medium | Partial | `month_sin` carries ~38% importance: model is seasonal climatology |
| D-016 | Low | Closed | No upward-drift extrapolation |

**The central research answer is currently negative:** multisource fusion has
not been shown to add predictive value, and no learned model has been shown to
beat persistence at p<0.05. The specification explicitly permits this result.

---

## 12. Research risks (INFERRED from verified data)

1. **Coverage is the binding constraint.** All-three-source overlap is 139 rows
   (4.7%), confined to 2024 H1. A seasonal model for any season other than
   winter/pre-monsoon has almost no multisource rows. M7 independent seasonal
   models on multisource features are likely **NOT_EVALUABLE** for monsoon and
   post-monsoon.
2. **MERRA-2 is untestable as a group** (172 days; ~5 matched test rows).
3. **AOD missingness is informative** (no-AOD days are more polluted,
   p≈0.032), so adding AOD changes the evaluation population — common-date
   comparison is mandatory.
4. **Seasonal climatology dominance** (D-011) means a "seasonal model" may
   simply be re-learning the annual cycle; interaction claims must be tested,
   not assumed.
5. **Data heterogeneity 2016→2023** (integer-valued readings 96.1% → 0.3%)
   means a model spanning the record crosses a measurement-character change.

---

## 13. Specification gap analysis

| Spec phase | Status | Note |
|---|---|---|
| 0 — Repository audit | **DONE (this file)** | |
| 1 — Daily alignment | **PARTIAL** | fused dataset exists; no `daily_source_coverage`/`missingness`/`alignment_report` artefacts as named |
| 2 — Availability contract | **PARTIAL** | code contract exists; per-feature doc/manifest missing |
| 3 — Forecast definition | **PARTIAL** | day-ahead defined in Track A; issue-time Track B blocked by D-002 |
| 4 — Seasonal feature pipeline | **PARTIAL** | seasons exist but duplicated; no `assign_season`, no season-year |
| 5 — Data sufficiency | **PARTIAL** | per-fold coverage diagnostics exist; no formal SUFFICIENT/INSUFFICIENT/NOT_EVALUABLE by season↔source |
| 6–10 — M0–M8 models | **PARTIAL** | group ablations + regime architectures; not one M0–M8 runner |
| 11 — Chronological validation | **DONE** | rolling-origin in `evaluation/splits.py` |
| 12 — Models/metrics | **DONE** | MAE/RMSE/R², paired tests, bootstrap CIs |
| 13 — Selection/abstention | **PARTIAL** | routing fallback exists; response taxonomy partly in API |
| 14 — Diagnostics | **PARTIAL** | by_season/by_year/cold_start artefacts; no observed-vs-pred plots per spec list |
| 15 — Reproducibility | **DONE** | manifests with hashes, seed, env |
| 16 — API | **PARTIAL** | 8 routes; no experiment-detail/ablation/seasonal-sufficiency routes |
| 17 — Frontend | **PARTIAL** | 10 routes exist; no M0–M8/seasonal-sufficiency views |
| 18 — Tests | **PARTIAL** | 15 modules incl. leakage/lags/availability; no M0–M8 or season-year tests |
| 19 — Documentation | **NOT DONE** | only this audit of the seven required docs |

---

## 14. Implementation plan (ordered, smallest-correct)

Consistent with the spec's own ordering and the repo's phase discipline:

1. **Phase 1** — Emit the named daily-alignment artefacts
   (`daily_source_coverage`, `daily_source_missingness`,
   `daily_alignment_report`) from the existing fused frame. No new data source.
2. **Phase 2** — Write `docs/reference/FEATURE_AVAILABILITY_CONTRACT.md` from
   `features/groups.py` and the fused columns; add a machine-readable
   per-feature availability table.
3. **Phase 4** — Add one authoritative `features/seasonal.py` with
   `assign_season`, `get_season_year`, `validate_season_assignment`; replace
   the three duplicated helpers with it; add season-year tests (Dec 2024–Feb
   2025 → one winter).
4. **Phase 5** — Add per-season × per-source sufficiency status
   (SUFFICIENT / INSUFFICIENT / WARNING / NOT_EVALUABLE) with documented
   criteria; expect monsoon/post-monsoon multisource to be NOT_EVALUABLE.
5. **Phase 6–11** — Add a single `tracks/experiment_matrix.py` runner that
   executes M0–M8 on an identical eligible-date intersection per track, reusing
   `evaluation/splits.py` and `models/`. Keep Track A (retrospective) and Track
   B (issue-time-valid) separate; Track B currently reduces to ground-only.
6. **Phase 16–17** — Add read-only API routes and UI panels that render
   persisted artefacts only; extend the existing eligibility-gated forecast,
   never reintroduce a fabricated value.
7. **Phase 18–19** — Tests and the remaining six documents, written **after**
   results exist, with negative findings reported honestly.

---

## 15. Phase 0 gate status

| Requirement | Status |
|---|---|
| Inspect AGENTS.md, README, source, data, config, tests, API, UI | **PASS** |
| Identify authoritative datasets | **PASS** — data/cpcb, data/aod, data/merra2 |
| Determine temporal resolution and timezone | **PASS** — §3, §5 |
| Determine daily aggregation rules | **PASS** — §5 |
| Determine station coverage and alignment | **PASS** — single station; 139 all-source rows |
| Identify existing models / experiments / availability evidence | **PASS** — §6, §8, §9 |
| Record known defects and risks | **PASS** — §11, §12 |
| Produce implementation plan | **PASS** — §14 |
| Begin large-scale implementation | **NOT STARTED** — correctly deferred until audit approval |

**Phase 0 is complete. Proceed to Phase 1 (named daily-alignment artefacts),
then Phase 4 (single seasonal authority), per the plan above.**
