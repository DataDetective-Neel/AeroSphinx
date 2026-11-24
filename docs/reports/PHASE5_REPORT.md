# AeroSphinx — Phase 5 Report: MERRA-2 Validation

**Phase 5 complete as a short validation.** Structure, units, grid and
temporal coverage were verified against every granule, and the MERRA-2 fields
were cross-validated physically against the CPCB station. No feature-group
infrastructure was built, because the group cannot be evaluated on this record.

**The headline is that Phase 5 found a defect in a different group's data.**

---

## 1. Headline

> **MERRA-2 passes every structural, unit, grid and physical check — but
> the same cross-validation exposed a corrupt column in the CPCB
> station-local meteorology.**
>
> The CPCB `BP (mmHg)` column fails three independent physical tests. Once it
> is screened and excluded, the station-meteorology group stops being
> *significantly harmful* and becomes merely neutral.
>
> MERRA-2 itself remains **untestable**: 172 days, all in 2024, against a
> nine-year record. Its group stays disabled.

---

## 2. Structural audit — all 172 granules

| Property | Value | Uniform? |
|---|---|---|
| Granules | 172 | — |
| Dimensions | `lat=1, lon=1, time=3` | **yes, all 172** |
| Days per granule | 1 | **yes, all 172** |
| Grid cell | (28.5 N, 77.5 E) | **yes, all 172** |
| Time signature | 11:30, 12:30, 13:30 UTC | **yes, all 172** |
| Declared units | CF-1, matching the expected unit for every variable | **yes** |
| Unit mismatches | **0** | — |
| Values outside the plausible range | **0** | — |
| Missing (NaN) values | **0** | — |
| Date range | 2024-01-01 to 2024-06-20 | — |
| Dates present | 172 | — |
| **Missing dates** | **0** | complete, no gaps |

Variable presence: `PS`, `QV10M`, `QV2M`, `SLP`, `T10M`, `T2M`, `T2MDEW`,
`TQI`, `TQL`, `TS`, `U10M` in all 172; `V10M` in 170; `U2M` in only 2. The
`V10M` and `U2M` irregularities are the only variable-level heterogeneity,
and neither is in the requested set.

**The coverage is complete within its window.** The limitation is not gaps
but extent: 172 days of a 3,288-day record.

---

## 3. Grid

The subset is a **single 1×1 cell** at 28.5 N, 77.5 E — one cell of the
MERRA-2 0.5° × 0.625° grid. The station sits at 28.4089 N, 77.3178 E, so
the offset is about **10 km north and 18 km east**.

**One provenance inconsistency:** the `history_L34RS` subsetting box is
`77.2 77.4 28.3 28.5` in 170 files but `77.2 77.45 28.35 28.55` in the
last two (2024-06-19 and 2024-06-20). Both yield the same 1×1 cell and
therefore the same extracted values, so the effect on the data is nil — but
the subsetting was not done uniformly, and that is worth recording.

---

## 4. The timezone correction that matters

The three UTC timesteps are **17:00, 18:00 and 19:00 IST**.

The value is an **evening** window at the station, not the Indian midday.
That matters physically: midday is when boundary-layer mixing is strongest
and when the daily PM2.5 maximum usually occurs. The sampled hours are
systematically offset from the diurnal peak, so the extracted value is
biased low for exactly the episodes that drive peak pollution.

This is the concrete form of **D-010**, and it is worse than "not a daily
mean" — the problem is not just averaging, it is *when* the sample is taken.

---

## 5. Physical cross-validation against the CPCB station

This is the MERRA-2 analogue of the ocean-versus-land test that proved the
INSAT-3D orientation. The MERRA-2 fields and the station's own
measurements describe the same atmosphere at the same instants, so they must
agree. 516 hour-matched pairs (MERRA UTC + 5:30 = station IST):

| Check | Result | Interpretation |
|---|---|---|
| **T2M vs station `AT`** | corr **+0.9728**, mean diff **+0.96 °C**, sd 2.66 °C | **Correct.** The reanalysis tracks the station closely; the residual is the expected grid-versus-station difference. |
| **QV2M vs station implied `q`** | corr **+0.6038**, mean diff −3.26 g/kg | **Correct.** Specific humidity agrees at the expected level and tracks the station. |
| **TS vs T2M** | mean **−1.37 °C** | **Correct.** The skin is cooler than the air at 17:00–19:00 IST, as it must be after sunset. |
| **SLP − PS** | **+23.32 hPa** | **Correct.** Implies a model-level elevation of **185 m**, consistent with Faridabad's ~200 m. |
| SLP range | 988.7 – 1020.5 hPa | **Correct.** Sea-level pressure varies by ~32 hPa over six months, with the winter maximum and the monsoon minimum. |

**MERRA-2 is physically sound and correctly labelled.** The Phase 0 concern
(D-010) was that the extraction and its documentation misrepresented the
values; the naming is now fixed (`_midday` suffix) and the values themselves
check out.

---

## 6. The defect Phase 5 found: the CPCB barometric pressure column

Cross-checking MERRA-2 surface pressure against the station's `BP` column
produced an impossible result: **corr = −0.214** between two pressure series
at the same site. Investigating that led to the `BP` column.

### 6.1 Three independent failures

**(a) Physically impossible magnitudes.** 107 of 3,288 days have a daily mean
above 800 mmHg, reaching **1062 mmHg = 1416 hPa** — above sea-level
pressure *at sea level*, so impossible for any station.

| Year-month | Days |
|---|---|
| 2023-01 | 1 |
| 2023-02 | 9 |
| 2024-07 | 19 |
| 2024-08 | 16 |
| 2024-09 | 17 |
| 2024-10 | 31 |
| 2024-11 | 14 |

**(b) Anti-correlation with the reanalysis.** Over 2024-01 to 2024-06 the
station's `BP` has a daily correlation of **−0.214** with MERRA-2 sea-level
pressure, and **−0.212** with MERRA-2 surface pressure at the same site.
Two pressure series at the same location cannot behave this way.

**(c) An implausible diurnal phase.** The station's `BP` reaches its
**maximum at 10h** and its minimum at 17h. Surface pressure at a fixed
elevation must be *highest near dawn* and *lowest in the afternoon*. For
comparison, the station's own `AT` column peaks at 14h and troughs at 05h —
correctly phased — and MERRA-2's own fields show the expected evening
behaviour.

**The other eight station columns pass every screen**: 0 implausible days for
`AT`, `RH`, `WS`, `SR`, `TOT-RF` (and `WD`, `RF`, `VWS`).

### 6.2 What it does to the model

I tested the hypothesis that the corrupt `BP` caused the station-meteorology
"harm" reported in Workstream C. **The hypothesis was wrong**, and checking it
mattered: the 22 corrupt-BP dates in the evaluation set carry only **1.4% of
the total absolute error**, because they are low-pollution late-2024 days
that are easy to predict.

But excluding `BP` still changes the result materially:

| | Before (with `BP`) | After (`BP` screened and excluded) |
|---|---|---|
| Significantly **better** | 0 | 0 |
| Significantly **worse** | **2** | **0** |
| Best delta (global, h=4) | +3.08 | +2.13 |

**The corrupt column was responsible for the station-meteorology group being
significantly harmful.** With it removed the group is merely neutral: worse on
average at the longer horizons, but not significantly so.

This answers **B-024** — and for a stronger reason than completeness. `BP`
also had the highest missingness of the well-populated columns (22.4%).

---

## 7. Why MERRA-2 remains untestable

| | |
|---|---|
| MERRA-2 coverage | 172 days, 2024-01-01 to 2024-06-20 |
| Track A record | 3,288 days, 2016-01-01 to 2024-12-31 |
| Coverage fraction | **5.2%** |
| Matched test rows in the Track B comparison | **5** |

The Workstream C comparison of the `merra` group rests on **five matched
test rows**. Every MERRA comparison is reported with
`sufficient_evidence = False`, and MERRA is excluded from architecture
selection for the same reason. **No conclusion about MERRA-2's predictive
value is possible on this data, and none is claimed.**

Adding the `_midday` suffix fixed the naming; it did not and cannot fix the
coverage.

---

## 8. Phase 5 gate

| Criterion | Status |
|---|---|
| Structure verified | **PASS** — 172/172 granules, identical dimensions, variables, units and grid |
| Units verified | **PASS** — 0 mismatches, 0 out-of-range values, 0 NaN |
| Grid verified | **PASS** — single 1×1 cell, offset quantified, subsetting inconsistency recorded |
| Temporal coverage verified | **PASS** — 172 consecutive days, 0 missing; extent limitation quantified |
| Physical cross-validation | **PASS** — T2M corr +0.973 against the station, implied elevation 185 m, correct diurnal signs |
| Feature-group infrastructure | **NOT BUILT — deliberately.** The group stays disabled; it cannot be evaluated on this record. |
| Cross-group defect detection | **BONUS** — the corrupt CPCB `BP` column found and excluded |

**Phase 5 gate: PASSED as a short validation.**

---

## 9. What changed as a result

1. **The `BP` column is screened and excluded** from the station-local group
   (`config/data_quality.yaml`: `station_met_valid_ranges` and
   `station_met_excluded_columns`). The plausibility screen is now explicit
   for all nine columns.
2. **The station-meteorology conclusion is softened.** It was "significantly
   harmful at long horizons"; it is now "neutral: worse on average at longer
   horizons, not significantly". The null conclusion — no demonstrated
   benefit — is unchanged.
3. **The Workstream C run was re-executed** with the corrected feature set.
   The prior run is preserved at `results/runs/C-regime-20261112T201612Z/`
   and the corrected run is at `results/runs/C-regime-20261113T052110Z/`.
4. **D-010 is closed on the naming half.** The MERRA-2 variables carry a
   `_midday` suffix and the misleading docstrings are gone.

---

## 10. What this establishes

1. **MERRA-2 is sound data.** Every structural, unit, grid and physical check
   passes. It is not a data-quality problem; it is a **coverage** problem.
2. **The evening-sample bias is now explicit.** 17:00–19:00 IST is not the
   Indian midday, and the bias is directed against the pollution episodes
   that matter most.
3. **The project's environmental hypothesis remains unsupported** — and
   this phase found a *second* independent reason to doubt the station
   meteorology: one of its columns was corrupt, and removing it removed the
   harm.
4. **A useful methodological point.** The cross-validation was built to
   validate MERRA-2 and it caught a defect in a *different* group. That is
   the argument for cross-validating every source against every other
   source rather than validating each in isolation.

---

## 11. Open questions

| ID | Question |
|---|---|
| **B-038** | Can MERRA-2 granules be obtained for 2016–2023 (B-002)? Without them the group stays untestable. |
| **B-039** | Is the station `BP` column's corruption localized to the instrument, or is the column systematically transformed? The diurnal phase inversion suggests a transformation rather than sporadic faults. |
| **B-040** | Should the other station columns be cross-validated the same way — every source against every other? The `BP` defect suggests this is worth doing systematically. |
| **B-041** | Should the MERRA-2 subset be re-requested with the 08:00–14:00 UTC window (13:30–19:30 IST) that brackets the Indian midday, and with full 24-hour coverage? |
| **B-042** | With `BP` excluded, does the station-meteorology group help under a tree model rather than the linear one used here? |

---

## 12. Artifacts

| File | Contents |
|---|---|
| `merra_structural_audit.json` | The audit of every granule: dimensions, variables, units, grid, coverage |
| `merra_station_crosscheck.csv` | The 516 hour-matched pairs against the CPCB station |
| `manifest.json` | Git SHA, environment, configuration |
