# AeroSphinx — Phase 6 Report: Multisource Fusion and Availability

**Phase 6 complete.** The fused Track B dataset now carries an explicit
per-source availability mask. Forward-fill is gone from the research package:
a missing environmental observation stays missing and is flagged, never
manufactured.

Run directory: `results/runs/F-fusion-20261113T071533Z/`

---

## 1. Headline

> **The forward-fill defect (D-003) is closed in the research package.**
>
> The fused dataset preserves all **2,979 ground rows** unchanged and adds
> **zero** environmental values that are not backed by a same-day source
> record. All **33** alignment checks pass.
>
> The binding limitation is now explicit and quantified: all three
> environmental sources are simultaneously observed on **139 of 2,979 rows
> (4.7%)**. The station meteorology covers 100% of the ground record, but
> MERRA-2 covers 5.8% and AOD 4.7% — both confined to 2024.
>
> **Track B ablations are not run here.** They are Phase 7, and they must use
> this fused frame's matched dates rather than a complete-case frame built by
> filling.

---

## 2. What Phase 6 built

| Artefact | Purpose |
|---|---|
| `src/aerosphinx/features/availability.py` | Canonical fusion with an explicit availability mask; the alignment gate |
| `config/fusion.yaml` | The fusion policy as configuration: no fill, source order, observation rule, matched-row threshold |
| `scripts/run_phase6_fusion.py` | Builds the fused dataset, runs the gate, writes the audit and manifest |
| `data/processed/fused_dataset.csv` | The published fused frame: 2,979 rows, 42 columns |
| `tests/unit/test_fusion_availability.py` | 18 unit tests, including the exact D-003 failure mode |
| `tests/integration/test_phase6_fusion.py` | 7 tests on the real record, including same-day provenance for every value |

The fused frame keeps the ground series exactly as it is: same rows, same
dates, same target. Environmental sources are left-joined on `Date`; a source
cannot add a row or extend the calendar.

### 2.1 The availability columns

| Column | Meaning |
|---|---|
| `<source>_observed` | Boolean. True only when that source has a genuine observation on that exact date. |
| `<source>_observation_age_days` | Calendar days since the most recent genuine observation of that source. **Diagnostic only** — it never supplies a value. |
| `aod_valid_pixels`, `aod_overpasses` | AOD support metadata. Set to `0` when no AOD observation exists; these are counts, not measurements. |
| `n_sources_observed` | Number of requested sources observed on the row. |
| `all_sources_observed` | Whether every requested source is observed on the row. |

`<source>` is one of `station_met`, `merra`, `aod`.

---

## 3. The D-003 correction

The defect was that `harmonise` in the legacy pipeline applied `ffill(limit=3)`
to AOD and every MERRA-2 variable, leaving 7 forward-filled AOD rows and 1
forward-filled MERRA-2 row in the 166-row modelling frame with no presence
indicator.

| | Legacy baseline | Phase 6 research package |
|---|---|---|
| Missing environmental value | Forward-filled up to 3 days | Preserved as `NaN` |
| Presence indicator | None | `<source>_observed` on every row |
| Staleness | Invisible | `<source>_observation_age_days` |
| Same-day provenance | Not checkable | Verified by 12 per-column checks |
| Unobserved rows carrying a value | 8 in 166 | **0 in 2,979** |

The legacy `aerosphinx_pipeline.py` is retained unchanged as the frozen Phase 0
baseline, per the repository-preservation rule. It is not the research path
and its forward-fill is documented as the baseline defect, not reused.

### 3.1 Tests that pin the correction

* `test_missing_environmental_values_are_not_forward_filled` — a source gap
  stays `NaN`, and the next observed value is not the stale one.
* `test_observation_age_records_staleness_without_supplying_a_value` — ages
  advance while values stay absent.
* `test_alignment_gate_rejects_a_forward_filled_value` — copying yesterday's
  value into a gap fails the gate with
  `merra:value_without_observation_flag`.
* `test_no_environmental_value_without_a_same_day_observation` — the same
  property on all 2,979 real rows.
* `test_no_source_value_is_forward_filled` — every present value equals the
  source's own record on that exact date.

---

## 4. The alignment gate

`validate_fused_dataset` runs 33 checks. A fused dataset that fails any check
is refused by `assert_fusion_alignment` and by the runner.

| Check family | Checks | Result |
|---|---|---|
| Calendar integrity — unique, chronological, ground row count, ground calendar, target preserved | 5 | **PASS** |
| Per-source mask consistency — value without flag, flag without value, observation-age consistency | 9 | **PASS** |
| AOD support — non-negative integer counts, value requires a valid pixel | 2 | **PASS** |
| Same-day provenance — every measurement and support column against its source record | 17 | **PASS** |
| **Total** | **33** | **PASS — 0 failed** |

The same-day provenance family is the important one: it compares the fused
value against the source frame on the identical date, so a filled or shifted
value cannot pass even if its mask were set consistently.

---

## 5. Availability on the real record

The fused frame spans the full Track A record: **2016-01-01 to 2024-12-31,
2,979 rows**.

| Source | Observed rows | Fraction | First | Last | Median age (days) | Max age (days) |
|---|---|---|---|---|---|---|
| `station_met` | **2,979** | **100.0%** | 2016-01-01 | 2024-12-31 | 0.0 | 0.0 |
| `merra` | **172** | **5.8%** | 2024-01-01 | 2024-06-20 | 11.5 | 194 |
| `aod` | **139** | **4.7%** | 2024-01-07 | 2024-06-18 | 16.5 | 196 |

| Combined availability | Rows | Fraction |
|---|---|---|
| All three sources observed | **139** | **4.7%** |
| At least one source observed | **2,979** | **100.0%** |
| No source observed | **0** | 0.0% |

"At least one source observed" is 100% only because the station meteorology
covers the whole record. That does **not** make the multisource fusion usable
across the record: the two sources that carry new atmospheric information —
AOD and MERRA-2 — exist only for 2024.

### 5.1 AOD support is thinner than the date count suggests

Of 166 AOD dates, **139** carry at least one valid pixel and become
observations; **27** are zero-pixel days and are correctly flagged
unobserved. The median AOD observation age across the record is 16.5 days,
reflecting that the source exists only in 2024 and the ground record is nine
years long. This is the Phase 4 finding carried into the fused schema, not a
new result.

### 5.2 Station meteorology discards 309 dates

The station-meteorology loader accepts **3,288** days, but the ground PM2.5
series has **2,979**. The fusion keeps only ground dates, so **309**
station-met days that correspond to rejected ground days do not enter the
fused frame. They are counted in
`per_source.station_met.source_dates_outside_ground_calendar` = 309 rather
than silently dropped.

---

## 6. Informative missingness: what this does and does not show

The summary reports the target mean on observed versus unobserved dates for
each source:

| Source | Target mean when observed | Target mean when unobserved | Difference |
|---|---|---|---|
| `merra` | 106.93 | 92.60 | +14.33 |
| `aod` | 105.34 | 92.84 | +12.50 |

**These full-record contrasts are confounded by coverage and must not be read
as informative missingness.** Both sources exist only in 2024, while the
unobserved set is dominated by 2016–2023; the difference partly measures
which years are covered, not which pollution conditions are covered.

The valid informative-missingness result is the Phase 4 one, computed inside
the 2024 window: days with no valid AOD pixel have a mean ground PM2.5 of
**119.38 µg/m³** against **105.34** on days with one (Mann-Whitney
p = 0.032). Phase 6 preserves that structure in the mask but adds no new
claim about it.

The same caution applies to the p-values reported in
`availability_summary.json` for the full-record contrast: they are computed
and reported for completeness, not as evidence.

---

## 7. Phase 6 gate

| Criterion | Status |
|---|---|
| Explicit availability mask replaces forward-fill | **PASS** — 5 mask columns, 0 filled values |
| No environmental value without a same-day source record | **PASS** — 0 violations across 2,979 rows and 17 provenance checks |
| Ground calendar preserved exactly | **PASS** — 2,979 fused rows, 0 extended, 0 dropped, target unchanged |
| AOD support metadata consistent | **PASS** — every AOD value has ≥1 valid pixel; 27 zero-pixel days flagged unobserved |
| Alignment tests | **PASS** — 18 unit + 7 integration, 202 tests in the suite |
| Fused dataset published | **PASS** — `data/processed/fused_dataset.csv` |
| Track B ablations and robustness | **NOT RUN — Phase 7** |
| Operational forecasting | **NOT RUN — Phase 8** |

**Phase 6 gate: PASSED.**

---

## 8. What changed as a result

1. **`src/aerosphinx/features/availability.py` is the canonical fusion path.**
   It is exported from `aerosphinx.features` and used by the Phase 6 runner
   and integration tests.
2. **`config/fusion.yaml` records the policy** rather than leaving it in code:
   no fill, the per-source observation rule, and the reused 30-row matched
   evidence threshold from the Workstream C comparisons.
3. **`data/processed/fused_dataset.csv` is the published Track B frame.**
   Phase 7 must build its matched-date comparisons from this file or the same
   builder, not from a filled frame.
4. **`make run-phase6`** rebuilds the dataset and reruns the gate.
5. **No existing result was overwritten.** The legacy outputs and every
   earlier run directory are untouched.

---

## 9. What this establishes

1. **The availability contract is now executable end to end.** The feature
   groups declare their availability class, the fusion emits the mask, and
   the gate refuses any value that is not backed by a same-day record.
2. **Coverage, not code, is now the Track B bottleneck.** 4.7% of rows have
   all three sources. Phase 7 can compare groups only on matched dates, and
   MERRA-2 remains untestable at 172 days (Phase 5).
3. **Fusion is estimation, not forecasting.** Every environmental group is
   retrospective-only, so this dataset supports Track B estimation. The
   operational contract still admits the ground-only groups alone (D-002,
   Phase 8).
4. **A masked frame is not a complete-case frame.** Dropping unobserved rows
   would reduce the fusion to 139 rows, all in 2024, with no winter and no
   pre-monsoon. Carrying the mask is what lets a model use the ground history
   while knowing when the environmental source is absent.

---

## 10. What remains

| Phase | Work |
|---|---|
| **7** | Track B ablations (ground-only, AOD-only, met-only, pairs, all three), matched-date discipline, missing-source robustness, coverage analysis |
| **8** | Operational issue-time availability contract and forecasting; closes D-002 and D-016 |
| **9** | CI, clean-environment reproducibility, consolidated research reporting |

Open data questions that still bind Phase 7: B-033 (why 27 AOD days are
empty), B-034 (whether thin-pixel AOD should be filtered), B-037/B-038
(granules for 2016–2023), B-041 (a MERRA-2 subset covering midday and a full
day).

---

## 11. Artifacts

| File | Contents |
|---|---|
| `data/processed/fused_dataset.csv` | The fused frame with all mask and provenance columns |
| `availability_mask.csv` | Date, target, per-source flags and ages, combined mask |
| `availability_summary.json` | Per-source coverage, observation ages, target contrasts, policy |
| `alignment_audit.json` | All 33 gate checks with their detail |
| `manifest.json` | Git SHA, environment, input paths, fusion rules, dataset counts |
