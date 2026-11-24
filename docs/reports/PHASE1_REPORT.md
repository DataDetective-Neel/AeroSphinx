# AeroSphinx — Phase 1 Report: CPCB Ingestion (Track A)

**Phase 1 complete.** Track A dataset built, validated and reconciled.
Awaiting approval to begin Phase 2 (Track A feature engineering and
baselines).

Reproduce everything in this report with:

```bash
make install-dev
make test
python scripts/build_track_a_dataset.py
```

---

## 1. What was built

| Component | Path | Purpose |
|---|---|---|
| CPCB ingestion | `src/aerosphinx/ingestion/cpcb.py` | Validated daily series with full audit trail |
| Calendar-aware features | `src/aerosphinx/features/lags.py` | The D-001 correction |
| Coverage analysis | `src/aerosphinx/quality/coverage.py` | Gap detection and calendar reconciliation |
| Provenance | `src/aerosphinx/quality/provenance.py` | Run manifests with Git SHA, env and input hashes |
| Dataset builder | `scripts/build_track_a_dataset.py` | End-to-end, read-only against raw data |
| Configuration | `config/common.yaml`, `config/data_quality.yaml`, `config/track_a.yaml` | Every rule in a file, not in code |
| Package metadata | `pyproject.toml`, `requirements.lock`, `Makefile` | Installable, pinned, reproducible |
| Tests | `tests/unit/`, `tests/integration/` | 40 tests |

**The legacy `aerosphinx_pipeline.py` was not modified.** It remains the
Phase 0 baseline and the compatibility entry point for the Track B pipeline
built in Phases 4–7.

---

## 2. Test results

```
$ .venv/bin/python -m pytest tests/ -q
40 passed in 0.64s
```

| Suite | Count | Status |
|---|---|---|
| Unit — CPCB ingestion | 13 | **PASS** |
| Unit — calendar-aware lags | 14 | **PASS** |
| Integration — Track A ingestion on real data | 13 | **PASS** |

All four CPCB test categories required by `AGENTS.md` §8 are covered, plus
the feature tests required for calendar-aware lags and rolling-window
leakage.

---

## 3. The Track A dataset

| Quantity | Value |
|---|---|
| **Accepted daily rows** | **2,979** |
| Date range | 2016-01-01 to 2024-12-31 |
| Calendar span | 3,288 days |
| Coverage | 90.6% |
| Calendar gaps | 123 (longest 24 missing days) |
| Rows with a complete 1-day lag | 2,855 |
| Features per row | 17 |

Per-year accepted days: 2016: 261, 2017: 254, 2018: 340, 2019: 310, 2020: 359,
2021: 362, 2022: 363, 2023: 364, 2024: 366.

**This is 18× the 166-row multisource frame.** Track A now has the data volume
its research question requires.

---

## 4. Defects corrected

### D-001 (Critical) — calendar-aware lags

`src/aerosphinx/features/lags.py` reindexes onto the full calendar *before*
shifting, so a missing date becomes an explicit `NaN` rather than being
bridged by a row-position shift.

**Measured on the real record:** 123 rows change from a stale value to `NaN`.
Mean staleness 3.5 days, worst case **25 days** (2017-11-03, which previously
received 2017-10-09's value of 54.23 µg/m³ while calling it "yesterday").

| Date | Legacy "yesterday" | True previous calendar day | Actual staleness |
|---|---|---|---|
| 2017-11-03 | 54.23 | 2017-11-02 **missing** | 25 days |
| 2016-11-13 | 308.15 | 2016-11-12 **missing** | 13 days |
| 2016-12-14 | 307.77 | 2016-12-13 **missing** | 12 days |
| 2019-02-06 | 70.46 | 2019-02-05 **missing** | 12 days |
| 2017-10-01 | 51.85 | 2017-09-30 **missing** | 10 days |

Rows where both legacy and calendar-aware lags have a value: **0
differences**. The correction is surgical — it changes exactly the rows that
were wrong.

The module also provides `observation_age_days`, which reports the staleness
directly, and `calendar_rolling_stats`, which reports how many observations
contributed to each rolling window (`PM25_Roll3_n`) so a mean built from one
observation is distinguishable from one built from three.

### D-013 (Medium) — hour-weighted aggregation

`aggregate_daily` pools every unique hourly observation across all files and
computes one hourly-weighted mean per date, instead of averaging file-level
daily means.

**The corrected output is identical to the legacy output on the real record**
(max absolute difference **0.0** across all 2,979 days), confirming the
correction is behaviour-preserving where files do not overlap. It is now
correct *by construction* rather than by accident of file layout.

The proof test uses a 20/4 hour split: the mean-of-file-means would score
55.0, the hour-weighted mean scores 25.0.

---

## 5. Two bugs found and fixed during Phase 1

Both were caught by tests written to the protocol, not by inspection.

**Bug 1 — silently dropped days.** A day whose *every* hour was rejected never
entered the groupby, so it vanished from both the output and the audit. The
day counts failed to reconcile: 3,232 accounted for against a 3,288-day
calendar. Fixed by reindexing onto the full calendar before aggregating.

**Bug 2 — truncated calendar span.** The reindex used the span of the
*surviving* rows, so days rejected at the hourly stage fell outside the span
entirely. Fixed by carrying the raw record's start and end through the
audit. The span now reconciles exactly: 2,979 + 253 + 56 = 3,288.

**This is precisely the class of defect Phase 0 warned about** — a silent
undercount that a reviewer reading only the output would never notice. The
reconciliation test now guards it permanently.

---

## 6. Audit trail

Every rejected record is attributed to a named reason.

| Reason | Count |
|---|---|
| `non_numeric_pm25` | 4,695 |
| `pm25_out_of_range` | 736 |
| `day_below_min_hours` | 253 |
| `day_no_valid_observations` | 56 |
| `unparseable_timestamp` | 0 |
| `off_grid_timestamp` | 0 |
| `duplicate_hour` | 0 |

**Reconciliation:** 78,912 hourly rows read → 73,481 used → 5,431 rejected
(4,695 + 736 = 5,431 ✓). Days: 2,979 accepted + 253 insufficient + 56 all
missing = **3,288 = calendar span** ✓.

The run manifest at `data/manifests/track_a_ingestion.json` records the Git
SHA, branch and dirty state, Python and package versions, SHA-256 of every
input file, dataset counts and date range, coverage statistics, the full rule
set in force, and the output paths — as `AGENTS.md` §9 requires.

---

## 7. Data quality rules now explicit

Every rule lives in `config/data_quality.yaml`, not in code:

| Rule | Value | Note |
|---|---|---|
| PM2.5 valid range | [0, 500] µg/m³ | Out-of-range values are **set to missing, never clipped** |
| Minimum hours per day | 18 of 24 | 75% coverage |
| Duplicate hour policy | `exclude_day` | A conflicting hour disqualifies the day rather than being silently resolved |
| Unparseable timestamp | reject | Counted and recorded |
| Off-grid hour | reject | Whole-hour timestamps required |

The duplicate policy deserves note: when the same hour appears in two files
with different values, choosing either one would fabricate an observation and
averaging them would double-weight the hour. The day is excluded instead.
**No duplicate hour exists in the current record** (CPCB files are
partitioned by year), but the rule is now enforced and tested.

---

## 8. Phase 1 gate

| Criterion | Status |
|---|---|
| CPCB ingestion produces a validated daily series | **PASS** — 2,979 days, every rejection attributed |
| A provenance record is produced | **PASS** — `data/manifests/track_a_ingestion.json` |
| Duplicate-hour test | **PASS** — unit + integration |
| Unequal-file-size aggregation test | **PASS** — the 20/4 hour split |
| Timestamp and missing-hour tests | **PASS** — unit + integration |
| Daily means are hour-weighted | **PASS** — D-013 corrected |
| Lags are calendar-aware | **PASS** — D-001 corrected, 123 stale rows eliminated |
| Day counts reconcile against the calendar | **PASS** — 2,979 + 253 + 56 = 3,288 |
| Raw data unmodified | **PASS** — `data/cpcb/` is read-only; SHA-256 verified unchanged |
| No new model trained | **PASS** — Phase 1 is ingestion only |
| No forward filling or imputation | **PASS** — gaps remain gaps |

**Phase 1 gate: PASSED.**

---

## 9. Remaining risks

1. **The record's character changes across years** (Phase 0, Finding 10).
   Integer-valued hourly readings fall from 96.1% in 2016 to 0.3% in 2023.
   This is unresolved and now matters more, because Track A spans the whole
   period. **Phase 2 must test for distribution shift** before any model is
   fitted across it, or a "generalisation failure" may be a measurement
   artefact.

2. **123 calendar gaps are now NaN in the lag features.** This is correct but
   costs rows. 2,855 of 2,979 rows have a complete 1-day lag. Phase 2 must
   decide how to handle the remaining rows — dropping them, or using
   shorter-horizon lags that skip gaps — and must report the choice.

3. **Station-local meteorology is still unused.** The CPCB exports carry
   `AT`, `RH`, `WS`, `WD`, `RF`, `SR`, `BP` and `VWS` for the full
   2016–2024 record at ~99% completeness. They are disabled pending the
   provenance check (B-007 / assumption H4). If provenance is confirmed they
   are a zero-cost source of genuine signal, and the Phase 0 audit found the
   model leans on `month_sin` for 37.9% of its importance.

4. **The climatology baseline is still mis-specified** (D-005). Phase 2 owns
   this.

5. **No CI yet.** The test suite runs locally but nothing enforces it.
   Phase 9 owns CI; the `Makefile` provides the entry point in the meantime.

---

## 10. Recommended Phase 2 scope

Phase 2 is Track A feature engineering and baselines. Recommended order:

1. **Fit the persistence and seasonal-naive baselines first**, on the full
   2,979-day record, with the D-005 climatology corrected.
2. **Add the regularized linear regression** as the simplest learned model.
3. **Test for distribution shift** across the record's character change
   before fitting RF and XGBoost — otherwise a spurious "poor generalisation"
   finding is likely.
4. **Report every comparison with uncertainty** (paired tests + bootstrap
   CI), per `docs/reference/EXPERIMENT_PROTOCOL.md` §5.3.
5. **Report by season and by year**, not just in aggregate, so the
   heterogeneity in the record is visible rather than averaged away.

Do **not** start Track B work in Phase 2. Track B needs pre-2024 INSAT
granules (B-002) to be worth anything, and Phase 4–6 are blocked on that
decision.

---

## 11. Open questions for the reviewer

| ID | Question | Impact if unanswered |
|---|---|---|
| **B-007** | What is the provenance of the root-level legacy CSVs, and of the CPCB export's own QA conventions? | Blocks the station-meteorology decision (risk 3) |
| **B-008** | Should the 124 rows without a complete 1-day lag be dropped, or handled with gap-skipping lags? | Changes the Track A dataset size by ~4% |
| **B-009** | Is a day with a duplicate hour better excluded entirely, or resolved by a documented precedence rule? | No effect on current data; matters if overlapping files are ever supplied |
