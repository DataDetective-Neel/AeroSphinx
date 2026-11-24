# AeroSphinx — Research Findings

**Phase 0.** Findings established by the repository autopsy and baseline
reproduction.

Every finding is classified **VERIFIED** (directly supported by evidence
reproduced in this repository), **INFERRED** (logically derived but not
directly verified) or **BLOCKED** (insufficient evidence available).

Reproduce every number in this document with:

```bash
.venv/bin/python scripts/audit_repository.py
.venv/bin/python scripts/audit_data.py
.venv/bin/python aerosphinx_pipeline.py \
  --results-dir /tmp/repro/results --figures-dir /tmp/repro/figures --models-dir /tmp/repro/models
```

---

## Summary

Phase 0 did not produce a new predictive result. It produced a **reproducible
baseline** and an **evidence-backed account of what that baseline does and
does not establish**.

The headline conclusion is a negative one, and it is the most important
output of this phase:

> **The current repository does not demonstrate that multisource fusion
> improves PM2.5 prediction, and it does not contain a valid operational
> forecast.** Its best model has R² = 0.017 and its improvement over
> persistence is not statistically significant.

This is not a failure of the project. It is precisely what the Phase 0 gate
was designed to surface before further model work was built on top of an
unexamined foundation.

---

## Finding 1 — The baseline is exactly reproducible (VERIFIED)

The shipped benchmark reproduces **81 of 81 numeric metrics exactly** from a
clean output directory, with no modification to any input or output. Three
CSV artefacts are byte-identical; two differ only at ~1e-13 in floating-point
representation with identical metrics.

The environment is deterministic: `random_state=42` throughout, fixed
hyperparameters, a chronological split, and a validation window that ends
exactly where the test window begins.

**Implication.** Any metric change in a later phase is a genuine behavioural
change, not environmental noise. This is the precondition for the
experiment protocol's usefulness.

---

## Finding 2 — The fused multisource dataset is 166 rows (VERIFIED)

| Quantity | Value |
|---|---|
| CPCB daily rows available (2016-2024) | **2,979** |
| Fused modelling rows after requiring AOD + MERRA-2 | **166** |
| Coverage of the fused frame | 2024-01-07 to 2024-06-21 |
| Train / test split | 130 / 33 |
| Seasons represented | one (late spring to early summer) |

**Implication.** The three-source intersection, not the nine-year CPCB
record, determines what can currently be claimed. The plan's two-track
architecture is not merely good practice here — it is the only structure that
exposes the 18× difference in available data. Track A should be built first
and is not blocked by any external dependency.

---

## Finding 3 — Fusion does not beat the ground-only ablation (VERIFIED)

On **identical test rows**:

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Ground history only** | **16.617** | 23.671 | -0.050 |
| No satellite AOD | 17.205 | 22.976 | 0.011 |
| **Full multisource ensemble** | **17.298** | 22.902 | **0.017** |
| Persistence baseline | 19.918 | 24.186 | -0.096 |
| Satellite AOD only | 19.260 | 24.355 | -0.112 |
| Meteorology only | 20.119 | 26.811 | -0.347 |

The full ensemble is **worse on MAE** than the model that uses only ground
history and calendar features. Both the satellite-only and meteorology-only
models are worse than persistence.

**Implication.** There is currently **no evidence** that satellite or
meteorological information adds predictive value. This is the project's
central research question, and the current data does not support a positive
answer. The result must be reported as the finding it is — the plan's
Phase 7 gate requires fusion to "demonstrate measurable improvement on
matched observations to support an improvement claim", and that gate is
**not** met by the current evidence.

---

## Finding 4 — The improvement over persistence is not statistically significant (VERIFIED)

- Ensemble MAE 17.298 vs persistence MAE 19.918 (Δ = -2.621, n = 33).
- **Paired Wilcoxon signed-rank: p = 0.1495.**
- **Paired bootstrap 95% CI on the MAE difference: [-6.47, +1.39]** — spans
  zero. P(ensemble better) = 0.905.
- Without the 0.4 persistence anchor, the ensemble's MAE is **20.118** —
  *worse* than persistence.

**Implication.** The headline advantage is directionally consistent but not
established, and it is largely produced by the anchor rather than by the
learned model. Any write-up must publish the paired test and the interval,
not the point estimate.

---

## Finding 5 — The five-day "forecast" is a retrospective estimation (VERIFIED)

The forecast issued at origin 2024-06-16 received, for each horizon:

| Horizon | Date | AOD supplied | Pixel support | T2M supplied |
|---|---|---|---|---|
| 1 | 2024-06-17 | 0.6911 — **observed satellite** | 6 px | 317.73 |
| 2 | 2024-06-18 | 0.8607 — **observed satellite** | 6 px | 317.22 |
| 3 | 2024-06-19 | 0.8607 — **forward-filled** | — | 317.06 |
| 4 | 2024-06-20 | 0.8607 — **forward-filled** | — | 314.25 |
| 5 | 2024-06-21 | 0.8607 — **forward-filled** | — | 314.25 |

No external meteorological forecast product exists in the repository, despite
the docstring asserting one.

**Implication.** The repository contains **no valid operational forecast**.
The shipped Figure 7 and `five_day_recursive_forecast.csv` present
retrospective estimation under perfect environmental knowledge as
operational forecasting. This is the exact error the plan's §4 "critical
correction" exists to prevent, and it is the strongest single argument for
the plan's retrospective/operational separation.

---

## Finding 6 — Row-based lags silently misrepresent time (VERIFIED)

The CPCB daily series has **123 calendar gaps**, the longest **25 days**.
Because lags are computed with `shift()` on row position, **123 rows carry a
`PM25_Lag_1` that is not the previous calendar day** — a value up to 25 days
old presented as "yesterday's PM2.5".

**Implication.** The feature semantics in the published data dictionary are
wrong for 4.1% of the CPCB record. This does not materially affect the
current 166-row model (whose window is dense), but it will corrupt Track A
across the full nine-year record unless fixed in Phase 1.

---

## Finding 7 — The model is a seasonal climatology, not a physical model (VERIFIED)

| Feature | Importance |
|---|---|
| `month_sin` | **37.93%** |
| `PM25_Roll3` | 11.22% |
| `month_cos` | 6.29% |
| `PM25_Lag_1` | 6.19% |
| `PM25_Roll7` | 5.85% |
| `V10M` | 4.52% |

`month_sin` alone outweighs the three PM2.5 history features combined.

**Implication.** The model has learned "what does May look like" rather than
"what drives aerosol loading". The physical covariates are noise-fillers.
Reporting this as a multisource *environmental* model would misrepresent
what it does. The climatology baseline's own poor calibration (bias
-35.85 µg/m³) is consistent with this picture.

---

## Finding 8 — The AOD orientation is correct, but the support is thin (VERIFIED)

- **Orientation VERIFIED** via HDF5 dimension scales: `AOD.dim1 -> /latitude`,
  `AOD.dim2 -> /longitude`. The pipeline's `np.ix_` indexing is correct.
  *(This was the highest-risk silent-corruption candidate in the repository;
  it is clean.)*
- **Support is thin**: median **12 valid pixels** per granule in the +/-0.25°
  station window (max 25); **27 of 169 granules yield zero valid pixels**.
- **No QA or cloud-flag dataset exists** in the granules, so "valid" cannot
  be distinguished from "clear-sky retrieval".

**Implication.** A daily AOD derived from a handful of pixels carries real
uncertainty that the model cannot see. Pixel support should become a feature
and a documented quality gate.

---

## Finding 9 — MERRA-2 features are not what the code says they are (VERIFIED)

The granules hold a **1×1 grid cell** at (28.5, 77.5) with **three hourly
timesteps** (11:30, 12:30, 13:30 UTC). The granule's own `history_L34RS`
attribute records `Spatial: 77.2 77.4 28.3 28.5. Timesteps: 12 13 14`.

The pipeline's docstring claims "spatially averaged over the Faridabad
bounding box" and the code treats the result as a daily mean. Neither
statement is supported.

Additionally, **`PBLH` and `PBLTOP` are both absent from all 172 granules.**
The pipeline correctly drops the feature, but it carries a dead
`PBLTOP -> PBLH` fallback that would silently produce a physically invalid
feature if a future granule contained `PBLTOP`.

**Implication.** Documentation and feature naming must be corrected before
any MERRA-based claim is made, and the dead fallback removed.

---

## Finding 10 — The CPCB record changes character across the nine years (INFERRED)

| Year | Integer-valued hourly readings | Mean abs. successive-hour difference |
|---|---|---|
| 2016 | 96.1% | 19.26 µg/m³ |
| 2019 | 40.8% | 14.46 µg/m³ |
| 2023 | 0.3% | 9.92 µg/m³ |

Earlier years are raw hourly instrument values; later years are progressively
smoothed or pre-aggregated.

**INFERRED** because the cause cannot be established from local files: no
upstream CPCB documentation accompanies the exports. **UNKNOWN** whether the
smoothing is a temporal average (which would blunt sub-daily peaks) or a
different measurement mode.

**Implication.** Track A across the full record is trained across a change in
measurement character. Any backtest spanning 2016-2024 must test for
distribution shift, or it will report a "generalisation failure" that is
actually a measurement artefact.

---

## Finding 11 — The climatology baseline is mis-specified (VERIFIED)

The seasonal-naive baseline predicts a mean of 64.78 µg/m³ against an
observed test-window mean of 100.63 — a systematic **-35.85 µg/m³** bias,
MAE 35.87, R² -2.68.

The cause is that the baseline averages day-of-year across 2016-2023 while
the test window sits in a higher-PM2.5 period than that average.

**Implication.** A baseline this poorly calibrated flatters every model
compared against it. The baseline must be reported with its own bias, and
built from a more appropriate reference period.

---

## Finding 12 — Undocumented data is being discarded (VERIFIED)

The CPCB exports contain 25 columns. The pipeline uses **two**: `Timestamp`
and `PM2.5`. The remainder — including station-local `AT`, `RH`, `WS`, `WD`,
`RF`, `SR`, `BP` and `VWS` — is discarded.

**Implication.** These are same-station, same-timestamp meteorological
observations, available for the **entire 2016-2024 record**, at zero download
cost. They are a candidate source of genuine predictive signal for Track A,
subject to a provenance check (the columns' QA status is undocumented
locally). The current model leans on `month_sin` for 37.9% of its importance
while ignoring same-station temperature and humidity.

---

## Finding 13 — Legacy experimental history is recoverable but not preserved (VERIFIED)

407 tracked paths are deleted in the working tree. 341 are byte-identical to
files relocated under `data/`. **66 are absent from disk**; of those, 11 are
exact duplicates of another tracked path and **55 are unique**, recoverable
only from commit `859fd08`.

The unique set includes every legacy pipeline and notebook:
`scripts/run_research_pipeline.py` (662 lines), `ML model/ML_Model.py`,
`ML_Model.py`, `pawanai-project/pawanai.py` (1,106 lines),
`Data_Reading.ipynb`, plus chapter drafts and the original
`results/audit_report.md`.

**Implication.** The plan's repository-preservation rule requires legacy
results and model artefacts be preserved. They have been restored to
`legacy/` before committing the reorganisation, so the experimental history
remains inspectable and no content was lost.

---

## Finding 14 — The environment floor is wrong (VERIFIED)

`requirements.txt` states "Python 3.9 or later", but the pipeline uses PEP 604
union syntax (`X | None`) in return annotations at `aerosphinx_pipeline.py:263`
and `:493`, which requires **Python 3.10+**.

Separately, the tracked model artefacts (`models/*.joblib`) have no recorded
provenance linking them to a dataset, code version or metrics file.

---

## Open questions (BLOCKED)

| ID | Question | Blocked on |
|---|---|---|
| B-001 | Can a MERRA-2 subset with `PBLH` and full 24-hour coverage be produced? | NASA GES DISC subsetting service (network) |
| B-002 | Do INSAT-3D granules for 2016-2023 exist at the required cadence? | ISRO/SAC data request |
| B-003 | Are the 27 zero-pixel AOD granules cloud-flagged or an extraction artefact? | No QA dataset in the local granules |
| B-004 | Why did the CPCB record's character change after 2019? | Upstream CPCB provenance documentation |
| B-005 | Should the chapter draft, `.pbix` and `.docx` artefacts stay in the repository? | Project decision |

---

## Findings that constrain the next phase

1. **Build Track A first.** It is unblocked, has 18× the data, and two of the
   three highest-severity defects (D-001 lag correctness, D-013 aggregation)
   live in the CPCB ingestion path Phase 1 rebuilds anyway.
2. **Do not present the current multisource result as a positive finding.**
   The ablation evidence is negative and must be reported as such.
3. **Publish uncertainty with every metric.** The current point estimates
   would not survive review.
4. **Do not call anything an operational forecast** until a future covariate
   supply is demonstrated.
5. **Restore the legacy artefacts** to `legacy/` before committing the
   reorganisation. — **DONE in Phase 0**; 50 artefacts restored, 0 content
   loss.
