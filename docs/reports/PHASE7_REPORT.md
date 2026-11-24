# AeroSphinx — Phase 7 Report: M0–M8 Matrix, Seasonal Sufficiency and Delivery

This report covers the multisource extension: the M0–M8 experiment ladder
(spec Phases 6–11), the named daily-alignment and sufficiency artefacts
(Phases 1 and 5), and the read-only API and UI delivery (Phases 16–17). It
reports what the artefacts contain, including the negative results.

Every number below is copied from a persisted artefact under `results/`. The
run is `matrix-20261122T085518Z`, generated 2026-11-22T08:55:20Z.

---

## 1. Headline

1. The matrix runs all nine configurations on an identical eligible-date
   intersection within each track: **97 rolling-origin folds / 2,615
   predictions for Track A**, **3 folds / 80 predictions for Track B**.
2. **Track A:** the learned ground-only model (M1, ridge) beats persistence
   by **0.65 MAE** on 2,615 matched predictions (21.44 vs 22.09, R² 0.740 vs
   0.706). The gain is seasonal: it holds in winter, pre-monsoon and
   post-monsoon and reverses in monsoon.
3. **Track B:** on the 49 dates where every configuration predicts, adding
   AOD costs **+0.21 MAE** (M2 vs M1), adding MERRA-2 costs **+4.24** (M3 vs
   M1), and full fusion costs **+4.70** (M4 vs M1). No environmental group
   adds value.
4. On the **full** Track B set (80 predictions), persistence (22.98) beats
   every learned configuration by at least 5.6 MAE. The ranking changes on the
   common-date subset, which is the informative-missingness trap the protocol
   exists to prevent.
5. No configuration reaches stable seasonal specialisation: M7 can only
   predict **49 of 80** dates, and M8's fallback reproduces M7 exactly on
   those 49.

The central research answer remains **negative**: multisource fusion does not
improve day-ahead PM2.5 on this record, and the learned ground-only model's
advantage over persistence is small and season-dependent.

---

## 2. What was built

| Component | Path | Purpose |
|---|---|---|
| Experiment matrix | `src/aerosphinx/tracks/experiment_matrix.py` | One runner for M0–M8, one eligible frame and identical folds per track |
| Runner | `scripts/run_experiment_matrix.py` | Writes metrics, per-fold metrics, predictions, ablations, by-season/year tables |
| Alignment artefacts | `scripts/run_daily_alignment.py` | `daily_source_coverage`, `daily_source_missingness`, `daily_alignment_report` |
| Sufficiency | `src/aerosphinx/evaluation/sufficiency.py`, `scripts/run_season_sufficiency.py` | SUFFICIENT / WARNING / INSUFFICIENT / NOT_EVALUABLE per season × source |
| Seasonal contract | `src/aerosphinx/features/seasonal.py` | Single `assign_season`, `get_season_year` (season-start year), validators |
| API | `api/server.py` | Read-only routes for matrix, ablations, sufficiency, alignment |
| UI | `ui/src/modules/research-lab`, `data-observatory` | Panels bound to those routes, Track A/B separated |
| Contract document | `docs/reference/FEATURE_AVAILABILITY_CONTRACT.md` | Per-feature availability table, pinned by a test |

The matrix reuses `evaluation/splits.py` and `models/`; it does not implement a
second evaluation stack. Track B remains retrospective by construction: AOD
and MERRA-2 are same-day observations, so no Track B number is an operational
forecast (D-002).

---

## 3. Track A — ground-history ladder

Two configurations are eligible for Track A by design: M0 persistence and M1
ridge on ground history plus calendar. M2–M8 add environmental groups and
therefore belong to Track B; they are not silently run on Track A.

| Config | Model | n | MAE | RMSE | Median AE | R² | Mean bias |
|---|---|---|---|---|---|---|---|
| M0 | persistence | 2,615 | 22.09 | 33.71 | 13.61 | 0.706 | +0.06 |
| M1 | ridge | 2,615 | **21.44** | **31.71** | 14.37 | **0.740** | +0.52 |

Paired delta, M1 − M0: **−0.65 MAE** across 97 folds, 2017-01-01 to 2024-12-19.
M1 has a higher median absolute error (14.37 vs 13.61): it improves the tail,
not the typical day.

**By season (MAE, M0 → M1):**

| Season | M0 | M1 | Delta |
|---|---|---|---|
| winter | 35.01 | 33.39 | −1.61 |
| pre_monsoon | 19.80 | 18.42 | −1.38 |
| monsoon | 11.08 | 11.69 | **+0.61** |
| post_monsoon | 27.96 | 27.41 | −0.55 |

**By year:** M1 is better in 2018 and 2020–2024, and worse in 2017 (+1.70) and
2019 (+0.68). The advantage is not uniform across the record, consistent with
the documented change in measurement character (D-019 context, Finding 10).

---

## 4. Track B — environmental ladder

Eligible dates: 2024-03-07 to 2024-06-03, three 30-day folds. The full set has
80 predictions per configuration; the common-date intersection across all nine
configurations is **49 dates**, because M7 abstains elsewhere.

| Config | Groups added | n | MAE | RMSE | R² |
|---|---|---|---|---|---|
| M0 | none (persistence) | 80 | **22.98** | 31.97 | 0.023 |
| M1 | none (ridge) | 80 | 28.59 | 35.90 | −0.232 |
| M2 | AOD | 80 | 28.55 | 36.68 | −0.286 |
| M3 | MERRA-2 | 80 | 35.13 | 42.85 | −0.756 |
| M4 | AOD + MERRA-2 | 80 | 35.99 | 45.44 | −0.974 |
| M5 | M4 + season categorical | 80 | 35.25 | 44.74 | −0.914 |
| M6 | M4 + season interactions | 80 | 30.94 | 40.63 | −0.578 |
| M7 | independent seasonal models | 49 | 25.90 | 36.65 | −0.074 |
| M8 | M7 with global fallback | 80 | 33.86 | 44.13 | −0.862 |

**Common-date metrics (n = 49, identical target observations):**

| Config | MAE |
|---|---|
| M0 | 24.93 |
| M1 | **24.68** |
| M2 | 24.89 |
| M3 | 28.92 |
| M4 | 29.39 |
| M5 | 28.44 |
| M6 | 27.95 |
| M7 | 25.90 |
| M8 | 25.90 |

**Paired ablations on those 49 dates (delta MAE, negative is better):**

| Comparison | Delta |
|---|---|
| M2 vs M1 (AOD) | +0.21 |
| M3 vs M1 (MERRA-2) | +4.24 |
| M4 vs M1 (fusion) | +4.70 |
| M4 vs M0 (fusion vs persistence) | +4.45 |
| M5 vs M4 (season categorical) | −0.95 |
| M6 vs M4 (season interactions) | −1.43 |
| M7 vs M4 (seasonal specialists) | −3.48 |
| M8 vs M4 | −3.48 |
| M7 vs M1 | +1.22 |
| M8 vs M1 | +1.22 |

Reading these honestly:

- **AOD is neutral-to-harmful.** +0.21 MAE is not an improvement, and the AOD
  days are systematically more polluted (target mean 105.3 vs 92.8), so the
  comparison population itself changes.
- **MERRA-2 is harmful** as configured. The midday-only features are not a
  daily meteorological series (D-010).
- **Season terms reduce the damage but do not create skill.** M6 recovers
  1.43 of the 4.70 fusion penalty and still trails M1 by 3.27.
- **Specialists abstain rather than generalise.** M7 produces predictions on
  49/80 dates; on the remaining 31 the pre-monsoon training block has fewer
  than the 30-day specialist threshold. M8's fallback fills them but performs
  poorly (33.86 overall), and on the 49 common dates M8 is identical to M7 by
  construction.
- **The rank flip is the point.** Persistence wins by 5.6 MAE on the full set
  and loses by 0.25 on the common set. Any claim must name its population.

No paired significance test was run for these matrix deltas. They are
descriptive comparisons on identical dates, not evidence of a significant
difference; the earlier block-aware tests found no significant improvement over
persistence for the ensemble (D-006).

---

## 5. Seasonal sufficiency

Thresholds are declared in the sufficiency report: **< 30 observed rows** is
below the inference threshold, **30–89** supports a comparison but is a
WARNING, **≥ 90** (one full season) is SUFFICIENT, and zero observations is
NOT_EVALUABLE.

| Season | station_met | merra | aod |
|---|---|---|---|
| winter | SUFFICIENT (742) | WARNING (60) | WARNING (41) |
| pre_monsoon | SUFFICIENT (761) | SUFFICIENT (92) | WARNING (83) |
| monsoon | SUFFICIENT (997) | INSUFFICIENT (20) | INSUFFICIENT (15) |
| post_monsoon | SUFFICIENT (479) | NOT_EVALUABLE (0) | NOT_EVALUABLE (0) |

Counts: 5 SUFFICIENT, 3 WARNING, 2 INSUFFICIENT, 2 NOT_EVALUABLE. This is why
seasonal environmental claims are restricted to winter and pre-monsoon, and why
the matrix does not report monsoon or post-monsoon fusion skill.

---

## 6. Alignment

`results/alignment/daily_alignment_report.json`, 2,979 days (2016-01-01 to
2024-12-31):

| Source | Observed days | Fraction | Window |
|---|---|---|---|
| station_met | 2,979 | 100% | 2016-01-01 → 2024-12-31 |
| merra | 172 | 5.8% | 2024-01-01 → 2024-06-20 |
| aod | 139 | 4.7% | 2024-01-07 → 2024-06-18 |

All three sources are observed together on **139 days (4.7%)**, all in 2024 H1.
There are no days with zero sources observed. Missingness is informative:
days with AOD average 105.3 µg/m³, days without 92.8; MERRA-2 shows the same
direction (106.9 vs 92.6).

---

## 7. Delivery

- **API (13 read-only routes):** `/api/health`, `/status`, `/dataset`,
  `/experiments`, `/series`, `/coverage`, `/manifests`, `/experiment-matrix`,
  `/ablations`, `/seasonal-sufficiency`, `/alignment`, and the
  eligibility-gated `POST /api/forecast` (which abstains rather than inventing
  a value).
- **UI (10 routes):** Command Center, Research Lab, Data Observatory,
  Validation Matrix, Defect Register, Model Intelligence, Forecast Studio,
  Atmospheric Explorer, Reports & Reproducibility, Settings. All render
  persisted artefacts; empty states state why data is unavailable.
- **Deploy parity:** `scripts/check_deploy.py` compares the deployed commit to
  local HEAD; `make check-deploy` runs it. Cold starts are retried.
- **Tests:** 260 Python tests (unit + integration) and 25 UI tests.

---

## 8. Limitations

1. **Track B is retrospective.** Same-day AOD and MERRA-2 cannot enter an
   operational forecast; no operational Track B forecast exists (D-002, OPEN).
2. **Three Track B folds.** The environmental window is 2024 H1 only; fold
   noise is large, and no confidence interval is attached to the matrix
   deltas.
3. **M7/M8 are not a fair test of seasonal specialisation on this window.**
   Specialists abstain on 31/80 dates; the fallback result is dominated by the
   global model's errors.
4. **MERRA-2 midday features are not a daily series** (D-010), so the MERRA-2
   ladder tests a weaker construct than its name suggests.
5. **AOD has no QA/cloud flag**; a present pixel means a retrieval was
   attempted, not that it is trustworthy.

---

## 9. Reproduce

```bash
make run-matrix        # M0–M8 matrix -> results/experiment_matrix/
make run-sufficiency   # season × source status -> results/sufficiency/
make run-alignment     # coverage, missingness, report -> results/alignment/
make test              # 260 tests
make check-deploy      # deployed commit == local HEAD
```

---

## 10. Revision 2026-11-23 — inference added to Track B

The point estimates in §§1.2–1.4 are unchanged, but they were published
without uncertainty while Track A carried a full block bootstrap and a Holm
correction. Both tracks now go through the same treatment, and the Track A
ridge-vs-persistence pair joins the family so its own headline has an interval.

| Track | Comparison | Δ MAE | Day-level 95% CI | Year-blocked 95% CI | Holm p | Distinguishable from zero |
|---|---|---|---|---|---|---|
| A | M1 ridge − M0 persistence | −0.65 | [−1.14, −0.17] | [−1.20, +0.06] | 0.008 | days: yes · years: no |
| B | M1 ridge − M0 persistence (49 common dates) | −0.25 | [−5.66, +5.11] | [−5.66, +5.11] | 1.000 | no |
| B | M2 − M1 (AOD) | +0.21 | [−2.19, +2.61] | same | 1.000 | no |
| B | M3 − M1 (MERRA-2) | +4.24 | [+1.26, +7.18] | same | 0.059 | no |
| B | M4 − M1 (full fusion) | +4.70 | [+0.30, +10.41] | same | 0.340 | no |

Consequences for the claims in this report:

1. **§1.2 must be read as ordinal, not measured.** "Adding AOD costs +0.21,
   MERRA-2 +4.24, fusion +4.70" is a ranking of point estimates on 49 days.
   None of the three is distinguishable from zero, and the MERRA-2 pair sits
   at Holm p = 0.059. The claim supported by the data is "no environmental
   group demonstrably adds value on this window", which is what the report's
   narrative already says; the numerical attribution was stronger than the
   evidence.
2. **§1.4's 5.6 MAE gap is a coverage effect, not a modelling one.** On the
   49 common dates ridge and persistence are indistinguishable (−0.25). The
   gap lives in the 31 dates outside the common set, which is the
   informative-missingness structure the protocol exists to expose.
3. **The Track A headline is small.** −0.65 MAE against a 22.09 baseline is
   about 3% of the observed MAE, and the year-blocked interval includes zero.
   "Beats persistence" is defensible for the record as a whole; "generalises"
   is not.

Every row above is reproduced by `aerosphinx check-claims` against
`results/experiment_matrix/matrix_ablations.csv`; the register in
`results/CANONICAL.json` is the authority if this table and the CSV ever
disagree.
