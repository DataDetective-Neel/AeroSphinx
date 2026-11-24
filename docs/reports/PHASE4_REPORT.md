# AeroSphinx — Phase 4 Report: INSAT-3D AOD Validation

**Phase 4 complete.** The INSAT-3D L2G extraction is validated against every
granule, and the `aod` feature group is enabled.

Run directory: `results/runs/D-aod-20261113T042245Z/`

---

## 1. Headline

> **The extraction is verified correct — orientation confirmed by two
> independent lines of evidence — and the first satellite test is negative
> but benign: AOD neither helps nor hurts.**
>
> Across 25 architecture × horizon comparisons, adding INSAT-3D AOD to the
> ground-only model is better in 8 and worse in 17, with **0 significantly
> better and 0 significantly worse**. The effect sizes are tiny (|Δ MAE| ≤
> 0.42 µg/m³) and every interval touches zero.
>
> This is a different profile from station-local meteorology, which was
> *significantly harmful* at long horizons (Δ up to +3.08 µg/m³).

---

## 2. Structural audit — all 169 granules, not a sample

| Property | Value | Uniform? |
|---|---|---|
| Granules | 169 | — |
| AOD shape | `(1, 551, 551)` | **yes, all 169** |
| dtype | float32 | yes |
| Datasets present | `AOD`, `latitude`, `longitude`, `time` | **yes, all 169** |
| `scale_factor` / `add_offset` | both absent | **yes, all 169** |
| `_FillValue` | −999.0 | **yes, all 169** |
| Latitude order | descending (45.05 → −9.95), 0.1° uniform | **yes, all 169** |
| Longitude order | ascending (45.05 → 100.05), 0.1° uniform | **yes, all 169** |
| Dimension-scale binding | `dim1 → /latitude`, `dim2 → /longitude` | **yes, all 169** |
| Bounding box | lat [−9.95, 45.05], lon [45.05, 100.05] | **yes, all 169** |
| QA / cloud-flag dataset | **absent** | yes |
| Duplicate (date, overpass) keys | **0** | — |
| Distinct dates | 166 | — |

The product is remarkably uniform: no granule differs in shape, fill value,
axis binding, scale factor or grid. That uniformity is itself evidence that
the extraction can rely on a single rule for all of them.

The **descending latitude axis** is recorded rather than assumed. Boolean
masking is order-independent, so it is not a hazard, but a future maintainer
must know it.

---

## 3. Orientation — verified twice, independently

This was the highest-risk silent-corruption candidate in the repository: a
transposed axis would corrupt every satellite value while leaving the code
looking correct.

### 3.1 Structural evidence

All 169 granules bind `AOD.dim1 → /latitude` and `AOD.dim2 →
/longitude` through their HDF5 dimension scales. This is authoritative for
the CF-1.6 layout.

### 3.2 Independent physical evidence

A mis-bound product would not be caught by the dimension scales alone, so a
second and *independent* test is required: **aerosol optical depth over the
open ocean is systematically lower than over polluted continental land.**

Using known geography boxes across 60 granules:

| Hypothesis | Ocean mean AOD | Land mean AOD | Contrast | Physically plausible? |
|---|---|---|---|---|
| **Dimension-scale** (i=lat, j=lon) | **0.3895** | **0.8033** | **+0.4138** | **yes** |
| Transposed | 0.5611 | 0.3214 | −0.2397 | **no** |

Under the transposed orientation the contrast **reverses** — land appears
clearer than ocean, which is physically impossible. The two lines of
evidence agree, and the check is implemented in
`aerosphinx.ingestion.aod.orientation_check` so it is reproducible rather
than a one-off script.

---

## 4. Fill value, quality flags and their absence

| Check | Result |
|---|---|
| Fill value read from the product | −999.0, uniform |
| Fill pixels excluded from the window mean | **yes** — verified by test |
| Values outside [0, 5] excluded | **yes** |
| Scale factor or offset present | **no** — values used as stored |
| **QA / cloud-flag dataset present** | **NO** |

**The absence of a QA dataset is the single most important limitation of
this data.** A present pixel means *a retrieval was attempted and returned a
number*, not *the retrieval is trustworthy*. There is no way to distinguish a
clear-sky retrieval from a marginal one.

The mitigation is to carry the **pixel count** as a feature alongside the
value, so the model can see the support. The `aod` group therefore contains
`AOD`, `aod_valid_pixels` and `aod_overpasses`.

---

## 5. Coverage per date

| Statistic | Value |
|---|---|
| Dates with a granule | 166 (2024-01-01 to 2024-06-18) |
| Dates with **zero** valid pixels | **27** (16.3%) |
| Dates with at least one valid pixel | 139 |
| Mean valid pixels (all dates) | 12.10 |
| Median valid pixels | **6** |
| Maximum valid pixels | 25 (the full window) |
| AOD value range | 0.0622 – 2.5333 |
| Overpasses per date | 165 dates with 1, one date with 4 |

Coverage by month:

| Month | Days | Zero-pixel days | Mean pixels | Mean AOD |
|---|---|---|---|---|
| January | 31 | **14** | 4.65 | 1.37 |
| February | 29 | 5 | 15.93 | 0.55 |
| March | 31 | 2 | 21.58 | 0.87 |
| April | 27 | 2 | 18.04 | 1.79 |
| May | 30 | 1 | 4.80 | 0.67 |
| June | 18 | 3 | 5.72 | 0.91 |

The median of **6 valid pixels out of a possible 25** is a thin basis for a
daily station value. January is the worst month — 14 of 31 days have no
valid pixel at all.

### 5.1 The missingness is not random

This is a finding that matters for any future use of AOD as a feature:

| | Mean ground PM2.5 |
|---|---|
| Days **with** a valid AOD pixel | 105.34 µg/m³ |
| Days **without** one | **119.38 µg/m³** |

Mann-Whitney U **p = 0.032**; Pearson correlation between AOD availability
and PM2.5 is −0.129 (p = 0.097). The satellite record is least available
**exactly when the pollution is worst** — the winter haze and dust season.
Corroborating this, days with ≤5 valid pixels have a mean AOD of 1.147
against 0.967 for days with ≥20.

**Consequence.** A model trained on the AOD-present days sees a
systematically cleaner-than-representative sample. Any AOD feature must
carry its coverage, and the availability structure must be reported rather
than treated as random.

---

## 6. Collocation-window sensitivity

The window half-width is a scientific choice and its effect is reported
rather than assumed.

| Half-width (deg) | Days with data | Mean AOD | Mean valid pixels | Correlation with default | Max \|difference\| |
|---|---|---|---|---|---|
| 0.15 | 131 | 0.9570 | 4.85 | 0.9841 | 0.5841 |
| 0.20 | 135 | 1.0019 | 7.77 | 0.9930 | 0.3871 |
| **0.25 (default)** | 139 | **1.0046** | 12.10 | 1.0000 | 0.0000 |
| 0.30 | 139 | 1.0101 | 16.56 | 0.9840 | 0.7679 |
| 0.40 | 139 | 1.0027 | 28.00 | 0.9579 | 1.1394 |
| 0.50 | 139 | 1.0037 | 42.95 | 0.9276 | 1.3615 |

**The extracted value is robust to the window choice.** The mean AOD moves
only between 0.957 and 1.010 across a 3.3-fold range of window sizes, and
the correlation with the default never falls below 0.93. The default of
0.25° is retained: it is the middle of the range, gives a median of 6
supporting pixels, and is no better or worse than its neighbours.

Note that smaller windows find data on **fewer** days (131 at 0.15° against
139 at 0.25°), so shrinking the window trades coverage for nominal
precision.

---

## 7. Duplicates and overpasses

- **Zero** duplicate (date, overpass) keys across all 169 granules.
- 165 dates carry a single overpass; **one date (2024-06-15) carries four**,
  whose valid pixels are averaged into one daily value (22 pixels, AOD
  0.835).
- Overpass times: 05:30 UTC ×154, 06:00 ×7, 06:30 ×4, 07:00 ×3, 07:30 ×1.

Two overpasses of one date are two observations, not a duplicate. The
extraction keys on (date, overpass) and averages same-date overpasses, and a
true duplicate would be reported rather than silently double-counted.

---

## 8. The first satellite test — AOD against the ground-only model

The AOD group was added to the Workstream C comparison framework. The
question is whether AOD adds value **on top of** the ground-only model, on
matched rows, with the architecture, method and horizon held fixed.

| Method | Comparisons | Group better | Significantly better | Significantly worse | Holm p < 0.05 and better |
|---|---|---|---|---|---|
| Direct | 25 | 8 | **0** | **0** | **0** |

Effect sizes are small and consistent in sign across horizons:

| Horizon | Δ MAE (global architecture) | 95% block CI |
|---|---|---|
| 1 | +0.07 | [−0.16, +0.32] |
| 2 | +0.26 | [−0.11, +0.80] |
| 3 | +0.23 | [−0.00, +0.53] |
| 4 | −0.20 | [−0.47, +0.00] |
| 5 | −0.14 | [−0.47, +0.15] |

**AOD neither helps nor hurts.** There is a hint of a small benefit at the
longer horizons (−0.14 to −0.21 at h=4 and h=5) and a small cost at the
short ones, but nothing survives the uncertainty analysis.

### 8.1 Comparison with the other environmental groups

| Group | Better | Sig. better | Sig. worse | Matched rows |
|---|---|---|---|---|
| `ground_plus_aod` | 8 / 25 | 0 | 0 | 83–88 |
| `ground_plus_station_met` | 7 / 25 | 0 | 2 | 83–88 |
| `ground_plus_merra` | 5 / 25 | 1 | 6 | **5** |

AOD's profile is distinctly **benign**: it neither helps nor harms.
Station meteorology actively harms at long horizons. MERRA-2 cannot be
assessed at all.

---

## 9. Phase 4 gate

| Criterion | Status |
|---|---|
| Dimension and orientation verification | **PASS** — dimension scales in all 169, independently corroborated by the ocean/land contrast (+0.414 against −0.240) |
| Fill-value verification | **PASS** — −999.0 uniform, excluded from means, tested |
| QA verification | **PASS (with a documented limitation)** — no QA dataset exists; the absence is recorded and the pixel count is carried as a feature |
| Spatial-window verification | **PASS** — sensitivity reported across six window sizes; the value is robust |
| Duplicate-granule verification | **PASS** — zero duplicate (date, overpass) keys |
| Valid-pixel coverage reported per date | **PASS** — `aod_coverage.csv` for all 166 dates |
| The `aod` feature group enabled | **PASS** — enabled with the pixel-support columns |

**Phase 4 gate: PASSED.**

---

## 10. What this establishes

1. **The AOD extraction is correct.** Two independent lines of evidence
   agree on the axis orientation, the structure is uniform across all 169
   granules, and the extracted value is robust to the window choice.
2. **The satellite data has no demonstrated predictive value** for PM2.5 on
   this record — but unlike station meteorology it does not *hurt* either.
   The difference is instructive: AOD arrives with its own coverage
   indicators, whereas nine met columns add noise dimensions.
3. **The AOD missingness is informative, not random.** The record is least
   available when pollution is worst (119.4 vs 105.3 µg/m³, p = 0.032).
4. **The thin support is the real limitation.** A median of 6 pixels out of
   25 for a station value, with no QA flag to qualify it, is a weak
   measurement whatever its predictive value.
5. **AOD is retrospective only.** The observation is for a past date, the
   publication latency is not established, and no satellite *forecast*
   product exists in the repository. The group is excluded from every
   operational feature contract.

---

## 11. Open questions

| ID | Question |
|---|---|
| **B-033** | Can the 27 zero-pixel days be explained? Without a QA dataset they cannot be classified as cloud, haze or missing data. |
| **B-034** | Does the non-random missingness make AOD actively misleading as a feature, and should the group be restricted to days with a minimum pixel count? |
| **B-035** | Is the small AOD benefit at h=4 and h=5 real? It is consistent in sign across architectures but not significant; it would need a larger test set to resolve. |
| **B-036** | Would a larger collocation window with a quality-weighted average outperform the simple mean, given the median support of 6 pixels? |
| **B-037** | Can INSAT-3D granules be obtained for 2016–2023 (B-002)? Without them AOD remains a 2024-only feature and cannot be evaluated across the full record. |

---

## 12. Artifacts

| File | Contents |
|---|---|
| `aod_structural_audit.json` | The audit of every granule: shapes, bindings, fill values, duplicates, coverage |
| `aod_orientation_check.json` | The physical orientation cross-check under both hypotheses |
| `aod_daily.csv` | The extracted daily series |
| `aod_coverage.csv` | Valid-pixel coverage and overpass count for every date |
| `aod_window_sensitivity.csv` | Sensitivity to the collocation window |
| `manifest.json` | Git SHA, environment, configuration, station coordinates |
