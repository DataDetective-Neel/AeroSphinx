# AeroSphinx — availability-aware multisource PM2.5 research platform

[![CI](https://github.com/AkshatGo/AeroSphinx/actions/workflows/ci.yml/badge.svg)](https://github.com/AkshatGo/AeroSphinx/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Live](https://img.shields.io/badge/live-aerosphinx.onrender.com-2ea44f)](https://aerosphinx.onrender.com)

> **A Multi-Source Machine Learning Framework for Satellite-Fused PM2.5 Estimation
> and Five-Day Atmospheric Forecasting over Urban India**
> Raju Bhakta and Indraneel Chatterjee
> Mathematics and Basic Sciences, NIIT University, NH-8, Delhi–Jaipur Highway,
> Alwar, Neemrana, Rajasthan-301705, India

AeroSphinx is being rebuilt as **two independently evaluated research tracks**
over a shared, evidence-first infrastructure:

| Track | Question | Data |
|---|---|---|
| **A** | How far can historical ground measurements alone predict future PM2.5? | CPCB, 2016–2024 (2,979 daily days) |
| **B** | Does satellite and meteorological information actually add value? | CPCB + INSAT-3D AOD + MERRA-2 |

The current scientific position, established by the Phase 0 audit, is
documented in `docs/reports/RESEARCH_FINDINGS.md`. In short: the shipped baseline
reproduces exactly (81/81 metrics) but **does not** demonstrate that fusion
beats ground-only prediction, and its five-day "forecast" is a retrospective
estimation rather than an operational forecast. Those are the open questions
the two tracks exist to answer.

> **Start here:** [`docs/README.md`](docs/README.md) (index of every document) ·
> `AGENTS.md` (engineering contract) · `docs/reports/REPOSITORY_AUDIT.md`
> (audit) · `docs/reference/DEFECT_REGISTER.md` (defects) · `docs/reports/PHASE7_REPORT.md`
> (matrix and its negative result) · `docs/reference/FEATURE_AVAILABILITY_CONTRACT.md`
> (per-feature availability) · `docs/reference/EXPERIMENT_PROTOCOL.md` (how experiments
> must be run).

---

## Quick start

```bash
make install-dev          # install the package and test dependencies
make test                 # 253 Python tests
make build-track-a        # build the Track A daily dataset and manifest
make run-track-a          # run the Track A rolling-origin backtest
make run-phase6           # build the fused dataset and run the availability gate
make run-matrix           # run the M0-M8 matrix on identical eligible dates
make run-sufficiency      # per-season source sufficiency
make run-alignment        # daily coverage, missingness and alignment report
make check-deploy         # deployed commit == local HEAD
make check-claims         # README numbers vs persisted artefacts
```

Runtime is roughly two to four minutes on a standard laptop CPU. No GPU and no
network access are required; all input files are local.

### Current scientific position

Every number below is reproducible from the persisted artefacts:
`aerosphinx check-claims` re-derives each one from `results/` and fails if the
prose and the artefacts disagree. `results/CANONICAL.json` is the registry it
checks against.

The canonical Track A result is the M0–M8 matrix: 97 rolling-origin folds over
the full nine-year record, with 2,615 matched predictions for every
configuration. Ridge on ground history plus calendar reaches **21.44 MAE
against persistence's 22.09**, a delta of -0.65 whose day-level block bootstrap
interval [-1.14, -0.17] excludes zero. Resampling whole calendar years instead
of days widens it to [-1.20, +0.06], which does not. The honest reading is
therefore narrow: a gain of about 3% of the observed MAE, which holds in
winter, pre-monsoon and post-monsoon, reverses in monsoon, and is not robust to
which years happen to be in the record. It is not a general improvement in
day-ahead skill.

That matrix figure is the published Track A comparison. `scripts/run-track-a`
is a protocol-discovery tool with a narrower eligible set (2,510 predictions,
because it requires a complete seven-day warm-up) and its ridge delta has the
opposite sign (+0.34, p = 0.13); it is a diagnostic, not the result. The
pre-audit Phase 2 numbers are marked superseded in `docs/reports/PHASE2_REPORT.md`.

Phase 6 publishes the fused Track B dataset with an explicit availability
mask: all 2,979 ground rows are preserved, missing environmental values are
never filled, and all three sources are observed together on only 139 rows
(4.7%).

Phase 7 runs the M0–M8 ladder on identical eligible dates within each track.
On Track B's 80 fused dates persistence scores 22.98 and the ground-only ridge
28.59, but on the 49 dates where every configuration predicts the two are
indistinguishable (-0.25, interval [-5.66, +5.11]); the 5.6 MAE gap is
concentrated in the 31 dates outside that common set, which is exactly the
informative-missingness structure the protocol exists to expose. On those 49
dates adding AOD changes MAE by +0.21 [-2.19, +2.61], MERRA-2 by +4.24
[+1.26, +7.18] (Holm-adjusted p = 0.06) and full fusion by +4.70
[+0.30, +10.41] (Holm-adjusted p = 0.34). **No environmental ablation is
distinguishable from zero.** The result is negative and underpowered, and is
now reported as both: not "fusion costs 4.70 MAE" but "nothing here separates
fusion from the ground-only model on 49 days".

Full analysis: `docs/reports/PHASE2_REPORT.md` (Track A), `docs/reports/PHASE6_REPORT.md`
(fusion and availability), `docs/reports/PHASE7_REPORT.md` (matrix, ablations,
sufficiency), `docs/reference/FEATURE_AVAILABILITY_CONTRACT.md` (per-feature contract).

### Web application

`api/server.py` serves 13 read-only routes over the persisted artefacts, and
the Vite frontend in `ui/` renders them in ten modules. Nothing is fabricated:
an unavailable artefact produces an explicit empty state. The forecast endpoint
abstains rather than inventing a value.

```bash
python api/server.py            # API + built frontend on :8787 (or $PORT)
cd ui && npm ci && npm run dev  # frontend dev server on :5173
make check-deploy               # verify the deployment matches local HEAD
```

## Repository layout

```text
README.md                       What this is, how to run it, what it concludes
LICENSE                         MIT
CITATION.cff                    How to cite the work
CONTRIBUTING.md                 How to change the code without breaking the science
CHANGELOG.md                    Dated record of structural and scientific changes
AGENTS.md                       Research-engineering contract (read before editing)
Makefile                        Every reproducible entry point, in one place

config/                         Scientific rules as data, not code
  common.yaml                   Paths, station, calendar conventions
  data_quality.yaml             Quality rules applied during ingestion
  track_a.yaml                  Track A features, evaluation and baselines
  fusion.yaml                   Availability policy for the fused Track B dataset

src/aerosphinx/                 The research package (installable, typed)
  ingestion/                    Validated CPCB series with an audit trail
  features/                     Calendar-aware lags, availability masks, seasons
  evaluation/                   Splits, metrics, paired uncertainty
  models/                       Baselines and supervised regressors
  tracks/                       Track A runner, regime diagnostics, M0-M8 matrix
  quality/                      Coverage, alignment, run provenance

api/server.py                   Read-only artefact API (stdlib only) + SPA host
ui/                             React dashboard over that API
scripts/                        Audits, dataset builds, experiments, gates
tests/                          266 unit and integration tests

data/                           Inputs (read-only) and derived datasets
  cpcb/ aod/ merra2/            Raw provider granules, unchanged
  processed/                    Track A series and the fused Track B frame
  manifests/                    Run manifests: Git SHA, environment, input hashes

results/                        Every persisted result
  CANONICAL.json                Registry of published numbers
  experiment_matrix/            M0-M8 matrix — the canonical evaluation
  alignment/  sufficiency/      Coverage, missingness, per-season sufficiency
  baseline/                     Phase 0 data audit
  runs/                         Timestamped exploratory runs, one directory each
  legacy_chapter/               Chapter-pipeline outputs (33-day window)

docs/                           Documentation
  README.md                     Index of everything in this directory
  reference/                    Living specifications and registers
  reports/                      Dated phase reports and audits
  figures/                      Chapter figures (generated by the legacy pipeline)
  screenshots/                  Dashboard captures

legacy/                         Frozen pre-Phase-0 record, including the
                                single-script chapter pipeline
models/                         Trained joblib artefacts from the legacy pipeline

## Data provenance and known constraints

| Stream | Provider | Local coverage |
|---|---|---|
| PM2.5 | CPCB National Air Quality Monitoring Programme | 2016-01-01 to 2024-12-31, hourly |
| AOD 550 nm | ISRO / SAC, INSAT-3D Imager L2G V02R00 | 2024-01-01 to 2024-06-18 |
| Meteorology | NASA GMAO, MERRA-2 `tavg1_2d_slv_Nx` | 2024-01-01 to 2024-06-20 |

Two constraints follow directly from these files and are reported by the script
rather than worked around:

1. **The multi-source model is a 2024 model.** A modelling row needs CPCB, AOD
   and MERRA-2 on the same day. The satellite and reanalysis granules held
   locally cover January to June 2024 only, so although the CPCB record reaches
   back to 2016, the fused dataset is confined to 2024. Extending the study to
   earlier years requires downloading the corresponding AOD and MERRA-2
   granules; the script picks them up automatically once they are in `data/`.

2. **PBLH is unavailable.** Planetary boundary layer height appears in the
   theoretical formulation of Section 3, but the subsetted MERRA-2 granules
   distributed with this study contain neither `PBLH` nor `PBLTOP`. The script
   detects this, drops the feature, and records it in `metrics.json` under
   `features_unavailable` instead of substituting a proxy. Re-subsetting
   MERRA-2 with `PBLH` included is the single highest-value data improvement
   available to this framework.

## Reproducibility

`random_state=42` throughout, fixed hyperparameters, no automated search, and a
chronological split, so repeated runs on the same files give identical numbers.
The exact configuration of each run is written to `results/legacy_chapter/metrics.json`.

## Citing

`CITATION.cff` carries machine-readable citation metadata; GitHub renders it as
a "Cite this repository" button. If you use the code rather than the paper, cite
the repository and the commit you ran.

## Licence

MIT — see `LICENSE`. Provider data under `data/` retains its own terms: CPCB
observations, ISRO/SAC INSAT-3D products and NASA GMAO MERRA-2 fields are
redistributed here only to the extent their terms allow. Check the provider's
terms before reusing them.

## Acknowledgements

INSAT-3D Imager AOD products: Indian Space Research Organisation (ISRO), Space
Applications Centre (SAC), Ahmedabad. MERRA-2 reanalysis: NASA Global Modeling
and Assimilation Office (GMAO). Ground-level air-quality observations: Central
Pollution Control Board (CPCB), Government of India.
