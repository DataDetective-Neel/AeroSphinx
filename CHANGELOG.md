# Changelog

All notable changes to this repository are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html) for the package.

Because this is a research repository, a **scientific** change and a
**structural** change are called out separately: the first can change a
published number, the second never should.

## [0.2.0] — 2026-11-23

### Changed — structure

* Repository root reduced to the files that belong there. The legacy
  single-script pipeline moved to `legacy/aerosphinx_pipeline.py`; the chapter
  draft and corrections moved to `docs/reports/`; figures to `docs/figures/`.
* `docs/` split into `reference/` (authoritative for the current code) and
  `reports/` (dated, append-only), with an index at `docs/README.md`.
* `results/` grouped by producer, with `results/README.md` explaining what each
  directory is and which one is canonical. Legacy chapter outputs moved to
  `results/legacy_chapter/`; exploratory runs live only under `results/runs/`.
* The legacy-pipeline documentation moved out of the README into
  `docs/reference/LEGACY_CHAPTER_PIPELINE.md`; the README now leads with the
  current research position instead of the 33-day chapter numbers.

### Added

* `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`,
  `.editorconfig`, `.gitattributes`, issue and pull-request templates.
* `scripts/check_docs.py` (`make check-docs`): every relative link, heading
  anchor and backticked artefact path must resolve. Runs in CI.
* `results/CANONICAL.json` and `scripts/check_claims.py`: one registry of
  published numbers, re-derived from the artefacts on every CI run.
* `scripts/check_reproducibility.py`: regenerates the matrix and compares it
  field by field with what is committed.
* `src/aerosphinx/cli.py`: the console script declared in `pyproject.toml` now
  exists, with `doctor`, `serve`, `run-matrix`, `check-claims`,
  `check-reproducibility` and `check-docs`.

### Changed — science

* **Track B now carries uncertainty.** Every ablation pair has a paired
  moving-block bootstrap interval, a two-sided p-value and a Holm adjustment
  across its track's family. Track A's ridge-vs-persistence pair joined the
  family, so its headline has an interval for the first time: −0.65 MAE,
  day-level [−1.14, −0.17], year-blocked [−1.20, +0.06].
* Consequence: "adding AOD costs 0.21 MAE, MERRA-2 4.24, fusion 4.70" is a
  *ranking of point estimates* on 49 days, not a measurement. No Track B
  ablation separates from zero. The phase report, defect register, limitations
  and assumptions documents carry dated revisions saying so.
* Metrics computed on fewer than 30 paired days are suppressed rather than
  published; a season with two days previously carried an R² of −125.
* Artefacts are written with ten significant digits, and a manifest no longer
  reports `dirty: true` merely because the run wrote its own output.

### Fixed

* `pip install -e .` installed a console script that crashed: the entry point
  resolved to a module that did not exist.
* `make install-dev` failed on a clean checkout; it now creates `.venv`.
* No `LICENSE` file despite MIT in the metadata.
* CI was red and unrunnable for anyone else: the lint job failed and the test
  job ran inside a private container image.
* `requirements.txt` omitted `PyYAML` and `scipy` and claimed Python 3.9.
* `station_audit.__all__` exported a function that does not exist.

### Changed — front end

* Routes are code-split: the entry chunk fell from 1,029 kB to 15.7 kB, with
  ECharts (495 kB) fetched only by pages that draw.
* The font stack named a family that is not served, so every page had been
  rendering in the system fallback.
* Skip link, keyboard-dismissable drawer, scroll and focus reset on navigation,
  reduced-motion support, and a legible disabled button state.
* Timestamps render as `2026-11-23 16:17 UTC` everywhere instead of raw ISO.

## [0.1.0] — 2026-11-22

Initial state: Phase 0 audit through Phase 7 matrix, the dual-track evaluation,
the artefact API and the ten-module dashboard, with the chapter pipeline
retained as the reproducibility baseline.
