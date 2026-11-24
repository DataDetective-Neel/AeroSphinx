# PR #1 integration — 2026-11-24

## Evidence and scope

**VERIFIED:** GitHub's pull-request API identified PR #1 (`hellomebaja`) as
the latest PR in DataDetective-Neel/AeroSphinx, merged on 2026-11-24.
The checkout was on `main` at `47b220b4f54ef010ef3e6a3b716c1847e3e8c4eb`,
with an interrupted merge of `441e3da50d5bc3a94ae71d1e977adb8ba2f42277`
(the PR merge commit), 415 status entries and 71 unmerged paths.

The incoming versions resolve conflicts where available. Files absent on
the incoming side are retained from the local side or merge base. Local-only
scripts, legacy figures and historical outputs are retained. This is an
integration of the existing research package, API, UI and documentation.

Before editing, the working files, index, merge metadata, Git history,
tracked-file inventory and SHA-256 hashes were saved under
`C:\Users\Indraneel\AppData\Local\Temp\AeroSphinx-pre-PR1-jisa8ia2`.
That local backup contains `working-tree.zip`, `history.bundle`,
`manifest.json` and `status.bin`.

## Defects corrected

* **VERIFIED — `.gitattributes`, CSV and legacy conversion rules:** staging
  converted four incoming raw CPCB files and four legacy text files from
  CRLF to LF. Committing those bytes would alter the preserved research
  record. Added `-text` overrides for raw CPCB CSVs and the frozen legacy
  tree, then restaged them. Proof: all 59 conflicted paths with an incoming
  version match its Git blob byte for byte in both the worktree and index.
* **VERIFIED — `tests/unit/test_claims_and_cli.py`, both CLI subprocess
  tests:** replacing the entire subprocess environment with POSIX paths
  caused Python's Windows random-number initialization to fail. Preserve
  the process environment and use `os.defpath`. Proof: both tests pass in
  the complete 240-test unit run after failing before the correction.
* **VERIFIED — `tests/unit/test_cpcb_ingestion.py`,
  `test_aggregation_is_independent_of_file_split`:** an unused write to
  `/dev/null` failed on Windows before the split-invariance assertion.
  Removed the unused write. Proof: the test and complete unit suite pass.

## Validation

Commands use `.venv/Scripts/python.exe` on this Windows checkout. Pytest
was run with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` and `PYTHONUTF8=1`.
Filesystem approval was needed for pytest's temporary fixtures.

| Command | Observed result |
|---|---|
| `python -m pytest tests/ -q --tb=short` | PASS: 268 tests, including the real-data availability gate; 124.93 seconds |
| `python -m pytest tests/unit -q --tb=short` | PASS: 240 tests |
| `python -m compileall -q src scripts tests api` | PASS |
| `python -m black --check src scripts tests api` | PASS: 78 files unchanged |
| `python -m isort --check-only src scripts tests api` | PASS with UTF-8 enabled |
| `python -m ruff check src scripts tests api` | PASS |
| `aerosphinx check-claims` | PASS: all 13 published claims |
| `aerosphinx check-reproducibility` | PASS: all three compared matrix tables within 1e-6 |
| `python scripts/check_docs.py` | PASS: 42 files, every reference resolves |
| `npm.cmd run typecheck` | PASS |
| `npm.cmd run lint` | PASS: six warnings, zero errors |
| `npm.cmd test` | PASS: 26 tests |
| `npm.cmd run build` | PASS |
| `aerosphinx doctor` | PASS: all listed resources present, including built UI |

**VERIFIED:** Track A daily, Track A features and the fused dataset each
retain 2,979 rows, identical to the incoming PR. Published artefacts and
scientific algorithms were not regenerated or changed by these fixes.

## Remaining limits

The installed Python 3.10 environment shares existing scientific packages;
the tests emit a NumPy binary-compatibility warning during netCDF imports.
Frontend lint warnings remain. Live deployment was not changed or checked.
Track B remains retrospective; D-002 still blocks operational multisource
forecasting. Nothing was pushed to GitHub.
