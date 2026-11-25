# 07_Test_Evidence

Every file here comes from one evidence run on **24 Nov 2026** against commit `7bbddda`, on a copy of the
repository (so the original `results/` folder was not modified).

Environment: Linux, Python 3.10.12, Node 22.23.5, scikit-learn 1.7.2, XGBoost 3.2.0.

| Folder | File | What it shows | Result |
|---|---|---|---|
| `01_python_tests/` | `pytest_output.txt` | Verbose console output of `pytest tests/ -v` | 268 passed |
| | `pytest_junit.xml` | Machine-readable results (also listed in 06_Test_Cases.xlsx › Automated Tests) | 0 failures |
| `02_quality_gates/` | `gate_run_phase6_fusion.txt` | Fused dataset build + availability/alignment gate | PASS |
| | `gate_check_claims.txt` | 13 README numbers re-derived from artefacts | 13/13 PASS |
| | `gate_check_docs.txt` | Documentation links and artefact paths | PASS |
| | `gate_check_reproducibility.txt` | Matrix regenerated and compared field by field | PASS |
| | `gate_cli_doctor.txt` | `aerosphinx --version` and `aerosphinx doctor` | PASS |
| | `python_lint.txt` | ruff, black --check, isort --check-only | Clean |
| `03_api_tests/` | `api_smoke_test.txt` | 15 GET requests and 8 POST /api/forecast cases, with timings | All as specified |
| | `api_security_test.txt` | Raw path-traversal requests against the static-file handler | **FAIL (TC-50)** |
| `04_frontend_tests/` | `ui_vitest_output.txt` | `npx vitest run` | 26 passed |
| | `ui_typecheck.txt`, `ui_lint.txt` | `npm run typecheck`, `npm run lint` | 0 errors (6 lint warnings) |
| `05_ui_screenshots/` | `*-desktop.png`, `*-mobile.png`, `report.json` | Captures of the deployed dashboard: 10 modules + not-found page at desktop and mobile widths, with the capture report (HTTP status, console errors). Captured by the team with `ui/scripts/screenshots.mjs` and committed to `docs/screenshots/deployed/` | All routes HTTP 200, no console errors |

Each requirement-based test case in `06_Test_Cases.xlsx` names the exact test or evidence file that verifies it.
