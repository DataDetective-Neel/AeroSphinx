# AeroSphinx — Software Engineering Project Submission

**AeroSphinx: An Availability-Aware Multi-Source Platform for PM2.5 Estimation and Forecasting over Faridabad, Delhi-NCR**

Team: Indraneel Chatterjee · Akshat Godika · Ansh
NIIT University, Neemrana, Rajasthan, India · November 2026

- Repository: <https://github.com/AkshatGo/AeroSphinx>
- Live dashboard: <https://aerosphinx.onrender.com>
- Baseline commit documented and tested: `7bbddda` (24 Nov 2026)

The combined report covering every artefact below is `AeroSphinx_Final_Report.pdf`, one level up next to this folder.

## What is in this folder

| Item | Contents |
|---|---|
| `01_Project_Proposal.pdf` | Problem, users, objectives, scope (in / out), deliverables, team |
| `02_SRS.pdf` | 28 functional and 10 non-functional requirements, external interfaces (13 API routes), 9 use cases, data requirements, acceptance criteria |
| `03_SDD.pdf` | Layered architecture, module and class design, data model, API contract, sequence and activity diagrams, error handling, security, technology stack, deployment, traceability matrix |
| `04_Wireframes.pdf` | Low-fidelity wireframes of all 10 dashboard modules plus the not-found page, mobile layout and command-line flow, with the guideline's review checklist |
| `05_Source_Code/` | Source code: Python package `src/aerosphinx`, `api/server.py`, React UI `ui/`, `scripts/`, `tests/`, `config/`, build and CI files, and the original chapter pipeline in `legacy_chapter_pipeline/` |
| `06_Test_Cases.xlsx` | 66 requirement-based test cases (positive / negative / boundary) with results, traceability to every requirement, NFR verification, use cases, and all 294 automated tests |
| `07_Test_Evidence/` | Logs and reports from the evidence run, plus UI screenshots — see `07_Test_Evidence/README.md` |

## Results at a glance

| Check | Result |
|---|---|
| Python tests (pytest) | 268 / 268 passed |
| UI tests (Vitest) | 26 / 26 passed |
| Requirement-based test cases | 65 / 66 passed (TC-50 failed — see below) |
| Scientific gates | Phase 6 availability gate PASS · check_claims 13/13 · check_docs PASS · check_reproducibility PASS |
| Static checks | ruff, black, isort clean · tsc clean · oxlint 0 errors |

**Open defect found during testing (TC-50, High):** `api/server.py` serves static files without checking that the
resolved path stays inside `ui/dist`, so a raw request such as `GET /../../api/server.py` returns files from outside the
web root. The proposed fix and a regression test are described in `03_SDD.pdf` §8. This was tested only against a local
copy, not against the live site.

## How to run the source code

Requires Python ≥ 3.10. Node.js ≥ 22.12 is needed only to build the UI.

The raw data (`data/`, ~87 MB) and the persisted results (`results/`) are **not** copied into `05_Source_Code/` to keep the
submission small. Run the commands from a full checkout of the repository, which contains them.

```bash
git clone https://github.com/AkshatGo/AeroSphinx.git && cd AeroSphinx
make install-dev            # creates .venv and installs the package with test tools
make test                   # 268 Python tests
make run-phase6             # fused dataset + availability gate
make run-matrix             # M0–M8 experiment matrix
make verify                 # tests + lint + check-claims + check-docs
python api/server.py        # API + built dashboard on http://localhost:8787
cd ui && npm ci && npm run dev    # dashboard dev server on http://localhost:5173
```

## Note on the implementation language

The guideline's sample prototype is a Java console application. AeroSphinx is written in Python, with a TypeScript/React
front end, because its scientific inputs (HDF5, NetCDF) and models (scikit-learn, XGBoost) depend on that ecosystem.
The concepts the guideline asks for are all demonstrated in this stack: layered design, OOP, validation, exception
handling and requirement-mapped testing.
