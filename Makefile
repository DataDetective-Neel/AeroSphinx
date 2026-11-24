# AeroSphinx - developer entry points.
#
# Every target is safe to run from a clean checkout. Raw data under
# data/cpcb, data/aod and data/merra2 is never modified.

PYTHON ?= .venv/bin/python
PIP    ?= .venv/bin/python -m pip
ROOT   := $(shell pwd)
# Interpreter used to create the venv itself. Never the venv's own python.
BASE_PYTHON ?= python3

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev test test-unit test-integration audit \
        build-track-a run-track-a run-phase6 run-matrix run-sufficiency \
        run-alignment lint fmt check-claims check-docs check-deploy verify clean

venv: .venv/bin/python ## Create the virtual environment if it is absent

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.venv/bin/python:
	@echo "creating .venv with $(BASE_PYTHON)"
	$(BASE_PYTHON) -m venv .venv
	$(PIP) install --quiet --upgrade pip

install: .venv/bin/python ## Install the package
	$(PIP) install -e .

install-dev: .venv/bin/python ## Install the package with test dependencies
	$(PIP) install -e ".[dev]"

test: ## Run the full test suite
	$(PYTHON) -m pytest tests/ -v

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit -v

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration -v

audit: ## Regenerate the Phase 0 baseline audit artefacts
	$(PYTHON) scripts/audit_repository.py
	$(PYTHON) scripts/audit_data.py

build-track-a: ## Build the Track A daily dataset and manifest
	$(PYTHON) scripts/build_track_a_dataset.py

run-track-a: ## Run the Track A rolling-origin backtest
	$(PYTHON) scripts/run_track_a.py

run-phase6: ## Build the fused dataset and run the Phase 6 alignment gate
	$(PYTHON) scripts/run_phase6_fusion.py

run-matrix: ## Run the M0-M8 experiment matrix
	$(PYTHON) scripts/run_experiment_matrix.py

run-sufficiency: ## Compute per-season source sufficiency
	$(PYTHON) scripts/run_season_sufficiency.py

run-alignment: ## Emit daily coverage, missingness and alignment report
	$(PYTHON) scripts/run_daily_alignment.py

lint: ## Byte-compile, then check formatting and lint
	$(PYTHON) -m compileall -q src scripts tests
	$(PYTHON) -m black --check src scripts tests api
	$(PYTHON) -m isort --check-only src scripts tests api
	$(PYTHON) -m ruff check src scripts tests api

fmt: ## Apply black and isort; legacy/ and the legacy pipeline are untouched
	$(PYTHON) -m isort src scripts tests api
	$(PYTHON) -m black src scripts tests api

check-claims: ## Verify README numbers against persisted artefacts
	$(PYTHON) scripts/check_claims.py --verbose

check-docs: ## Verify every internal documentation link and artefact path
	$(PYTHON) scripts/check_docs.py

check-deploy: ## Verify the deployment serves the local HEAD commit
	$(PYTHON) scripts/check_deploy.py

verify: test lint check-claims check-docs ## Everything CI runs, locally
	@echo "verify: tests, formatting, lint, published numbers and doc links all pass"

clean: ## Remove build and cache artefacts
	rm -rf build dist *.egg-info .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
