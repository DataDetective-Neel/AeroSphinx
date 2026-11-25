"""Tests for the delivery contract: claims registry, CLI, reportable n.

These guard the pieces that make the published numbers traceable. They are
cheap, they run in CI, and they exist because each one was broken once:
the console script pointed at a module that did not exist, the README quoted
numbers no artefact produced, and a season table published an R2 on two days.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_claims  # noqa: E402

from aerosphinx.cli import subcommands  # noqa: E402
from aerosphinx.tracks.experiment_matrix import (  # noqa: E402
    MIN_REPORTABLE_N,
    _grouped,
)

pytestmark = pytest.mark.unit


def test_canonical_registry_parses_and_is_namespaced():
    registry = json.loads((ROOT / "results" / "CANONICAL.json").read_text())
    assert registry["claims"], "the registry must carry at least one claim"
    ids = [claim["id"] for claim in registry["claims"]]
    assert len(ids) == len(set(ids)), "claim ids must be unique"
    for claim in registry["claims"]:
        assert claim["artifact"], claim["id"]
        assert "value" in claim, claim["id"]
        assert claim.get("where") or claim.get("json_path"), claim["id"]


def test_every_canonical_claim_holds():
    """The registry, the artefacts and the README must agree right now."""
    root = ROOT
    registry = check_claims._load_registry(root)
    results = [check_claims.check_claim(root, claim) for claim in registry["claims"]]
    failures = [result for result in results if not result["passed"]]
    assert not failures, failures


def test_checker_fails_a_claim_whose_value_moved(tmp_path):
    """A claim whose artefact value changed must be reported, not tolerated."""
    root = tmp_path
    (root / "results").mkdir()
    (root / "README.md").write_text("MAE is 1.00 today\n", encoding="utf-8")
    frame = pd.DataFrame([{"config": "M1", "MAE": 2.5}])
    frame.to_csv(root / "results" / "metrics.csv", index=False)
    claim = {
        "id": "moved",
        "artifact": "results/metrics.csv",
        "where": {"config": "M1"},
        "column": "MAE",
        "value": 1.0,
        "tolerance": 1e-6,
        "readme": "1.00",
    }
    result = check_claims.check_claim(root, claim)
    assert result["passed"] is False
    assert "2.5" in result["detail"]


def test_checker_fails_a_claim_missing_from_the_readme(tmp_path):
    root = tmp_path
    (root / "results").mkdir()
    (root / "README.md").write_text("no numbers here\n", encoding="utf-8")
    pd.DataFrame([{"config": "M1", "MAE": 1.0}]).to_csv(
        root / "results" / "metrics.csv", index=False
    )
    claim = {
        "id": "absent",
        "artifact": "results/metrics.csv",
        "where": {"config": "M1"},
        "column": "MAE",
        "value": 1.0,
        "readme": "1.00",
    }
    result = check_claims.check_claim(root, claim)
    assert result["passed"] is False
    assert "README" in result["detail"]


def test_cli_reports_a_version():
    result = subprocess.run(
        [sys.executable, "-m", "aerosphinx.cli", "--version"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "aerosphinx" in result.stdout.lower()


def test_cli_without_a_subcommand_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "aerosphinx.cli"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        check=False,
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()


def _predictions(n_by_season: dict[str, int]) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(42)
    for season, n in n_by_season.items():
        for index in range(n):
            target = 100.0 + rng.normal(0, 10)
            rows.append(
                {
                    "track": "B",
                    "config": "M1",
                    "model": "ridge",
                    "season": season,
                    "Target": target,
                    "Prediction": target + rng.normal(0, 5),
                }
            )
    return pd.DataFrame(rows)


def test_grouped_suppresses_a_season_below_the_threshold():
    frame = _predictions({"pre_monsoon": MIN_REPORTABLE_N + 5, "monsoon": 2})
    table = _grouped(frame, "season", "season")
    monsoon = table[table["season"] == "monsoon"].iloc[0]
    assert bool(monsoon["reported"]) is False
    assert pd.isna(monsoon["MAE"])
    assert pd.isna(monsoon["R2"])
    assert int(monsoon["n"]) == 2

    healthy = table[table["season"] == "pre_monsoon"].iloc[0]
    assert bool(healthy["reported"]) is True
    assert not pd.isna(healthy["MAE"])


def test_every_command_named_in_the_readme_exists():
    """The README is the front door; it may not name a command that is gone."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    known = set(subcommands())
    named = set(re.findall(r"aerosphinx ([\w-]+)", readme))
    assert named, "expected at least one aerosphinx invocation in the README"
    assert not sorted(named - known), sorted(named - known)


def test_every_make_target_named_in_the_readme_exists():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z][\w-]*):", makefile, re.MULTILINE))
    named = set(re.findall(r"^make ([\w-]+)", readme, re.MULTILINE))
    assert named, "expected at least one make invocation in the README"
    assert not sorted(named - targets), sorted(named - targets)
