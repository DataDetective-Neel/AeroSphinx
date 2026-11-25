"""Provenance records for reproducible runs.

Every experiment must save a manifest containing the Git SHA, environment
and package versions, input hashes, dataset counts and date range, feature
list, source availability rules, split boundaries, model configuration,
random seed, metrics and prediction artefact locations.

This module builds the subset of that manifest that applies to an
ingestion run. Later phases extend it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["ProvenanceRecord", "build_provenance", "write_provenance", "git_state"]


@dataclass
class ProvenanceRecord:
    """Provenance for one ingestion or experiment run."""

    run_id: str
    generated_utc: str
    git: dict
    environment: dict
    inputs: dict
    dataset: dict
    rules: dict
    outputs: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path, ignore: tuple[str, ...] = ()) -> dict:
    """Capture the Git SHA, branch and dirty state of the checkout.

    ``ignore`` lists path prefixes that do not count as dirt: a run that writes
    its own artefacts makes the tree dirty by definition, and that must not be
    recorded as 'this result came from uncommitted code'. Anything outside the
    output directory still counts, which is the case the field exists for.
    """

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True
        )
        return result.stdout.strip()

    try:
        # Read the raw porcelain output: ``run`` strips leading whitespace,
        # which would eat the first line's status column and corrupt the
        # path slice below. Porcelain v1 is exactly "XY <path>".
        raw = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True
        ).stdout
        status = [line for line in raw.splitlines() if line.strip()]
        if ignore:
            normalised = tuple(prefix.rstrip("/") for prefix in ignore)
            status = [
                line
                for line in status
                if not any(
                    line[3:].strip().strip('"').startswith(prefix)
                    for prefix in normalised
                )
            ]
        return {
            "head_sha": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "dirty_entries": len(status),
        }
    except FileNotFoundError:  # pragma: no cover
        return {"head_sha": None, "branch": None, "dirty": None}


def _package_versions() -> dict:
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "joblib",
        "matplotlib",
        "h5py",
        "xarray",
        "netCDF4",
        "scipy",
        "PyYAML",
    ):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:
            versions[name] = None
    return versions


def build_provenance(
    run_id: str,
    root: Path,
    inputs: dict,
    dataset: dict,
    rules: dict,
    outputs: dict | None = None,
    ignore_dirty: tuple[str, ...] = (),
) -> ProvenanceRecord:
    """Assemble a provenance record for an ingestion run.

    ``ignore_dirty`` names path prefixes that are written by this run, so the
    record does not report the run as having been produced from uncommitted
    code merely because it wrote its own output.
    """
    root = Path(root)
    return ProvenanceRecord(
        run_id=run_id,
        generated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        git=git_state(root, ignore=ignore_dirty),
        environment={
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": _package_versions(),
        },
        inputs=inputs,
        dataset=dataset,
        rules=rules,
        outputs=outputs or {},
    )


def write_provenance(record: ProvenanceRecord, path: Path) -> Path:
    """Write a provenance record to JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.as_dict(), indent=2) + "\n")
    return path
