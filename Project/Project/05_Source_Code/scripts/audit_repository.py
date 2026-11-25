#!/usr/bin/env python
"""AeroSphinx Phase 0 - repository autopsy and baseline manifest.

Records the Git SHA, working-tree state, file inventory, dataset hashes,
dependency versions and the reproduced baseline metrics, then writes a
machine-readable manifest under results/baseline/.

This script is READ-ONLY with respect to tracked content: it inspects the
repository and writes only its own manifest.

Usage:
    python scripts/audit_repository.py [--root .] [--out results/baseline/manifest.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SKIP_DIRS = {".git", ".venv", "__pycache__", "node_modules", ".mypy_cache"}
EXT_COUNTS: dict[str, int] = {}


def run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return result.stdout.strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_state(root: Path) -> dict:
    sha = run(["git", "rev-parse", "HEAD"], root)
    branch = run(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    status = run(["git", "status", "--porcelain"], root)
    counts: dict[str, int] = {}
    for line in status.split("\n"):
        if not line:
            continue
        code = line[:2]
        counts[code] = counts.get(code, 0) + 1
    remotes = {}
    for line in run(["git", "remote", "-v"], root).split("\n"):
        if not line:
            continue
        parts = line.split()
        remotes.setdefault(parts[0], {})[parts[2]] = parts[1]
    return {
        "head_sha": sha,
        "branch": branch,
        "working_tree": {
            "porcelain_code_counts": counts,
            "total_entries": sum(counts.values()),
            "is_clean": not status,
        },
        "remotes": remotes,
        "commits": int(run(["git", "rev-list", "--count", "HEAD"], root) or 0),
        "tracked_files": len(
            [p for p in run(["git", "ls-files"], root).split("\n") if p]
        ),
    }


def file_inventory(root: Path) -> dict:
    counts: dict[str, int] = {}
    total = 0
    total_bytes = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            suffix = path.suffix.lower() or "(none)"
            counts[suffix] = counts.get(suffix, 0) + 1
            total += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
    return {
        "files_excluding_venv_and_git": total,
        "bytes_excluding_venv_and_git": total_bytes,
        "by_extension": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
    }


def dataset_hashes(root: Path) -> dict:
    """SHA-256 every raw input file so the baseline is tamper-evident."""
    manifest: dict[str, list[dict]] = {}
    for label, rel in (
        ("cpcb", "data/cpcb"),
        ("insat_aod", "data/aod"),
        ("merra2", "data/merra2"),
    ):
        directory = root / rel
        entries = []
        if directory.exists():
            for path in sorted(directory.iterdir()):
                if path.is_file():
                    entries.append(
                        {
                            "name": path.name,
                            "bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
        manifest[label] = entries
    return manifest


def dependency_versions(root: Path) -> dict:
    packages = [
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
    ]
    versions = {}
    for name in packages:
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:
            versions[name] = None
    return {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    out_path = args.out or (root / "results" / "baseline" / "manifest.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project": "AeroSphinx",
        "phase": "0 - repository autopsy and baseline freeze",
        "note": (
            "git.head_sha records the audited baseline commit, i.e. the state "
            "described by the Phase 0 audit. It is intentionally not updated "
            "to the commit that adds this manifest."
        ),
        "root_relative": str(root.name),
        "git": git_state(root),
        "inventory": file_inventory(root),
        "environment": dependency_versions(root),
        "dataset_hashes": dataset_hashes(root),
    }

    metrics_path = root / "results" / "metrics.json"
    if metrics_path.exists():
        try:
            manifest["baseline_metrics"] = json.loads(metrics_path.read_text())
        except json.JSONDecodeError as exc:
            manifest["baseline_metrics"] = {"error": repr(exc)}

    data_audit = root / "results" / "baseline" / "data_audit.json"
    if data_audit.exists():
        try:
            manifest["data_audit_file"] = str(data_audit.relative_to(root))
            manifest["data_audit_sha256"] = sha256_file(data_audit)
        except OSError:
            pass

    out_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"baseline manifest written -> {out_path}")
    print(f"  git HEAD          : {manifest['git']['head_sha']}")
    print(f"  tracked files     : {manifest['git']['tracked_files']}")
    print(f"  worktree entries  : {manifest['git']['working_tree']['total_entries']}")
    print(
        f"  inventory files   : {manifest['inventory']['files_excluding_venv_and_git']}"
    )
    for label, entries in manifest["dataset_hashes"].items():
        print(f"  {label:17s}: {len(entries)} files hashed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
