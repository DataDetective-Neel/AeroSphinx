"""Command-line entry point for the AeroSphinx research package.

The console script declared in ``pyproject.toml``
(``aerosphinx = "aerosphinx.cli:main"``) resolves to this module. Until
this file existed the entry point was declared but unimplemented, so
``pip install -e .`` produced a command that failed on first use.

Every subcommand is a thin, argument-complete wrapper around an existing
script or module. Nothing scientific lives here: the CLI exists so that a
reader of the README can run the platform without memorising script paths,
and so that ``--help`` documents the supported surface.

Usage::

    aerosphinx --version
    aerosphinx audit                 # Phase 0 repository + data audit
    aerosphinx build-track-a         # rebuild the Track A daily dataset
    aerosphinx run-track-a           # rolling-origin Track A backtest
    aerosphinx run-fusion            # Phase 6 fused dataset and gate
    aerosphinx run-matrix            # M0-M8 experiment matrix
    aerosphinx check-claims          # README numbers vs persisted artefacts
    aerosphinx check-docs            # every internal link and artefact path
    aerosphinx check-reproducibility # re-run the matrix, compare with results/
    aerosphinx serve --port 8787     # API + built frontend

Subcommands that need the raw inputs print the missing path and exit
non-zero instead of failing with a traceback.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["main", "subcommands"]

ROOT = Path(__file__).resolve().parents[2]
"""Repository root, when the package is used from a source checkout."""

#: Script names under ``scripts/`` that the CLI can dispatch to.
_SCRIPTS = {
    "audit": ["audit_repository.py", "audit_data.py"],
    "build-track-a": ["build_track_a_dataset.py"],
    "run-track-a": ["run_track_a.py"],
    "run-fusion": ["run_phase6_fusion.py"],
    "run-matrix": ["run_experiment_matrix.py"],
    "run-sufficiency": ["run_season_sufficiency.py"],
    "run-alignment": ["run_daily_alignment.py"],
    "check-claims": ["check_claims.py"],
    "check-docs": ["check_docs.py"],
    "check-reproducibility": ["check_reproducibility.py"],
    "check-deploy": ["check_deploy.py"],
}


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("aerosphinx")
    except Exception:  # pragma: no cover - metadata absent in a bare checkout
        return "0.1.0"


def _repo_root(explicit: str | None) -> Path:
    """Locate the repository root, allowing an explicit override."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    for candidate in (ROOT, Path.cwd()):
        if (candidate / "scripts").is_dir() and (candidate / "config").is_dir():
            return candidate
    return ROOT


def _dispatch(name: str, argv: list[str], root: Path) -> int:
    scripts = _SCRIPTS[name]
    missing = [s for s in scripts if not (root / "scripts" / s).is_file()]
    if missing:
        print(
            f"aerosphinx {name}: missing script(s) under {root / 'scripts'}: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    for script in scripts:
        command = [sys.executable, str(root / "scripts" / script), *argv]
        result = subprocess.run(command, cwd=root)
        if result.returncode != 0:
            return result.returncode
    return 0


def _serve(root: Path, argv: list[str]) -> int:
    server = root / "api" / "server.py"
    if not server.is_file():
        print(f"aerosphinx serve: {server} not found", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, str(server), *argv], cwd=root).returncode


def _doctor(root: Path) -> int:
    """Report what is present and what a run would need."""
    checks = [
        ("package source", root / "src" / "aerosphinx"),
        ("scripts", root / "scripts"),
        ("config", root / "config"),
        ("CPCB raw hourlies", root / "data" / "cpcb"),
        ("INSAT-3D AOD granules", root / "data" / "aod"),
        ("MERRA-2 granules", root / "data" / "merra2"),
        ("processed datasets", root / "data" / "processed"),
        ("experiment matrix", root / "results" / "experiment_matrix"),
        ("built frontend", root / "ui" / "dist"),
    ]
    width = max(len(label) for label, _ in checks)
    for label, path in checks:
        print(
            f"{label.ljust(width)}  {'ok      ' if path.exists() else 'MISSING '}  {path}"
        )
    node = shutil.which("node")
    print(
        f"{'node (optional)'.ljust(width)}  {'ok      ' if node else 'absent  '}  {node or '-'}"
    )
    print(f"{'python'.ljust(width)}  ok        {sys.version.split()[0]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerosphinx",
        description="Availability-aware multisource PM2.5 research platform.",
        epilog="Every subcommand delegates to a script under scripts/.",
    )
    parser.add_argument(
        "--version", action="version", version=f"aerosphinx {_version()}"
    )
    parser.add_argument(
        "--root",
        default=None,
        help="repository root (default: auto-detected from the package location)",
    )

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    for name, help_text in [
        ("audit", "rebuild the Phase 0 repository and data audit artefacts"),
        ("build-track-a", "rebuild the Track A daily dataset and manifest"),
        ("run-track-a", "run the Track A rolling-origin backtest"),
        ("run-fusion", "build the fused dataset and run the availability gate"),
        ("run-matrix", "run the M0-M8 experiment matrix"),
        ("run-sufficiency", "compute per-season source sufficiency"),
        ("run-alignment", "emit daily coverage, missingness and alignment report"),
        ("check-claims", "verify README numbers against persisted artefacts"),
        ("check-docs", "verify internal documentation links and artefact paths"),
        (
            "check-reproducibility",
            "re-run the matrix and compare it with the committed artefacts",
        ),
        ("check-deploy", "verify the deployment serves the local HEAD commit"),
    ]:
        child = sub.add_parser(name, help=help_text, description=help_text)
        child.add_argument(
            "args",
            nargs=argparse.REMAINDER,
            help="arguments forwarded to the underlying script",
        )

    serve = sub.add_parser("serve", help="serve the read-only API and built frontend")
    serve.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8787)))
    serve.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))

    sub.add_parser("doctor", help="report which inputs and artefacts are present")
    return parser


def subcommands() -> list[str]:
    """Every subcommand name, as argparse will accept it.

    Public so documentation and CI can check that a command named in the
    README exists, without shelling out to a subprocess or reaching into
    argparse's private ``_SubParsersAction``.
    """
    parser = build_parser()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            return sorted(choices)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = _repo_root(getattr(args, "root", None))

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "doctor":
        return _doctor(root)
    if args.command == "serve":
        forwarded = ["--port", str(args.port), "--host", args.host]
        return _serve(root, forwarded)
    forwarded = list(getattr(args, "args", []) or [])
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    return _dispatch(args.command, forwarded, root)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
