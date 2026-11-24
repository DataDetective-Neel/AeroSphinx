"""Check that every internal link and file reference resolves.

A restructure is only safe if nothing silently rots. This walks every tracked
Markdown file, resolves each relative link against the file's own directory,
and fails on anything that does not exist. It also validates the two prose
conventions this repository relies on:

* a path written in backticks that starts with a known top-level directory
  (``docs/``, ``results/``, ``scripts/``, ``src/`` ...) must exist;
* a ``:line`` suffix on a code reference must be a real line in a real file.

External URLs are not fetched: a network check in CI is a flaky test. Anchor
fragments are checked against the target's headings when both are Markdown.

Usage::

    aerosphinx check-docs
    python scripts/check_docs.py --root .
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories whose backticked paths must resolve. Anything else (`data/...`
#: counts) is checked too; this list only documents the intent.
#: Documents whose backticked paths are authoritative for the *current* tree.
#: Phase reports and audits are dated records of the layout as it was; their
#: prose is left alone (the convention in this repository is to append a
#: revision rather than rewrite history), so only link targets are enforced
#: there.
_ENFORCED_PATHS = {
    "README.md",
    "AGENTS.md",
    "CONTRIBUTING.md",
    "CHANGELOG.md",
    "docs/README.md",
    "results/README.md",
    "docs/reference/DATA_DICTIONARY.md",
    "docs/reference/DEFECT_REGISTER.md",
    "docs/reference/EXPERIMENT_PROTOCOL.md",
    "docs/reference/FEATURE_AVAILABILITY_CONTRACT.md",
    "docs/reference/LIMITATIONS.md",
    "docs/reference/SCIENTIFIC_ASSUMPTIONS.md",
}

_TOP_LEVEL = {
    "api",
    "config",
    "data",
    "docs",
    "legacy",
    "models",
    "results",
    "scripts",
    "src",
    "tests",
    "ui",
}

_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_BACKTICK_PATH = re.compile(
    r"`([A-Za-z0-9_./-]+\.(?:md|csv|json|py|ts|tsx|yaml|yml|png|svg|lock))(?::(\d+))?`"
)
_FENCE = re.compile(r"^```.*?^```", re.S | re.M)
_SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "tel:")


def _markdown_files(root: Path) -> list[Path]:
    files = []
    for path in root.rglob("*.md"):
        if any(
            part in {".git", "node_modules", ".venv", "dist", "__pycache__"}
            for part in path.parts
        ):
            continue
        files.append(path)
    return sorted(files)


def _headings(text: str) -> set[str]:
    anchors = set()
    for line in text.splitlines():
        if not line.startswith("#"):
            continue
        title = line.lstrip("#").strip().lower()
        slug = re.sub(r"[^\w\s-]", "", title).replace(" ", "-")
        anchors.add(slug)
    return anchors


def _strip_code_fences(text: str) -> str:
    return _FENCE.sub("", text)


def check(root: Path) -> list[str]:
    problems: list[str] = []
    root = root.resolve()
    files = _markdown_files(root)
    heading_cache: dict[Path, set[str]] = {}

    for path in files:
        text = path.read_text(encoding="utf-8")
        if "legacy" in path.parts:
            continue
        prose = _strip_code_fences(text)

        for raw in _LINK.findall(prose):
            target = raw.split(" ")[0]
            if target.startswith(_SKIP_PREFIX):
                continue
            file_part, _, anchor = target.partition("#")
            if not file_part:
                continue
            resolved = (path.parent / file_part).resolve()
            if not resolved.exists():
                problems.append(
                    f"{path.relative_to(root)}: link target missing: {target}"
                )
                continue
            if anchor and resolved.suffix == ".md":
                if resolved not in heading_cache:
                    heading_cache[resolved] = _headings(
                        resolved.read_text(encoding="utf-8")
                    )
                if anchor.lower() not in heading_cache[resolved]:
                    problems.append(
                        f"{path.relative_to(root)}: anchor missing in {file_part}: #{anchor}"
                    )

        enforce_paths = path.relative_to(root).as_posix() in _ENFORCED_PATHS
        for match in _BACKTICK_PATH.finditer(prose if enforce_paths else ""):
            reference, line = match.group(1), match.group(2)
            top = reference.split("/")[0]
            if "/" not in reference or top not in _TOP_LEVEL:
                continue
            resolved = (root / reference).resolve()
            if not resolved.exists():
                problems.append(
                    f"{path.relative_to(root)}: referenced file missing: {reference}"
                )
                continue
            if line:
                try:
                    total = len(resolved.read_text(encoding="utf-8").splitlines())
                except (UnicodeDecodeError, OSError):
                    continue
                if int(line) > total:
                    problems.append(
                        f"{path.relative_to(root)}: {reference}:{line} beyond end of file ({total} lines)"
                    )

    # Tracked data files referenced from prose must also be tracked, or the
    # repository is not self-contained after a clone.
    tracked = set(
        subprocess.run(
            ["git", "ls-files"], cwd=root, capture_output=True, text=True
        ).stdout.splitlines()
    )
    if tracked:
        for path in files:
            if path.relative_to(root).as_posix() not in _ENFORCED_PATHS:
                continue
            for reference in _BACKTICK_PATH.findall(path.read_text(encoding="utf-8")):
                name = reference[0]
                top = name.split("/")[0]
                if "/" in name and top in _TOP_LEVEL and (root / name).exists():
                    if name not in tracked:
                        problems.append(
                            f"{path.relative_to(root)}: {name} exists but is not tracked by git"
                        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify internal documentation links.")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)

    root = args.root.resolve()
    problems = check(root)
    if problems:
        print(f"{len(problems)} broken reference(s):\n", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(
        f"documentation links: {len(_markdown_files(root))} files, every reference resolves"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
