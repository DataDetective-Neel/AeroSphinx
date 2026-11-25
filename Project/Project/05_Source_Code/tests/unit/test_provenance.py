"""Tests for run provenance.

The dirty-state filter exists because a run that writes its own artefacts
made the tree dirty and the manifest then reported the result as having come
from uncommitted code. Its first implementation was wrong in a way that only
a test catches: stripping the porcelain output removed the first line's
status column, so exactly one artefact was always counted as dirt.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from aerosphinx.quality.provenance import git_state  # noqa: E402

pytestmark = pytest.mark.unit


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "test",
            "GIT_AUTHOR_EMAIL": "test@example.com",
            "GIT_COMMITTER_NAME": "test",
            "GIT_COMMITTER_EMAIL": "test@example.com",
            "PATH": "/usr/bin:/bin:/usr/local/bin",
            "HOME": str(root),
        },
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "metrics.csv").write_text("MAE\n1\n")
    (tmp_path / "src.py").write_text("x = 1\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "initial")
    return tmp_path


def test_clean_tree_reports_no_dirt(repo: Path):
    state = git_state(repo)
    assert state["dirty"] is False
    assert state["dirty_entries"] == 0


def test_an_ignored_output_directory_is_not_dirt(repo: Path):
    (repo / "results" / "metrics.csv").write_text("MAE\n2\n")
    assert git_state(repo)["dirty_entries"] == 1
    ignored = git_state(repo, ignore=("results",))
    assert ignored["dirty"] is False
    assert ignored["dirty_entries"] == 0


def test_the_first_porcelain_line_is_also_filtered(repo: Path):
    """The bug: line one lost its status column and never matched."""
    (repo / "results" / "metrics.csv").write_text("MAE\n3\n")
    (repo / "results" / "other.csv").write_text("MAE\n3\n")
    assert git_state(repo, ignore=("results",))["dirty_entries"] == 0


def test_a_source_change_is_still_dirt_alongside_an_ignored_output(repo: Path):
    (repo / "results" / "metrics.csv").write_text("MAE\n4\n")
    (repo / "src.py").write_text("x = 2\n")
    state = git_state(repo, ignore=("results",))
    assert state["dirty"] is True
    assert state["dirty_entries"] == 1
