"""Tests for the deploy-parity verdict.

A deployment that reports a different commit, or no commit at all, must never
be treated as current.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.check_deploy import verdict  # noqa: E402

LOCAL = "9b0738baa2c3cf791b4c0a582f1c2675e4fc551d"


def test_exact_match_passes():
    matches, message = verdict(LOCAL, {"repo": {"sha": LOCAL}})
    assert matches is True
    assert "matches" in message


def test_short_sha_matches():
    matches, message = verdict(LOCAL, {"repo": {"sha": "9b0738b"}})
    assert matches is True
    assert "9b0738b" in message


def test_different_commit_fails():
    matches, message = verdict(
        LOCAL, {"repo": {"sha": "f080d8fe0fbc4994990022b41d3d865644defa3b"}}
    )
    assert matches is False
    assert "f080d8f" in message


def test_missing_sha_fails():
    matches, message = verdict(LOCAL, {"repo": {}})
    assert matches is False
    assert "repo.sha" in message
