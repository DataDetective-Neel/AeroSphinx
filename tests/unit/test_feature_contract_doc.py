"""The written feature-availability contract must match the code.

`docs/reference/FEATURE_AVAILABILITY_CONTRACT.md` is a Phase 2 deliverable. A document
that drifts from `FEATURE_GROUPS` would be worse than no document, so this test
parses its per-feature table and checks every feature, group, class and
operational-eligibility flag against the registry.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aerosphinx.features.groups import FEATURE_GROUPS, AvailabilityClass

pytestmark = pytest.mark.unit

DOC = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "reference"
    / "FEATURE_AVAILABILITY_CONTRACT.md"
)

ROW = re.compile(
    r"^\|\s*`(?P<feature>[^`]+)`\s*"
    r"\|\s*`(?P<group>[^`]+)`\s*"
    r"\|\s*`(?P<availability>[^`]+)`\s*"
    r"\|\s*(?P<operational>Yes|No)\s*\|"
)


def documented_rows() -> dict[str, tuple[str, str, bool]]:
    rows: dict[str, tuple[str, str, bool]] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            rows[match.group("feature")] = (
                match.group("group"),
                match.group("availability"),
                match.group("operational") == "Yes",
            )
    return rows


def test_document_exists():
    assert DOC.is_file(), f"missing deliverable: {DOC}"


def test_every_registered_feature_is_documented_exactly_once():
    rows = documented_rows()
    registered = {
        feature: group.name
        for group in FEATURE_GROUPS.values()
        for feature in group.features
    }
    assert set(rows) == set(registered), (
        "documented features differ from FEATURE_GROUPS: "
        f"missing={sorted(set(registered) - set(rows))}, "
        f"extra={sorted(set(rows) - set(registered))}"
    )
    assert len(rows) == len(registered), "a feature is documented more than once"


def test_documented_group_class_and_eligibility_match_the_registry():
    rows = documented_rows()
    for feature, (group_name, availability, operational) in rows.items():
        group = FEATURE_GROUPS[group_name]
        assert feature in group.features
        assert availability == group.availability.value
        assert operational is group.operational_eligible


def test_only_ground_history_and_calendar_are_operational():
    eligible = {
        group.availability
        for group in FEATURE_GROUPS.values()
        if group.operational_eligible
    }
    assert eligible == {
        AvailabilityClass.OBSERVED_GROUND_HISTORY,
        AvailabilityClass.CALENDAR,
    }
    documented = {
        availability
        for _, availability, operational in documented_rows().values()
        if operational
    }
    assert documented == {cls.value for cls in eligible}
