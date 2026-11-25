"""Feature groups and their availability classes.

The availability contract is the mechanism that keeps operational
forecasting honest. Every feature belongs to a group, every group has an
availability class, and only groups that are demonstrably available at
forecast issue time may enter an operational feature contract.

The classes are:

``observed_ground_history``
    PM2.5 observed on or before the issue date. Available.
``calendar``
    Deterministic functions of the target date. Available.
``observed_satellite``
    INSAT-3D AOD observed for a past date. Retrospective only: the
    publication latency is not established, and there is no satellite
    *forecast* product in the repository.
``reanalysis``
    MERRA-2. Retrospective only, and **never** operational: reanalysis is
    not available at forecast issue time at all, which is the plan's
    section 4 correction.
``station_local_unknown_availability``
    CPCB station-local meteorological measurements. Retrospective only:
    the measurement time is known, the *publication* time is not, so a
    forecast cannot claim to have used them.

Two functions enforce the contract:

* :func:`assert_operational_contract` refuses an operational feature set
  that contains an ineligible group.
* :func:`partition_by_availability` separates a group list into the
  operationally eligible and retrospective-only parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

__all__ = [
    "AvailabilityClass",
    "FeatureGroup",
    "FEATURE_GROUPS",
    "GROUND_ONLY_GROUPS",
    "RETROSPECTIVE_ONLY_GROUPS",
    "get_group",
    "assemble_features",
    "distinctive_features",
    "assert_operational_contract",
    "partition_by_availability",
    "describe_groups",
]


class AvailabilityClass(str, Enum):
    """When a feature group is available to a forecaster."""

    OBSERVED_GROUND_HISTORY = "observed_ground_history"
    CALENDAR = "calendar"
    OBSERVED_SATELLITE = "observed_satellite"
    REANALYSIS = "reanalysis"
    STATION_LOCAL_UNKNOWN_AVAILABILITY = "station_local_unknown_availability"


#: Groups whose values are available at forecast issue time.
OPERATIONAL_ELIGIBLE = frozenset(
    {AvailabilityClass.OBSERVED_GROUND_HISTORY, AvailabilityClass.CALENDAR}
)


@dataclass(frozen=True)
class FeatureGroup:
    """A named set of features with a single availability class."""

    name: str
    features: tuple[str, ...]
    availability: AvailabilityClass
    description: str
    enabled: bool = True
    note: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def operational_eligible(self) -> bool:
        """Whether this group may enter an operational feature contract."""
        return self.availability in OPERATIONAL_ELIGIBLE

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "features": list(self.features),
            "availability": self.availability.value,
            "operational_eligible": self.operational_eligible,
            "description": self.description,
            "enabled": self.enabled,
            "note": self.note,
            "metadata": self.metadata,
        }


FEATURE_GROUPS: dict[str, FeatureGroup] = {
    "pm25_history": FeatureGroup(
        name="pm25_history",
        features=(
            "PM25_Lag_1",
            "PM25_Lag_2",
            "PM25_Lag_3",
            "PM25_Lag_7",
            "PM25_Roll3",
            "PM25_Roll7",
        ),
        availability=AvailabilityClass.OBSERVED_GROUND_HISTORY,
        description=(
            "Calendar-aware PM2.5 history: lags at 1, 2, 3 and 7 calendar "
            "days and 3- and 7-day rolling means, all computed from strictly "
            "prior calendar days."
        ),
    ),
    "calendar": FeatureGroup(
        name="calendar",
        features=("day_sin", "day_cos", "month_sin", "month_cos"),
        availability=AvailabilityClass.CALENDAR,
        description=(
            "Cyclic encodings of day-of-week and month, preserving the "
            "circular topology of the weekly and annual cycles."
        ),
    ),
    "station_met": FeatureGroup(
        name="station_met",
        features=(
            "AT",
            "RH",
            "WS",
            "WD",
            "RF",
            "TOT_RF",
            "SR",
            "VWS",
        ),
        availability=AvailabilityClass.STATION_LOCAL_UNKNOWN_AVAILABILITY,
        description=(
            "CPCB station-local meteorological measurements, daily "
            "aggregated from the hourly export: air temperature, relative "
            "humidity, wind speed and direction, rainfall, cumulative "
            "rainfall, solar radiation and vertical wind speed. The "
            "barometric pressure column is excluded: it fails the Phase 5 "
            "physical cross-validation against MERRA-2."
        ),
        note=(
            "RETROSPECTIVE ONLY. The Phase 3 Workstream D audit confirmed "
            "these are genuine hourly station-local measurements (zero "
            "identical values against MERRA-2 across 516 matched hours), "
            "but the CPCB export carries no publication or upload "
            "timestamp, so the time at which each value became available "
            "to a forecaster cannot be established. No publication lag is "
            "assumed. Excluded from every operational feature contract. The "
            "barometric pressure column BP is excluded from this group "
            "because the Phase 5 cross-validation found daily means up to "
            "1062 mmHg (1416 hPa, above sea-level pressure at sea level) "
            "on 107 days, anti-correlation with the MERRA-2 surface "
            "pressure at the same site, and a diurnal maximum at 10h "
            "rather than near dawn."
        ),
    ),
    "merra": FeatureGroup(
        name="merra",
        features=(
            "T2M_midday",
            "TS_midday",
            "QV2M_midday",
            "U10M_midday",
            "V10M_midday",
            "SLP_midday",
        ),
        availability=AvailabilityClass.REANALYSIS,
        description=(
            "MERRA-2 reanalysis at the station grid cell, averaged over the "
            "three hourly timesteps (11:30, 12:30, 13:30 UTC) that the "
            "distributed granules contain."
        ),
        note=(
            "RETROSPECTIVE ONLY, and never operational. Reanalysis is not "
            "available at forecast issue time: it is an assimilation "
            "product for a past period, not a forecast of a future one. "
            "The feature names carry the '_midday' suffix because the "
            "granules contain a midday window, not a daily mean."
        ),
    ),
    "aod": FeatureGroup(
        name="aod",
        features=("AOD", "aod_valid_pixels", "aod_overpasses"),
        availability=AvailabilityClass.OBSERVED_SATELLITE,
        description=(
            "INSAT-3D aerosol optical depth in the station collocation "
            "window, together with the number of valid pixels supporting it "
            "and the number of overpasses averaged into the daily value. "
            "The support columns are included because the pixel count is "
            "the only available proxy for retrieval quality: the granules "
            "carry no QA or cloud-flag dataset."
        ),
        note=(
            "Enabled after the Phase 4 validation, which verified the "
            "extraction against every granule: uniform structure, the "
            "HDF5 dimension-scale orientation corroborated independently by "
            "the ocean-versus-land aerosol contrast, fill value -999.0, no "
            "duplicate date-overpass keys, and collocation-window "
            "sensitivity reported. Retrospective only: the observation is "
            "for a past date and the publication latency is not established, "
            "and no satellite forecast product exists in the repository."
        ),
        metadata={
            "validated_in": "Phase 4",
            "known_limitation": (
                "The granules carry no QA or cloud-flag dataset, so a "
                "present pixel means a retrieval was attempted, not that "
                "the retrieval is trustworthy."
            ),
        },
    ),
}


#: The groups that make up a purely ground-based model.
GROUND_ONLY_GROUPS = ("pm25_history", "calendar")

#: The groups that may only be used in retrospective estimation.
RETROSPECTIVE_ONLY_GROUPS = ("station_met", "merra", "aod")


def get_group(name: str) -> FeatureGroup:
    """Return a registered feature group by name."""
    if name not in FEATURE_GROUPS:
        raise KeyError(
            f"unknown feature group {name!r}; available: {sorted(FEATURE_GROUPS)}"
        )
    return FEATURE_GROUPS[name]


def distinctive_features(group_names) -> list[str]:
    """The features a group set *adds* beyond the ground baseline.

    The ground groups are available on every date, so including them in a
    coverage test would make every group set look fully covered. Coverage
    must therefore be judged on the features that distinguish a group set
    from the ground-only model: the environmental groups it adds.
    """
    features: list[str] = []
    for name in group_names:
        if name in GROUND_ONLY_GROUPS:
            continue
        for feature in get_group(name).features:
            if feature not in features:
                features.append(feature)
    return features


def assemble_features(group_names) -> list[str]:
    """The flat feature list for a set of groups, in a stable order.

    Raises
    ------
    KeyError
        If a group is unknown.
    ValueError
        If a group is disabled.
    """
    features: list[str] = []
    for name in group_names:
        group = get_group(name)
        if not group.enabled:
            raise ValueError(
                f"feature group {name!r} is disabled: {group.note or 'no reason given'}"
            )
        for feature in group.features:
            if feature not in features:
                features.append(feature)
    return features


def partition_by_availability(group_names) -> tuple[list[str], list[str]]:
    """Split groups into operationally eligible and retrospective-only."""
    operational, retrospective = [], []
    for name in group_names:
        group = get_group(name)
        (operational if group.operational_eligible else retrospective).append(name)
    return operational, retrospective


def assert_operational_contract(group_names) -> None:
    """Refuse an operational feature set containing an ineligible group.

    This is the stop condition from the Phase 3 authorisation: "If a
    feature's availability is unknown, it cannot enter an operational
    forecasting feature contract."

    Raises
    ------
    ValueError
        If any group is not operationally eligible.
    """
    operational, retrospective = partition_by_availability(group_names)
    if retrospective:
        details = "; ".join(
            f"{name} ({get_group(name).availability.value})" for name in retrospective
        )
        raise ValueError(
            "operational feature contract violated: the following groups "
            f"are not available at forecast issue time: {details}. "
            "Use retrospective estimation mode, or remove them."
        )
    return None


def describe_groups(group_names) -> list[dict]:
    """A JSON-friendly description of a set of groups."""
    return [get_group(name).as_dict() for name in group_names]
