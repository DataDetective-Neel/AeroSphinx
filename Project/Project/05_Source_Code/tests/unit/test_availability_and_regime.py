"""Tests for the feature-group availability contract and regime modelling.

The availability contract is the mechanism that keeps operational
forecasting honest, so its enforcement is tested directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aerosphinx.features.groups import (
    GROUND_ONLY_GROUPS,
    AvailabilityClass,
    assert_operational_contract,
    describe_groups,
    get_group,
    partition_by_availability,
)
from aerosphinx.tracks.regime import (
    REGIMES,
    assign_regime,
    build_architecture,
    interaction_features,
    regime_indicators,
)

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# Availability contract
# --------------------------------------------------------------------------


def test_ground_only_groups_are_operationally_eligible():
    """The ground-only set must satisfy the operational contract."""
    for name in GROUND_ONLY_GROUPS:
        assert get_group(name).operational_eligible is True
    assert_operational_contract(GROUND_ONLY_GROUPS)


def test_station_met_is_retrospective_only():
    """Station-local meteorology has an unknown availability timestamp."""
    group = get_group("station_met")
    assert group.availability is AvailabilityClass.STATION_LOCAL_UNKNOWN_AVAILABILITY
    assert group.operational_eligible is False


def test_merra_is_never_operational():
    """Reanalysis is never available at forecast issue time."""
    group = get_group("merra")
    assert group.availability is AvailabilityClass.REANALYSIS
    assert group.operational_eligible is False


def test_aod_is_retrospective_only():
    """Observed satellite data has no forecast product here."""
    group = get_group("aod")
    assert group.availability is AvailabilityClass.OBSERVED_SATELLITE
    assert group.operational_eligible is False


def test_operational_contract_refuses_a_retrospective_group():
    """The stop condition: unknown availability cannot enter a forecast."""
    with pytest.raises(ValueError, match="not available at forecast issue time"):
        assert_operational_contract(GROUND_ONLY_GROUPS + ("station_met",))
    with pytest.raises(ValueError):
        assert_operational_contract(GROUND_ONLY_GROUPS + ("merra",))


def test_partition_separates_the_two_classes():
    """The partition must place each group in the right class."""
    operational, retrospective = partition_by_availability(
        GROUND_ONLY_GROUPS + ("station_met", "merra")
    )
    assert set(operational) == set(GROUND_ONLY_GROUPS)
    assert set(retrospective) == {"station_met", "merra"}


def test_aod_group_is_enabled_after_phase_4_validation():
    """The AOD group was enabled by the Phase 4 validation.

    It carries the pixel-support columns alongside the value, because the
    granules carry no QA dataset and the pixel count is the only available
    proxy for retrieval quality.
    """
    group = get_group("aod")
    assert group.enabled is True
    assert "AOD" in group.features
    assert "aod_valid_pixels" in group.features
    assert "aod_overpasses" in group.features
    # It is still retrospective only.
    assert group.operational_eligible is False


def test_unknown_group_raises():
    """An unregistered group name is an error."""
    with pytest.raises(KeyError):
        get_group("no_such_group")


def test_group_description_is_serialisable():
    """The group record must be JSON-friendly for the run manifest."""
    import json

    json.dumps(describe_groups(list(GROUND_ONLY_GROUPS) + ["station_met"]))


# --------------------------------------------------------------------------
# Regime assignment
# --------------------------------------------------------------------------


def test_regime_assignment_uses_calendar_only():
    """Regime must depend only on the month, never on any observation."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    regimes = assign_regime(dates)
    months = dates.month.to_numpy()
    for month, regime in zip(months, regimes):
        expected = next(name for name, members in REGIMES.items() if month in members)
        assert regime == expected


def test_regime_assignment_is_stable_under_target_change():
    """Changing the target values must not change the routing.

    This is the leakage guard: the routing may depend on the calendar and
    on nothing else.
    """
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    first = assign_regime(dates)
    second = assign_regime(dates)
    assert list(first) == list(second)


def test_regime_indicators_sum_to_one():
    """Exactly one regime indicator must be active on each date."""
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    indicators = regime_indicators(dates)
    assert len(indicators) == len(dates)
    assert indicators.sum(axis=1).eq(1.0).all()


def test_interaction_features_multiply_by_the_regime():
    """An interaction must be the feature times the regime indicator."""
    dates = pd.to_datetime(["2024-01-15", "2024-07-15"])
    frame = pd.DataFrame({"PM25_Lag_1": [10.0, 20.0]})
    result = interaction_features(frame, dates, ["PM25_Lag_1"])

    winter = result.iloc[0]
    summer = result.iloc[1]
    assert winter["PM25_Lag_1_x_regime_winter"] == pytest.approx(10.0)
    assert summer["PM25_Lag_1_x_regime_monsoon"] == pytest.approx(20.0)


# --------------------------------------------------------------------------
# Architectures
# --------------------------------------------------------------------------


@pytest.fixture
def training_frame() -> pd.DataFrame:
    """A training frame with a seasonal signal and enough rows per regime."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n = len(dates)
    rng = np.random.default_rng(0)
    seasonal = 60 + 40 * np.sin((dates.month.to_numpy() - 3) * 2 * np.pi / 12)
    values = seasonal + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "Date": dates,
            "PM25": values,
            "PM25_Lag_1": np.roll(values, 1),
            "day_sin": np.sin(dates.day_of_week.to_numpy() * 2 * np.pi / 7),
        }
    )


def test_global_architecture_predicts(training_frame):
    """The global architecture must fit and predict."""
    features = ["PM25_Lag_1", "day_sin"]
    architecture = build_architecture("global", features, "ridge")
    architecture.fit(training_frame)
    predictions = architecture.predict(training_frame)
    assert len(predictions) == len(training_frame)
    assert np.isfinite(predictions).all()


def test_seasonal_specialist_requires_minimum_training_data(training_frame):
    """A specialist with too little data must fall back to the global model."""
    features = ["PM25_Lag_1", "day_sin"]
    # An impossible threshold makes every specialist ineligible.
    architecture = build_architecture(
        "seasonal", features, "ridge", min_specialist_days=100000
    )
    architecture.fit(training_frame)
    assert not any(architecture._specialist_eligible.values())
    predictions = architecture.predict(training_frame)
    assert np.isfinite(predictions).all()


def test_seasonal_specialist_is_used_when_data_suffices(training_frame):
    """A specialist with enough data must actually be used."""
    features = ["PM25_Lag_1", "day_sin"]
    architecture = build_architecture(
        "seasonal", features, "ridge", min_specialist_days=30
    )
    architecture.fit(training_frame)
    assert any(architecture._specialist_eligible.values())
    assert architecture._specialists


def test_routing_falls_back_to_global_for_ineligible_regimes(training_frame):
    """An ineligible regime must be served by the global model."""
    features = ["PM25_Lag_1", "day_sin"]
    architecture = build_architecture(
        "routing", features, "ridge", min_specialist_days=100000
    )
    architecture.fit(training_frame)
    predictions = architecture.predict(training_frame)
    global_predictions = architecture._global.predict(
        training_frame.reset_index(drop=True)
    )
    np.testing.assert_allclose(predictions, global_predictions)


def test_unknown_architecture_raises():
    """An unregistered architecture must fail loudly."""
    with pytest.raises(ValueError):
        build_architecture("no_such_architecture", ["PM25_Lag_1"])
