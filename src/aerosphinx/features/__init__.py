"""Feature engineering with explicit temporal semantics."""

from aerosphinx.features.availability import (
    SOURCE_GROUPS,
    FusedDataset,
    FusionRules,
    assert_fusion_alignment,
    build_fused_dataset,
    validate_fused_dataset,
)
from aerosphinx.features.lags import (
    LagSpec,
    calendar_lag,
    calendar_lags,
    calendar_rolling,
    calendar_rolling_stats,
    observation_age_days,
)
from aerosphinx.features.seasonal import (
    SEASON_MONTHS,
    SEASONS,
    assign_season,
    get_season_year,
    season_features,
    season_of_month,
    validate_season_assignment,
    validate_temporal_order,
)

__all__ = [
    "SOURCE_GROUPS",
    "FusedDataset",
    "FusionRules",
    "assert_fusion_alignment",
    "build_fused_dataset",
    "validate_fused_dataset",
    "LagSpec",
    "calendar_lag",
    "calendar_lags",
    "calendar_rolling",
    "calendar_rolling_stats",
    "observation_age_days",
    "SEASONS",
    "SEASON_MONTHS",
    "assign_season",
    "get_season_year",
    "season_features",
    "season_of_month",
    "validate_season_assignment",
    "validate_temporal_order",
]
