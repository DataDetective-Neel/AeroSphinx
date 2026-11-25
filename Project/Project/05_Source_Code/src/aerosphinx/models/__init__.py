"""Models: baselines and supervised regressors."""

from aerosphinx.models.baselines import (
    ClimatologyBuilder,
    PersistenceBaseline,
    SeasonalNaiveBaseline,
    persistence_forecast,
)
from aerosphinx.models.supervised import (
    MODEL_REGISTRY,
    RandomForestModel,
    RidgeModel,
    XGBoostModel,
    build_model,
)

__all__ = [
    "ClimatologyBuilder",
    "PersistenceBaseline",
    "SeasonalNaiveBaseline",
    "persistence_forecast",
    "MODEL_REGISTRY",
    "RandomForestModel",
    "RidgeModel",
    "XGBoostModel",
    "build_model",
]
