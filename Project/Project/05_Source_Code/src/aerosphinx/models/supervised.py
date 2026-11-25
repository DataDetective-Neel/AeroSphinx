"""Supervised models for AeroSphinx Track A.

Model candidates follow the order set in ``docs/reference/EXPERIMENT_PROTOCOL.md``
section 6: regularized linear regression, then Random Forest, then
XGBoost. Additional models are added only when a baseline failure or a
specific research question justifies them.

Two rules are enforced here rather than left to the caller:

1. **Trees are never asked to impute.** Random Forest and XGBoost handle
   missing values natively, and inventing a value for a missing lag would
   destroy the calendar-aware semantics that Phase 1 established. Only the
   linear model, which cannot handle NaN, uses an imputer - and that
   imputer is fitted on the **training block only**, never on the full
   dataset.
2. **No tuning on held-out data.** Hyperparameters are fixed per
   configuration and recorded in the run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "ModelSpec",
    "RidgeModel",
    "RandomForestModel",
    "XGBoostModel",
    "build_model",
    "MODEL_REGISTRY",
]


@dataclass(frozen=True)
class ModelSpec:
    """Fixed hyperparameters for one model.

    Every value is recorded in the run manifest so a result can be traced
    to the exact configuration that produced it.
    """

    name: str
    params: dict = field(default_factory=dict)


# Fixed hyperparameters. These are not searched in Phase 2; the Phase 0
# audit noted that the legacy pipeline's fixed values were defensible, and
# a per-fold search belongs in Phase 3 where it can be validated.
RF_PARAMS = ModelSpec(
    "random_forest",
    dict(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
)

XGB_PARAMS = ModelSpec(
    "xgboost",
    dict(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    ),
)

RIDGE_PARAMS = ModelSpec("ridge", dict(alpha=1.0, random_state=42))


class _BaseModel:
    """Common interface: fit on a frame, predict a frame."""

    name = "base"

    def __init__(self, features: list[str], target: str = "PM25"):
        self.features = list(features)
        self.target = target
        self.fitted = False

    def _check(self, frame: pd.DataFrame) -> None:
        missing = [f for f in self.features if f not in frame.columns]
        if missing:
            raise KeyError(f"{self.name}: frame is missing features {missing}")

    def fit(self, frame: pd.DataFrame):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, frame: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError


class RidgeModel(_BaseModel):
    """Regularized linear regression on the Track A feature set.

    The simplest learned model, and the one that exposes whether a signal
    exists at all.

    Two details matter and are easy to get wrong:

    * **Features are standardised.** The feature set mixes calendar
      encodings in [-1, 1] with PM2.5 lags in the tens or hundreds. Ridge
      penalises coefficient magnitude, so without scaling the penalty
      falls almost entirely on the large-scale features and the regularised
      model is not the one the penalty was chosen for.
    * **Missingness is made explicit.** The calendar-aware lags are
      ``NaN`` exactly where the previous calendar day is absent, which is
      information rather than an inconvenience. The imputer therefore
      records missingness as an indicator column, so the model can learn
      "yesterday was missing" instead of silently receiving a median value.

      The imputer and the scaler are fitted on the **training block only**.
    """

    name = "ridge"

    def __init__(
        self,
        features: list[str],
        target: str = "PM25",
        alpha: float = 1.0,
    ):
        super().__init__(features, target)
        self.alpha = alpha
        self._pipeline = None

    def fit(self, frame: pd.DataFrame) -> "RidgeModel":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self._check(frame)
        X = frame[self.features]
        y = pd.to_numeric(frame[self.target], errors="coerce")

        # Rows whose target is missing cannot train the model.
        keep = y.notna()
        X = X[keep]
        y = y[keep]

        self._pipeline = Pipeline(
            [
                # ``add_indicator`` preserves the fact that a lag was
                # missing instead of hiding it behind a median.
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=self.alpha, random_state=42)),
            ]
        )
        self._pipeline.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        self.fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("ridge model is not fitted")
        self._check(frame)
        X = frame[self.features].to_numpy(dtype=float)
        return np.asarray(self._pipeline.predict(X), dtype=float)

    def feature_importance(self) -> pd.Series:
        """Standardised coefficients, ordered by absolute magnitude.

        The coefficients are expressed per standard deviation of the
        feature, so they are directly comparable across features.
        """
        if not self.fitted:
            raise RuntimeError("ridge model is not fitted")
        coefficients = self._pipeline.named_steps["ridge"].coef_
        return pd.Series(coefficients, index=self.features).sort_values(
            key=np.abs, ascending=False
        )


class RandomForestModel(_BaseModel):
    """Random Forest regressor.

    Handles missing values natively, so no imputation is applied and the
    calendar-aware NaN lags keep their meaning.
    """

    name = "random_forest"

    def __init__(self, features: list[str], target: str = "PM25", **params):
        super().__init__(features, target)
        self.params = {**RF_PARAMS.params, **params}

    def fit(self, frame: pd.DataFrame) -> "RandomForestModel":
        from sklearn.ensemble import RandomForestRegressor

        self._check(frame)
        X = frame[self.features]
        y = pd.to_numeric(frame[self.target], errors="coerce")
        keep = y.notna()
        X = X[keep]
        y = y[keep]

        self._model = RandomForestRegressor(**self.params)
        self._model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        self.fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("random forest is not fitted")
        self._check(frame)
        X = frame[self.features].to_numpy(dtype=float)
        return np.asarray(self._model.predict(X), dtype=float)

    def feature_importance(self) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("random forest is not fitted")
        return pd.Series(
            self._model.feature_importances_, index=self.features
        ).sort_values(ascending=False)


class XGBoostModel(_BaseModel):
    """XGBoost regressor.

    Handles missing values natively, so no imputation is applied.
    """

    name = "xgboost"

    def __init__(self, features: list[str], target: str = "PM25", **params):
        super().__init__(features, target)
        self.params = {**XGB_PARAMS.params, **params}

    def fit(self, frame: pd.DataFrame) -> "XGBoostModel":
        from xgboost import XGBRegressor

        self._check(frame)
        X = frame[self.features]
        y = pd.to_numeric(frame[self.target], errors="coerce")
        keep = y.notna()
        X = X[keep]
        y = y[keep]

        self._model = XGBRegressor(**self.params)
        self._model.fit(X.to_numpy(dtype=float), y.to_numpy(dtype=float))
        self.fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("xgboost is not fitted")
        self._check(frame)
        X = frame[self.features].to_numpy(dtype=float)
        return np.asarray(self._model.predict(X), dtype=float)

    def feature_importance(self) -> pd.Series:
        if not self.fitted:
            raise RuntimeError("xgboost is not fitted")
        return pd.Series(
            self._model.feature_importances_, index=self.features
        ).sort_values(ascending=False)


MODEL_REGISTRY = {
    "ridge": RidgeModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
}


def build_model(name: str, features: list[str], target: str = "PM25", **params):
    """Instantiate a model by name from the registry."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](features=features, target=target, **params)
