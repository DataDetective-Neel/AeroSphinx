"""Regime-conditional modelling for AeroSphinx Track A.

Phase 3 Workstream C, expanded to cover regime-conditional modelling,
feature-group ablation and horizon-specific architecture.

Regimes
-------
Four meteorological seasons: winter, pre-monsoon, monsoon, post-monsoon.

**Regime assignment is a deterministic function of the target date's
calendar month**, which is known in advance. No assignment may use the
target value, any future observation, or any measured pollution quantity.
This is asserted by the tests.

Architectures
-------------
==========  ==============================================================
``global``  One model for all regimes.
``seasonal``  A specialist per regime, with a minimum-sample threshold
              and a fallback to the global model.
``interaction``  One global model, with regime x feature interaction terms
              added to the feature vector.
``routing``  Regime-aware dispatch: a specialist where it is eligible,
              the global model otherwise.
==========  ==============================================================

Selection discipline
--------------------
Every architecture and every threshold is chosen on **chronological
validation folds** that precede the test folds. The test folds are scored
once, with the selection already fixed. Selecting an architecture on test
scores is prohibited, and the code does not provide a way to do it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from aerosphinx.features.seasonal import SEASON_MONTHS, assign_season
from aerosphinx.models.supervised import build_model

__all__ = [
    "REGIMES",
    "assign_regime",
    "regime_indicators",
    "interaction_features",
    "Architecture",
    "build_architecture",
    "MIN_SPECIALIST_DAYS_CANDIDATES",
]


#: The four meteorological regimes, with the months they cover. The month
#: mapping is owned by :mod:`aerosphinx.features.seasonal`; this alias exists
#: for backward compatibility with callers importing ``REGIMES``.
REGIMES = SEASON_MONTHS

#: Candidate minimum training sizes for a specialist, selected on validation.
MIN_SPECIALIST_DAYS_CANDIDATES = (60, 120, 180, 240, 365)


def assign_regime(dates) -> np.ndarray:
    """Assign a regime to each date from the calendar alone.

    Parameters
    ----------
    dates:
        The target dates.

    Returns
    -------
    numpy.ndarray
        Regime labels, one per date.
    """
    return assign_season(pd.Series(dates)).to_numpy(dtype=object)


def regime_indicators(dates) -> pd.DataFrame:
    """One-hot regime indicators for the interaction architecture.

    Built from the calendar alone, so they are known at forecast time.
    """
    regimes = assign_regime(dates)
    frame = pd.DataFrame(
        {f"regime_{regime}": (regimes == regime).astype(float) for regime in REGIMES}
    )
    return frame


def interaction_features(frame: pd.DataFrame, dates, features) -> pd.DataFrame:
    """Add regime x feature interaction terms to a frame.

    The interactions allow one global model to behave differently by
    regime, without the regime-specific sample-size problem of separate
    specialists.
    """
    indicators = regime_indicators(dates).reset_index(drop=True)
    working = frame.reset_index(drop=True)
    for feature in features:
        if feature not in working.columns:
            continue
        for column in indicators.columns:
            working[f"{feature}_x_{column}"] = (
                working[feature].to_numpy(dtype=float) * indicators[column].to_numpy()
            )
    return working


@dataclass
class Architecture:
    """One model architecture and its fitted state.

    Attributes
    ----------
    name:
        Architecture identifier.
    kind:
        ``"global"``, ``"seasonal"``, ``"interaction"`` or ``"routing"``.
    min_specialist_days:
        Minimum training observations for a specialist to be used.
    """

    name: str
    kind: str
    min_specialist_days: int = 180
    model_name: str = "ridge"
    features: tuple[str, ...] = ()
    _global = None
    _specialists: dict = field(default_factory=dict)
    _specialist_eligible: dict = field(default_factory=dict)

    # -- fitting -----------------------------------------------------------

    def fit(self, train: pd.DataFrame, dates=None) -> "Architecture":
        """Fit the architecture on a training frame.

        Parameters
        ----------
        train:
            Training rows.
        dates:
            The target dates for the training rows, used for regime
            assignment. Defaults to the ``Date`` column.
        """
        if self._global is None and self.kind == "shared":
            raise ValueError(
                "the shared architecture is built by "
                "aerosphinx.tracks.workstream_c._shared_model_predictions"
            )
        if dates is None:
            dates = train["Date"]
        regimes = assign_regime(dates)

        counts = {regime: int((regimes == regime).sum()) for regime in REGIMES}
        self.record_specialist_counts(counts)

        if self.kind == "interaction":
            working = interaction_features(train, dates, self.features)
        else:
            working = train.reset_index(drop=True)

        self._global = build_model(
            self.model_name, list(self.features), target="PM25"
        ).fit(working)

        if self.kind in {"seasonal", "routing"}:
            for regime in REGIMES:
                mask = regimes == regime
                count = int(mask.sum())
                # A specialist is used only when its training set holds
                # enough observations. Otherwise the global model serves
                # that regime: the fallback is unconditional, so the
                # architecture never abstains and never trains a
                # specialist on too little data.
                self._specialist_eligible[regime] = count >= self.min_specialist_days
                if not self._specialist_eligible[regime]:
                    continue
                subset = working[mask]
                self._specialists[regime] = build_model(
                    self.model_name, list(self.features), target="PM25"
                ).fit(subset)

        return self

    # -- prediction --------------------------------------------------------

    def predict(self, frame: pd.DataFrame, dates=None) -> np.ndarray:
        """Predict with the fitted architecture.

        Specialist models are used only where they are eligible; otherwise
        the global model is used. This is the validated fallback, and it
        is unconditional: the global model is always fitted, so the
        fallback never abstains.
        """
        if self._global is None:
            raise RuntimeError("architecture is not fitted")
        if dates is None:
            dates = frame["Date"]

        if self.kind == "interaction":
            working = interaction_features(frame, dates, self.features)
            return self._global.predict(working)

        regimes = assign_regime(dates)
        predictions = self._global.predict(frame.reset_index(drop=True))

        if self.kind in {"seasonal", "routing"}:
            for index, regime in enumerate(regimes):
                if regime in self._specialists:
                    subset = frame.reset_index(drop=True).iloc[[index]]
                    predictions[index] = self._specialists[regime].predict(subset)[0]
        return predictions

    # -- reporting ---------------------------------------------------------

    def specialist_report(self) -> dict:
        """Which specialists were eligible, and on how much data."""
        return {
            "min_specialist_days": self.min_specialist_days,
            "specialist_training_counts": (
                {
                    regime: int(count)
                    for regime, count in self._specialist_counts.items()
                }
                if hasattr(self, "_specialist_counts")
                else {}
            ),
            "specialist_eligible": dict(self._specialist_eligible),
        }

    def record_specialist_counts(self, counts: dict) -> None:
        """Record how many training observations each regime received."""
        self._specialist_counts = dict(counts)


def build_architecture(
    kind: str,
    features,
    model_name: str = "ridge",
    min_specialist_days: int = 180,
    name: str | None = None,
) -> Architecture:
    """Construct an architecture by kind.

    Raises
    ------
    ValueError
        For an unknown architecture kind.
    """
    if kind == "shared":
        raise ValueError(
            "the 'shared' architecture trains one model across all horizons "
            "and is built by "
            "aerosphinx.tracks.workstream_c._shared_model_predictions, not "
            "by build_architecture"
        )
    if kind not in {"global", "seasonal", "interaction", "routing"}:
        raise ValueError(
            f"unknown architecture {kind!r}; expected one of global, "
            "seasonal, interaction, routing, shared"
        )
    return Architecture(
        name=name or kind,
        kind=kind,
        model_name=model_name,
        features=tuple(features),
        min_specialist_days=min_specialist_days,
    )
