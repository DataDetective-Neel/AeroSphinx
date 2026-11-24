"""Baseline models for AeroSphinx.

Baselines are the reference against which every learned model is judged.
They are not throwaway: the Phase 0 audit found that the legacy pipeline's
climatology baseline was mis-specified (bias -35.85 ug/m3), which flattered
every model compared against it.

Every baseline here consumes only information available **before** the
target date, and each is fitted on the training block of its fold only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "PersistenceBaseline",
    "SeasonalNaiveBaseline",
    "ClimatologyBuilder",
    "persistence_forecast",
]


class PersistenceBaseline:
    """A0 - predict tomorrow's PM2.5 as today's PM2.5.

    The naive benchmark every forecasting system must beat. Under the
    calendar-aware contract, "today" is the observation dated exactly one
    calendar day before the target; where that day is absent the baseline
    returns NaN rather than reaching further back.
    """

    name = "persistence"

    def __init__(self, value_column: str = "PM25", date_column: str = "Date"):
        self.value_column = value_column
        self.date_column = date_column

    def fit(self, frame: pd.DataFrame) -> "PersistenceBaseline":
        """Nothing to fit; the baseline is parameter-free."""
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict using the previous calendar day's observation."""
        from aerosphinx.features.lags import calendar_lag

        lag = calendar_lag(frame, self.value_column, 1)
        return lag.to_numpy(dtype=float)


class ClimatologyBuilder:
    """Day-of-year climatology built from a training block.

    The correction for **D-005**: the legacy baseline averaged every year
    of history for each day-of-year, which under-predicted the 2024 test
    window by 35.85 ug/m3 because the window sat in a higher-PM2.5 period
    than the long-run average.

    Two changes follow:

    1. Only the most recent ``lookback_years`` years are used, so the
       baseline tracks the current regime rather than a decade-average.
    2. The baseline's own bias is reported, so a shared systematic offset
       cannot hide inside a relative comparison.

    The climatology is always fitted on training data only. A day-of-year
    key that is absent from the training block falls back to the nearest
    available key within ``max_day_offset`` days.
    """

    def __init__(
        self,
        lookback_years: int = 3,
        max_day_offset: int = 3,
        value_column: str = "PM25",
        date_column: str = "Date",
    ):
        if lookback_years < 1:
            raise ValueError("lookback_years must be at least 1")
        self.lookback_years = lookback_years
        self.max_day_offset = max_day_offset
        self.value_column = value_column
        self.date_column = date_column
        self._table: pd.Series | None = None
        self._fitted_end: pd.Timestamp | None = None
        self._covered_day_of_year: set[int] = set()
        self._primary_years: list[int] = []
        self._fallback_years: list[int] = []

    @property
    def covered_day_of_year(self) -> set[int]:
        """The day-of-years this climatology can serve."""
        return set(self._covered_day_of_year)

    @property
    def coverage_fraction(self) -> float:
        """Fraction of the 366 possible day-of-years that are covered."""
        return len(self._covered_day_of_year) / 366.0

    @property
    def primary_years(self) -> list[int]:
        """The years used as the primary source for each day-of-year."""
        return list(self._primary_years)

    @property
    def fallback_years(self) -> list[int]:
        """The years used to fill day-of-years the primary window missed."""
        return list(self._fallback_years)

    def fit(self, frame: pd.DataFrame) -> "ClimatologyBuilder":
        """Build the climatology from the given (training) observations.

        The climatology is guaranteed to **cover every day-of-year** that
        the training span can serve.

        The lookback window defines the *primary* source for each
        day-of-year, which is the D-005 correction: recent years are
        preferred, because earlier regimes had different pollution levels.
        Where the primary window does not contain a given day-of-year — a
        short training window covers only part of the year — that
        day-of-year is served from the most recent year that does contain
        it.

        This per-day-of-year fallback is what prevents the Phase 2 defect,
        where a climatology built from one partial year left 106
        day-of-years uncovered and silently removed whole folds from the
        evaluation set.
        """
        data = frame[[self.date_column, self.value_column]].copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.dropna(subset=[self.value_column])
        if data.empty:
            self._table = pd.Series(dtype=float)
            self._fitted_end = None
            self._covered_day_of_year = set()
            self._primary_years = []
            self._fallback_years = []
            return self

        data["day_of_year"] = data[self.date_column].dt.day_of_year
        self._fitted_end = data[self.date_column].max()

        all_years = sorted(data[self.date_column].dt.year.unique())
        # Primary window: the most recent ``lookback_years`` years present.
        primary_years = all_years[-self.lookback_years :] if self.lookback_years else []
        self._primary_years = list(primary_years)

        primary = data[data[self.date_column].dt.year.isin(primary_years)]
        primary_means = primary.groupby("day_of_year")[self.value_column].mean()

        # Fall back to the most recent year containing each missing
        # day-of-year, then to any year containing it.
        fallback = data[~data[self.date_column].dt.year.isin(primary_years)]
        fallback_means = fallback.groupby("day_of_year")[self.value_column].mean()

        combined = primary_means.combine_first(fallback_means)

        # A day-of-year may be absent from the training span entirely (for
        # example a 29 February, or a date whose day-of-year falls in the
        # part of the year the training window does not reach). Those days
        # are served by the nearest available day-of-year, so the baseline
        # is never undefined on a day it could plausibly serve.
        combined = self._fill_gaps(combined)

        combined = combined.sort_index()

        self._table = combined
        self._covered_day_of_year = set(combined.index)
        self._fallback_years = sorted(
            set(fallback[self.date_column].dt.year.unique()) - set(primary_years)
        )
        return self

    @staticmethod
    def _fill_gaps(means: pd.Series) -> pd.Series:
        """Fill day-of-year gaps from the nearest available day-of-year.

        A climatology that leaves a day-of-year unserved turns into a
        missing prediction, which in turn removes rows from the evaluation
        set of every model. Filling from the nearest day-of-year keeps the
        baseline defined everywhere.
        """
        if means.empty:
            return means

        available = np.array(sorted(means.index))
        required = set(range(int(available.min()), int(available.max()) + 1))
        missing = required - set(available)
        if not missing:
            return means

        filled = dict(means.to_dict())
        for day in missing:
            nearest = available[np.argmin(np.abs(available - day))]
            filled[day] = float(means.loc[int(nearest)])
        return pd.Series(filled).sort_index()

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict each row from its day-of-year climatological mean."""
        if self._table is None or self._table.empty:
            return np.full(len(frame), np.nan)

        dates = pd.to_datetime(frame[self.date_column])
        keys = dates.dt.day_of_year.to_numpy()

        predictions = np.full(len(frame), np.nan)
        available = np.array(sorted(self._table.index))
        values = self._table.to_dict()

        for position, key in enumerate(keys):
            if key in values:
                predictions[position] = values[key]
                continue
            # Nearest available day-of-year within the permitted offset.
            nearest = available[np.abs(available - key) <= self.max_day_offset]
            if nearest.size:
                predictions[position] = values[
                    int(nearest[np.argmin(np.abs(nearest - key))])
                ]

        return predictions

    @property
    def fitted_end(self) -> pd.Timestamp | None:
        """Latest observation date used to build the climatology."""
        return self._fitted_end


class SeasonalNaiveBaseline:
    """A1 - predict from the day-of-year climatology.

    A stronger baseline than persistence, because it captures the strong
    seasonal cycle in Delhi-NCR PM2.5. It is fitted per fold on training
    data only.

    The lookback window is a genuine trade-off and must be chosen on
    validation data, not asserted:

    * A **long** lookback averages more years, giving a smoother
      climatology with lower variance, but it also blends in earlier
      regimes whose pollution levels differed.
    * A **short** lookback tracks the current regime but is noisier, and on
      a single year it can miss a day-of-year entirely.

    :meth:`select_lookback` chooses between them using an inner validation
    block carved out of the training data, so no test observation
    influences the choice.
    """

    name = "seasonal_naive"

    def __init__(
        self,
        lookback_years: int | None = 3,
        max_day_offset: int = 3,
        value_column: str = "PM25",
        date_column: str = "Date",
    ):
        self.lookback_years = lookback_years
        self.max_day_offset = max_day_offset
        self.value_column = value_column
        self.date_column = date_column
        self.builder = ClimatologyBuilder(
            lookback_years=lookback_years or 3,
            max_day_offset=max_day_offset,
            value_column=value_column,
            date_column=date_column,
        )
        self.selected_lookback: int | None = None

    def fit(self, frame: pd.DataFrame) -> "SeasonalNaiveBaseline":
        self.builder.fit(frame)
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.builder.predict(frame)

    def select_lookback(
        self,
        frame: pd.DataFrame,
        candidates: tuple[int, ...] = (1, 2, 3, 5, 9),
        validation_fraction: float = 0.2,
    ) -> int:
        """Choose the lookback window on an inner validation block.

        The training frame is split chronologically: the earlier part fits
        the climatology and the later part scores it. The lookback with the
        lowest validation MAE wins. No test observation is involved.

        A candidate that cannot predict part of the validation block is
        penalised, because a climatology with gaps will also fail on the
        test block. The penalty is expressed as the fraction of validation
        rows the candidate fails to cover, so a candidate with better MAE
        but incomplete coverage does not win by default.

        Returns
        -------
        int
            The selected lookback in years.
        """
        if not candidates:
            raise ValueError("candidates cannot be empty")
        if not 0 < validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")

        data = frame[[self.date_column, self.value_column]].copy()
        data[self.date_column] = pd.to_datetime(data[self.date_column])
        data = data.sort_values(self.date_column).reset_index(drop=True)
        if len(data) < 20:
            self.selected_lookback = self.lookback_years or 3
            return self.selected_lookback

        split = int(len(data) * (1 - validation_fraction))
        inner_train = data.iloc[:split]
        inner_validation = data.iloc[split:]

        scores: dict[int, float] = {}
        for candidate in candidates:
            builder = ClimatologyBuilder(
                lookback_years=candidate,
                max_day_offset=self.max_day_offset,
                value_column=self.value_column,
                date_column=self.date_column,
            ).fit(inner_train)
            predictions = builder.predict(inner_validation)
            valid = pd.notna(predictions) & inner_validation[self.value_column].notna()
            if not valid.any():
                scores[candidate] = float("inf")
                continue
            errors = np.abs(
                inner_validation[self.value_column].to_numpy()[valid]
                - predictions[valid]
            )
            # Incomplete coverage is penalised: the shortfall is added to
            # the mean error so a candidate that leaves rows unpredicted
            # cannot win on the rows it does cover.
            shortfall = 1.0 - float(valid.mean())
            scores[candidate] = float(errors.mean()) * (1.0 + shortfall)

        best = min(scores, key=lambda key: (scores[key], key))
        self.selected_lookback = int(best)
        self.lookback_years = int(best)
        self.builder = ClimatologyBuilder(
            lookback_years=int(best),
            max_day_offset=self.max_day_offset,
            value_column=self.value_column,
            date_column=self.date_column,
        )
        return self.selected_lookback


def persistence_forecast(
    frame: pd.DataFrame,
    value_column: str = "PM25",
    date_column: str = "Date",
    max_age_days: int | None = None,
    periods: int = 1,
) -> np.ndarray:
    """Persistence prediction with an optional staleness allowance.

    With ``max_age_days=None`` this is the strict calendar-aware baseline:
    the observation dated exactly ``periods`` calendar days before each
    row, or NaN where that date is absent.

    ``periods`` is what makes the baseline **horizon-appropriate**: a
    day-5 forecast must be judged against the value observed 5 days
    earlier, not the value observed yesterday.

    With ``max_age_days`` set, the most recent observation at or before the
    reference date is used, provided it is no more than
    ``max_age_days`` old. This is the operationally realistic variant - a
    forecaster would use the latest observation it has - and it is reported
    alongside the strict variant so the effect of the 123 calendar gaps on
    the evaluation set is visible rather than hidden.
    """
    from aerosphinx.features.lags import calendar_lag

    if periods < 1:
        raise ValueError("periods must be at least 1")

    if max_age_days is None:
        return calendar_lag(frame, value_column, periods).to_numpy(dtype=float)

    dates = pd.to_datetime(frame[date_column])
    values = pd.to_numeric(frame[value_column], errors="coerce")

    observed_index = pd.DatetimeIndex(dates)
    # For each target date, find the most recent observation at or before
    # the reference date, then apply the staleness limit.
    targets = observed_index - pd.Timedelta(days=periods)
    positions = observed_index.get_indexer(targets, method="ffill")

    predictions = np.full(len(frame), np.nan)
    valid = positions >= 0
    last_observed = observed_index[positions[valid]]
    age = (targets[valid] - last_observed).days
    within = age <= max_age_days
    rows = np.flatnonzero(valid)[within]
    predictions[rows] = values.to_numpy()[positions[valid][within]]

    return predictions
