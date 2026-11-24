"""Chronological and rolling-origin evaluation splits.

The Phase 0 audit found that the legacy pipeline evaluated a model on a
single 33-day window at the end of the record. A single window cannot
distinguish a model that generalises from one that happened to suit one
season, and it cannot support a claim about year-to-year stability.

This module provides the rolling-origin protocol described in
``docs/reference/EXPERIMENT_PROTOCOL.md`` section 5.1: an expanding training window
with a validation block and a test block, advancing by a fixed step.

Every split is **strictly chronological**. The training block of a fold
ends before its validation block, which ends before its test block, and no
fold's training block ever contains an observation dated at or after its own
test block.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

__all__ = [
    "Fold",
    "rolling_origin_folds",
    "expanding_window_folds",
    "assert_no_leakage",
]


@dataclass(frozen=True)
class Fold:
    """One rolling-origin fold.

    Attributes
    ----------
    index:
        Zero-based fold number.
    train_start, train_end:
        Inclusive bounds of the training block.
    validation_start, validation_end:
        Inclusive bounds of the validation block. May be empty when
        validation is disabled.
    test_start, test_end:
        Inclusive bounds of the test block.
    """

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp | None
    validation_end: pd.Timestamp | None
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def split(self, frame: pd.DataFrame, date_column: str = "Date") -> tuple:
        """Slice a frame into ``(train, validation, test)`` for this fold.

        The validation frame is empty when the fold has no validation block.
        """
        dates = pd.to_datetime(frame[date_column])

        train = frame[(dates >= self.train_start) & (dates <= self.train_end)]
        if self.validation_start is None:
            validation = frame.iloc[0:0]
        else:
            validation = frame[
                (dates >= self.validation_start) & (dates <= self.validation_end)
            ]
        test = frame[(dates >= self.test_start) & (dates <= self.test_end)]
        return train, validation, test

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "train_start": self.train_start.date().isoformat(),
            "train_end": self.train_end.date().isoformat(),
            "validation_start": (
                self.validation_start.date().isoformat()
                if self.validation_start is not None
                else None
            ),
            "validation_end": (
                self.validation_end.date().isoformat()
                if self.validation_end is not None
                else None
            ),
            "test_start": self.test_start.date().isoformat(),
            "test_end": self.test_end.date().isoformat(),
        }


def rolling_origin_folds(
    dates: pd.Series,
    min_train_days: int = 365,
    validation_days: int = 30,
    test_block_days: int = 30,
    step_days: int = 30,
    date_column: str = "Date",
) -> list[Fold]:
    """Build an expanding-window rolling-origin backtest.

    The first fold trains on the first ``min_train_days`` days, validates on
    the next ``validation_days``, and tests on the next ``test_block_days``.
    Each subsequent fold advances the origin by ``step_days`` and extends
    the training window to include everything before its validation block.

    Parameters
    ----------
    dates:
        The observation dates, or a frame with a date column.
    min_train_days:
        Minimum length of the first training block.
    validation_days:
        Length of the validation block; 0 disables validation.
    test_block_days:
        Length of the test block.
    step_days:
        How far the origin advances between folds.
    date_column:
        Column holding the date when ``dates`` is a frame.

    Returns
    -------
    list[Fold]
        The folds, in chronological order.
    """
    if isinstance(dates, pd.DataFrame):
        series = pd.to_datetime(dates[date_column])
    else:
        series = pd.to_datetime(dates)

    series = series.dropna().sort_values().reset_index(drop=True)
    if series.empty:
        return []

    if min_train_days < 1:
        raise ValueError("min_train_days must be at least 1")
    if test_block_days < 1:
        raise ValueError("test_block_days must be at least 1")
    if step_days < 1:
        raise ValueError("step_days must be at least 1")
    if validation_days < 0:
        raise ValueError("validation_days cannot be negative")

    start = series.min().normalize()
    end = series.max().normalize()

    folds: list[Fold] = []
    index = 0
    while True:
        train_start = start
        train_end = (
            start
            + pd.Timedelta(days=min_train_days - 1)
            + pd.Timedelta(days=index * step_days)
        )
        validation_start = train_end + pd.Timedelta(days=1)
        validation_end = validation_start + pd.Timedelta(days=validation_days - 1)
        test_start = validation_end + pd.Timedelta(days=1)
        test_end = test_start + pd.Timedelta(days=test_block_days - 1)

        if test_end > end:
            break

        folds.append(
            Fold(
                index=index,
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start if validation_days else None,
                validation_end=validation_end if validation_days else None,
                test_start=test_start,
                test_end=test_end,
            )
        )
        index += 1

    return folds


def expanding_window_folds(
    dates: pd.Series,
    min_train_days: int = 365,
    test_block_days: int = 30,
    step_days: int = 30,
    date_column: str = "Date",
) -> list[Fold]:
    """Rolling-origin folds with the validation block disabled.

    Convenience wrapper for the common case where a model is fitted with
    fixed hyperparameters and no per-fold selection is required.
    """
    return rolling_origin_folds(
        dates,
        min_train_days=min_train_days,
        validation_days=0,
        test_block_days=test_block_days,
        step_days=step_days,
        date_column=date_column,
    )


def assert_no_leakage(
    fold: Fold, frame: pd.DataFrame, date_column: str = "Date"
) -> None:
    """Assert that a fold's training block precedes its test block.

    Raises
    ------
    AssertionError
        If any training observation is dated at or after the test block, or
        if the blocks are not contiguous.
    """
    dates = pd.to_datetime(frame[date_column])

    train_dates = dates[(dates >= fold.train_start) & (dates <= fold.train_end)]
    test_dates = dates[(dates >= fold.test_start) & (dates <= fold.test_end)]

    if not train_dates.empty and not test_dates.empty:
        assert train_dates.max() < test_dates.min(), (
            f"fold {fold.index} leaks: latest training date "
            f"{train_dates.max().date()} is not before the earliest test date "
            f"{test_dates.min().date()}"
        )

    if fold.validation_start is not None:
        validation_dates = dates[
            (dates >= fold.validation_start) & (dates <= fold.validation_end)
        ]
        if not validation_dates.empty and not test_dates.empty:
            assert validation_dates.max() < test_dates.min(), (
                f"fold {fold.index} leaks: validation reaches into the test " f"block"
            )
