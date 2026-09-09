"""
Target construction for AI Swing Analyser.

This module creates the variables the ML models will learn to
predict.

Targets are deliberately separated from explanatory features.

Supported prediction horizons:
    1D
    3D
    5D
    10D
    20D

Target types:
    - Direction
    - Future return
    - Maximum favourable excursion
    - Maximum adverse excursion
    - Future high/low range

IMPORTANT:
Every target contains future information by definition.
Target columns must NEVER be supplied as model features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetSpec:
    """
    Definition of a single prediction target.
    """

    horizon: int
    direction_threshold: float = 0.0

    @property
    def direction_column(self) -> str:
        return f"Direction_{self.horizon}"

    @property
    def return_column(self) -> str:
        return f"Future_Return_{self.horizon}"

    @property
    def high_return_column(self) -> str:
        return (
            f"Future_High_Return_{self.horizon}"
        )

    @property
    def low_return_column(self) -> str:
        return (
            f"Future_Low_Return_{self.horizon}"
        )

    @property
    def mfe_column(self) -> str:
        return (
            f"MFE_{self.horizon}"
        )

    @property
    def mae_column(self) -> str:
        return (
            f"MAE_{self.horizon}"
        )

    @property
    def range_upper_column(self) -> str:
        return (
            f"Target_Upper_Return_{self.horizon}"
        )

    @property
    def range_lower_column(self) -> str:
        return (
            f"Target_Lower_Return_{self.horizon}"
        )


DEFAULT_HORIZONS = (
    1,
    3,
    5,
    10,
    20,
)


def validate_horizons(
    horizons: Iterable[int],
) -> tuple[int, ...]:
    """
    Validate and normalize prediction horizons.
    """

    normalized = tuple(
        sorted(
            set(
                int(horizon)
                for horizon in horizons
            )
        )
    )

    if not normalized:
        raise ValueError(
            "At least one prediction horizon is required."
        )

    if any(
        horizon <= 0
        for horizon in normalized
    ):
        raise ValueError(
            "Prediction horizons must be positive integers."
        )

    return normalized


def add_direction_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Create binary direction targets.

    Direction definition:

        1 -> future return > threshold
        0 -> future return <= threshold

    The default threshold is zero, but later experiments can use
    volatility-adjusted or transaction-cost-adjusted thresholds.
    """

    if "Close" not in data.columns:
        raise ValueError(
            "Close column is required."
        )

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

    for horizon in horizons:

        future_close = (
            result["Close"]
            .shift(-horizon)
        )

        future_return = (
            future_close
            / result["Close"]
            - 1
        )

        result[
            f"Future_Return_{horizon}"
        ] = future_return

        result[
            f"Direction_{horizon}"
        ] = pd.Series(
            np.where(
                future_return.isna(),
                np.nan,
                (
                    future_return
                    > threshold
                ).astype(int),
            ),
            index=result.index,
        )

    return result


def add_path_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Create future path targets.

    For each horizon:

        Future_High_Return
            Maximum upside reached during the future window.

        Future_Low_Return
            Maximum downside reached during the future window.

        MFE
            Maximum favourable excursion.

        MAE
            Maximum adverse excursion.

    These targets are useful for target-range modelling and risk
    analysis.
    """

    required = {
        "Close",
        "High",
        "Low",
    }

    missing = [
        column
        for column in required
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

    for horizon in horizons:

        future_highs = pd.concat(
            [
                result["High"].shift(-step)
                for step in range(
                    1,
                    horizon + 1,
                )
            ],
            axis=1,
        )

        future_lows = pd.concat(
            [
                result["Low"].shift(-step)
                for step in range(
                    1,
                    horizon + 1,
                )
            ],
            axis=1,
        )

        future_high = (
            future_highs.max(
                axis=1
            )
        )

        future_low = (
            future_lows.min(
                axis=1
            )
        )

        future_high_return = (
            future_high
            / result["Close"]
            - 1
        )

        future_low_return = (
            future_low
            / result["Close"]
            - 1
        )

        result[
            f"Future_High_Return_{horizon}"
        ] = future_high_return

        result[
            f"Future_Low_Return_{horizon}"
        ] = future_low_return

        # MFE represents the best favourable movement.
        result[
            f"MFE_{horizon}"
        ] = future_high_return.clip(
            lower=0
        )

        # MAE represents adverse movement.
        # Stored as a positive magnitude.
        result[
            f"MAE_{horizon}"
        ] = (
            -future_low_return
        ).clip(
            lower=0
        )

    return result


def add_range_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """
    Create target-range variables.

    These are deliberately expressed as returns rather than prices.

    The eventual prediction layer can convert:

        predicted return
            ->
        predicted price range

    using the latest available price.

    This makes the model more transferable across stocks with
    very different absolute prices.
    """

    result = data.copy()

    horizons = validate_horizons(
        horizons
    )

    # Path targets must exist first.
    required_targets = []

    for horizon in horizons:
        required_targets.extend(
            [
                f"Future_High_Return_{horizon}",
                f"Future_Low_Return_{horizon}",
            ]
        )

    missing = [
        column
        for column in required_targets
        if column not in result.columns
    ]

    if missing:
        result = add_path_targets(
            result,
            horizons=horizons,
        )

    for horizon in horizons:

        high_return = result[
            f"Future_High_Return_{horizon}"
        ]

        low_return = result[
            f"Future_Low_Return_{horizon}"
        ]

        result[
            f"Target_Upper_Return_{horizon}"
        ] = high_return

        result[
            f"Target_Lower_Return_{horizon}"
        ] = low_return

    return result


def build_all_targets(
    data: pd.DataFrame,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    direction_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Build the complete target set.

    Processing order:

        Direction
            ↓
        Future path
            ↓
        Target ranges
    """

    horizons = validate_horizons(
        horizons
    )

    result = add_direction_targets(
        data,
        horizons=horizons,
        threshold=direction_threshold,
    )

    result = add_path_targets(
        result,
        horizons=horizons,
    )

    result = add_range_targets(
        result,
        horizons=horizons,
    )

    return result


def target_columns(
    data: pd.DataFrame,
) -> list[str]:
    """
    Return every target column.

    This function is used as a safety barrier before modelling.
    """

    target_prefixes = (
        "Future_",
        "Direction_",
        "MFE_",
        "MAE_",
        "Target_",
    )

    return [
        column
        for column in data.columns
        if column.startswith(
            target_prefixes
        )
    ]


def direction_target(
    data: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Return one direction target."""

    column = (
        f"Direction_{horizon}"
    )

    if column not in data.columns:
        raise KeyError(
            f"Target not found: {column}"
        )

    return data[column]


def return_target(
    data: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    """Return one future-return target."""

    column = (
        f"Future_Return_{horizon}"
    )

    if column not in data.columns:
        raise KeyError(
            f"Target not found: {column}"
        )

    return data[column]


def range_targets(
    data: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """
    Return upper and lower future-range targets.
    """

    upper = (
        f"Target_Upper_Return_{horizon}"
    )

    lower = (
        f"Target_Lower_Return_{horizon}"
    )

    missing = [
        column
        for column in (
            upper,
            lower,
        )
        if column not in data.columns
    ]

    if missing:
        raise KeyError(
            f"Range targets not found: {missing}"
        )

    return data[
        [
            lower,
            upper,
        ]
    ].copy()


def price_range_from_returns(
    current_price: float,
    lower_return: float,
    upper_return: float,
) -> tuple[float, float]:
    """
    Convert predicted return bounds into price bounds.
    """

    if not np.isfinite(
        current_price
    ):
        raise ValueError(
            "current_price must be finite."
        )

    if current_price <= 0:
        raise ValueError(
            "current_price must be positive."
        )

    if not np.isfinite(
        lower_return
    ) or not np.isfinite(
        upper_return
    ):
        raise ValueError(
            "Return bounds must be finite."
        )

    lower_price = (
        current_price
        * (1 + lower_return)
    )

    upper_price = (
        current_price
        * (1 + upper_return)
    )

    return (
        float(lower_price),
        float(upper_price),
    )


def validate_target_columns(
    data: pd.DataFrame,
) -> None:
    """
    Verify that target columns are clearly separated from
    normal explanatory features.
    """

    targets = target_columns(
        data
    )

    for column in targets:

        if not (
            column.startswith("Future_")
            or column.startswith("Direction_")
            or column.startswith("MFE_")
            or column.startswith("MAE_")
            or column.startswith("Target_")
        ):
            raise ValueError(
                f"Unrecognized target column: {column}"
            )
