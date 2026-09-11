"""
AI Swing Analyser — Unified Feature Integration.

Combines:
- Technical indicators
- Price-action features
- Volume features
- Regime features
- Multi-timeframe context
- Indian market context
- Stock-vs-market relative strength

Design principles:
- Feature engineering remains causal.
- No target columns are allowed.
- No future columns are allowed.
- Original OHLCV data is not modified.
- Final holdout data is never used for fitting.
- This module only constructs research features.

It does NOT:
- train models
- select models
- calibrate probabilities
- generate BUY/SELL decisions
- approve production models
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features.engine import (
    FeatureSet,
    engineer_features,
)
from src.features.multi_timeframe import (
    build_multi_timeframe_dataset,
)
from .market_data_context import (
    MarketContextData,
)
from .market_integration import (
    MarketIntegrationConfig,
    integrate_market_context,
)


@dataclass(frozen=True)
class UnifiedFeatureConfig:
    """Configuration for unified feature construction."""

    include_market_context: bool = True

    include_relative_strength: bool = True

    include_multi_timeframe: bool = True

    benchmark: str = "NIFTY50"

    relative_strength_window: int = 20

    max_missing_fraction: float = 0.40

    reject_future_columns: bool = True

    def __post_init__(self) -> None:
        if not isinstance(
            self.benchmark,
            str,
        ) or not self.benchmark.strip():
            raise ValueError(
                "benchmark must be a non-empty string."
            )

        if (
            not isinstance(
                self.relative_strength_window,
                int,
            )
            or isinstance(
                self.relative_strength_window,
                bool,
            )
            or self.relative_strength_window < 2
        ):
            raise ValueError(
                "relative_strength_window must be >= 2."
            )

        if not (
            0.0
            <= self.max_missing_fraction
            <= 1.0
        ):
            raise ValueError(
                "max_missing_fraction must be between 0 and 1."
            )


@dataclass
class UnifiedFeatureResult:
    """Complete research feature-set result."""

    data: pd.DataFrame

    feature_names: list[str]

    base_feature_names: list[str]

    market_feature_names: list[str]

    multi_timeframe_feature_names: list[str]

    rows: int

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, object] = field(
        default_factory=dict
    )

    @property
    def feature_count(self) -> int:
        return len(
            self.feature_names
        )

    def summary(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "feature_count": self.feature_count,
            "base_features": len(
                self.base_feature_names
            ),
            "market_features": len(
                self.market_feature_names
            ),
            "multi_timeframe_features": len(
                self.multi_timeframe_feature_names
            ),
            "warning_count": len(
                self.warnings
            ),
            "research_only": self.metadata.get(
                "research_only",
                True,
            ),
        }


def _validate_ohlcv(
    data: pd.DataFrame,
) -> None:
    """Validate the primary stock dataframe."""

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "data cannot be empty."
        )

    if not isinstance(
        data.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "data must have a DatetimeIndex."
        )

    if data.index.has_duplicates:
        raise ValueError(
            "data contains duplicate timestamps."
        )

    if not data.index.is_monotonic_increasing:
        raise ValueError(
            "data must be chronologically sorted."
        )

    required = {
        "Open",
        "High",
        "Low",
        "Close",
    }

    missing = required.difference(
        data.columns
    )

    if missing:
        raise ValueError(
            "Missing required OHLCV columns: "
            f"{sorted(missing)}"
        )

    for column in required:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if values.isna().all():
            raise ValueError(
                f"{column} contains no valid numeric values."
            )

        finite = values.dropna()

        if not np.isfinite(
            finite.to_numpy()
        ).all():
            raise ValueError(
                f"{column} contains non-finite values."
            )


def _validate_feature_names(
    feature_names: list[str],
    *,
    reject_future_columns: bool,
) -> None:
    """Reject suspicious target/future columns."""

    if len(feature_names) != len(
        set(feature_names)
    ):
        raise ValueError(
            "Feature names contain duplicates."
        )

    if not reject_future_columns:
        return

    suspicious_tokens = (
        "Future_",
        "Direction_",
        "Target_",
        "Target",
        "Label",
    )

    suspicious = [
        column
        for column in feature_names
        if any(
            token.lower()
            in column.lower()
            for token in suspicious_tokens
        )
    ]

    if suspicious:
        raise ValueError(
            "Potential target/future columns detected "
            f"in features: {suspicious}"
        )


def _numeric_feature_names(
    data: pd.DataFrame,
    feature_names: list[str],
) -> list[str]:
    """Return only numeric feature columns."""

    numeric = []

    for column in feature_names:
        if pd.api.types.is_numeric_dtype(
            data[column]
        ):
            numeric.append(column)

    return numeric


def _remove_high_missing(
    data: pd.DataFrame,
    feature_names: list[str],
    threshold: float,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
]:
    """
    Remove features exceeding the missing-value threshold.

    This operation is deterministic and does not inspect targets.
    """

    missing_fraction = data[
        feature_names
    ].isna().mean()

    removed = [
        column
        for column in feature_names
        if missing_fraction[column]
        > threshold
    ]

    kept = [
        column
        for column in feature_names
        if column not in removed
    ]

    result = data.copy(
        deep=True
    )

    return (
        result,
        kept,
        removed,
    )


def _build_base_features(
    stock_data: pd.DataFrame,
) -> FeatureSet:
    """
    Build the existing base feature stack.

    The existing engine already combines:
    technical indicators,
    price action,
    volume and regime features.
    """

    return engineer_features(
        stock_data.copy(
            deep=True
        )
    )


def _integrate_multi_timeframe(
    stock_data: pd.DataFrame,
    timeframe_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Build multi-timeframe features.

    Higher-timeframe features are aligned through the existing
    leakage-safe multi-timeframe feature implementation.
    """

    if not timeframe_data:
        return stock_data.copy(
            deep=True
        )

    frames = {
        key: value.copy(
            deep=True
        )
        for key, value
        in timeframe_data.items()
    }

    if "4H" not in frames:
        frames["4H"] = stock_data.copy(
            deep=True
        )

    return build_multi_timeframe_dataset(
        frames
    )


def _identify_added_columns(
    original_columns: list[str],
    combined_columns: list[str],
) -> list[str]:
    original = set(
        original_columns
    )

    return [
        column
        for column in combined_columns
        if column not in original
    ]


def build_unified_features(
    stock_data: pd.DataFrame,
    *,
    market_context: MarketContextData | None = None,
    timeframe_data: dict[
        str,
        pd.DataFrame,
    ]
    | None = None,
    config: UnifiedFeatureConfig | None = None,
) -> UnifiedFeatureResult:
    """
    Construct the unified research feature dataframe.
    """

    _validate_ohlcv(
        stock_data
    )

    cfg = (
        config
        if config is not None
        else UnifiedFeatureConfig()
    )

    original = stock_data.copy(
        deep=True
    )

    warnings: list[str] = []

    # ---------------------------------------------------------------
    # 1. Base feature engineering
    # ---------------------------------------------------------------

    base_result = _build_base_features(
        original
    )

    if not isinstance(
        base_result,
        FeatureSet,
    ):
        raise TypeError(
            "engineer_features must return FeatureSet."
        )

    base_data = base_result.data.copy(
        deep=True
    )

    base_feature_names = list(
        base_result.feature_names
    )

    # ---------------------------------------------------------------
    # 2. Multi-timeframe integration
    # ---------------------------------------------------------------

    mtf_feature_names: list[str] = []

    if (
        cfg.include_multi_timeframe
        and timeframe_data is not None
    ):
        mtf_data = _integrate_multi_timeframe(
            original,
            timeframe_data,
        )

        # Only retain columns that were added by
        # multi-timeframe processing.
        mtf_added = _identify_added_columns(
            list(original.columns),
            list(mtf_data.columns),
        )

        mtf_added = [
            column
            for column in mtf_added
            if column not in base_data.columns
        ]

        for column in mtf_added:
            base_data[column] = (
                mtf_data[column]
                .reindex(
                    base_data.index
                )
            )

        mtf_feature_names = mtf_added

    # ---------------------------------------------------------------
    # 3. Market-context integration
    # ---------------------------------------------------------------

    market_feature_names: list[str] = []

    if (
        cfg.include_market_context
        and market_context is not None
    ):
        market_result = integrate_market_context(
            base_data,
            market_context,
            config=MarketIntegrationConfig(
                benchmark=cfg.benchmark,
                relative_strength_window=(
                    cfg.relative_strength_window
                ),
                allow_market_missing=True,
            ),
        )

        base_data = market_result.data.copy(
            deep=True
        )

        market_feature_names = list(
            market_result.market_columns
        )

        if cfg.include_relative_strength:
            market_feature_names.extend(
                market_result.relative_strength_columns
            )
        else:
            base_data = base_data.drop(
                columns=market_result.relative_strength_columns,
                errors="ignore",
            )

        warnings.extend(
            market_result.warnings
        )

    elif cfg.include_market_context:
        warnings.append(
            "Market context was requested but "
            "no market_context was supplied."
        )

    # ---------------------------------------------------------------
    # 4. Identify complete feature set
    # ---------------------------------------------------------------

    original_columns = set(
        original.columns
    )

    excluded = {
        "Symbol",
        "_source_time",
        "_base_time",
    }

    feature_names = [
        column
        for column in base_data.columns
        if column not in original_columns
        and column not in excluded
    ]

    # Include numeric original OHLCV columns as model inputs only
    # when they were already handled by the existing engine.
    #
    # The feature engine is authoritative for base feature names.
    for column in base_feature_names:
        if (
            column in base_data.columns
            and column not in feature_names
            and column not in excluded
        ):
            feature_names.append(
                column
            )

    # Add MTF features.
    for column in mtf_feature_names:
        if (
            column in base_data.columns
            and column not in feature_names
        ):
            feature_names.append(
                column
            )

    # Add market features.
    for column in market_feature_names:
        if (
            column in base_data.columns
            and column not in feature_names
        ):
            feature_names.append(
                column
            )

    # Keep only features that actually exist.
    feature_names = [
        column
        for column in feature_names
        if column in base_data.columns
    ]

    _validate_feature_names(
        feature_names,
        reject_future_columns=(
            cfg.reject_future_columns
        ),
    )

    numeric_features = _numeric_feature_names(
        base_data,
        feature_names,
    )

    if len(numeric_features) != len(
        feature_names
    ):
        non_numeric = [
            column
            for column in feature_names
            if column not in numeric_features
        ]

        raise TypeError(
            "Non-numeric model features detected: "
            f"{non_numeric}"
        )

    # ---------------------------------------------------------------
    # 5. Missing-value screening
    # ---------------------------------------------------------------

    (
        base_data,
        kept_features,
        removed_features,
    ) = _remove_high_missing(
        base_data,
        feature_names,
        cfg.max_missing_fraction,
    )

    if removed_features:
        warnings.append(
            "Removed high-missing features: "
            + ", ".join(
                removed_features
            )
        )

    feature_names = kept_features

    if not feature_names:
        raise ValueError(
            "No usable features remain after feature validation."
        )

    # ---------------------------------------------------------------
    # 6. Numeric/finite validation
    # ---------------------------------------------------------------

    values = base_data[
        feature_names
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    finite_check = np.isfinite(
        values.fillna(0.0).to_numpy()
    )

    if not finite_check.all():
        raise ValueError(
            "Feature dataframe contains non-finite values."
        )

    for column in feature_names:
        base_data[column] = pd.to_numeric(
            base_data[column],
            errors="coerce",
        )

    # ---------------------------------------------------------------
    # 7. Timeline integrity
    # ---------------------------------------------------------------

    if not base_data.index.equals(
        original.index
    ):
        raise RuntimeError(
            "Feature integration changed the stock timeline."
        )

    # ---------------------------------------------------------------
    # 8. Final metadata
    # ---------------------------------------------------------------

    metadata = {
        "research_only": True,
        "production_approved": False,
        "future_values_used": False,
        "target_columns_included": False,
        "final_holdout_used": False,
        "base_feature_count": len(
            base_feature_names
        ),
        "multi_timeframe_feature_count": len(
            mtf_feature_names
        ),
        "market_feature_count": len(
            market_feature_names
        ),
        "removed_high_missing_features": (
            removed_features
        ),
    }

    return UnifiedFeatureResult(
        data=base_data,
        feature_names=feature_names,
        base_feature_names=[
            column
            for column in base_feature_names
            if column in feature_names
        ],
        market_feature_names=[
            column
            for column in market_feature_names
            if column in feature_names
        ],
        multi_timeframe_feature_names=[
            column
            for column in mtf_feature_names
            if column in feature_names
        ],
        rows=len(base_data),
        warnings=warnings,
        metadata=metadata,
    )


def feature_matrix(
    result: UnifiedFeatureResult,
) -> pd.DataFrame:
    """Return the model-ready feature matrix."""

    if not isinstance(
        result,
        UnifiedFeatureResult,
    ):
        raise TypeError(
            "result must be UnifiedFeatureResult."
        )

    return result.data[
        result.feature_names
    ].copy(
        deep=True
    )


def unified_feature_names(
    result: UnifiedFeatureResult,
) -> list[str]:
    """Return a copy of the final feature list."""

    if not isinstance(
        result,
        UnifiedFeatureResult,
    ):
        raise TypeError(
            "result must be UnifiedFeatureResult."
        )

    return list(
        result.feature_names
    )


def unified_feature_summary(
    result: UnifiedFeatureResult,
) -> dict[str, object]:
    """Return a compact feature summary."""

    if not isinstance(
        result,
        UnifiedFeatureResult,
    ):
        raise TypeError(
            "result must be UnifiedFeatureResult."
        )

    return result.summary()


__all__ = [
    "UnifiedFeatureConfig",
    "UnifiedFeatureResult",
    "build_unified_features",
    "feature_matrix",
    "unified_feature_names",
    "unified_feature_summary",
]
