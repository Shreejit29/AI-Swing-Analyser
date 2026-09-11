"""
Tests for the unified feature integration layer.

No live market-data requests are used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.feature_integration import (
    UnifiedFeatureConfig,
    UnifiedFeatureResult,
    build_unified_features,
    feature_matrix,
    unified_feature_names,
    unified_feature_summary,
)
from src.research.market_context import (
    MarketContextResult,
)
from src.research.market_data_context import (
    MarketContextData,
)


def make_ohlcv(
    n: int = 320,
    start_price: float = 100.0,
    slope: float = 0.20,
) -> pd.DataFrame:
    """Create deterministic synthetic OHLCV data."""

    index = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D",
    )

    x = np.arange(n, dtype=float)

    close = (
        start_price
        + slope * x
        + 2.0 * np.sin(x / 9.0)
        + 0.8 * np.sin(x / 3.0)
    )

    open_price = (
        close
        * (
            1.0
            + 0.002 * np.sin(x / 5.0)
        )
    )

    high = np.maximum(
        open_price,
        close,
    ) * 1.01

    low = np.minimum(
        open_price,
        close,
    ) * 0.99

    volume = (
        1_000_000
        + 50_000
        * (
            1.0
            + np.sin(x / 7.0)
        )
        + 1_000 * x
    )

    return pd.DataFrame(
        {
            "Open": open_price,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=index,
    )


def make_market_context(
    n: int = 320,
) -> MarketContextData:
    """Create deterministic synthetic NIFTY50 context."""

    index = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D",
    )

    x = np.arange(n, dtype=float)

    close = (
        100.0
        + 0.15 * x
        + np.sin(x / 10.0)
    )

    context = pd.DataFrame(
        {
            "NIFTY50_Close": close,
            "NIFTY50_Market_Return_1": (
                0.001 + x * 0.0
            ),
            "NIFTY50_Market_Return_5": (
                0.005 + x * 0.0
            ),
            "NIFTY50_Market_Return_20": (
                0.020 + x * 0.0
            ),
            "NIFTY50_Market_MA_20": (
                close - 1.0
            ),
            "NIFTY50_Market_MA_50": (
                close - 2.0
            ),
            "NIFTY50_Market_MA_200": (
                close - 3.0
            ),
            "NIFTY50_Market_Volatility_10": (
                0.01 + x * 0.0
            ),
            "NIFTY50_Market_Volatility_20": (
                0.012 + x * 0.0
            ),
            "NIFTY50_Market_Volatility_60": (
                0.015 + x * 0.0
            ),
            "NIFTY50_Market_Momentum_5": (
                0.005 + x * 0.0
            ),
            "NIFTY50_Market_Momentum_10": (
                0.010 + x * 0.0
            ),
            "NIFTY50_Market_Momentum_20": (
                0.020 + x * 0.0
            ),
            "NIFTY50_Trend_Score": (
                0.75 + x * 0.0
            ),
            "NIFTY50_Regime": [
                "BULL"
                for _ in range(n)
            ],
        },
        index=index,
    )

    raw = make_ohlcv(
        n=n,
        start_price=100.0,
        slope=0.15,
    )

    return MarketContextData(
        raw_data={
            "NIFTY50": raw
        },
        context=MarketContextResult(
            data=context,
            feature_names=list(
                context.columns
            ),
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True,
            "future_values_used": False,
        },
        warnings=[],
    )


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


def test_unified_feature_config_defaults():
    config = UnifiedFeatureConfig()

    assert (
        config.include_market_context
        is True
    )

    assert (
        config.include_relative_strength
        is True
    )

    assert (
        config.include_multi_timeframe
        is True
    )

    assert (
        config.benchmark
        == "NIFTY50"
    )

    assert (
        config.relative_strength_window
        == 20
    )

    assert (
        config.max_missing_fraction
        == 0.40
    )

    assert (
        config.reject_future_columns
        is True
    )


def test_invalid_benchmark_fails():
    with pytest.raises(ValueError):
        UnifiedFeatureConfig(
            benchmark=""
        )


def test_invalid_relative_strength_window_fails():
    with pytest.raises(ValueError):
        UnifiedFeatureConfig(
            relative_strength_window=1
        )


def test_invalid_missing_fraction_fails():
    with pytest.raises(ValueError):
        UnifiedFeatureConfig(
            max_missing_fraction=1.5
        )


# ---------------------------------------------------------------------
# Basic feature construction
# ---------------------------------------------------------------------


def test_build_unified_features():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    assert isinstance(
        result,
        UnifiedFeatureResult,
    )

    assert (
        result.rows
        == len(stock)
    )

    assert (
        result.feature_count
        > 0
    )


def test_base_features_are_created():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    assert (
        len(
            result.base_feature_names
        )
        > 0
    )


def test_technical_features_are_present():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    expected = [
        "RSI_14",
        "MACD",
        "EMA_20",
        "EMA_50",
        "ATR_14",
        "ADX_14",
    ]

    for column in expected:
        assert column in result.data.columns


def test_price_action_features_are_present():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    candidates = [
        "Body_Size",
        "Upper_Wick",
        "Lower_Wick",
        "Candle_Range",
    ]

    assert any(
        column in result.data.columns
        for column in candidates
    )


def test_volume_features_are_present():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    candidates = [
        "RVOL_20",
        "OBV",
        "Volume_Change_20",
        "Price_Volume_Confirmation",
    ]

    assert any(
        column in result.data.columns
        for column in candidates
    )


def test_regime_features_are_present():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    candidates = [
        "Trend_Regime",
        "Volatility_Regime",
        "Market_State_Score",
    ]

    assert any(
        column in result.data.columns
        for column in candidates
    )


# ---------------------------------------------------------------------
# Market context
# ---------------------------------------------------------------------


def test_market_context_is_integrated():
    stock = make_ohlcv()
    market = make_market_context()

    result = build_unified_features(
        stock,
        market_context=market,
    )

    assert (
        len(
            result.market_feature_names
        )
        > 0
    )

    assert (
        "NIFTY50_Close"
        in result.data.columns
    )


def test_relative_strength_is_integrated():
    stock = make_ohlcv()
    market = make_market_context()

    result = build_unified_features(
        stock,
        market_context=market,
    )

    assert any(
        column.startswith(
            "NIFTY50_Relative_Strength"
        )
        for column in result.data.columns
    )


def test_market_context_can_be_disabled():
    stock = make_ohlcv()
    market = make_market_context()

    result = build_unified_features(
        stock,
        market_context=market,
        config=UnifiedFeatureConfig(
            include_market_context=False
        ),
    )

    assert (
        len(
            result.market_feature_names
        )
        == 0
    )

    assert (
        "NIFTY50_Close"
        not in result.data.columns
    )


def test_relative_strength_can_be_disabled():
    stock = make_ohlcv()
    market = make_market_context()

    result = build_unified_features(
        stock,
        market_context=market,
        config=UnifiedFeatureConfig(
            include_relative_strength=False
        ),
    )

    assert not any(
        column.startswith(
            "NIFTY50_Relative_Strength"
        )
        for column in result.data.columns
    )


def test_missing_market_context_generates_warning():
    stock = make_ohlcv()

    result = build_unified_features(
        stock,
        market_context=None,
        config=UnifiedFeatureConfig(
            include_market_context=True
        ),
    )

    assert len(
        result.warnings
    ) > 0


# ---------------------------------------------------------------------
# Feature / target separation
# ---------------------------------------------------------------------


def test_future_columns_are_not_features():
    stock = make_ohlcv()

    stock["Future_Return_5"] = (
        stock["Close"]
        .shift(-5)
        / stock["Close"]
        - 1.0
    )

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_direction_columns_are_not_features():
    stock = make_ohlcv()

    stock["Direction_5"] = (
        stock["Close"]
        .shift(-5)
        > stock["Close"]
    ).astype(int)

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_target_columns_are_not_features():
    stock = make_ohlcv()

    stock["Target_Return"] = (
        stock["Close"]
        .shift(-5)
    )

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_label_columns_are_not_features():
    stock = make_ohlcv()

    stock["Label"] = 1

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_feature_names_are_unique():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    assert len(
        result.feature_names
    ) == len(
        set(result.feature_names)
    )


# ---------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------


def test_feature_matrix_returns_dataframe():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    matrix = feature_matrix(
        result
    )

    assert isinstance(
        matrix,
        pd.DataFrame,
    )

    assert list(
        matrix.columns
    ) == result.feature_names


def test_feature_matrix_preserves_index():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    matrix = feature_matrix(
        result
    )

    assert matrix.index.equals(
        stock.index
    )


def test_feature_matrix_is_copy():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    matrix = feature_matrix(
        result
    )

    column = result.feature_names[0]

    original_value = result.data.loc[
        result.data.index[100],
        column,
    ]

    matrix.loc[
        matrix.index[100],
        column,
    ] = 999999.0

    assert (
        result.data.loc[
            result.data.index[100],
            column,
        ]
        == original_value
    )


def test_unified_feature_names_returns_copy():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    names = unified_feature_names(
        result
    )

    names.append(
        "FAKE_FEATURE"
    )

    assert (
        "FAKE_FEATURE"
        not in result.feature_names
    )


# ---------------------------------------------------------------------
# Timeline integrity
# ---------------------------------------------------------------------


def test_timeline_is_preserved():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    assert result.data.index.equals(
        stock.index
    )


def test_row_count_is_preserved():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    assert len(
        result.data
    ) == len(stock)


def test_unsorted_stock_data_fails():
    stock = make_ohlcv()

    stock = stock.iloc[
        ::-1
    ]

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_duplicate_stock_timestamps_fail():
    stock = make_ohlcv()

    stock.index = list(
        stock.index
    )

    stock.index = pd.DatetimeIndex(
        stock.index
    )

    stock.index = stock.index.where(
        np.arange(len(stock)) != 10,
        stock.index[9],
    )

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


# ---------------------------------------------------------------------
# Numeric and finite safety
# ---------------------------------------------------------------------


def test_non_numeric_feature_data_is_rejected():
    stock = make_ohlcv()

    stock["Close"] = "invalid"

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_infinite_ohlcv_is_rejected():
    stock = make_ohlcv()

    stock.loc[
        stock.index[100],
        "Close",
    ] = np.inf

    with pytest.raises(ValueError):
        build_unified_features(
            stock
        )


def test_feature_matrix_is_numeric():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    matrix = feature_matrix(
        result
    )

    for column in matrix.columns:
        assert pd.api.types.is_numeric_dtype(
            matrix[column]
        )


# ---------------------------------------------------------------------
# Missing feature handling
# ---------------------------------------------------------------------


def test_high_missing_features_are_removed():
    stock = make_ohlcv()

    stock["Artificial_Missing"] = np.nan

    result = build_unified_features(
        stock
    )

    assert (
        "Artificial_Missing"
        not in result.feature_names
    )


def test_missing_feature_removal_is_reported():
    stock = make_ohlcv()

    stock["Artificial_Missing"] = np.nan

    result = build_unified_features(
        stock
    )

    assert any(
        "Artificial_Missing"
        in warning
        for warning in result.warnings
    )


# ---------------------------------------------------------------------
# Multi-timeframe
# ---------------------------------------------------------------------


def test_multitimeframe_features_can_be_added():
    stock = make_ohlcv(
        n=320
    )

    daily = stock.copy()

    weekly = (
        stock.resample("W")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    monthly = (
        stock.resample("MS")
        .agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )
        .dropna()
    )

    timeframe_data = {
        "4H": daily,
        "1D": daily,
        "1W": weekly,
        "1M": monthly,
    }

    result = build_unified_features(
        stock,
        timeframe_data=timeframe_data,
    )

    assert isinstance(
        result,
        UnifiedFeatureResult,
    )

    assert (
        result.rows
        == len(stock)
    )


def test_multitimeframe_can_be_disabled():
    stock = make_ohlcv()

    result = build_unified_features(
        stock,
        timeframe_data=None,
        config=UnifiedFeatureConfig(
            include_multi_timeframe=False
        ),
    )

    assert isinstance(
        result,
        UnifiedFeatureResult,
    )


# ---------------------------------------------------------------------
# Future mutation resistance
# ---------------------------------------------------------------------


def test_base_features_do_not_depend_on_future_prices():
    stock = make_ohlcv()

    original = build_unified_features(
        stock
    )

    mutated = stock.copy(
        deep=True
    )

    future_start = 220

    mutated.loc[
        mutated.index[
            future_start:
        ],
        "Close",
    ] *= 100.0

    mutated.loc[
        mutated.index[
            future_start:
        ],
        "High",
    ] *= 100.0

    mutated.loc[
        mutated.index[
            future_start:
        ],
        "Low",
    ] *= 100.0

    changed = build_unified_features(
        mutated
    )

    timestamp = stock.index[
        150
    ]

    # A past feature should remain unchanged
    # after changing only future market prices.
    common = [
        column
        for column in original.feature_names
        if column in changed.feature_names
    ]

    for column in common:
        left = original.data.loc[
            timestamp,
            column,
        ]

        right = changed.data.loc[
            timestamp,
            column,
        ]

        if pd.isna(left) and pd.isna(right):
            continue

        assert left == pytest.approx(
            right
        )


# ---------------------------------------------------------------------
# Market future mutation
# ---------------------------------------------------------------------


def test_market_features_do_not_use_future_market_rows():
    stock = make_ohlcv()
    market = make_market_context()

    original = build_unified_features(
        stock,
        market_context=market,
    )

    mutated_context = (
        market.context.data.copy(
            deep=True
        )
    )

    future_start = 220

    for column in [
        "NIFTY50_Close",
        "NIFTY50_Market_MA_20",
        "NIFTY50_Market_Return_20",
        "NIFTY50_Trend_Score",
    ]:
        if column in mutated_context.columns:
            mutated_context.loc[
                mutated_context.index[
                    future_start:
                ],
                column,
            ] *= 100.0

    mutated_market = MarketContextData(
        raw_data={
            "NIFTY50": (
                market.raw_data[
                    "NIFTY50"
                ].copy(deep=True)
            )
        },
        context=MarketContextResult(
            data=mutated_context,
            feature_names=list(
                mutated_context.columns
            ),
            warnings=[],
            metadata={
                "relative_strength_window": 20
            },
        ),
        metadata={
            "research_only": True
        },
        warnings=[],
    )

    changed = build_unified_features(
        stock,
        market_context=mutated_market,
    )

    timestamp = stock.index[
        150
    ]

    common = [
        column
        for column in original.market_feature_names
        if column in changed.data.columns
    ]

    for column in common:
        left = original.data.loc[
            timestamp,
            column,
        ]

        right = changed.data.loc[
            timestamp,
            column,
        ]

        if pd.isna(left) and pd.isna(right):
            continue

        assert left == pytest.approx(
            right
        )


# ---------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------


def test_stock_input_is_not_modified():
    stock = make_ohlcv()

    original = stock.copy(
        deep=True
    )

    build_unified_features(
        stock
    )

    pd.testing.assert_frame_equal(
        stock,
        original,
    )


def test_market_input_is_not_modified():
    stock = make_ohlcv()
    market = make_market_context()

    original_context = (
        market.context.data.copy(
            deep=True
        )
    )

    original_raw = {
        name: frame.copy(
            deep=True
        )
        for name, frame
        in market.raw_data.items()
    }

    build_unified_features(
        stock,
        market_context=market,
    )

    pd.testing.assert_frame_equal(
        market.context.data,
        original_context,
    )

    for name in original_raw:
        pd.testing.assert_frame_equal(
            market.raw_data[name],
            original_raw[name],
        )


# ---------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------


def test_feature_generation_is_deterministic():
    stock = make_ohlcv()

    result_1 = build_unified_features(
        stock
    )

    result_2 = build_unified_features(
        stock
    )

    assert (
        result_1.feature_names
        == result_2.feature_names
    )

    pd.testing.assert_frame_equal(
        result_1.data,
        result_2.data,
    )


def test_market_feature_generation_is_deterministic():
    stock = make_ohlcv()
    market = make_market_context()

    result_1 = build_unified_features(
        stock,
        market_context=market,
    )

    result_2 = build_unified_features(
        stock,
        market_context=market,
    )

    assert (
        result_1.feature_names
        == result_2.feature_names
    )

    pd.testing.assert_frame_equal(
        result_1.data,
        result_2.data,
    )


# ---------------------------------------------------------------------
# Summary and metadata
# ---------------------------------------------------------------------


def test_summary():
    stock = make_ohlcv()
    market = make_market_context()

    result = build_unified_features(
        stock,
        market_context=market,
    )

    summary = unified_feature_summary(
        result
    )

    assert (
        summary["rows"]
        == len(stock)
    )

    assert (
        summary["feature_count"]
        > 0
    )

    assert (
        summary["base_features"]
        > 0
    )

    assert (
        summary["market_features"]
        > 0
    )


def test_metadata_is_research_only():
    stock = make_ohlcv()

    result = build_unified_features(
        stock
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )

    assert (
        result.metadata[
            "production_approved"
        ]
        is False
    )

    assert (
        result.metadata[
            "future_values_used"
        ]
        is False
    )

    assert (
        result.metadata[
            "target_columns_included"
        ]
        is False
    )

    assert (
        result.metadata[
            "final_holdout_used"
        ]
        is False
    )


def test_summary_requires_correct_type():
    with pytest.raises(TypeError):
        unified_feature_summary(
            object()
        )


def test_feature_matrix_requires_correct_type():
    with pytest.raises(TypeError):
        feature_matrix(
            object()
        )


def test_feature_names_requires_correct_type():
    with pytest.raises(TypeError):
        unified_feature_names(
            object()
        )
