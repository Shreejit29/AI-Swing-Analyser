"""
AI Swing Analyser - Research Dataset Builder.

Builds a model-ready research dataset from historical market data.

Pipeline:

    Historical OHLCV
        ↓
    Quality control
        ↓
    Feature engineering
        ↓
    Target construction
        ↓
    Leakage checks
        ↓
    Research dataset

The final holdout is NOT created here.
Temporal partitioning belongs to the validation layer.

Important:
    Future values are allowed only inside target columns.
    They must never enter the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.data.pipeline import (
    build_historical_dataset,
)

from src.features.engine import (
    engineer_features,
)

from src.models.targets import (
    TargetSpec,
    build_targets,
)

from .config import (
    ResearchPipelineConfig,
)


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


@dataclass
class ResearchDatasetResult:
    """
    Output of the research dataset builder.
    """

    symbol: str

    timeframe: str

    dataframe: pd.DataFrame

    feature_columns: list[str] = field(
        default_factory=list
    )

    target_columns: list[str] = field(
        default_factory=list
    )

    horizons: tuple[int, ...] = field(
        default_factory=tuple
    )

    start: pd.Timestamp | None = None

    end: pd.Timestamp | None = None

    rows: int = 0

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def feature_count(self) -> int:
        return len(
            self.feature_columns
        )

    @property
    def target_count(self) -> int:
        return len(
            self.target_columns
        )

    def summary(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "features": self.feature_count,
            "targets": self.target_count,
            "horizons": list(
                self.horizons
            ),
            "start": self.start,
            "end": self.end,
            "warnings": len(
                self.warnings
            ),
        }


# ----------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------


class ResearchDatasetBuilder:
    """
    Builds a leakage-safe research dataset.

    The builder is intentionally independent of model training.
    """

    def __init__(
        self,
        config: ResearchPipelineConfig,
        *,
        data_builder: Callable[..., Any] | None = None,
        feature_builder: Callable[
            [pd.DataFrame],
            Any,
        ]
        | None = None,
        target_builder: Callable[..., Any] | None = None,
    ) -> None:

        if not isinstance(
            config,
            ResearchPipelineConfig,
        ):
            raise TypeError(
                "config must be a ResearchPipelineConfig."
            )

        self.config = config

        self.data_builder = (
            data_builder
            or build_historical_dataset
        )

        self.feature_builder = (
            feature_builder
            or engineer_features
        )

        self.target_builder = (
            target_builder
            or build_targets
        )

    # ------------------------------------------------------------------
    # Public build method
    # ------------------------------------------------------------------

    def build(self) -> ResearchDatasetResult:
        """
        Build the complete research dataset.
        """

        raw_data = self._load_data()

        self._validate_raw_data(
            raw_data
        )

        features = self._build_features(
            raw_data
        )

        self._validate_features(
            features
        )

        dataset = self._build_targets(
            features
        )

        feature_columns = (
            self._identify_features(
                dataset
            )
        )

        target_columns = (
            self._identify_targets(
                dataset
            )
        )

        self._validate_feature_target_separation(
            feature_columns,
            target_columns,
        )

        self._validate_feature_names(
            feature_columns
        )

        self._validate_feature_values(
            dataset,
            feature_columns,
        )

        dataset = self._sort_and_deduplicate(
            dataset
        )

        warnings = self._build_warnings(
            dataset,
            feature_columns,
            target_columns,
        )

        return ResearchDatasetResult(
            symbol=self.config.symbol,
            timeframe=self.config.timeframe,
            dataframe=dataset,
            feature_columns=feature_columns,
            target_columns=target_columns,
            horizons=self.config.horizons,
            start=dataset.index.min(),
            end=dataset.index.max(),
            rows=len(dataset),
            warnings=warnings,
            metadata={
                "symbol": self.config.symbol,
                "timeframe": self.config.timeframe,
                "horizons": list(
                    self.config.horizons
                ),
                "feature_count": len(
                    feature_columns
                ),
                "target_count": len(
                    target_columns
                ),
            },
        )

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def _load_data(self) -> pd.DataFrame:
        """
        Load historical OHLCV data.
        """

        try:
            result = self.data_builder(
                self.config.symbol,
                timeframe=self.config.timeframe,
            )
        except TypeError:
            result = self.data_builder(
                self.config.symbol
            )

        dataframe = self._extract_dataframe(
            result
        )

        if dataframe is None:
            raise ValueError(
                "Data builder did not return a DataFrame."
            )

        return dataframe.copy()

    @staticmethod
    def _extract_dataframe(
        result: Any,
    ) -> pd.DataFrame | None:

        if isinstance(
            result,
            pd.DataFrame,
        ):
            return result

        for name in (
            "data",
            "dataset",
            "dataframe",
            "prepared_data",
            "daily",
        ):
            value = getattr(
                result,
                name,
                None,
            )

            if isinstance(
                value,
                pd.DataFrame,
            ):
                return value

        return None

    @staticmethod
    def _validate_raw_data(
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:
            raise ValueError(
                "Historical OHLCV dataset is empty."
            )

        required = {
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        }

        missing = (
            required
            - set(
                dataframe.columns
            )
        )

        if missing:
            raise ValueError(
                "Historical dataset is missing required columns: "
                f"{sorted(missing)}"
            )

        if not isinstance(
            dataframe.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "Historical dataset must use a DatetimeIndex."
            )

        if dataframe.index.has_duplicates:
            raise ValueError(
                "Historical dataset contains duplicate timestamps."
            )

        if not dataframe.index.is_monotonic_increasing:
            raise ValueError(
                "Historical dataset is not chronological."
            )

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def _build_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        output = self.feature_builder(
            dataframe.copy()
        )

        if isinstance(
            output,
            pd.DataFrame,
        ):
            return output.copy()

        # FeatureSet compatibility.
        feature_data = getattr(
            output,
            "data",
            None,
        )

        if isinstance(
            feature_data,
            pd.DataFrame,
        ):
            return feature_data.copy()

        raise TypeError(
            "Feature builder must return a DataFrame "
            "or FeatureSet containing a DataFrame."
        )

    @staticmethod
    def _validate_features(
        dataframe: pd.DataFrame,
    ) -> None:

        if dataframe.empty:
            raise ValueError(
                "Feature dataset is empty."
            )

        if dataframe.index.has_duplicates:
            raise ValueError(
                "Feature dataset contains duplicate timestamps."
            )

        if not dataframe.index.is_monotonic_increasing:
            raise ValueError(
                "Feature dataset is not chronological."
            )

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _build_targets(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        result = dataframe.copy()

        for horizon in self.config.horizons:

            target_spec = TargetSpec(
                horizon=int(
                    horizon
                )
            )

            target_frame = (
                self._call_target_builder(
                    result,
                    target_spec,
                )
            )

            if target_frame.empty:
                raise ValueError(
                    f"Target builder returned an empty frame "
                    f"for horizon {horizon}."
                )

            target_frame = (
                target_frame.copy()
            )

            overlapping = (
                set(
                    target_frame.columns
                )
                & set(
                    result.columns
                )
            )

            unexpected_overlap = (
                overlapping
                - {
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                }
            )

            if unexpected_overlap:
                target_frame = (
                    target_frame.drop(
                        columns=list(
                            unexpected_overlap
                        )
                    )
                )

            result = result.join(
                target_frame,
                how="left",
                rsuffix="_target",
            )

        return result

    def _call_target_builder(
        self,
        dataframe: pd.DataFrame,
        target_spec: TargetSpec,
    ) -> pd.DataFrame:

        try:
            output = self.target_builder(
                dataframe,
                target_spec,
            )

        except TypeError:

            try:
                output = self.target_builder(
                    dataframe,
                    horizon=target_spec.horizon,
                )

            except TypeError:
                output = self.target_builder(
                    dataframe,
                    target_spec.horizon,
                )

        if not isinstance(
            output,
            pd.DataFrame,
        ):
            raise TypeError(
                "Target builder must return a DataFrame."
            )

        return output

    # ------------------------------------------------------------------
    # Feature / target identification
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_features(
        dataframe: pd.DataFrame,
    ) -> list[str]:

        excluded_prefixes = (
            "Future_",
            "Direction_",
            "Target_",
        )

        excluded_names = {
            "Symbol",
            "_source_time",
            "_base_time",
        }

        features = []

        for column in dataframe.columns:

            name = str(
                column
            )

            if name in excluded_names:
                continue

            if any(
                name.startswith(
                    prefix
                )
                for prefix in excluded_prefixes
            ):
                continue

            features.append(
                name
            )

        return features

    @staticmethod
    def _identify_targets(
        dataframe: pd.DataFrame,
    ) -> list[str]:

        targets = []

        for column in dataframe.columns:

            name = str(
                column
            )

            if (
                name.startswith(
                    "Future_"
                )
                or name.startswith(
                    "Direction_"
                )
                or name.startswith(
                    "Target_"
                )
            ):
                targets.append(
                    name
                )

        return targets

    # ------------------------------------------------------------------
    # Leakage protection
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_feature_target_separation(
        feature_columns: list[str],
        target_columns: list[str],
    ) -> None:

        overlap = (
            set(
                feature_columns
            )
            & set(
                target_columns
            )
        )

        if overlap:
            raise ValueError(
                "Feature/target leakage detected. "
                f"Overlapping columns: {sorted(overlap)}"
            )

    @staticmethod
    def _validate_feature_names(
        feature_columns: list[str],
    ) -> None:

        suspicious = []

        for column in feature_columns:

            lower = str(
                column
            ).lower()

            suspicious_tokens = (
                "future",
                "target",
                "forward",
                "next_day",
                "nextday",
            )

            if any(
                token in lower
                for token in suspicious_tokens
            ):
                suspicious.append(
                    column
                )

        if suspicious:
            raise ValueError(
                "Potential forward-looking feature names detected: "
                f"{sorted(suspicious)}"
            )

    @staticmethod
    def _validate_feature_values(
        dataframe: pd.DataFrame,
        feature_columns: list[str],
    ) -> None:

        if not feature_columns:
            raise ValueError(
                "No feature columns were identified."
            )

        for column in feature_columns:

            series = dataframe[
                column
            ]

            if not pd.api.types.is_numeric_dtype(
                series
            ):
                raise TypeError(
                    f"Feature '{column}' is not numeric."
                )

            values = series.to_numpy(
                dtype=float
            )

            if np.isinf(
                values
            ).any():
                raise ValueError(
                    f"Feature '{column}' contains infinite values."
                )

    # ------------------------------------------------------------------
    # Index handling
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_and_deduplicate(
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:

        result = dataframe.copy()

        result = result.sort_index()

        result = result[
            ~result.index.duplicated(
                keep="last"
            )
        ]

        return result

    # ------------------------------------------------------------------
    # Warnings
    # ------------------------------------------------------------------

    @staticmethod
    def _build_warnings(
        dataframe: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str],
    ) -> list[str]:

        warnings: list[str] = []

        if len(
            dataframe
        ) < 500:

            warnings.append(
                "Dataset contains fewer than 500 observations; "
                "model estimates may be unstable."
            )

        if len(
            feature_columns
        ) > max(
            1,
            len(
                dataframe
            ) // 5,
        ):

            warnings.append(
                "Feature count is high relative to sample size; "
                "feature selection and regularization are important."
            )

        missing_feature_fraction = float(
            dataframe[
                feature_columns
            ]
            .isna()
            .mean()
            .mean()
        )

        if (
            missing_feature_fraction
            > 0.20
        ):

            warnings.append(
                "Feature matrix contains substantial missing data."
            )

        if not target_columns:

            warnings.append(
                "No targets are available."
            )

        return warnings


# ----------------------------------------------------------------------
# Convenience function
# ----------------------------------------------------------------------


def build_research_dataset(
    config: ResearchPipelineConfig,
    *,
    data_builder: Callable[..., Any] | None = None,
    feature_builder: Callable[
        [pd.DataFrame],
        Any,
    ]
    | None = None,
    target_builder: Callable[..., Any] | None = None,
) -> ResearchDatasetResult:
    """
    Convenience wrapper around ResearchDatasetBuilder.
    """

    builder = ResearchDatasetBuilder(
        config,
        data_builder=data_builder,
        feature_builder=feature_builder,
        target_builder=target_builder,
    )

    return builder.build()


__all__ = [
    "ResearchDatasetResult",
    "ResearchDatasetBuilder",
    "build_research_dataset",
]
