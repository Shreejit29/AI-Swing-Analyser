"""
AI Swing Analyser — Protected Final Holdout Pipeline.

Execution boundary:

    Model Selection
          ↓
    Frozen Development Model
          ↓
    Final Holdout Gate
          ↓
    Final Holdout Evaluation
          ↓
    Holdout Evidence

This module does NOT perform:
    - model selection
    - feature selection
    - hyperparameter tuning
    - calibration
    - threshold optimization
    - production approval

The final holdout remains an untouched evaluation set until this stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from .holdout_gate import HoldoutGateInput
from .holdout_stage import (
    HoldoutStageResult,
    run_final_holdout_stage,
)


@dataclass(frozen=True)
class HoldoutPipelineResult:
    """Result returned by the protected holdout pipeline."""

    model_id: str
    stage: HoldoutStageResult

    completed: bool
    production_approved: bool

    accuracy: float | None
    passed_accuracy_gate: bool

    errors: tuple[str, ...] = field(
        default_factory=tuple
    )

    warnings: tuple[str, ...] = field(
        default_factory=tuple
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def eligible(self) -> bool:
        """Whether the holdout evaluation was permitted."""
        return self.stage.eligible_for_evaluation

    @property
    def evaluated(self) -> bool:
        """Whether the final holdout was actually evaluated."""
        return self.stage.evaluated

    @property
    def final_holdout_used(self) -> bool:
        """Whether the holdout was consumed for final evaluation."""
        return self.stage.final_holdout_used

    def summary(self) -> dict[str, Any]:
        """Return a compact pipeline summary."""

        return {
            "model_id": self.model_id,
            "completed": self.completed,
            "eligible": self.eligible,
            "evaluated": self.evaluated,
            "final_holdout_used": self.final_holdout_used,
            "accuracy": self.accuracy,
            "passed_accuracy_gate": (
                self.passed_accuracy_gate
            ),
            "production_approved": (
                self.production_approved
            ),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class ProtectedHoldoutPipeline:
    """
    Controlled wrapper around the final holdout stage.

    This class intentionally has no training methods.
    """

    def __init__(
        self,
        accuracy_threshold: float = 0.95,
    ) -> None:
        if not 0.0 <= accuracy_threshold <= 1.0:
            raise ValueError(
                "accuracy_threshold must be between 0 and 1."
            )

        self.accuracy_threshold = (
            accuracy_threshold
        )

    @staticmethod
    def _validate_model_id(
        model_id: str,
    ) -> None:
        if not isinstance(model_id, str):
            raise TypeError(
                "model_id must be a string."
            )

        if not model_id.strip():
            raise ValueError(
                "model_id cannot be empty."
            )

    @staticmethod
    def _validate_gate(
        gate_input: HoldoutGateInput,
    ) -> None:
        if not isinstance(
            gate_input,
            HoldoutGateInput,
        ):
            raise TypeError(
                "gate_input must be a HoldoutGateInput."
            )

    @staticmethod
    def _validate_data(
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
    ) -> None:
        if not isinstance(
            development_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "development_data must be a pandas DataFrame."
            )

        if not isinstance(
            holdout_data,
            pd.DataFrame,
        ):
            raise TypeError(
                "holdout_data must be a pandas DataFrame."
            )

        if development_data.empty:
            raise ValueError(
                "development_data cannot be empty."
            )

        if holdout_data.empty:
            raise ValueError(
                "holdout_data cannot be empty."
            )

        if not isinstance(
            development_data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "development_data must use a DatetimeIndex."
            )

        if not isinstance(
            holdout_data.index,
            pd.DatetimeIndex,
        ):
            raise TypeError(
                "holdout_data must use a DatetimeIndex."
            )

    @staticmethod
    def _validate_model_identity(
        model_id: str,
        gate_input: HoldoutGateInput,
    ) -> None:
        if model_id != gate_input.model_id:
            raise ValueError(
                "model_id does not match the holdout gate."
            )

    @staticmethod
    def _build_metadata(
        model_id: str,
        stage_result: HoldoutStageResult,
    ) -> dict[str, Any]:
        metadata = {
            "stage": "protected_final_holdout",
            "model_id": model_id,
            "research_only": True,
            "production_approved": False,
            "holdout_used_for_selection": False,
            "holdout_used_for_feature_selection": False,
            "holdout_used_for_hyperparameter_tuning": False,
            "holdout_used_for_calibration": False,
            "holdout_used_for_threshold_optimization": False,
            "model_fitted_on_holdout": False,
            "preprocessor_fitted_on_holdout": False,
            "holdout_gate_status": (
                stage_result.gate.status.value
            ),
        }

        metadata.update(
            stage_result.metadata
        )

        return metadata

    def run(
        self,
        *,
        model_id: str,
        model: Any,
        preprocessor: Any,
        development_data: pd.DataFrame,
        holdout_data: pd.DataFrame,
        feature_columns: Sequence[str],
        target_column: str,
        gate_input: HoldoutGateInput,
    ) -> HoldoutPipelineResult:
        """
        Execute the protected final holdout evaluation.

        No fitting occurs in this method.
        """

        self._validate_model_id(
            model_id
        )

        self._validate_gate(
            gate_input
        )

        self._validate_model_identity(
            model_id,
            gate_input,
        )

        self._validate_data(
            development_data,
            holdout_data,
        )

        try:
            stage_result = (
                run_final_holdout_stage(
                    model_id=model_id,
                    model=model,
                    preprocessor=preprocessor,
                    development_data=development_data,
                    holdout_data=holdout_data,
                    feature_columns=feature_columns,
                    target_column=target_column,
                    gate_input=gate_input,
                    accuracy_threshold=(
                        self.accuracy_threshold
                    ),
                )
            )

        except Exception as exc:
            return HoldoutPipelineResult(
                model_id=model_id,
                stage=HoldoutStageResult(
                    model_id=model_id,
                    gate=(
                        # The stage should normally construct
                        # this itself. This branch only records
                        # unexpected execution failure.
                        gate_input_to_blocked_result(
                            gate_input
                        )
                    ),
                    evaluation=None,
                    successful=False,
                    final_holdout_used=False,
                    warnings=(
                        str(exc),
                    ),
                    metadata={
                        "stage": (
                            "protected_final_holdout"
                        ),
                        "research_only": True,
                        "production_approved": False,
                    },
                ),
                completed=False,
                production_approved=False,
                accuracy=None,
                passed_accuracy_gate=False,
                errors=(str(exc),),
                warnings=(),
                metadata={
                    "stage": (
                        "protected_final_holdout"
                    ),
                    "evaluation_started": False,
                    "final_holdout_used": False,
                    "research_only": True,
                    "production_approved": False,
                },
            )

        accuracy = (
            stage_result.holdout_accuracy
        )

        warnings = tuple(
            stage_result.warnings
        )

        errors: tuple[str, ...] = ()

        completed = (
            stage_result.successful
            and stage_result.evaluated
        )

        # This pipeline never grants production approval.
        production_approved = False

        metadata = self._build_metadata(
            model_id,
            stage_result,
        )

        return HoldoutPipelineResult(
            model_id=model_id,
            stage=stage_result,
            completed=completed,
            production_approved=(
                production_approved
            ),
            accuracy=accuracy,
            passed_accuracy_gate=(
                stage_result.passed_accuracy_gate
            ),
            errors=errors,
            warnings=warnings,
            metadata=metadata,
        )


def gate_input_to_blocked_result(
    gate_input: HoldoutGateInput,
):
    """
    Create a blocked gate result for unexpected pipeline failures.

    This helper keeps the failure path fail-closed.
    """

    from .holdout_gate import (
        FinalHoldoutGate,
    )

    gate = FinalHoldoutGate()

    try:
        result = gate.evaluate(
            gate_input
        )
    except Exception:
        # Import locally to avoid changing normal gate behavior.
        from .holdout_gate import (
            HoldoutGateResult,
            HoldoutGateStatus,
        )

        return HoldoutGateResult(
            status=HoldoutGateStatus.BLOCKED,
            eligible=False,
            safe_to_evaluate=False,
            failures=(
                "Unexpected failure occurred before final "
                "holdout evaluation.",
            ),
            warnings=(),
            metadata={
                "research_only": True,
                "final_holdout_protected": True,
            },
        )

    return result


def run_protected_holdout(
    *,
    model_id: str,
    model: Any,
    preprocessor: Any,
    development_data: pd.DataFrame,
    holdout_data: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    gate_input: HoldoutGateInput,
    accuracy_threshold: float = 0.95,
) -> HoldoutPipelineResult:
    """Convenience API for protected holdout evaluation."""

    pipeline = ProtectedHoldoutPipeline(
        accuracy_threshold=accuracy_threshold,
    )

    return pipeline.run(
        model_id=model_id,
        model=model,
        preprocessor=preprocessor,
        development_data=development_data,
        holdout_data=holdout_data,
        feature_columns=feature_columns,
        target_column=target_column,
        gate_input=gate_input,
    )


__all__ = [
    "HoldoutPipelineResult",
    "ProtectedHoldoutPipeline",
    "run_protected_holdout",
]
