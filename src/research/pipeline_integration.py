"""
AI Swing Analyser — Research Pipeline Integration.

Connects the independent research stages into a single auditable
workflow.

The integration layer is intentionally conservative:

    DATA
      ↓
    FEATURES
      ↓
    TARGETS
      ↓
    MODEL DEVELOPMENT
      ↓
    WALK-FORWARD VALIDATION
      ↓
    CALIBRATION
      ↓
    RANGE VALIDATION
      ↓
    REGIME VALIDATION
      ↓
    BACKTEST
      ↓
    ROBUSTNESS
      ↓
    FINAL HOLDOUT
      ↓
    APPROVAL

This module coordinates stages and records evidence.

It does not silently bypass failed stages.
It does not turn research metrics into a live signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.research.integration import (
    EvidenceStatus,
    ResearchEvidence,
    ResearchEvidenceBuilder,
)


# ---------------------------------------------------------------------
# Pipeline status
# ---------------------------------------------------------------------


class IntegrationStatus(str, Enum):
    """
    Overall state of the integrated research workflow.
    """

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------
# Stage record
# ---------------------------------------------------------------------


@dataclass
class IntegrationStage:
    """
    Records the outcome of one integrated research stage.
    """

    name: str

    status: EvidenceStatus = (
        EvidenceStatus.NOT_EVALUATED
    )

    started: bool = False

    completed: bool = False

    score: float | None = None

    metrics: dict[str, Any] = field(
        default_factory=dict
    )

    details: str = ""

    error: str | None = None


# ---------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------


@dataclass
class IntegratedResearchResult:
    """
    Complete state of the integrated research workflow.
    """

    model_id: str | None = None

    symbol: str | None = None

    timeframe: str | None = None

    horizon: int | None = None

    status: IntegrationStatus = (
        IntegrationStatus.CREATED
    )

    stages: list[IntegrationStage] = field(
        default_factory=list
    )

    evidence: ResearchEvidence | None = None

    production_eligible: bool = False

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    # -----------------------------------------------------------------
    # Stage helpers
    # -----------------------------------------------------------------

    def get_stage(
        self,
        name: str,
    ) -> IntegrationStage | None:

        for stage in self.stages:
            if stage.name == name:
                return stage

        return None

    def completed_stages(
        self,
    ) -> list[IntegrationStage]:

        return [
            stage
            for stage in self.stages
            if stage.completed
        ]

    def failed_stages(
        self,
    ) -> list[IntegrationStage]:

        return [
            stage
            for stage in self.stages
            if stage.status
            == EvidenceStatus.FAIL
        ]

    def pending_stages(
        self,
    ) -> list[IntegrationStage]:

        return [
            stage
            for stage in self.stages
            if not stage.completed
        ]

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(
        self,
    ) -> dict[str, Any]:

        return {
            "model_id": self.model_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "horizon": self.horizon,
            "status": self.status.value,
            "production_eligible": (
                self.production_eligible
            ),
            "completed_stages": [
                stage.name
                for stage in self.completed_stages()
            ],
            "failed_stages": [
                stage.name
                for stage in self.failed_stages()
            ],
            "pending_stages": [
                stage.name
                for stage in self.pending_stages()
            ],
            "errors": list(
                self.errors
            ),
            "warnings": list(
                self.warnings
            ),
            "metadata": dict(
                self.metadata
            ),
        }


# ---------------------------------------------------------------------
# Default stage definitions
# ---------------------------------------------------------------------


DEFAULT_INTEGRATION_STAGES = (
    "data",
    "features",
    "targets",
    "model_development",
    "walk_forward_validation",
    "calibration",
    "range_validation",
    "regime_validation",
    "backtest",
    "robustness",
    "final_holdout",
    "leakage_audit",
)


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------


class ResearchPipelineIntegrator:
    """
    Coordinates evidence from the independent research stages.

    The integrator deliberately separates:

        stage execution
        evidence recording
        production eligibility

    This prevents a successful early stage from accidentally
    being interpreted as a production-ready model.
    """

    def __init__(
        self,
        model_id: str | None = None,
        symbol: str | None = None,
        timeframe: str | None = None,
        horizon: int | None = None,
        stage_names: tuple[str, ...] | None = None,
    ) -> None:

        names = (
            stage_names
            if stage_names is not None
            else DEFAULT_INTEGRATION_STAGES
        )

        if not names:
            raise ValueError(
                "At least one integration stage is required."
            )

        if len(set(names)) != len(names):
            raise ValueError(
                "Integration stage names must be unique."
            )

        self.result = (
            IntegratedResearchResult(
                model_id=model_id,
                symbol=symbol,
                timeframe=timeframe,
                horizon=horizon,
                stages=[
                    IntegrationStage(
                        name=name
                    )
                    for name in names
                ],
                metadata={
                    "research_only": True,
                    "model_fitted": False,
                    "final_holdout_used": False,
                    "production_signal_generated": False,
                },
            )
        )

    # -----------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------

    def start(self) -> None:

        if (
            self.result.status
            not in (
                IntegrationStatus.CREATED,
                IntegrationStatus.RUNNING,
            )
        ):
            raise RuntimeError(
                "Pipeline cannot be started from "
                f"status {self.result.status.value}."
            )

        self.result.status = (
            IntegrationStatus.RUNNING
        )

    def complete(self) -> None:

        if self.result.status not in (
            IntegrationStatus.RUNNING,
            IntegrationStatus.COMPLETE,
        ):
            raise RuntimeError(
                "Pipeline must be running before "
                "it can be completed."
            )

        if self.result.errors:
            self.result.status = (
                IntegrationStatus.FAILED
            )
            self.result.production_eligible = False
            return

        failed = (
            self.result.failed_stages()
        )

        if failed:
            self.result.status = (
                IntegrationStatus.BLOCKED
            )
            self.result.production_eligible = False
            return

        self.result.status = (
            IntegrationStatus.COMPLETE
        )

        self._refresh_production_eligibility()

    # -----------------------------------------------------------------
    # Stage state
    # -----------------------------------------------------------------

    def start_stage(
        self,
        name: str,
    ) -> IntegrationStage:

        self.start()

        stage = self._require_stage(
            name
        )

        if stage.completed:
            raise RuntimeError(
                f"Stage '{name}' has already completed."
            )

        stage.started = True

        return stage

    def complete_stage(
        self,
        name: str,
        status: EvidenceStatus = (
            EvidenceStatus.PASS
        ),
        score: float | None = None,
        metrics: dict[str, Any] | None = None,
        details: str = "",
    ) -> IntegrationStage:

        stage = self._require_stage(
            name
        )

        if not stage.started:
            raise RuntimeError(
                f"Stage '{name}' must be started first."
            )

        if status == (
            EvidenceStatus.NOT_EVALUATED
        ):
            raise ValueError(
                "A completed stage cannot have "
                "NOT_EVALUATED status."
            )

        stage.status = status
        stage.completed = True
        stage.score = score
        stage.metrics = (
            dict(metrics)
            if metrics is not None
            else {}
        )
        stage.details = details

        if status == EvidenceStatus.FAIL:
            self.result.warnings.append(
                f"Stage failed: {name}"
            )

        self._refresh_production_eligibility()

        return stage

    def fail_stage(
        self,
        name: str,
        error: str,
    ) -> IntegrationStage:

        stage = self._require_stage(
            name
        )

        stage.started = True
        stage.completed = True
        stage.status = (
            EvidenceStatus.FAIL
        )
        stage.error = str(error)

        self.result.errors.append(
            f"{name}: {error}"
        )

        self.result.production_eligible = False

        return stage

    # -----------------------------------------------------------------
    # Evidence integration
    # -----------------------------------------------------------------

    def attach_evidence(
        self,
        evidence: ResearchEvidence,
    ) -> None:

        if not isinstance(
            evidence,
            ResearchEvidence,
        ):
            raise TypeError(
                "evidence must be a ResearchEvidence instance."
            )

        if (
            self.result.model_id is not None
            and evidence.model_id is not None
            and self.result.model_id
            != evidence.model_id
        ):
            raise ValueError(
                "Evidence model_id does not match "
                "the integration pipeline."
            )

        if (
            self.result.symbol is not None
            and evidence.symbol is not None
            and self.result.symbol
            != evidence.symbol
        ):
            raise ValueError(
                "Evidence symbol does not match "
                "the integration pipeline."
            )

        self.result.evidence = evidence

        self._refresh_production_eligibility()

    # -----------------------------------------------------------------
    # Build evidence from completed stages
    # -----------------------------------------------------------------

    def build_evidence(
        self,
    ) -> ResearchEvidence:

        builder = ResearchEvidenceBuilder(
            model_id=self.result.model_id,
            symbol=self.result.symbol,
            timeframe=self.result.timeframe,
            horizon=self.result.horizon,
        )

        stage_to_evidence = {
            "walk_forward_validation":
                "walk_forward_validation",
            "final_holdout":
                "final_holdout",
            "calibration":
                "probability_calibration",
            "range_validation":
                "target_range_validation",
            "regime_validation":
                "regime_validation",
            "backtest":
                "backtest",
            "robustness":
                "robustness",
            "leakage_audit":
                "leakage_audit",
        }

        for stage_name, evidence_name in (
            stage_to_evidence.items()
        ):

            stage = self.result.get_stage(
                stage_name
            )

            if stage is None:
                continue

            if not stage.completed:
                continue

            builder.set_evidence(
                name=evidence_name,
                status=stage.status,
                score=stage.score,
                metrics=stage.metrics,
                details=stage.details,
                critical=True,
                source=stage_name,
            )

        for warning in (
            self.result.warnings
        ):
            builder.add_warning(
                warning
            )

        for error in (
            self.result.errors
        ):
            builder.add_warning(
                error
            )

        evidence = builder.build()

        self.result.evidence = evidence

        self._refresh_production_eligibility()

        return evidence

    # -----------------------------------------------------------------
    # Safety
    # -----------------------------------------------------------------

    def can_proceed(
        self,
        stage_name: str,
    ) -> bool:

        stage = self._require_stage(
            stage_name
        )

        if self.result.errors:
            return False

        previous = self._previous_stages(
            stage_name
        )

        for previous_stage in previous:

            if not previous_stage.completed:
                return False

            if previous_stage.status == (
                EvidenceStatus.FAIL
            ):
                return False

        return True

    def assert_can_proceed(
        self,
        stage_name: str,
    ) -> None:

        if not self.can_proceed(
            stage_name
        ):
            stage = self._require_stage(
                stage_name
            )

            raise RuntimeError(
                "Cannot proceed to stage "
                f"'{stage.name}'. A previous stage "
                "is incomplete or has failed."
            )

    def _refresh_production_eligibility(
        self,
    ) -> None:

        evidence = self.result.evidence

        if evidence is None:
            self.result.production_eligible = (
                False
            )
            return

        self.result.production_eligible = (
            evidence.production_eligible()
            and not self.result.errors
        )

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _require_stage(
        self,
        name: str,
    ) -> IntegrationStage:

        stage = self.result.get_stage(
            name
        )

        if stage is None:
            raise ValueError(
                f"Unknown integration stage: {name}"
            )

        return stage

    def _previous_stages(
        self,
        name: str,
    ) -> list[IntegrationStage]:

        names = [
            stage.name
            for stage in self.result.stages
        ]

        index = names.index(
            name
        )

        return self.result.stages[
            :index
        ]


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def create_research_integrator(
    model_id: str | None = None,
    symbol: str | None = None,
    timeframe: str | None = None,
    horizon: int | None = None,
) -> ResearchPipelineIntegrator:

    return ResearchPipelineIntegrator(
        model_id=model_id,
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
    )


__all__ = [
    "IntegrationStatus",
    "IntegrationStage",
    "IntegratedResearchResult",
    "DEFAULT_INTEGRATION_STAGES",
    "ResearchPipelineIntegrator",
    "create_research_integrator",
]
