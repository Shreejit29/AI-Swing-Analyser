"""
AI Swing Analyser — Research Evidence Collector.

Collects evidence produced by independent research stages and converts
it into the common ResearchEvidence representation.

This module intentionally does NOT approve models.

Approval remains fail-closed and is handled by the dedicated approval
layer after all required evidence has been collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
    ResearchEvidenceBuilder,
)
from .stage_adapters import (
    adapt_backtest_result,
    adapt_calibration_result,
    adapt_holdout_result,
    adapt_leakage_result,
    adapt_range_result,
    adapt_regime_result,
    adapt_robustness_result,
    adapt_walk_forward_result,
)


# ---------------------------------------------------------------------
# Stage names
# ---------------------------------------------------------------------


WALK_FORWARD = "walk_forward_validation"
HOLDOUT = "final_holdout"
CALIBRATION = "calibration"
RANGE_VALIDATION = "range_validation"
REGIME_VALIDATION = "regime_validation"
BACKTEST = "backtest"
ROBUSTNESS = "robustness"
LEAKAGE_AUDIT = "leakage_audit"


REQUIRED_EVIDENCE_STAGES = (
    WALK_FORWARD,
    HOLDOUT,
    CALIBRATION,
    RANGE_VALIDATION,
    REGIME_VALIDATION,
    BACKTEST,
    ROBUSTNESS,
    LEAKAGE_AUDIT,
)


# ---------------------------------------------------------------------
# Collector result
# ---------------------------------------------------------------------


@dataclass
class EvidenceCollectionResult:
    """
    Result of collecting evidence from research stages.
    """

    evidence: ResearchEvidence

    collected_stages: list[str] = field(
        default_factory=list
    )

    missing_stages: list[str] = field(
        default_factory=list
    )

    errors: list[str] = field(
        default_factory=list
    )

    warnings: list[str] = field(
        default_factory=list
    )

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def successful(self) -> bool:
        """
        Evidence collection itself succeeded.

        This does not mean the model passed the research gates.
        """

        return not self.errors

    @property
    def production_eligible(self) -> bool:
        """
        Whether the collected evidence is sufficient for production
        eligibility according to the fail-closed evidence framework.
        """

        return self.evidence.production_eligible


# ---------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------


class ResearchEvidenceCollector:
    """
    Collects evidence from completed research-stage results.

    The collector is deliberately conservative:

    - Missing evidence is not converted into PASS.
    - Ambiguous stage results are not converted into PASS.
    - Adapter failures are recorded as errors.
    - Final holdout contamination remains a failure.
    - Leakage failures remain critical.
    """

    def __init__(
        self,
        *,
        required_stages: tuple[str, ...] = (
            REQUIRED_EVIDENCE_STAGES
        ),
    ) -> None:

        self.required_stages = tuple(
            required_stages
        )

        unknown = set(
            self.required_stages
        ) - set(
            REQUIRED_EVIDENCE_STAGES
        )

        if unknown:
            raise ValueError(
                "Unknown evidence stage(s): "
                f"{sorted(unknown)}"
            )

    # -----------------------------------------------------------------
    # Individual adapters
    # -----------------------------------------------------------------

    @staticmethod
    def adapt_stage(
        stage_name: str,
        result: Any,
    ) -> EvidenceItem:

        if stage_name == WALK_FORWARD:
            return adapt_walk_forward_result(
                result
            )

        if stage_name == HOLDOUT:
            return adapt_holdout_result(
                result
            )

        if stage_name == CALIBRATION:
            return adapt_calibration_result(
                result
            )

        if stage_name == RANGE_VALIDATION:
            return adapt_range_result(
                result
            )

        if stage_name == REGIME_VALIDATION:
            return adapt_regime_result(
                result
            )

        if stage_name == BACKTEST:
            return adapt_backtest_result(
                result
            )

        if stage_name == ROBUSTNESS:
            return adapt_robustness_result(
                result
            )

        if stage_name == LEAKAGE_AUDIT:
            return adapt_leakage_result(
                result
            )

        raise ValueError(
            f"Unsupported evidence stage: {stage_name}"
        )

    # -----------------------------------------------------------------
    # Collection
    # -----------------------------------------------------------------

    def collect(
        self,
        results: Mapping[str, Any],
        *,
        model_id: str | None = None,
        experiment_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvidenceCollectionResult:

        builder = ResearchEvidenceBuilder()

        collected: list[str] = []
        missing: list[str] = []
        errors: list[str] = []
        warnings: list[str] = []

        for stage_name in self.required_stages:

            if stage_name not in results:
                missing.append(
                    stage_name
                )

                warnings.append(
                    f"Missing required evidence stage: "
                    f"{stage_name}"
                )

                continue

            result = results[stage_name]

            if result is None:
                missing.append(
                    stage_name
                )

                warnings.append(
                    f"Evidence result is None: "
                    f"{stage_name}"
                )

                continue

            try:
                evidence_item = self.adapt_stage(
                    stage_name,
                    result,
                )

                self._attach_item(
                    builder,
                    stage_name,
                    evidence_item,
                )

                collected.append(
                    stage_name
                )

                if (
                    evidence_item.status
                    == EvidenceStatus.WARNING
                ):
                    warnings.append(
                        f"Research stage produced a warning: "
                        f"{stage_name}"
                    )

                if (
                    evidence_item.status
                    == EvidenceStatus.NOT_EVALUATED
                ):
                    warnings.append(
                        f"Research stage was not fully evaluated: "
                        f"{stage_name}"
                    )

                if (
                    evidence_item.status
                    == EvidenceStatus.FAIL
                ):
                    warnings.append(
                        f"Research stage failed its evidence gate: "
                        f"{stage_name}"
                    )

            except Exception as exc:
                errors.append(
                    f"Failed to adapt {stage_name}: "
                    f"{type(exc).__name__}: {exc}"
                )

        evidence_metadata = dict(
            metadata or {}
        )

        evidence_metadata.update(
            {
                "required_stages": list(
                    self.required_stages
                ),
                "collected_stages": collected,
                "missing_stages": missing,
                "collection_errors": list(
                    errors
                ),
                "research_only": True,
            }
        )

        evidence = builder.build(
            model_id=model_id,
            experiment_id=experiment_id,
            metadata=evidence_metadata,
        )

        # Missing or adapter-error stages must never silently disappear.
        if missing:
            evidence.warnings.extend(
                [
                    f"Missing required evidence: {stage}"
                    for stage in missing
                ]
            )

        if errors:
            evidence.warnings.extend(
                [
                    f"Evidence collection error: {error}"
                    for error in errors
                ]
            )

        return EvidenceCollectionResult(
            evidence=evidence,
            collected_stages=collected,
            missing_stages=missing,
            errors=errors,
            warnings=warnings,
            metadata=evidence_metadata,
        )

    # -----------------------------------------------------------------
    # Builder attachment
    # -----------------------------------------------------------------

    @staticmethod
    def _attach_item(
        builder: ResearchEvidenceBuilder,
        stage_name: str,
        item: EvidenceItem,
    ) -> None:

        if stage_name == WALK_FORWARD:
            builder.set_validation(
                item
            )
            return

        if stage_name == HOLDOUT:
            builder.set_holdout(
                item
            )
            return

        if stage_name == CALIBRATION:
            builder.set_calibration(
                item
            )
            return

        if stage_name == RANGE_VALIDATION:
            builder.set_range_validation(
                item
            )
            return

        if stage_name == REGIME_VALIDATION:
            builder.set_regime(
                item
            )
            return

        if stage_name == BACKTEST:
            builder.set_backtest(
                item
            )
            return

        if stage_name == ROBUSTNESS:
            builder.set_robustness(
                item
            )
            return

        if stage_name == LEAKAGE_AUDIT:
            builder.set_leakage(
                item
            )
            return

        raise ValueError(
            f"Unsupported evidence stage: {stage_name}"
        )

    # -----------------------------------------------------------------
    # Convenience checks
    # -----------------------------------------------------------------

    def missing_required_stages(
        self,
        results: Mapping[str, Any],
    ) -> list[str]:

        return [
            stage
            for stage in self.required_stages
            if stage not in results
            or results[stage] is None
        ]

    def all_required_results_present(
        self,
        results: Mapping[str, Any],
    ) -> bool:

        return not self.missing_required_stages(
            results
        )

    def summarize(
        self,
        result: EvidenceCollectionResult,
    ) -> dict[str, Any]:

        evidence = result.evidence

        return {
            "successful_collection": (
                result.successful
            ),
            "production_eligible": (
                result.production_eligible
            ),
            "required_stage_count": len(
                self.required_stages
            ),
            "collected_stage_count": len(
                result.collected_stages
            ),
            "missing_stage_count": len(
                result.missing_stages
            ),
            "error_count": len(
                result.errors
            ),
            "warning_count": len(
                result.warnings
            ),
            "failed_evidence": [
                item.name
                for item in evidence.failed_items()
            ],
            "missing_evidence": [
                item.name
                for item in evidence.missing_items()
            ],
            "critical_failures": [
                item.name
                for item in evidence.critical_failures()
            ],
            "evidence_score": (
                evidence.evidence_score()
            ),
        }


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def collect_research_evidence(
    results: Mapping[str, Any],
    *,
    model_id: str | None = None,
    experiment_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> EvidenceCollectionResult:
    """
    Convenience wrapper around ResearchEvidenceCollector.
    """

    collector = ResearchEvidenceCollector()

    return collector.collect(
        results,
        model_id=model_id,
        experiment_id=experiment_id,
        metadata=metadata,
    )


__all__ = [
    "WALK_FORWARD",
    "HOLDOUT",
    "CALIBRATION",
    "RANGE_VALIDATION",
    "REGIME_VALIDATION",
    "BACKTEST",
    "ROBUSTNESS",
    "LEAKAGE_AUDIT",
    "REQUIRED_EVIDENCE_STAGES",
    "EvidenceCollectionResult",
    "ResearchEvidenceCollector",
    "collect_research_evidence",
]
