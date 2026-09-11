"""
AI Swing Analyser — Final Holdout Evidence Integration.

Adds the protected final-holdout result to the centralized
ResearchEvidence collection framework.

The collector remains fail-closed:
    missing / invalid holdout evidence
        -> NOT production eligible

A passing holdout result is evidence only.
It does not grant production approval.
"""

from __future__ import annotations

from typing import Any, Mapping

from .final_holdout_evidence import (
    adapt_final_holdout_result,
)
from .integration import (
    EvidenceItem,
    EvidenceStatus,
    ResearchEvidence,
)
from .pipeline_holdout import (
    HoldoutPipelineResult,
)


FINAL_HOLDOUT_STAGE = "final_holdout"


def collect_final_holdout_evidence(
    result: HoldoutPipelineResult | None,
    *,
    critical: bool = True,
) -> EvidenceItem:
    """
    Convert the final holdout pipeline result into an EvidenceItem.

    None is intentionally treated as NOT_EVALUATED rather than PASS.
    """

    if result is None:
        return EvidenceItem(
            name=FINAL_HOLDOUT_STAGE,
            status=EvidenceStatus.NOT_EVALUATED,
            score=None,
            threshold=0.95,
            metrics={},
            details=(
                "Final holdout evidence is missing. "
                "The final holdout cannot be considered evaluated."
            ),
            critical=critical,
            source="FinalHoldoutEvidenceCollector",
        )

    return adapt_final_holdout_result(
        result,
        critical=critical,
    )


def attach_final_holdout_to_evidence(
    evidence: ResearchEvidence,
    result: HoldoutPipelineResult | None,
    *,
    critical: bool = True,
) -> EvidenceItem:
    """
    Add final holdout evidence to ResearchEvidence.

    This function supports the common dictionary-based evidence
    structure used by the research evidence framework.
    """

    if not isinstance(
        evidence,
        ResearchEvidence,
    ):
        raise TypeError(
            "evidence must be a ResearchEvidence object."
        )

    item = collect_final_holdout_evidence(
        result,
        critical=critical,
    )

    # Preferred interface when available.
    setter = getattr(
        evidence,
        "set_evidence",
        None,
    )

    if callable(setter):
        setter(
            item.name,
            item,
        )
        return item

    # Support the existing category-style representation.
    for attribute_name in (
        "final_holdout",
        "holdout",
    ):
        container = getattr(
            evidence,
            attribute_name,
            None,
        )

        if isinstance(
            container,
            dict,
        ):
            container[item.name] = item
            return item

    # Last-resort dictionary-like evidence storage.
    items = getattr(
        evidence,
        "evidence",
        None,
    )

    if isinstance(
        items,
        dict,
    ):
        items[item.name] = item
        return item

    raise AttributeError(
        "ResearchEvidence does not expose a supported "
        "evidence insertion interface."
    )


def build_final_holdout_evidence(
    *,
    model_id: str,
    result: HoldoutPipelineResult | None,
    critical: bool = True,
) -> EvidenceItem:
    """
    Build final holdout evidence with model identity checks.

    The model identity is recorded in the evidence details/metrics
    but does not alter the evaluation itself.
    """

    item = collect_final_holdout_evidence(
        result,
        critical=critical,
    )

    metrics = dict(
        item.metrics
    )

    metrics["model_id"] = model_id

    details = item.details

    if model_id:
        details = (
            f"Model '{model_id}': "
            f"{details}"
        )

    return EvidenceItem(
        name=item.name,
        status=item.status,
        score=item.score,
        threshold=item.threshold,
        metrics=metrics,
        details=details,
        critical=item.critical,
        source=item.source,
    )


def final_holdout_passed(
    result: HoldoutPipelineResult | None,
) -> bool:
    """Return True only when final holdout evidence explicitly passes."""

    item = collect_final_holdout_evidence(
        result
    )

    return (
        item.status
        == EvidenceStatus.PASS
    )


def final_holdout_evaluated(
    result: HoldoutPipelineResult | None,
) -> bool:
    """Return True only when a completed evaluation exists."""

    if result is None:
        return False

    return bool(
        result.evaluated
        and result.completed
        and result.final_holdout_used
    )


def final_holdout_status(
    result: HoldoutPipelineResult | None,
) -> EvidenceStatus:
    """Return the evidence status for the final holdout."""

    return collect_final_holdout_evidence(
        result
    ).status


def final_holdout_summary(
    result: HoldoutPipelineResult | None,
) -> dict[str, Any]:
    """Return a dashboard-safe final holdout summary."""

    item = collect_final_holdout_evidence(
        result
    )

    return {
        "name": item.name,
        "status": item.status.value,
        "score": item.score,
        "threshold": item.threshold,
        "critical": item.critical,
        "metrics": dict(item.metrics),
        "details": item.details,
        "source": item.source,
        "evaluated": (
            final_holdout_evaluated(result)
        ),
        "passed": (
            item.status
            == EvidenceStatus.PASS
        ),
        "production_approved": False,
        "research_only": True,
    }


def validate_final_holdout_identity(
    result: HoldoutPipelineResult | None,
    expected_model_id: str,
) -> bool:
    """
    Verify that the holdout result belongs to the expected model.

    This prevents evidence from one model being accidentally attached
    to another model's approval record.
    """

    if result is None:
        return False

    if not expected_model_id:
        return False

    return (
        result.model_id
        == expected_model_id
    )


def collect_holdout_evidence_map(
    results: Mapping[
        str,
        HoldoutPipelineResult | None,
    ],
    *,
    critical: bool = True,
) -> dict[str, EvidenceItem]:
    """
    Convert multiple model holdout results into an evidence map.

    This is useful when several horizons/models are evaluated.
    """

    collected: dict[
        str,
        EvidenceItem,
    ] = {}

    for model_id, result in results.items():
        if not model_id:
            continue

        if result is not None:
            if not validate_final_holdout_identity(
                result,
                model_id,
            ):
                collected[model_id] = (
                    EvidenceItem(
                        name=FINAL_HOLDOUT_STAGE,
                        status=EvidenceStatus.FAIL,
                        score=None,
                        threshold=0.95,
                        metrics={
                            "model_id": model_id,
                        },
                        details=(
                            "Final holdout evidence belongs to a "
                            "different model identity."
                        ),
                        critical=critical,
                        source=(
                            "FinalHoldoutEvidenceCollector"
                        ),
                    )
                )
                continue

        collected[model_id] = (
            build_final_holdout_evidence(
                model_id=model_id,
                result=result,
                critical=critical,
            )
        )

    return collected


__all__ = [
    "FINAL_HOLDOUT_STAGE",
    "collect_final_holdout_evidence",
    "attach_final_holdout_to_evidence",
    "build_final_holdout_evidence",
    "final_holdout_passed",
    "final_holdout_evaluated",
    "final_holdout_status",
    "final_holdout_summary",
    "validate_final_holdout_identity",
    "collect_holdout_evidence_map",
]
