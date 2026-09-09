"""
Tests for the integrated research pipeline.

The tests verify:

- pipeline lifecycle
- stage ordering
- stage completion
- failure blocking
- evidence attachment
- evidence generation
- production eligibility
- model/symbol consistency
- fail-closed behavior
"""

from __future__ import annotations

import pytest

from src.research.integration import (
    EvidenceStatus,
    ResearchEvidence,
)
from src.research.pipeline_integration import (
    DEFAULT_INTEGRATION_STAGES,
    IntegratedResearchResult,
    IntegrationStage,
    IntegrationStatus,
    ResearchPipelineIntegrator,
    create_research_integrator,
)


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------


@pytest.fixture
def integrator():
    return ResearchPipelineIntegrator(
        model_id="MODEL_001",
        symbol="RELIANCE",
        timeframe="1D",
        horizon=5,
    )


@pytest.fixture
def complete_integrator(
    integrator,
):
    integrator.start()

    for stage_name in DEFAULT_INTEGRATION_STAGES:
        stage = integrator.start_stage(
            stage_name
        )

        integrator.complete_stage(
            stage_name,
            status=EvidenceStatus.PASS,
            score=1.0,
        )

    return integrator


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_integrator_can_be_created():
    result = ResearchPipelineIntegrator()

    assert result is not None


def test_integrator_identity(
    integrator,
):
    result = integrator.result

    assert (
        result.model_id
        == "MODEL_001"
    )

    assert (
        result.symbol
        == "RELIANCE"
    )

    assert (
        result.timeframe
        == "1D"
    )

    assert (
        result.horizon
        == 5
    )


def test_initial_status(
    integrator,
):
    assert (
        integrator.result.status
        == IntegrationStatus.CREATED
    )


def test_default_stages_are_present(
    integrator,
):
    names = [
        stage.name
        for stage in integrator.result.stages
    ]

    assert (
        tuple(names)
        == DEFAULT_INTEGRATION_STAGES
    )


def test_duplicate_custom_stages_fail():
    with pytest.raises(ValueError):
        ResearchPipelineIntegrator(
            stage_names=(
                "data",
                "data",
            )
        )


def test_empty_custom_stages_fail():
    with pytest.raises(ValueError):
        ResearchPipelineIntegrator(
            stage_names=()
        )


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------


def test_integration_stage_defaults():
    stage = IntegrationStage(
        name="test"
    )

    assert stage.name == "test"
    assert (
        stage.status
        == EvidenceStatus.NOT_EVALUATED
    )
    assert stage.started is False
    assert stage.completed is False
    assert stage.error is None


def test_integrated_result_defaults():
    result = IntegratedResearchResult()

    assert (
        result.status
        == IntegrationStatus.CREATED
    )

    assert result.evidence is None
    assert (
        result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------


def test_start_changes_status(
    integrator,
):
    integrator.start()

    assert (
        integrator.result.status
        == IntegrationStatus.RUNNING
    )


def test_start_is_idempotent_while_running(
    integrator,
):
    integrator.start()
    integrator.start()

    assert (
        integrator.result.status
        == IntegrationStatus.RUNNING
    )


def test_complete_without_start_fails(
    integrator,
):
    with pytest.raises(RuntimeError):
        integrator.complete()


def test_complete_empty_pipeline_blocks(
    integrator,
):
    integrator.start()
    integrator.complete()

    assert (
        integrator.result.status
        == IntegrationStatus.BLOCKED
    )

    assert (
        integrator.result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Stage lookup
# ---------------------------------------------------------------------


def test_get_stage(
    integrator,
):
    stage = integrator.result.get_stage(
        "backtest"
    )

    assert stage is not None
    assert stage.name == "backtest"


def test_unknown_stage_returns_none(
    integrator,
):
    assert (
        integrator.result.get_stage(
            "unknown"
        )
        is None
    )


def test_require_unknown_stage_fails(
    integrator,
):
    with pytest.raises(ValueError):
        integrator.start_stage(
            "unknown"
        )


# ---------------------------------------------------------------------
# Stage lifecycle
# ---------------------------------------------------------------------


def test_start_stage(
    integrator,
):
    stage = integrator.start_stage(
        "data"
    )

    assert (
        stage.started
        is True
    )

    assert (
        integrator.result.status
        == IntegrationStatus.RUNNING
    )


def test_complete_stage_requires_start(
    integrator,
):
    with pytest.raises(RuntimeError):
        integrator.complete_stage(
            "data"
        )


def test_complete_stage(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    stage = integrator.complete_stage(
        "data",
        status=EvidenceStatus.PASS,
        score=0.99,
        metrics={
            "rows": 1000
        },
        details="Data passed quality checks.",
    )

    assert (
        stage.completed
        is True
    )

    assert (
        stage.status
        == EvidenceStatus.PASS
    )

    assert (
        stage.score
        == 0.99
    )

    assert (
        stage.metrics[
            "rows"
        ]
        == 1000
    )


def test_completed_stage_cannot_restart(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    integrator.complete_stage(
        "data"
    )

    with pytest.raises(RuntimeError):
        integrator.start_stage(
            "data"
        )


def test_not_evaluated_cannot_complete(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    with pytest.raises(ValueError):
        integrator.complete_stage(
            "data",
            status=EvidenceStatus.NOT_EVALUATED,
        )


# ---------------------------------------------------------------------
# Stage failure
# ---------------------------------------------------------------------


def test_fail_stage(
    integrator,
):
    stage = integrator.fail_stage(
        "features",
        "Feature construction failed.",
    )

    assert (
        stage.completed
        is True
    )

    assert (
        stage.status
        == EvidenceStatus.FAIL
    )

    assert (
        stage.error
        == "Feature construction failed."
    )

    assert (
        integrator.result.production_eligible
        is False
    )


def test_failed_stage_creates_error(
    integrator,
):
    integrator.fail_stage(
        "features",
        "Bad feature.",
    )

    assert len(
        integrator.result.errors
    ) == 1


# ---------------------------------------------------------------------
# Stage ordering
# ---------------------------------------------------------------------


def test_cannot_proceed_before_previous_stage(
    integrator,
):
    assert (
        integrator.can_proceed(
            "features"
        )
        is False
    )


def test_can_proceed_after_previous_stage_passes(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    integrator.complete_stage(
        "data",
        EvidenceStatus.PASS,
    )

    assert (
        integrator.can_proceed(
            "features"
        )
        is True
    )


def test_failed_previous_stage_blocks_next_stage(
    integrator,
):
    integrator.fail_stage(
        "data",
        "Data failure.",
    )

    assert (
        integrator.can_proceed(
            "features"
        )
        is False
    )


def test_assert_can_proceed_raises_when_blocked(
    integrator,
):
    with pytest.raises(RuntimeError):
        integrator.assert_can_proceed(
            "features"
        )


def test_assert_can_proceed_after_success(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    integrator.complete_stage(
        "data",
        EvidenceStatus.PASS,
    )

    integrator.assert_can_proceed(
        "features"
    )


# ---------------------------------------------------------------------
# Completed / failed / pending stages
# ---------------------------------------------------------------------


def test_pending_stages_initially_all(
    integrator,
):
    assert len(
        integrator.result.pending_stages()
    ) == len(
        DEFAULT_INTEGRATION_STAGES
    )


def test_completed_stage_is_removed_from_pending(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    integrator.complete_stage(
        "data",
        EvidenceStatus.PASS,
    )

    completed = (
        integrator.result.completed_stages()
    )

    assert len(completed) == 1
    assert completed[0].name == "data"


def test_failed_stage_is_reported(
    integrator,
):
    integrator.fail_stage(
        "data",
        "failure",
    )

    failed = (
        integrator.result.failed_stages()
    )

    assert len(failed) == 1
    assert failed[0].name == "data"


# ---------------------------------------------------------------------
# Evidence attachment
# ---------------------------------------------------------------------


def test_attach_evidence(
    integrator,
):
    evidence = ResearchEvidence(
        model_id="MODEL_001",
        symbol="RELIANCE",
    )

    integrator.attach_evidence(
        evidence
    )

    assert (
        integrator.result.evidence
        is evidence
    )


def test_wrong_model_id_is_rejected(
    integrator,
):
    evidence = ResearchEvidence(
        model_id="OTHER_MODEL",
        symbol="RELIANCE",
    )

    with pytest.raises(ValueError):
        integrator.attach_evidence(
            evidence
        )


def test_wrong_symbol_is_rejected(
    integrator,
):
    evidence = ResearchEvidence(
        model_id="MODEL_001",
        symbol="TCS",
    )

    with pytest.raises(ValueError):
        integrator.attach_evidence(
            evidence
        )


def test_wrong_evidence_type_is_rejected(
    integrator,
):
    with pytest.raises(TypeError):
        integrator.attach_evidence(
            None
        )


# ---------------------------------------------------------------------
# Evidence building
# ---------------------------------------------------------------------


def test_build_evidence_from_completed_stages(
    integrator,
):
    integrator.start_stage(
        "walk_forward_validation"
    )

    integrator.complete_stage(
        "walk_forward_validation",
        status=EvidenceStatus.PASS,
        score=0.96,
        metrics={
            "accuracy": 0.96
        },
    )

    evidence = (
        integrator.build_evidence()
    )

    assert isinstance(
        evidence,
        ResearchEvidence,
    )

    assert (
        evidence.validation.status
        == EvidenceStatus.PASS
    )

    assert (
        evidence.validation.score
        == 0.96
    )


def test_uncompleted_stage_remains_missing(
    integrator,
):
    evidence = (
        integrator.build_evidence()
    )

    assert (
        evidence.validation.status
        == EvidenceStatus.NOT_EVALUATED
    )


def test_failed_stage_becomes_failed_evidence(
    integrator,
):
    integrator.fail_stage(
        "backtest",
        "Backtest failed.",
    )

    evidence = (
        integrator.build_evidence()
    )

    assert (
        evidence.backtest.status
        == EvidenceStatus.FAIL
    )


# ---------------------------------------------------------------------
# Complete pipeline
# ---------------------------------------------------------------------


def test_complete_pipeline_can_build_evidence(
    complete_integrator,
):
    evidence = (
        complete_integrator.build_evidence()
    )

    assert (
        len(
            evidence.evaluated_items()
        )
        == 8
    )


def test_complete_pipeline_evidence_is_ready(
    complete_integrator,
):
    evidence = (
        complete_integrator.build_evidence()
    )

    assert (
        evidence.production_eligible()
        is True
    )


def test_complete_pipeline_can_complete(
    complete_integrator,
):
    complete_integrator.complete()

    assert (
        complete_integrator.result.status
        == IntegrationStatus.COMPLETE
    )

    assert (
        complete_integrator.result.production_eligible
        is True
    )


# ---------------------------------------------------------------------
# Failed pipeline
# ---------------------------------------------------------------------


def test_failed_pipeline_cannot_be_production_ready(
    integrator,
):
    integrator.fail_stage(
        "backtest",
        "Backtest failed.",
    )

    integrator.build_evidence()

    integrator.complete()

    assert (
        integrator.result.production_eligible
        is False
    )


def test_pipeline_with_errors_becomes_failed(
    integrator,
):
    integrator.fail_stage(
        "data",
        "Data failure.",
    )

    integrator.complete()

    assert (
        integrator.result.status
        == IntegrationStatus.FAILED
    )

    assert (
        integrator.result.production_eligible
        is False
    )


# ---------------------------------------------------------------------
# Metadata safety
# ---------------------------------------------------------------------


def test_research_only_metadata(
    integrator,
):
    metadata = (
        integrator.result.metadata
    )

    assert (
        metadata["research_only"]
        is True
    )

    assert (
        metadata["model_fitted"]
        is False
    )

    assert (
        metadata["final_holdout_used"]
        is False
    )

    assert (
        metadata[
            "production_signal_generated"
        ]
        is False
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_pipeline_state(
    integrator,
):
    summary = (
        integrator.result.summary()
    )

    assert (
        summary["status"]
        == "CREATED"
    )

    assert (
        summary["production_eligible"]
        is False
    )

    assert (
        len(
            summary["pending_stages"]
        )
        == len(
            DEFAULT_INTEGRATION_STAGES
        )
    )


def test_summary_after_stage_completion(
    integrator,
):
    integrator.start_stage(
        "data"
    )

    integrator.complete_stage(
        "data",
        EvidenceStatus.PASS,
    )

    summary = (
        integrator.result.summary()
    )

    assert (
        "data"
        in summary[
            "completed_stages"
        ]
    )

    assert (
        "data"
        not in summary[
            "pending_stages"
        ]
    )


# ---------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------


def test_create_research_integrator():
    integrator = create_research_integrator(
        model_id="MODEL_X",
        symbol="INFY",
        timeframe="1D",
        horizon=3,
    )

    assert isinstance(
        integrator,
        ResearchPipelineIntegrator,
    )

    assert (
        integrator.result.model_id
        == "MODEL_X"
    )

    assert (
        integrator.result.symbol
        == "INFY"
    )


# ---------------------------------------------------------------------
# Fail-closed production safety
# ---------------------------------------------------------------------


def test_missing_holdout_prevents_production(
    integrator,
):
    for stage_name in DEFAULT_INTEGRATION_STAGES:
        integrator.start_stage(
            stage_name
        )

        integrator.complete_stage(
            stage_name,
            EvidenceStatus.PASS,
        )

    # Rebuild evidence after deliberately replacing
    # the holdout stage with an unevaluated state.
    holdout = integrator.result.get_stage(
        "final_holdout"
    )

    assert holdout is not None

    holdout.status = (
        EvidenceStatus.NOT_EVALUATED
    )
    holdout.completed = False

    evidence = (
        integrator.build_evidence()
    )

    assert (
        evidence.production_eligible()
        is False
    )


def test_failed_leakage_audit_prevents_production(
    complete_integrator,
):
    leakage = (
        complete_integrator.result.get_stage(
            "leakage_audit"
        )
    )

    assert leakage is not None

    leakage.status = (
        EvidenceStatus.FAIL
    )

    evidence = (
        complete_integrator.build_evidence()
    )

    assert (
        evidence.production_eligible()
        is False
    )
