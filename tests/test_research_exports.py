"""
Tests for public research-package exports.
"""

from __future__ import annotations

import src.research as research


def test_holdout_gate_exports():
    assert hasattr(
        research,
        "FinalHoldoutGate",
    )

    assert hasattr(
        research,
        "HoldoutGateInput",
    )

    assert hasattr(
        research,
        "HoldoutGateResult",
    )

    assert hasattr(
        research,
        "HoldoutGateStatus",
    )

    assert hasattr(
        research,
        "create_holdout_gate",
    )

    assert hasattr(
        research,
        "evaluate_holdout_gate",
    )


def test_holdout_stage_exports():
    assert hasattr(
        research,
        "FinalHoldoutStage",
    )

    assert hasattr(
        research,
        "HoldoutStageResult",
    )

    assert hasattr(
        research,
        "run_final_holdout_stage",
    )


def test_protected_holdout_pipeline_exports():
    assert hasattr(
        research,
        "HoldoutPipelineResult",
    )

    assert hasattr(
        research,
        "ProtectedHoldoutPipeline",
    )

    assert hasattr(
        research,
        "run_protected_holdout",
    )


def test_holdout_gate_imports_are_callable():
    assert callable(
        research.FinalHoldoutGate
    )

    assert callable(
        research.create_holdout_gate
    )

    assert callable(
        research.evaluate_holdout_gate
    )


def test_holdout_stage_imports_are_callable():
    assert callable(
        research.FinalHoldoutStage
    )

    assert callable(
        research.run_final_holdout_stage
    )


def test_pipeline_imports_are_callable():
    assert callable(
        research.ProtectedHoldoutPipeline
    )

    assert callable(
        research.run_protected_holdout
    )


def test_public_all_contains_holdout_components():
    exported = set(
        research.__all__
    )

    expected = {
        "FinalHoldoutGate",
        "HoldoutGateInput",
        "HoldoutGateResult",
        "HoldoutGateStatus",
        "create_holdout_gate",
        "evaluate_holdout_gate",
        "FinalHoldoutStage",
        "HoldoutStageResult",
        "run_final_holdout_stage",
        "HoldoutPipelineResult",
        "ProtectedHoldoutPipeline",
        "run_protected_holdout",
    }

    assert expected.issubset(
        exported
    )


def test_holdout_gate_objects_can_be_created():
    gate = research.FinalHoldoutGate()

    evidence = research.create_holdout_gate(
        model_id="test_model",
        selection_completed=True,
        walk_forward_completed=True,
        feature_selection_frozen=True,
        hyperparameters_frozen=True,
        preprocessing_frozen=True,
    )

    result = gate.evaluate(
        evidence
    )

    assert result.eligible is True
    assert result.safe_to_evaluate is True


def test_pipeline_can_be_constructed():
    pipeline = (
        research.ProtectedHoldoutPipeline()
    )

    assert (
        pipeline.accuracy_threshold
        == 0.95
    )


def test_legacy_research_exports_remain_available():
    expected = {
        "ResearchPipeline",
        "ResearchPipelineResult",
        "ResearchStage",
        "ResearchPipelineConfig",
        "ResearchValidationConfig",
    }

    exported = set(
        research.__all__
    )

    assert expected.issubset(
        exported
    )


def test_integrated_configuration_exports_remain_available():
    expected = {
        "IntegratedResearchConfig",
        "IntegratedValidationConfig",
        "IntegratedHoldoutConfig",
        "IntegratedApprovalConfig",
        "default_integrated_research_config",
    }

    exported = set(
        research.__all__
    )

    assert expected.issubset(
        exported
    )


def test_model_selection_exports_remain_available():
    expected = {
        "ControlledModelSelector",
        "ModelCandidateScore",
        "ModelSelectionConfig",
        "ModelSelectionResult",
        "select_best_model",
    }

    exported = set(
        research.__all__
    )

    assert expected.issubset(
        exported
    )


def test_model_card_exports_remain_available():
    expected = {
        "ModelCard",
        "ModelCardBuilder",
        "model_card_from_selection",
        "save_model_card",
        "load_model_card",
        "ModelCardRegistry",
        "register_model_card",
    }

    exported = set(
        research.__all__
    )

    assert expected.issubset(
        exported
    )


def test_selection_stage_exports_remain_available():
    expected = {
        "ModelSelectionStage",
        "ModelSelectionStageResult",
        "run_model_selection_stage",
    }

    exported = set(
        research.__all__
    )

    assert expected.issubset(
        exported
    )
