"""
Tests for the integrated research evidence collector.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.research.evidence_collector import (
    BACKTEST,
    CALIBRATION,
    HOLDOUT,
    LEAKAGE_AUDIT,
    RANGE_VALIDATION,
    REGIME_VALIDATION,
    REQUIRED_EVIDENCE_STAGES,
    ROBUSTNESS,
    WALK_FORWARD,
    ResearchEvidenceCollector,
    collect_research_evidence,
)
from src.research.integration import (
    EvidenceStatus,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def passing_result(**kwargs):
    """
    Create a permissive synthetic result object.

    The adapters intentionally use dynamic attributes, so these
    lightweight objects are sufficient for collector-level tests.
    """

    defaults = {
        "passed": True,
        "production_ready": False,
        "final_holdout_used": False,
        "leakage_free": True,
        "research_only": True,
        "accuracy": 0.96,
        "mean_accuracy": 0.96,
        "median_accuracy": 0.96,
        "minimum_accuracy": 0.95,
        "accuracy_std": 0.02,
        "brier_score": 0.10,
        "ece": 0.05,
        "coverage": 0.80,
        "range_coverage": 0.80,
        "stability_score": 0.80,
        "profit_factor": 1.50,
        "sharpe": 1.00,
        "max_drawdown": -0.15,
        "total_trades": 50,
        "robustness_score": 0.80,
    }

    defaults.update(kwargs)

    return SimpleNamespace(**defaults)


def complete_results():
    """
    Return synthetic results for every mandatory evidence stage.
    """

    return {
        WALK_FORWARD: passing_result(),
        HOLDOUT: passing_result(),
        CALIBRATION: passing_result(),
        RANGE_VALIDATION: passing_result(),
        REGIME_VALIDATION: passing_result(),
        BACKTEST: passing_result(),
        ROBUSTNESS: passing_result(),
        LEAKAGE_AUDIT: passing_result(),
    }


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------


def test_default_required_stages():

    collector = ResearchEvidenceCollector()

    assert collector.required_stages == (
        REQUIRED_EVIDENCE_STAGES
    )


def test_custom_required_stages_are_supported():

    collector = ResearchEvidenceCollector(
        required_stages=(
            WALK_FORWARD,
            HOLDOUT,
        )
    )

    assert collector.required_stages == (
        WALK_FORWARD,
        HOLDOUT,
    )


def test_unknown_stage_is_rejected():

    with pytest.raises(ValueError):

        ResearchEvidenceCollector(
            required_stages=(
                WALK_FORWARD,
                "unknown_stage",
            )
        )


# ---------------------------------------------------------------------
# Stage adaptation
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "stage_name",
    [
        WALK_FORWARD,
        HOLDOUT,
        CALIBRATION,
        RANGE_VALIDATION,
        REGIME_VALIDATION,
        BACKTEST,
        ROBUSTNESS,
        LEAKAGE_AUDIT,
    ],
)
def test_supported_stage_is_adapted(
    stage_name,
):

    collector = ResearchEvidenceCollector()

    result = passing_result()

    item = collector.adapt_stage(
        stage_name,
        result,
    )

    assert item is not None
    assert item.name


def test_unknown_stage_adapter_is_rejected():

    collector = ResearchEvidenceCollector()

    with pytest.raises(ValueError):

        collector.adapt_stage(
            "unknown_stage",
            passing_result(),
        )


# ---------------------------------------------------------------------
# Missing evidence
# ---------------------------------------------------------------------


def test_missing_stage_is_not_treated_as_pass():

    results = complete_results()

    del results[BACKTEST]

    collector = ResearchEvidenceCollector()

    collected = collector.collect(
        results
    )

    assert BACKTEST in collected.missing_stages

    assert (
        BACKTEST
        not in collected.collected_stages
    )

    assert not collected.production_eligible


def test_none_stage_is_missing():

    results = complete_results()

    results[CALIBRATION] = None

    collector = ResearchEvidenceCollector()

    collected = collector.collect(
        results
    )

    assert (
        CALIBRATION
        in collected.missing_stages
    )

    assert not collected.production_eligible


def test_empty_results_are_not_production_eligible():

    collector = ResearchEvidenceCollector()

    result = collector.collect({})

    assert not result.production_eligible

    assert (
        len(result.missing_stages)
        == len(
            REQUIRED_EVIDENCE_STAGES
        )
    )


def test_missing_required_stages_helper():

    results = complete_results()

    del results[ROBUSTNESS]
    del results[LEAKAGE_AUDIT]

    collector = ResearchEvidenceCollector()

    missing = collector.missing_required_stages(
        results
    )

    assert ROBUSTNESS in missing
    assert LEAKAGE_AUDIT in missing


def test_all_required_results_present():

    collector = ResearchEvidenceCollector()

    assert collector.all_required_results_present(
        complete_results()
    )


def test_all_required_results_present_returns_false_when_missing():

    results = complete_results()

    del results[HOLDOUT]

    collector = ResearchEvidenceCollector()

    assert not collector.all_required_results_present(
        results
    )


# ---------------------------------------------------------------------
# Complete collection
# ---------------------------------------------------------------------


def test_complete_collection_records_all_stages():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results()
    )

    assert set(
        result.collected_stages
    ) == set(
        REQUIRED_EVIDENCE_STAGES
    )

    assert not result.missing_stages


def test_complete_collection_has_no_adapter_errors():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results()
    )

    assert not result.errors


def test_collection_result_reports_success():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results()
    )

    assert result.successful


# ---------------------------------------------------------------------
# Fail-closed behavior
# ---------------------------------------------------------------------


def test_failed_stage_remains_failed():

    results = complete_results()

    results[BACKTEST] = passing_result(
        passed=False,
        profit_factor=0.80,
        sharpe=0.10,
    )

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        results
    )

    assert not result.production_eligible

    assert BACKTEST in [
        item.name
        for item in result.evidence.failed_items()
    ]


def test_failed_leakage_audit_is_not_ignored():

    results = complete_results()

    results[LEAKAGE_AUDIT] = passing_result(
        passed=False,
        leakage_free=False,
    )

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        results
    )

    assert not result.production_eligible

    assert result.evidence.critical_failures()


def test_holdout_contamination_cannot_pass():

    results = complete_results()

    results[HOLDOUT] = passing_result(
        passed=True,
        final_holdout_used=True,
    )

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        results
    )

    assert not result.production_eligible


def test_ambiguous_stage_does_not_become_pass():

    results = complete_results()

    results[CALIBRATION] = SimpleNamespace()

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        results
    )

    calibration = [
        item
        for item in result.evidence.items()
        if item.name == CALIBRATION
    ]

    if calibration:
        assert calibration[0].status != (
            EvidenceStatus.PASS
        )

    assert not result.production_eligible


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------


def test_model_and_experiment_identity_are_preserved():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results(),
        model_id="MODEL-001",
        experiment_id="EXP-001",
    )

    assert (
        result.evidence.model_id
        == "MODEL-001"
    )

    assert (
        result.evidence.experiment_id
        == "EXP-001"
    )


def test_custom_metadata_is_preserved():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results(),
        metadata={
            "symbol": "RESEARCH",
            "timeframe": "1D",
            "horizon": 5,
        },
    )

    assert (
        result.evidence.metadata["symbol"]
        == "RESEARCH"
    )

    assert (
        result.evidence.metadata["timeframe"]
        == "1D"
    )

    assert (
        result.evidence.metadata["horizon"]
        == 5
    )


def test_collection_metadata_records_stage_counts():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results()
    )

    assert (
        result.metadata[
            "required_stages"
        ]
        == list(
            REQUIRED_EVIDENCE_STAGES
        )
    )

    assert (
        len(
            result.metadata[
                "collected_stages"
            ]
        )
        == len(
            REQUIRED_EVIDENCE_STAGES
        )
    )

    assert (
        result.metadata[
            "research_only"
        ]
        is True
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------


def test_summary_contains_collection_status():

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        complete_results()
    )

    summary = collector.summarize(
        result
    )

    assert (
        "successful_collection"
        in summary
    )

    assert (
        "production_eligible"
        in summary
    )

    assert (
        "collected_stage_count"
        in summary
    )

    assert (
        "missing_stage_count"
        in summary
    )

    assert (
        "error_count"
        in summary
    )


def test_summary_reports_missing_evidence():

    results = complete_results()

    del results[HOLDOUT]

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        results
    )

    summary = collector.summarize(
        result
    )

    assert (
        summary["missing_stage_count"]
        == 1
    )

    assert (
        HOLDOUT
        in summary["missing_evidence"]
    )


def test_summary_reports_failed_evidence():

    results = complete_results()

    results[BACKTEST] = passing_result(
        passed=False,
        profit_factor=0.50,
    )

    collector = ResearchEvidenceCollector()

    result = collector.collect(
        results
    )

    summary = collector.summarize(
        result
    )

    assert BACKTEST in (
        summary["failed_evidence"]
    )


# ---------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------


def test_convenience_function_returns_collection_result():

    result = collect_research_evidence(
        complete_results()
    )

    assert result.successful
    assert result.evidence is not None


def test_convenience_function_preserves_identity():

    result = collect_research_evidence(
        complete_results(),
        model_id="MODEL-X",
        experiment_id="EXP-X",
    )

    assert (
        result.evidence.model_id
        == "MODEL-X"
    )

    assert (
        result.evidence.experiment_id
        == "EXP-X"
    )


# ---------------------------------------------------------------------
# Immutability of source results
# ---------------------------------------------------------------------


def test_collection_does_not_mutate_source_mapping():

    results = complete_results()

    original_keys = set(
        results.keys()
    )

    collector = ResearchEvidenceCollector()

    collector.collect(
        results
    )

    assert set(
        results.keys()
    ) == original_keys


def test_collection_is_deterministic():

    results = complete_results()

    collector = ResearchEvidenceCollector()

    first = collector.collect(
        results
    )

    second = collector.collect(
        results
    )

    assert (
        collector.summarize(first)
        == collector.summarize(second)
    )
