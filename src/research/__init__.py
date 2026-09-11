"""
AI Swing Analyser — Research Package.

Public exports for the research and model-development layers.
"""

# ---------------------------------------------------------------------
# Core research configuration
# ---------------------------------------------------------------------

from .config import (
    ResearchValidationConfig,
    ResearchFeatureConfig,
    ResearchHyperparameterConfig,
    ResearchCalibrationConfig,
    ResearchRangeConfig,
    ResearchBacktestConfig,
    ResearchRobustnessConfig,
    ResearchApprovalConfig,
    ResearchPipelineConfig,
)

from .research_config import (
    IntegratedValidationConfig,
    IntegratedCalibrationConfig,
    IntegratedRangeConfig,
    IntegratedRegimeConfig,
    IntegratedBacktestConfig,
    IntegratedRobustnessConfig,
    IntegratedHoldoutConfig,
    IntegratedApprovalConfig,
    IntegratedTradingConfig,
    IntegratedResearchConfig,
    default_integrated_research_config,
)

# ---------------------------------------------------------------------
# Main research pipeline
# ---------------------------------------------------------------------

from .pipeline import (
    ResearchPipeline,
    ResearchPipelineResult,
    ResearchStage,
)

# ---------------------------------------------------------------------
# Dataset and leakage control
# ---------------------------------------------------------------------

from .dataset_builder import (
    ResearchDatasetResult,
    ResearchDatasetBuilder,
    build_research_dataset,
)

from .leakage_audit import (
    LeakageFinding,
    LeakageAuditReport,
    LeakageAuditConfig,
    LeakageAuditor,
    audit_research_dataset,
    assert_leakage_free,
    compare_future_mutation,
)

# ---------------------------------------------------------------------
# Temporal research
# ---------------------------------------------------------------------

from .temporal_split import (
    TemporalSplit,
    HoldoutSplit,
    TemporalSplitter,
    create_holdout_split,
    create_walk_forward_splits,
)

from .walk_forward_research import (
    WalkForwardFoldResult,
    WalkForwardResearchResult,
    WalkForwardResearchEngine,
    run_walk_forward_research,
)

# ---------------------------------------------------------------------
# Model development and selection
# ---------------------------------------------------------------------

from .model_development import (
    DevelopmentPartitions,
    ModelDevelopmentResult,
    ModelDevelopmentOrchestrator,
)

from .experiment_runner import (
    ExperimentCandidate,
    ExperimentRunResult,
    ResearchExperimentRunner,
)

from .model_selection import (
    ModelCandidateScore,
    ModelSelectionResult,
    ModelSelectionConfig,
    ControlledModelSelector,
    select_best_model,
)

from .selection_pipeline import (
    SelectionCandidate,
    SelectionPipelineResult,
    SelectionPipeline,
    run_model_selection,
)

from .selection_registry import (
    SelectionRecord,
    SelectionRegistry,
    register_selection,
)

from .selection_stage import (
    ModelSelectionStageResult,
    ModelSelectionStage,
    run_model_selection_stage,
)

# ---------------------------------------------------------------------
# Model documentation
# ---------------------------------------------------------------------

from .model_card import (
    ModelCard,
    ModelCardBuilder,
    model_card_from_selection,
    save_model_card,
    load_model_card,
)

from .model_card_registry import (
    ModelCardRegistry,
    register_model_card,
)

# ---------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------

from .calibration_research import (
    CalibrationResearchResult,
    CalibrationResearchEngine,
    calibrate_research_probabilities,
)

# ---------------------------------------------------------------------
# Range research
# ---------------------------------------------------------------------

from .range_research import (
    RangeResearchResult,
    RangeResearchEngine,
    run_range_research,
)

# ---------------------------------------------------------------------
# Regime research
# ---------------------------------------------------------------------

from .regime_research import (
    RegimeResearchResult,
    RegimeResearchEngine,
    run_regime_research,
)

# ---------------------------------------------------------------------
# Backtesting and robustness
# ---------------------------------------------------------------------

from .backtest_research import (
    BacktestResearchResult,
    BacktestResearchEngine,
    calculate_return_distribution,
    stress_trade_returns,
)

from .robustness_research import (
    MonteCarloResult,
    StressResult,
    TradeRemovalResult,
    RobustnessResearchResult,
    RobustnessResearchEngine,
    run_robustness_research,
)

# ---------------------------------------------------------------------
# Research evidence and approval
# ---------------------------------------------------------------------

from .integration import (
    EvidenceStatus,
    EvidenceItem,
    ResearchEvidence,
    ResearchEvidenceBuilder,
    create_research_evidence,
    assert_research_ready,
)

from .stage_adapters import (
    adapt_walk_forward_result,
    adapt_holdout_result,
    adapt_calibration_result,
    adapt_range_result,
    adapt_regime_result,
    adapt_backtest_result,
    adapt_robustness_result,
    adapt_leakage_result,
    adapt_stage_result,
)

from .approval_bridge import (
    ApprovalBridgeResult,
    ResearchApprovalBridge,
    evaluate_research_approval,
    assert_research_approved,
)

# ---------------------------------------------------------------------
# Final holdout protection
# ---------------------------------------------------------------------

from .holdout_gate import (
    HoldoutGateStatus,
    HoldoutGateInput,
    HoldoutGateResult,
    FinalHoldoutGate,
    create_holdout_gate,
    evaluate_holdout_gate,
)

from .holdout_stage import (
    HoldoutStageResult,
    FinalHoldoutStage,
    run_final_holdout_stage,
)

from .pipeline_holdout import (
    HoldoutPipelineResult,
    ProtectedHoldoutPipeline,
    run_protected_holdout,
)

from .final_holdout_evidence import (
    FINAL_HOLDOUT_EVIDENCE_NAME,
    adapt_final_holdout_result,
    attach_final_holdout_evidence,
)

# ---------------------------------------------------------------------
# Integrated pipeline
# ---------------------------------------------------------------------

from .pipeline_integration import (
    IntegrationStatus,
    IntegrationStage,
    IntegratedResearchResult,
    ResearchPipelineIntegrator,
    create_research_integrator,
    integrate_final_holdout,
)

# ---------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------

from .evidence_collector import (
    collect_final_holdout_evidence,
    attach_final_holdout_to_evidence,
    build_final_holdout_evidence,
    final_holdout_passed,
    final_holdout_evaluated,
    final_holdout_status,
    final_holdout_summary,
    validate_final_holdout_identity,
    collect_holdout_evidence_map,
)

# ---------------------------------------------------------------------
# Production prediction
# ---------------------------------------------------------------------

from .production_prediction import (
    ProductionPredictionStatus,
    ProductionPredictionConfig,
    ProductionPrediction,
    ProductionPredictionGateway,
    production_prediction_summary,
)

from .multi_horizon_prediction import (
    MultiHorizonConfig,
    MultiHorizonPrediction,
    MultiHorizonPredictionEngine,
    build_multi_horizon_engine,
    multi_horizon_prediction_summary,
)

# ---------------------------------------------------------------------
# Indian market context
# ---------------------------------------------------------------------

from .market_context import (
    MarketContextConfig,
    MarketContextResult,
    build_benchmark_features,
    build_market_context,
    add_relative_strength,
    market_strength_score,
)

from .market_data_context import (
    MarketContextRequest,
    MarketContextData,
    MarketContextDataLoader,
    load_market_context,
    build_stock_market_context,
)

from .market_integration import (
    MarketIntegrationConfig,
    MarketIntegrationResult,
    integrate_market_context,
    add_market_context_to_stock,
    market_integration_summary,
)

# ---------------------------------------------------------------------
# Unified feature integration
# ---------------------------------------------------------------------

from .feature_integration import (
    UnifiedFeatureConfig,
    UnifiedFeatureResult,
    build_unified_features,
    feature_matrix,
    unified_feature_names,
    unified_feature_summary,
)


__all__ = [
    # Core configuration
    "ResearchValidationConfig",
    "ResearchFeatureConfig",
    "ResearchHyperparameterConfig",
    "ResearchCalibrationConfig",
    "ResearchRangeConfig",
    "ResearchBacktestConfig",
    "ResearchRobustnessConfig",
    "ResearchApprovalConfig",
    "ResearchPipelineConfig",

    # Integrated configuration
    "IntegratedValidationConfig",
    "IntegratedCalibrationConfig",
    "IntegratedRangeConfig",
    "IntegratedRegimeConfig",
    "IntegratedBacktestConfig",
    "IntegratedRobustnessConfig",
    "IntegratedHoldoutConfig",
    "IntegratedApprovalConfig",
    "IntegratedTradingConfig",
    "IntegratedResearchConfig",
    "default_integrated_research_config",

    # Pipeline
    "ResearchPipeline",
    "ResearchPipelineResult",
    "ResearchStage",

    # Dataset / leakage
    "ResearchDatasetResult",
    "ResearchDatasetBuilder",
    "build_research_dataset",
    "LeakageFinding",
    "LeakageAuditReport",
    "LeakageAuditConfig",
    "LeakageAuditor",
    "audit_research_dataset",
    "assert_leakage_free",
    "compare_future_mutation",

    # Temporal research
    "TemporalSplit",
    "HoldoutSplit",
    "TemporalSplitter",
    "create_holdout_split",
    "create_walk_forward_splits",
    "WalkForwardFoldResult",
    "WalkForwardResearchResult",
    "WalkForwardResearchEngine",
    "run_walk_forward_research",

    # Model development
    "DevelopmentPartitions",
    "ModelDevelopmentResult",
    "ModelDevelopmentOrchestrator",
    "ExperimentCandidate",
    "ExperimentRunResult",
    "ResearchExperimentRunner",

    # Model selection
    "ModelCandidateScore",
    "ModelSelectionResult",
    "ModelSelectionConfig",
    "ControlledModelSelector",
    "select_best_model",
    "SelectionCandidate",
    "SelectionPipelineResult",
    "SelectionPipeline",
    "run_model_selection",
    "SelectionRecord",
    "SelectionRegistry",
    "register_selection",
    "ModelSelectionStageResult",
    "ModelSelectionStage",
    "run_model_selection_stage",

    # Model cards
    "ModelCard",
    "ModelCardBuilder",
    "model_card_from_selection",
    "save_model_card",
    "load_model_card",
    "ModelCardRegistry",
    "register_model_card",

    # Research stages
    "CalibrationResearchResult",
    "CalibrationResearchEngine",
    "calibrate_research_probabilities",
    "RangeResearchResult",
    "RangeResearchEngine",
    "run_range_research",
    "RegimeResearchResult",
    "RegimeResearchEngine",
    "run_regime_research",

    # Backtesting / robustness
    "BacktestResearchResult",
    "BacktestResearchEngine",
    "calculate_return_distribution",
    "stress_trade_returns",
    "MonteCarloResult",
    "StressResult",
    "TradeRemovalResult",
    "RobustnessResearchResult",
    "RobustnessResearchEngine",
    "run_robustness_research",

    # Evidence
    "EvidenceStatus",
    "EvidenceItem",
    "ResearchEvidence",
    "ResearchEvidenceBuilder",
    "create_research_evidence",
    "assert_research_ready",

    # Stage adapters
    "adapt_walk_forward_result",
    "adapt_holdout_result",
    "adapt_calibration_result",
    "adapt_range_result",
    "adapt_regime_result",
    "adapt_backtest_result",
    "adapt_robustness_result",
    "adapt_leakage_result",
    "adapt_stage_result",

    # Approval
    "ApprovalBridgeResult",
    "ResearchApprovalBridge",
    "evaluate_research_approval",
    "assert_research_approved",

    # Holdout
    "HoldoutGateStatus",
    "HoldoutGateInput",
    "HoldoutGateResult",
    "FinalHoldoutGate",
    "create_holdout_gate",
    "evaluate_holdout_gate",
    "HoldoutStageResult",
    "FinalHoldoutStage",
    "run_final_holdout_stage",
    "HoldoutPipelineResult",
    "ProtectedHoldoutPipeline",
    "run_protected_holdout",
    "FINAL_HOLDOUT_EVIDENCE_NAME",
    "adapt_final_holdout_result",
    "attach_final_holdout_evidence",

    # Integrated pipeline
    "IntegrationStatus",
    "IntegrationStage",
    "IntegratedResearchResult",
    "ResearchPipelineIntegrator",
    "create_research_integrator",
    "integrate_final_holdout",

    # Evidence collector
    "collect_final_holdout_evidence",
    "attach_final_holdout_to_evidence",
    "build_final_holdout_evidence",
    "final_holdout_passed",
    "final_holdout_evaluated",
    "final_holdout_status",
    "final_holdout_summary",
    "validate_final_holdout_identity",
    "collect_holdout_evidence_map",

    # Production prediction
    "ProductionPredictionStatus",
    "ProductionPredictionConfig",
    "ProductionPrediction",
    "ProductionPredictionGateway",
    "production_prediction_summary",
    "MultiHorizonConfig",
    "MultiHorizonPrediction",
    "MultiHorizonPredictionEngine",
    "build_multi_horizon_engine",
    "multi_horizon_prediction_summary",

    # Market context
    "MarketContextConfig",
    "MarketContextResult",
    "build_benchmark_features",
    "build_market_context",
    "add_relative_strength",
    "market_strength_score",
    "MarketContextRequest",
    "MarketContextData",
    "MarketContextDataLoader",
    "load_market_context",
    "build_stock_market_context",

    # Market integration
    "MarketIntegrationConfig",
    "MarketIntegrationResult",
    "integrate_market_context",
    "add_market_context_to_stock",
    "market_integration_summary",

    # Unified features
    "UnifiedFeatureConfig",
    "UnifiedFeatureResult",
    "build_unified_features",
    "feature_matrix",
    "unified_feature_names",
    "unified_feature_summary",
]
