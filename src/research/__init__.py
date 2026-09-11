"""
AI Swing Analyser — Research package.

Central exports for the research-first modelling framework.
"""

from .config import (
    ResearchApprovalConfig,
    ResearchBacktestConfig,
    ResearchCalibrationConfig,
    ResearchFeatureConfig,
    ResearchHyperparameterConfig,
    ResearchPipelineConfig,
    ResearchRangeConfig,
    ResearchRobustnessConfig,
    ResearchValidationConfig,
)

from .research_config import (
    IntegratedApprovalConfig,
    IntegratedBacktestConfig,
    IntegratedCalibrationConfig,
    IntegratedHoldoutConfig,
    IntegratedRangeConfig,
    IntegratedRegimeConfig,
    IntegratedResearchConfig,
    IntegratedRobustnessConfig,
    IntegratedTradingConfig,
    IntegratedValidationConfig,
    default_integrated_research_config,
)

from .pipeline import (
    ResearchPipeline,
    ResearchPipelineResult,
    ResearchStage,
)

from .model_selection import (
    ControlledModelSelector,
    ModelCandidateScore,
    ModelSelectionConfig,
    ModelSelectionResult,
    select_best_model,
)

from .selection_pipeline import (
    SelectionCandidate,
    SelectionPipeline,
    SelectionPipelineResult,
    run_model_selection,
)

from .selection_registry import (
    SelectionRecord,
    SelectionRegistry,
    register_selection,
)

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

from .selection_stage import (
    ModelSelectionStage,
    ModelSelectionStageResult,
    run_model_selection_stage,
)

from .holdout_gate import (
    FinalHoldoutGate,
    HoldoutGateInput,
    HoldoutGateResult,
    HoldoutGateStatus,
    create_holdout_gate,
    evaluate_holdout_gate,
)

from .holdout_stage import (
    FinalHoldoutStage,
    HoldoutStageResult,
    run_final_holdout_stage,
)

__all__ = [
    # ------------------------------------------------------------------
    # Legacy / modular research configuration
    # ------------------------------------------------------------------
    "ResearchValidationConfig",
    "ResearchFeatureConfig",
    "ResearchHyperparameterConfig",
    "ResearchCalibrationConfig",
    "ResearchRangeConfig",
    "ResearchBacktestConfig",
    "ResearchRobustnessConfig",
    "ResearchApprovalConfig",
    "ResearchPipelineConfig",

    # ------------------------------------------------------------------
    # Integrated research configuration
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Main research pipeline
    # ------------------------------------------------------------------
    "ResearchPipeline",
    "ResearchPipelineResult",
    "ResearchStage",

    # ------------------------------------------------------------------
    # Controlled model selection
    # ------------------------------------------------------------------
    "ModelCandidateScore",
    "ModelSelectionConfig",
    "ModelSelectionResult",
    "ControlledModelSelector",
    "select_best_model",

    # ------------------------------------------------------------------
    # Selection pipeline
    # ------------------------------------------------------------------
    "SelectionCandidate",
    "SelectionPipeline",
    "SelectionPipelineResult",
    "run_model_selection",

    # ------------------------------------------------------------------
    # Selection registry
    # ------------------------------------------------------------------
    "SelectionRecord",
    "SelectionRegistry",
    "register_selection",

    # ------------------------------------------------------------------
    # Model cards
    # ------------------------------------------------------------------
    "ModelCard",
    "ModelCardBuilder",
    "model_card_from_selection",
    "save_model_card",
    "load_model_card",

    # ------------------------------------------------------------------
    # Model card registry
    # ------------------------------------------------------------------
    "ModelCardRegistry",
    "register_model_card",

    # ------------------------------------------------------------------
    # Integrated model selection stage
    # ------------------------------------------------------------------
    "ModelSelectionStage",
    "ModelSelectionStageResult",
    "run_model_selection_stage",

    # ------------------------------------------------------------------
    # Final holdout protection
    # ------------------------------------------------------------------
    "FinalHoldoutGate",
    "HoldoutGateInput",
    "HoldoutGateResult",
    "HoldoutGateStatus",
    "create_holdout_gate",
    "evaluate_holdout_gate",

    # ------------------------------------------------------------------
    # Final holdout evaluation stage
    # ------------------------------------------------------------------
    "FinalHoldoutStage",
    "HoldoutStageResult",
    "run_final_holdout_stage",
]
