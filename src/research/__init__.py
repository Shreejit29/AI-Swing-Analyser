"""
AI Swing Analyser — Research package.

Public exports for the research, validation, model-selection,
model documentation, integration, and approval framework.
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
    load_model_card,
    model_card_from_selection,
    model_card_summary,
    save_model_card,
)

from .model_card_registry import (
    ModelCardRegistry,
    register_model_card,
)

from .pipeline import (
    ResearchPipeline,
    ResearchPipelineResult,
    ResearchStage,
)

__all__ = [
    # Existing research configuration
    "ResearchApprovalConfig",
    "ResearchBacktestConfig",
    "ResearchCalibrationConfig",
    "ResearchFeatureConfig",
    "ResearchHyperparameterConfig",
    "ResearchPipelineConfig",
    "ResearchRangeConfig",
    "ResearchRobustnessConfig",
    "ResearchValidationConfig",

    # Integrated configuration
    "IntegratedApprovalConfig",
    "IntegratedBacktestConfig",
    "IntegratedCalibrationConfig",
    "IntegratedHoldoutConfig",
    "IntegratedRangeConfig",
    "IntegratedRegimeConfig",
    "IntegratedResearchConfig",
    "IntegratedRobustnessConfig",
    "IntegratedTradingConfig",
    "IntegratedValidationConfig",
    "default_integrated_research_config",

    # Controlled model selection
    "ControlledModelSelector",
    "ModelCandidateScore",
    "ModelSelectionConfig",
    "ModelSelectionResult",
    "select_best_model",

    # Selection pipeline
    "SelectionCandidate",
    "SelectionPipeline",
    "SelectionPipelineResult",
    "run_model_selection",

    # Selection registry
    "SelectionRecord",
    "SelectionRegistry",
    "register_selection",

    # Model cards
    "ModelCard",
    "ModelCardBuilder",
    "load_model_card",
    "model_card_from_selection",
    "model_card_summary",
    "save_model_card",

    # Model-card registry
    "ModelCardRegistry",
    "register_model_card",

    # Main research pipeline
    "ResearchPipeline",
    "ResearchPipelineResult",
    "ResearchStage",
]
