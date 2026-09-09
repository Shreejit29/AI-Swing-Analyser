"""
AI Swing Analyser — Research package.

Public exports for the research and validation framework.
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

    # Research pipeline
    "ResearchPipeline",
    "ResearchPipelineResult",
    "ResearchStage",
]
