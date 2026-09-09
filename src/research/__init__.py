"""
AI Swing Analyser - Research Orchestration Package.

The research package coordinates the complete model-development workflow:

    Data
      ↓
    Quality Control
      ↓
    Feature Engineering
      ↓
    Target Construction
      ↓
    Temporal Splitting
      ↓
    Feature Selection
      ↓
    Hyperparameter Search
      ↓
    Walk-Forward Validation
      ↓
    Probability Calibration
      ↓
    Target-Range Validation
      ↓
    Regime Stability
      ↓
    Walk-Forward Backtest
      ↓
    Robustness Testing
      ↓
    Production Approval
      ↓
    Model Registry

The package is intentionally separate from:

    src.models
        Individual model implementations

    src.data
        Data acquisition and preparation

    src.features
        Feature engineering

    src.evaluation
        Reporting and visualisation

This separation reduces accidental coupling and makes it possible to
test every stage independently.
"""

from .config import (
    ResearchPipelineConfig,
)

from .pipeline import (
    ResearchPipeline,
    ResearchPipelineResult,
)

__all__ = [
    "ResearchPipelineConfig",
    "ResearchPipeline",
    "ResearchPipelineResult",
]
