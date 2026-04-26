"""NewEDS — пакет для анализа связности временных рядов и групповых fMRI-данных."""

from __future__ import annotations

from neweds.config import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_MAX_LAG,
    DEFAULT_PVALUE_ALPHA,
    PYINFORM_AVAILABLE,
    AnalysisConfig,
    ComputationContract,
)
from neweds.core.pipeline import run_analysis
from neweds.core.results import AnalysisResult, MetricResult, WindowResult

__version__ = "0.2.0"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ComputationContract",
    "DEFAULT_EDGE_THRESHOLD",
    "DEFAULT_MAX_LAG",
    "DEFAULT_PVALUE_ALPHA",
    "MetricResult",
    "PYINFORM_AVAILABLE",
    "WindowResult",
    "__version__",
    "run_analysis",
]
