"""NewEDS: tools for connectivity analysis of multivariate time-series."""

from __future__ import annotations

from neweds.config import (
    DEFAULT_EDGE_THRESHOLD,
    DEFAULT_MAX_LAG,
    DEFAULT_PVALUE_ALPHA,
    PYINFORM_AVAILABLE,
    AnalysisConfig,
    ComputationContract,
)
from neweds.core.fmri_roi_audit import FmriRoiAuditResult, run_fmri_roi_audit
from neweds.core.fmri_gm_signals import (
    AtlasDefinition,
    RegionalSignalResult,
    VoxelRecoveryConfig,
    VoxelRecoveryResult,
)
from neweds.core.fmri_stage2_benchmark import (
    FmriStage2Config,
    MetricMatrixResult,
    NestedLoocvBenchmarkResult,
    PairedRepresentationComparisonResult,
    PreprocessingBranchConfig,
    RegionalInputManifest,
    SubjectFeatureLedger,
)
from neweds.core.fmri_tissue_audit import FmriTissueAuditResult, run_fmri_tissue_audit
from neweds.core.pipeline import run_analysis
from neweds.core.results import AnalysisResult, MetricResult, WindowResult

__version__ = "0.1.0"

__all__ = [
    "AnalysisConfig",
    "AnalysisResult",
    "ComputationContract",
    "DEFAULT_EDGE_THRESHOLD",
    "DEFAULT_MAX_LAG",
    "DEFAULT_PVALUE_ALPHA",
    "FmriRoiAuditResult",
    "VoxelRecoveryConfig",
    "VoxelRecoveryResult",
    "AtlasDefinition",
    "RegionalSignalResult",
    "FmriStage2Config",
    "RegionalInputManifest",
    "PreprocessingBranchConfig",
    "MetricMatrixResult",
    "SubjectFeatureLedger",
    "NestedLoocvBenchmarkResult",
    "PairedRepresentationComparisonResult",
    "FmriTissueAuditResult",
    "MetricResult",
    "PYINFORM_AVAILABLE",
    "WindowResult",
    "__version__",
    "run_analysis",
    "run_fmri_roi_audit",
    "run_fmri_tissue_audit",
]
