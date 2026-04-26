"""Structured result contracts for the public analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.config import AnalysisConfig, ComputationContract


@dataclass(slots=True)
class MetricResult:
    """One computed connectivity metric and its execution metadata."""

    name: str
    matrix: np.ndarray
    directed: bool
    lag: int | None
    pvalue_based: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    contract: ComputationContract | None = None


@dataclass(slots=True)
class WindowResult:
    """Windowed or scan analysis for a metric."""

    metric_name: str
    window_size: int | None = None
    start: int | None = None
    end: int | None = None
    lag: int | None = None
    matrix: np.ndarray | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AnalysisResult:
    """Public output of run_analysis."""

    input_info: dict[str, Any]
    config: AnalysisConfig
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    graphs: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    windows: dict[str, WindowResult | list[WindowResult] | dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)


class AnalysisResultToolAdapter:
    """Compatibility shim for legacy report generators."""

    def __init__(self, result: AnalysisResult) -> None:
        import pandas as pd

        self.result = result
        self.config = result.config

        self.results = {name: item.matrix for name, item in result.metrics.items()}
        self.results_meta = {name: item.metadata for name, item in result.metrics.items()}
        self.variant_lags = {name: item.lag for name, item in result.metrics.items() if item.lag is not None}

        self.window_analysis = dict(result.windows or {})
        self.graph_results = dict(result.graphs or {})

        data = (result.artifacts or {}).get("data")
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = pd.DataFrame(columns=result.input_info.get("columns", []))

        self.data_normalized = self.data.copy()
        self.data_raw = self.data.copy()
        self.data_preprocessed = self.data.copy()
        self.data_after_autodiff = self.data.copy()

        self.preprocessing_report = (result.artifacts or {}).get("preprocess_report")
        self.autodiff_report = {"enabled": False, "differenced": []}
        self.fs = 1.0
        self.pairwise_summaries = {}
        self.log = type("_Log", (), {"items": list(result.logs)})()

    def build_pairwise_summaries(self, *_, **__) -> None:
        return None

    def export_series_bundle(self, save_path: str) -> str:
        self.data.to_excel(save_path, index=False)
        return save_path

    def get_preprocessing_summary(self) -> dict[str, Any]:
        return {"preprocess": {}, "autodiff": self.autodiff_report}

    def get_harmonics(self, *_, **__) -> dict[str, Any]:
        return {}

    def get_diagnostics(self) -> dict[str, Any]:
        return {}


def tool_adapter_from_result(result: AnalysisResult) -> Any:
    """Return a report-compatible adapter for AnalysisResult."""

    return AnalysisResultToolAdapter(result)
