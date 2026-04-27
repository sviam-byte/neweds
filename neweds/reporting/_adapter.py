"""Адаптер AnalysisResult для генераторов отчётов.

HTML/Excel-генераторы исходно работали с legacy BigMasterTool; этот тонкий
duck-typed слой подаёт им современный AnalysisResult без изменения генераторов.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from neweds.core.results import AnalysisResult


class _LogShim:
    __slots__ = ("items",)

    def __init__(self, items: list[str]) -> None:
        self.items = list(items)


class ReportAdapter:
    """Duck-typed обёртка AnalysisResult для генераторов отчётов."""

    def __init__(self, result: AnalysisResult) -> None:
        self.result = result
        self.config = result.config

        self.results = {name: item.matrix for name, item in result.metrics.items()}
        self.results_meta = {name: item.metadata for name, item in result.metrics.items()}
        self.variant_lags = {
            name: item.lag for name, item in result.metrics.items() if item.lag is not None
        }

        self.window_analysis = dict(result.windows or {})
        self.graph_results = dict(result.graphs or {})

        data = (result.artifacts or {}).get("analysis_data")
        if data is None:
            data = (result.artifacts or {}).get("data")
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = pd.DataFrame(columns=result.input_info.get("columns", []))

        self.analysis_data = self.data

        self.preprocessing_report = (result.artifacts or {}).get("preprocess_report")
        self.autodiff_report = {"enabled": False, "differenced": []}
        self.fs = float((result.artifacts or {}).get("fs", 1.0))
        self.pairwise_summaries: dict[str, Any] = {}
        self.log = _LogShim(list(result.logs))

    def build_pairwise_summaries(self, *_: Any, **__: Any) -> None:
        return None

    def export_series_bundle(self, save_path: str) -> str:
        self.data.to_excel(save_path, index=False)
        return save_path

    def get_preprocessing_summary(self) -> dict[str, Any]:
        return {"preprocess": {}, "autodiff": self.autodiff_report}

    def get_harmonics(self, *_: Any, **__: Any) -> dict[str, Any]:
        return {}

    def get_diagnostics(self) -> dict[str, Any]:
        return {}


def adapter_from_result(result: AnalysisResult) -> ReportAdapter:
    return ReportAdapter(result)


__all__ = ["ReportAdapter", "adapter_from_result"]
