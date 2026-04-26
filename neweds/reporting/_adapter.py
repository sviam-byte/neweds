"""Internal adapter that exposes ``AnalysisResult`` to the existing reporting code.

The HTML and Excel writers were originally driven by the legacy ``BigMasterTool``
object, which exposed many ad-hoc attributes (``data_normalized``, ``data_raw``,
``preprocessing_report``, etc.). After legacy removal those writers consume the
modern :class:`AnalysisResult` contract, but their internals still walk the same
attribute names. Rather than rewrite the report templates wholesale, this thin
adapter re-shapes ``AnalysisResult`` into the duck-typed surface the writers
expect. It lives in the ``reporting`` package precisely to keep the legacy-shape
out of the public API.
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
    """Duck-typed view of :class:`AnalysisResult` for the report writers."""

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

        data = (result.artifacts or {}).get("data")
        if isinstance(data, pd.DataFrame):
            self.data = data.copy()
        else:
            self.data = pd.DataFrame(columns=result.input_info.get("columns", []))

        # Layered views are aliased to the same data — there is only one logical
        # layer in the modern pipeline, but the report templates ask for them by
        # name.
        self.data_normalized = self.data
        self.data_raw = self.data
        self.data_preprocessed = self.data
        self.data_after_autodiff = self.data

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
