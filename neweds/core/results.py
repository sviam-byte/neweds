"""Структурированные контракты результатов публичного пайплайна."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from neweds.config import AnalysisConfig, ComputationContract


@dataclass(slots=True)
class MetricResult:
    """Одна посчитанная метрика связности + метаданные её вычисления."""

    name: str
    matrix: np.ndarray
    directed: bool
    lag: int | None
    pvalue_based: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    contract: ComputationContract | None = None


@dataclass(slots=True)
class WindowResult:
    """Результат оконного / сканирующего анализа одной метрики."""

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
    """Публичный результат :func:`neweds.core.pipeline.run_analysis`."""

    input_info: dict[str, Any]
    config: AnalysisConfig
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    graphs: dict[str, Any] = field(default_factory=dict)
    reports: dict[str, Any] = field(default_factory=dict)
    logs: list[str] = field(default_factory=list)
    windows: dict[str, WindowResult | list[WindowResult] | dict[str, Any]] = field(
        default_factory=dict
    )
    artifacts: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "AnalysisResult",
    "MetricResult",
    "WindowResult",
]
