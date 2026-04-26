"""Connectivity metrics: registry + implementations."""

from . import connectivity
from .registry import (
    METRICS_REGISTRY,
    Metric,
    MetricFunc,
    get_metric,
    get_metric_func,
    list_metrics,
    register_metric,
)

__all__ = [
    "METRICS_REGISTRY",
    "Metric",
    "MetricFunc",
    "connectivity",
    "get_metric",
    "get_metric_func",
    "list_metrics",
    "register_metric",
]
