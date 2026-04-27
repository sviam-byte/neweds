"""Connectivity metrics: registry + per-category implementations.

Регистрация метрик ленивая: ``import neweds`` (и ``import neweds.metrics``) сами по себе
НЕ тянут statsmodels/scipy.signal. Тяжёлые импорты выполняются внутри
``ensure_builtins()`` или при первом обращении к ``METRICS_REGISTRY[...]``.
"""

from .registry import (
    METRICS_REGISTRY,
    Metric,
    MetricFunc,
    PartialMode,
    ensure_builtins,
    get_metric,
    get_metric_func,
    list_metrics,
    register_metric,
)

__all__ = [
    "METRICS_REGISTRY",
    "Metric",
    "MetricFunc",
    "PartialMode",
    "ensure_builtins",
    "get_metric",
    "get_metric_func",
    "list_metrics",
    "register_metric",
]
