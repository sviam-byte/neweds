"""Метрики связности: плагин-реестр + реализации по категориям.

Регистрация метрик ленивая: сам по себе ``import neweds`` (и ``import neweds.metrics``)
НЕ тянет statsmodels и scipy.signal. Тяжёлые импорты происходят внутри
``ensure_builtins()`` либо при первом обращении к ``METRICS_REGISTRY[...]``.
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
