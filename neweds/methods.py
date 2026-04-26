"""Compatibility facade for connectivity method metadata.

The central source of truth is ``neweds.metrics.registry``. This module keeps
the historical names (``STABLE_METHODS``, ``METHOD_INFO`` and friends) for code
that imports them from ``neweds.methods`` or ``neweds.config``.
"""

from __future__ import annotations

from neweds.metrics.registry import list_metrics

_METRICS = list_metrics()

STABLE_METHODS: list[str] = [metric.name for metric in _METRICS if metric.stable]
EXPERIMENTAL_METHODS_BASE: list[str] = [metric.name for metric in _METRICS if metric.experimental]
EXPERIMENTAL_METHODS: list[str] = list(EXPERIMENTAL_METHODS_BASE)
PVAL_METHODS: set[str] = {metric.name for metric in _METRICS if metric.pvalue_based}
DIRECTED_METHODS: set[str] = {metric.name for metric in _METRICS if metric.directed}

METHOD_INFO: dict[str, dict[str, str]] = {
    metric.name: {
        "title": metric.name.replace("_", " ").title(),
        "meaning": metric.description,
        "category": metric.category,
    }
    for metric in _METRICS
}


def is_pvalue_method(variant: str) -> bool:
    """Return True if ``variant`` produces p-values rather than effect sizes."""

    return variant.lower() in PVAL_METHODS


def is_directed_method(variant: str) -> bool:
    """Return True if ``variant`` is direction-sensitive."""

    return variant.lower() in DIRECTED_METHODS


def is_control_sensitive_method(variant: str) -> bool:
    """Return True if ``variant`` accepts explicit control variables."""

    try:
        return bool(next(metric.supports_control for metric in _METRICS if metric.name == variant.lower()))
    except StopIteration:
        return "_partial" in variant.lower()
