"""Compatibility facade for connectivity method metadata.

The central source of truth is ``neweds.metrics.registry``. This module keeps
the historical names (``STABLE_METHODS``, ``METHOD_INFO`` and friends) for code
that imports them from ``neweds.methods`` or ``neweds.config``.

Все таблицы вычисляются лениво через ``__getattr__``, чтобы ``import neweds``
не триггерил ``ensure_builtins()`` (который тянет statsmodels/scipy.signal).
"""

from __future__ import annotations

from typing import Any


def _build_tables() -> dict[str, Any]:
    from neweds.metrics.registry import list_metrics

    metrics = list_metrics()
    return {
        "STABLE_METHODS": [m.name for m in metrics if m.stable],
        "EXPERIMENTAL_METHODS_BASE": [m.name for m in metrics if m.experimental],
        "EXPERIMENTAL_METHODS": [m.name for m in metrics if m.experimental],
        "PVAL_METHODS": {m.name for m in metrics if m.pvalue_based},
        "DIRECTED_METHODS": {m.name for m in metrics if m.directed},
        "METHOD_INFO": {
            m.name: {
                "title": m.name.replace("_", " ").title(),
                "meaning": m.description,
                "category": m.category,
            }
            for m in metrics
        },
    }


_LAZY_NAMES = {
    "STABLE_METHODS",
    "EXPERIMENTAL_METHODS_BASE",
    "EXPERIMENTAL_METHODS",
    "PVAL_METHODS",
    "DIRECTED_METHODS",
    "METHOD_INFO",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_NAMES:
        tables = _build_tables()
        for k, v in tables.items():
            globals()[k] = v
        return globals()[name]
    raise AttributeError(f"module 'neweds.methods' has no attribute {name!r}")


def is_pvalue_method(variant: str) -> bool:
    """Return True if ``variant`` produces p-values rather than effect sizes."""
    from neweds.metrics.registry import get_metric

    try:
        return bool(get_metric(variant.lower()).pvalue_based)
    except ValueError:
        return False


def is_directed_method(variant: str) -> bool:
    """Return True if ``variant`` is direction-sensitive."""
    from neweds.metrics.registry import get_metric

    try:
        return bool(get_metric(variant.lower()).directed)
    except ValueError:
        return False


def is_control_sensitive_method(variant: str) -> bool:
    """Return True if ``variant`` accepts explicit control variables."""
    from neweds.metrics.registry import get_metric

    try:
        return bool(get_metric(variant.lower()).supports_control)
    except ValueError:
        return "_partial" in variant.lower()


__all__ = [
    "DIRECTED_METHODS",
    "EXPERIMENTAL_METHODS",
    "EXPERIMENTAL_METHODS_BASE",
    "METHOD_INFO",
    "PVAL_METHODS",
    "STABLE_METHODS",
    "is_control_sensitive_method",
    "is_directed_method",
    "is_pvalue_method",
]
