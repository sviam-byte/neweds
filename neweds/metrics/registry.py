"""Plugin-реестр метрик связности.

Реализации регистрируются декоратором ``@register_metric(...)`` в категорийных
модулях (``correlation.py``, ``information.py``, ``causal.py``, ``spectral.py``,
``ordinal.py``). Регистрация ленивая: ``ensure_builtins()`` импортирует модули
по требованию, чтобы ``import neweds`` не тянул statsmodels/scipy на старте.

Классический ``METRICS_REGISTRY: Mapping[str, MetricFunc]`` оставлен как
read-only view для обратной совместимости со старыми импортами.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

MetricFunc = Callable[..., np.ndarray]
Category = Literal["correlation", "information", "spectral", "ordinal", "causal"]
PartialMode = Literal[
    "none",
    "precision_matrix",
    "explicit_controls_residualization",
]


@dataclass(frozen=True, slots=True)
class Metric:
    """Метрика: метаданные + сама функция."""

    name: str
    func: MetricFunc
    category: Category
    description: str
    directed: bool = False
    pvalue_based: bool = False
    supports_control: bool = False
    experimental: bool = False
    stable: bool = False
    partial_mode: PartialMode = "none"


_METRICS: dict[str, Metric] = {}
_BUILTINS_LOADED: bool = False


def register_metric(
    name: str,
    *,
    category: Category,
    description: str,
    directed: bool = False,
    pvalue_based: bool = False,
    supports_control: bool = False,
    experimental: bool = False,
    stable: bool = False,
    partial_mode: PartialMode = "none",
) -> Callable[[MetricFunc], MetricFunc]:
    """Декоратор: регистрирует обёрнутую функцию в реестре метрик."""

    def decorator(func: MetricFunc) -> MetricFunc:
        if name in _METRICS:
            # Идемпотентно при повторных импортах модулей (например, в тестах).
            return func
        _METRICS[name] = Metric(
            name=name,
            func=func,
            category=category,
            description=description,
            directed=directed,
            pvalue_based=pvalue_based,
            supports_control=supports_control,
            experimental=experimental,
            stable=stable,
            partial_mode=partial_mode,
        )
        return func

    return decorator


def ensure_builtins() -> None:
    """Лениво регистрирует встроенные метрики.

    Вызывается из публичных операций реестра (``get_metric``, ``list_metrics``,
    ``METRICS_REGISTRY[...]``) и из ``core/pipeline.py`` перед поиском метрик.
    Импорт ``neweds`` сам по себе сюда НЕ заходит.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    # Помечаем заранее, чтобы рекурсивные ре-импорты не зацикливались.
    _BUILTINS_LOADED = True
    from neweds.metrics import causal, correlation, information, ordinal, spectral

    correlation._register()
    spectral._register()
    ordinal._register()
    information._register()
    causal._register()


def get_metric(name: str) -> Metric:
    """Возвращает полную запись о метрике. Для неизвестного имени бросает ``ValueError``."""
    ensure_builtins()
    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"Неизвестная метрика: {name}") from exc


def get_metric_func(name: str) -> MetricFunc:
    """Возвращает только функцию метрики (совместимо со старым API)."""
    return get_metric(name).func


def list_metrics() -> list[Metric]:
    """Возвращает все зарегистрированные метрики в порядке регистрации."""
    ensure_builtins()
    return list(_METRICS.values())


class _RegistryView(Mapping[str, MetricFunc]):
    """Read-only ``Mapping`` (имя → функция) для старого кода, ожидавшего dict."""

    def __getitem__(self, key: str) -> MetricFunc:
        ensure_builtins()
        return _METRICS[key].func

    def __iter__(self):
        ensure_builtins()
        return iter(_METRICS)

    def __len__(self) -> int:
        ensure_builtins()
        return len(_METRICS)

    def __contains__(self, key: object) -> bool:
        ensure_builtins()
        return key in _METRICS


METRICS_REGISTRY: Mapping[str, MetricFunc] = _RegistryView()


__all__ = [
    "Category",
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
