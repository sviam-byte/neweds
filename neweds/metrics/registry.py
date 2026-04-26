"""Plugin-реестр метрик связности.

Реализации метрик регистрируются декоратором ``@register_metric(...)``,
а публичный ``run_analysis`` достаёт их по имени. У каждой записи есть
структурированная метадата (флаги directed / p-value, категория, описание),
чтобы отчёты и CLI могли красиво показывать документацию без захардкоженных
таблиц.

Классический ``METRICS_REGISTRY: dict[str, MetricFunc]`` оставлен как
read-only view — для обратной совместимости со старыми импортами.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

MetricFunc = Callable[..., np.ndarray]
Category = Literal["correlation", "information", "spectral", "ordinal", "causal"]


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


_METRICS: dict[str, Metric] = {}


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
) -> Callable[[MetricFunc], MetricFunc]:
    """Декоратор: регистрирует обёрнутую функцию в реестре метрик."""

    def decorator(func: MetricFunc) -> MetricFunc:
        if name in _METRICS:
            raise ValueError(f"Метрика '{name}' уже зарегистрирована")
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
        )
        return func

    return decorator


def get_metric(name: str) -> Metric:
    """Возвращает полную запись о метрике. Для неизвестного имени бросает ``ValueError``."""

    try:
        return _METRICS[name]
    except KeyError as exc:
        raise ValueError(f"Неизвестная метрика: {name}") from exc


def get_metric_func(name: str) -> MetricFunc:
    """Возвращает только функцию метрики (совместимо со старым API)."""

    return get_metric(name).func


def list_metrics() -> list[Metric]:
    """Возвращает все зарегистрированные метрики в порядке регистрации."""

    return list(_METRICS.values())


class _RegistryView(Mapping[str, MetricFunc]):
    """Read-only ``Mapping`` (имя → функция) для старого кода, ожидавшего dict."""

    def __getitem__(self, key: str) -> MetricFunc:
        return _METRICS[key].func

    def __iter__(self):
        return iter(_METRICS)

    def __len__(self) -> int:
        return len(_METRICS)

    def __contains__(self, key: object) -> bool:
        return key in _METRICS


METRICS_REGISTRY: Mapping[str, MetricFunc] = _RegistryView()


def _bootstrap_builtin_metrics() -> None:
    """Регистрирует встроенный набор метрик из :mod:`neweds.metrics.connectivity`.

    Это мост: реализации остаются в connectivity.py, а тут мы вешаем на них
    метаданные плагин-реестра, не переписывая код метрик.
    """

    from neweds.metrics import connectivity as c

    def _h2_full(
        df: pd.DataFrame,
        lag: int = 1,
        control: list[str] | None = None,
        **kw,
    ) -> np.ndarray:
        return c.correlation_matrix(df, lag=lag, control=control, **kw) ** 2

    def _h2_directed(
        df: pd.DataFrame,
        lag: int = 1,
        control: list[str] | None = None,
        **kw,
    ) -> np.ndarray:
        return c.lagged_directed_correlation(df, lag=lag, control=control, **kw) ** 2

    builtins: list[tuple[str, MetricFunc, dict]] = [
        (
            "correlation_full",
            c.correlation_matrix,
            {
                "category": "correlation",
                "description": "Корреляция Пирсона, [-1, 1].",
            },
        ),
        (
            "correlation_spearman",
            c.spearman_correlation_matrix,
            {
                "category": "correlation",
                "description": "Ранговая корреляция Спирмена, [-1, 1].",
            },
        ),
        (
            "correlation_kendall",
            c.kendall_correlation_matrix,
            {
                "category": "correlation",
                "description": "Кендалл tau-b — ранговая согласованность, [-1, 1].",
            },
        ),
        (
            "correlation_partial",
            c.partial_correlation_matrix,
            {
                "category": "correlation",
                "description": "Корреляция Пирсона при контроле остальных переменных.",
                "supports_control": True,
            },
        ),
        (
            "correlation_directed",
            c.lagged_directed_correlation,
            {
                "category": "correlation",
                "description": "Лаговая направленная корреляция между каналами.",
                "directed": True,
            },
        ),
        (
            "h2_full",
            _h2_full,
            {
                "category": "correlation",
                "description": "Квадрат корреляции Пирсона (улавливает нелинейность).",
            },
        ),
        (
            "h2_partial",
            c.partial_h2_matrix,
            {
                "category": "correlation",
                "description": "H2 при контроле других каналов.",
                "supports_control": True,
            },
        ),
        (
            "h2_directed",
            _h2_directed,
            {
                "category": "correlation",
                "description": "Направленная H2.",
                "directed": True,
            },
        ),
        (
            "mutinf_full",
            c.mutual_info_matrix,
            {
                "category": "information",
                "description": "Взаимная информация между парами каналов.",
                "experimental": True,
            },
        ),
        (
            "mutinf_partial",
            c.mutual_info_matrix_partial,
            {
                "category": "information",
                "description": "MI при контроле других каналов.",
                "supports_control": True,
                "experimental": True,
            },
        ),
        (
            "coherence_full",
            c.coherence_matrix,
            {
                "category": "spectral",
                "description": "Magnitude-squared coherence, [0, 1].",
            },
        ),
        (
            "coherence_partial",
            c.coherence_matrix_partial,
            {
                "category": "spectral",
                "description": "Частичная когерентность с контрольными переменными.",
                "supports_control": True,
            },
        ),
        (
            "granger_full",
            c.granger_matrix,
            {
                "category": "causal",
                "description": "p-values F-теста причинности по Грейнджеру.",
                "directed": True,
                "pvalue_based": True,
            },
        ),
        (
            "granger_partial",
            c.granger_matrix_partial,
            {
                "category": "causal",
                "description": "Грейнджер после линейной регрессии по контрольным переменным.",
                "directed": True,
                "pvalue_based": True,
                "supports_control": True,
            },
        ),
        (
            "te_full",
            c.transfer_entropy_matrix,
            {
                "category": "information",
                "description": "Transfer entropy между каналами.",
                "directed": True,
                "experimental": True,
            },
        ),
        (
            "te_partial",
            c.transfer_entropy_matrix_partial,
            {
                "category": "information",
                "description": "Transfer entropy при контроле других каналов.",
                "directed": True,
                "supports_control": True,
                "experimental": True,
            },
        ),
        (
            "dcor_full",
            c.dcor_matrix,
            {
                "category": "information",
                "description": "Дистанционная корреляция dCor, [0, 1]; 0 ⟺ независимость.",
            },
        ),
        (
            "dcor_partial",
            c.dcor_matrix_partial,
            {
                "category": "information",
                "description": "Частичная dCor через резидуализацию по контрольным.",
                "supports_control": True,
                "experimental": True,
            },
        ),
        (
            "dcor_directed",
            c.dcor_matrix_directed,
            {
                "category": "information",
                "description": "Лаговая dCor (направленная).",
                "directed": True,
                "experimental": True,
            },
        ),
        (
            "ordinal_full",
            c.ordinal_matrix,
            {
                "category": "ordinal",
                "description": "Ordinal MI по Bandt–Pompe (порядковые паттерны).",
            },
        ),
        (
            "ordinal_directed",
            c.ordinal_matrix_directed,
            {
                "category": "ordinal",
                "description": "Направленная ordinal MI.",
                "directed": True,
                "experimental": True,
            },
        ),
        (
            "ah_full",
            c.AH_matrix,
            {
                "category": "information",
                "description": "Active information storage (AH).",
                "directed": True,
                "experimental": True,
            },
        ),
        (
            "ah_partial",
            c.compute_partial_AH_matrix,
            {
                "category": "information",
                "description": "Частичная AH с контрольными переменными.",
                "directed": True,
                "supports_control": True,
                "experimental": True,
            },
        ),
        (
            "ah_directed",
            c.AH_matrix_directed,
            {
                "category": "information",
                "description": "Направленная AH.",
                "directed": True,
                "experimental": True,
            },
        ),
    ]

    stable_builtin_names = {
        "correlation_full",
        "correlation_spearman",
        "correlation_kendall",
        "correlation_partial",
        "coherence_full",
        "coherence_partial",
        "granger_full",
        "dcor_full",
        "ordinal_full",
    }

    for metric_name, func, meta in builtins:
        if metric_name in _METRICS:
            continue
        if metric_name in stable_builtin_names:
            meta = {**meta, "stable": True}
        _METRICS[metric_name] = Metric(name=metric_name, func=func, **meta)


_bootstrap_builtin_metrics()


__all__ = [
    "Category",
    "METRICS_REGISTRY",
    "Metric",
    "MetricFunc",
    "get_metric",
    "get_metric_func",
    "list_metrics",
    "register_metric",
]
