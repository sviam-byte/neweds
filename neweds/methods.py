"""Каталог connectivity-методов NewEDS.

Здесь живёт классификация: какие метрики стабильные, какие экспериментальные,
какие направленные, какие отдают p-values, и как описать каждую в отчёте.

Реестр метрик (``neweds.metrics.registry``) опирается на эти таблицы, когда
собирает метаданные для ``MetricResult``.
"""

from __future__ import annotations

# Стабильные методы — то, что показываем по умолчанию.
STABLE_METHODS: list[str] = [
    "correlation_full",
    "correlation_spearman",
    "correlation_kendall",
    "correlation_partial",
    "coherence_full",
    "coherence_partial",
    "granger_full",
    "dcor_full",
    "ordinal_full",
]

# Экспериментальные: чувствительные к параметрам или ещё не до конца отлаженные.
EXPERIMENTAL_METHODS_BASE: list[str] = [
    "mutinf_full",
    "mutinf_partial",
    "te_full",
    "te_partial",
    "ah_full",
    "ah_partial",
    "ah_directed",
    "dcor_partial",
    "dcor_directed",
    "ordinal_directed",
]

# TE-методы оставляем включёнными: при отсутствии pyinform
# ядро откатится на numpy-fallback, но работать будет.
EXPERIMENTAL_METHODS: list[str] = list(EXPERIMENTAL_METHODS_BASE)

# Методы, у которых результат — p-values, а не сила связи.
PVAL_METHODS: set[str] = {
    "granger_full",
    "granger_partial",
}

# Методы с асимметричной матрицей (направление имеет значение).
DIRECTED_METHODS: set[str] = {
    "correlation_directed",
    "h2_directed",
    "granger_full",
    "granger_partial",
    "te_full",
    "te_partial",
    "ah_full",
    "ah_partial",
    "ah_directed",
    "dcor_directed",
    "ordinal_directed",
}

# Описания методов: заголовок и расшифровка для отчётов и UI.
METHOD_INFO: dict[str, dict[str, str]] = {
    "correlation_full": {
        "title": "Корреляция Пирсона (полная)",
        "meaning": "Линейная связь, [-1, 1]. |value| ближе к 1 — связь сильнее.",
    },
    "correlation_spearman": {
        "title": "Корреляция Спирмена",
        "meaning": "Ранговая монотонная связь, [-1, 1]. Устойчива к выбросам и нелинейной монотонности.",
    },
    "correlation_kendall": {
        "title": "Кендалл tau-b",
        "meaning": "Ранговая согласованность пар, [-1, 1]. Устойчивее к ties, но медленнее Спирмена.",
    },
    "correlation_partial": {
        "title": "Частичная корреляция",
        "meaning": "Линейная связь при контроле остальных переменных, [-1, 1].",
    },
    "correlation_directed": {
        "title": "Лаговая направленная корреляция",
        "meaning": "Направленная связь через сдвиг по лагу. |value| ближе к 1 — сильнее.",
    },
    "mutinf_full": {
        "title": "Взаимная информация (MI)",
        "meaning": "Нелинейная зависимость, ≥ 0. Больше — сильнее.",
    },
    "mutinf_partial": {
        "title": "Частичная MI",
        "meaning": "MI при контроле, ≥ 0. Больше — сильнее.",
    },
    "coherence_full": {
        "title": "Когерентность",
        "meaning": "Частотная синхронизация, обычно [0, 1]. Больше — сильнее.",
    },
    "coherence_partial": {
        "title": "Частичная когерентность",
        "meaning": "Когерентность при контроле других каналов.",
    },
    "dcor_full": {
        "title": "Дистанционная корреляция (dCor)",
        "meaning": "Непараметрическая нелинейная зависимость, [0, 1]. dCor=0 ⟺ независимость.",
    },
    "dcor_partial": {
        "title": "Частичная dCor",
        "meaning": "dCor с резидуализацией по контрольным переменным, [0, 1].",
    },
    "dcor_directed": {
        "title": "Лаговая dCor (направленная)",
        "meaning": "dCor(X(t), Y(t+lag)) — нелинейная направленная связь.",
    },
    "ordinal_full": {
        "title": "Ordinal MI (Bandt–Pompe)",
        "meaning": "Зависимость через порядковые паттерны, ≥ 0. Устойчива к шуму.",
    },
    "ordinal_directed": {
        "title": "Ordinal MI (направленная)",
        "meaning": "Лаговая ordinal MI. Больше — сильнее направленная связь.",
    },
    "h2_full": {
        "title": "H2 (полная)",
        "meaning": "Нелинейная связность, обычно [0, 1].",
    },
    "h2_partial": {
        "title": "H2 (частичная)",
        "meaning": "H2 при контроле других каналов.",
    },
    "h2_directed": {
        "title": "H2 (направленная)",
        "meaning": "Направленная H2.",
    },
    "granger_full": {
        "title": "Грейнджер (p-values)",
        "meaning": "p-value F-теста. Меньше — сильнее свидетельство причинности.",
    },
    "granger_partial": {
        "title": "Грейнджер частичный (p-values)",
        "meaning": "Грейнджер после линейной регрессии по контрольным, лучший лаг до L.",
    },
    "te_full": {
        "title": "Transfer Entropy",
        "meaning": "Направленный поток информации. Больше — сильнее.",
    },
    "te_partial": {
        "title": "Частичная Transfer Entropy",
        "meaning": "TE при контроле других каналов.",
    },
    "ah_full": {
        "title": "Active information storage (AH)",
        "meaning": "Информационная мера памяти системы. Больше — сильнее.",
    },
    "ah_partial": {
        "title": "AH (частичная)",
        "meaning": "AH при контроле других каналов.",
    },
    "ah_directed": {
        "title": "AH (направленная)",
        "meaning": "Направленная AH.",
    },
}


def is_pvalue_method(variant: str) -> bool:
    """True, если метрика возвращает p-values, а не силу связи."""

    return variant.lower() in PVAL_METHODS


def is_directed_method(variant: str) -> bool:
    """True, если метрика направленная (матрица асимметрична)."""

    return variant.lower() in DIRECTED_METHODS


def is_control_sensitive_method(variant: str) -> bool:
    """True, если метрика умеет работать с контрольными переменными."""

    return "_partial" in variant.lower()
