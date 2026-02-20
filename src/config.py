#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Конфигурация и константы для Time Series Analysis Tool.
"""

import importlib.util
import os
from dataclasses import dataclass

# Базовые пути
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVE_FOLDER = os.path.join(BASE_PATH, "TimeSeriesAnalysis")
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Параметры по умолчанию
DEFAULT_MAX_LAG = 5
DEFAULT_K_MI = 5
DEFAULT_BINS = 8
DEFAULT_OUTLIER_Z = 5
DEFAULT_REGULARIZATION = 1e-8
DEFAULT_EMBED_DIM = 3
DEFAULT_EMBED_TAU = 1
DEFAULT_PVALUE_ALPHA = 0.05
DEFAULT_EDGE_THRESHOLD = 0.2

# Параметр регуляризации
REG_ALPHA = 1e-5

# Проверка наличия pyinform
PYINFORM_AVAILABLE = importlib.util.find_spec("pyinform") is not None

# Стабильные методы
STABLE_METHODS = [
    "correlation_full",
    "correlation_partial",
    "coherence_full",
    "coherence_partial",
    "granger_full",
    "dcor_full",
    "ordinal_full",
]

# Экспериментальные методы (базовый список)
EXPERIMENTAL_METHODS_BASE = [
    "mutinf_full",
    "mutinf_partial",
    "te_full",
    "te_partial",
    "te_directed",
    "ah_full",
    "ah_partial",
    "ah_directed",
    "dcor_partial",
    "dcor_directed",
    "ordinal_directed",
]

# Если pyinform недоступен, скрываем TE-методы
EXPERIMENTAL_METHODS = [
    method
    for method in EXPERIMENTAL_METHODS_BASE
    if PYINFORM_AVAILABLE or not method.startswith("te_")
]

# P-value методы
PVAL_METHODS = {
    "granger_full",
    "granger_partial",
    "granger_directed",
}

# Directed методы
DIRECTED_METHODS = {
    "correlation_directed",
    "h2_directed",
    "granger_full",
    "granger_partial",
    "granger_directed",
    "te_full",
    "te_partial",
    "te_directed",
    "ah_full",
    "ah_partial",
    "ah_directed",
    "dcor_directed",
    "ordinal_directed",
}

# Информация о методах
METHOD_INFO = {
    "correlation_full": {
        "title": "Корреляция (полная)",
        "meaning": "Линейная связь. Значение в [-1, 1]. |value| ближе к 1 = сильнее.",
    },
    "correlation_partial": {
        "title": "Частичная корреляция",
        "meaning": "Линейная связь при контроле остальных переменных. [-1, 1].",
    },
    "correlation_directed": {
        "title": "Лаговая корреляция (directed)",
        "meaning": "Оценка направленной связи через сдвиг по лагу. Чем больше |value|, тем сильнее.",
    },
    "mutinf_full": {
        "title": "Взаимная информация (MI)",
        "meaning": "Нелинейная зависимость. >= 0. Больше = сильнее.",
    },
    "mutinf_partial": {
        "title": "Частичная MI",
        "meaning": "MI при контроле переменных. >= 0. Больше = сильнее.",
    },
    "coherence_full": {
        "title": "Когерентность",
        "meaning": "Частотная синхронизация. Обычно в [0, 1]. Больше = сильнее.",
    },
    "coherence_partial": {
        "title": "Частичная когерентность",
        "meaning": "Частотная синхронизация при контроле. Обычно в [0, 1]. Больше = сильнее.",
    },
    "dcor_full": {
        "title": "Дистанционная корреляция (dCor)",
        "meaning": "Нелинейная зависимость без параметров. [0, 1]. dCor=0 ⟺ независимость.",
    },
    "dcor_partial": {
        "title": "Partial dCor",
        "meaning": "dCor при контроле (через резидуализацию). [0, 1].",
    },
    "dcor_directed": {
        "title": "Lagged dCor (directed)",
        "meaning": "dCor(X(t), Y(t+lag)). Нелинейная направленная зависимость.",
    },
    "ordinal_full": {
        "title": "Ordinal MI (permutation)",
        "meaning": "Зависимость через порядковые паттерны (Bandt-Pompe). ≥0. Устойчива к шуму.",
    },
    "ordinal_directed": {
        "title": "Ordinal MI (directed)",
        "meaning": "Лаговая ordinal MI. Больше = сильнее направленная зависимость.",
    },
    "h2_full": {
        "title": "H2 (полная)", 
        "meaning": "Нелинейная связность. Обычно в [0, 1]. Больше = сильнее."
    },
    "h2_partial": {
        "title": "H2 (partial)", 
        "meaning": "H2 при контроле. Обычно в [0, 1]. Больше = сильнее."
    },
    "h2_directed": {
        "title": "H2 (directed)", 
        "meaning": "Направленная H2. Больше = сильнее."
    },
    "granger_full": {
        "title": "Granger (p-values)", 
        "meaning": "p-value теста. Меньше = сильнее свидетельство причинности."
    },
    "granger_partial": {
        "title": "Granger partial (p-values)",
        "meaning": "Granger partial (linear control; best lag up to L): p-value после удаления влияния control. Меньше = сильнее.",
    },
    "granger_directed": {
        "title": "Granger directed (p-values)",
        "meaning": "То же семейство p-values. Меньше = сильнее.",
    },
    "te_full": {
        "title": "Transfer Entropy",
        "meaning": "Направленный поток информации. Больше = сильнее.",
    },
    "te_partial": {
        "title": "Transfer Entropy (partial)",
        "meaning": "TE при контроле. Больше = сильнее.",
    },
    "te_directed": {
        "title": "Transfer Entropy (directed)",
        "meaning": "TE (directed). Больше = сильнее.",
    },
    "ah_full": {
        "title": "Active information storage (AH)",
        "meaning": "Информационная мера памяти системы. Больше = сильнее.",
    },
    "ah_partial": {
        "title": "AH (partial)",
        "meaning": "AH при контроле. Больше = сильнее.",
    },
    "ah_directed": {
        "title": "AH (directed)",
        "meaning": "AH направленная. Больше = сильнее.",
    },
}




@dataclass(slots=True)
class ComputationContract:
    """Контракт вычисления — явная декларация что/как/почему считалось.

    Каждый результат (матрица) сопровождается контрактом, позволяющим
    однозначно воспроизвести и интерпретировать результат.
    """

    variant: str = ""
    input_channels: int = 0
    input_T: int = 0
    input_missing_frac: float = 0.0

    preprocess_steps: list = None  # type: ignore[assignment]
    controls: list = None  # type: ignore[assignment]
    control_strategy: str = "none"

    directed: bool = False
    directed_lag: int = 1
    lag_selection: str = "fixed"

    validity_warnings: list = None  # type: ignore[assignment]

    output_shape: tuple = (0, 0)
    output_type: str = "matrix_NxN"

    seed: int | None = None
    config_hash: str = ""

    def __post_init__(self):
        if self.preprocess_steps is None:
            self.preprocess_steps = []
        if self.controls is None:
            self.controls = []
        if self.validity_warnings is None:
            self.validity_warnings = []

    def as_dict(self) -> dict:
        return {
            "variant": self.variant,
            "input": {"channels": self.input_channels, "T": self.input_T, "missing_frac": self.input_missing_frac},
            "preprocess": list(self.preprocess_steps),
            "controls": {"strategy": self.control_strategy, "variables": list(self.controls)},
            "directed": {"is_directed": self.directed, "lag": self.directed_lag, "lag_selection": self.lag_selection},
            "validity": list(self.validity_warnings),
            "output": {"shape": list(self.output_shape), "type": self.output_type},
            "repro": {"seed": self.seed, "config_hash": self.config_hash},
        }

    def summary_text(self) -> str:
        lines = [
            f"Считали: {self.variant}",
            f"Вход: {self.input_channels} каналов × {self.input_T} точек (пропуски: {self.input_missing_frac:.1%})",
            f"Предобработка: {', '.join(self.preprocess_steps) if self.preprocess_steps else 'нет'}",
            f"Контроль: {self.control_strategy} ({', '.join(self.controls) if self.controls else '—'})",
        ]
        if self.directed:
            lines.append(f"Направление: lag={self.directed_lag}, выбор={self.lag_selection}")
        if self.validity_warnings:
            lines.append(f"Предупреждения: {'; '.join(self.validity_warnings)}")
        lines.append(f"Выход: {self.output_type} {self.output_shape}")
        return "\n".join(lines)


@dataclass(slots=True)
class AnalysisConfig:
    """Контейнер конфигурации анализа."""

    max_lag: int = DEFAULT_MAX_LAG
    p_value_alpha: float = DEFAULT_PVALUE_ALPHA
    graph_threshold: float = DEFAULT_EDGE_THRESHOLD
    enable_experimental: bool = False
    # Автоматически делать diff() если ряд нестационарен.
    auto_difference: bool = False
    # Поправка на множественные сравнения: 'none' | 'fdr_bh'.
    pvalue_correction: str = "none"

    # Настройки окон/лагов для связности
    # Если задано, расчёты могут выполняться на скользящих окнах.
    # В результирующую матрицу по методу попадёт "лучшая" матрица (см. window_policy).
    window_sizes: list[int] | None = None
    # Шаг окна (в точках). Если None — дефолт = max(1, window_size//5).
    window_stride: int | None = None
    # Политика агрегации по окнам: 'best' | 'mean'.
    window_policy: str = "best"

    # Подбор лага: 'fixed' | 'optimize'
    # - fixed: используем lag=1 (или явно заданный)
    # - optimize: перебираем 1..max_lag и выбираем лучший по quality-score
    lag_selection: str = "optimize"

def is_pvalue_method(variant: str) -> bool:
    """Проверяет, является ли метод p-value методом."""
    return variant.lower() in PVAL_METHODS


def is_directed_method(variant: str) -> bool:
    """Проверяет, является ли метод направленным."""
    return variant.lower() in DIRECTED_METHODS


def is_control_sensitive_method(variant: str) -> bool:
    """Проверяет, зависит ли метод от control-переменных."""
    return "_partial" in variant.lower()
