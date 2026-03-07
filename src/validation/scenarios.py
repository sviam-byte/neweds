"""Синтетические сценарии с известным ground truth для валидации метрик связности.

Каждый сценарий возвращает DataFrame + словарь ожиданий (expectations),
описывающий, что каждая метрика *должна* показать.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Expectation:
    """Ожидание для одной метрики на одном сценарии.

    metric_group: 'undirected' | 'directed' | 'pvalue'
    pair: (i, j) -- индексы колонок
    check: 'near_zero' | 'significantly_positive' | 'greater_than' |
           'pvalue_high' | 'pvalue_low' | 'asymmetry'
    """
    metric_group: str  # e.g. 'correlation_full', 'mutinf_full', или '*_directed'
    pair: tuple[int, int]
    check: str
    threshold: float = 0.0
    compare_pair: Optional[tuple[int, int]] = None  # для 'greater_than'
    description: str = ""
    tolerance: float = 0.15  # допустимая погрешность для near_zero


@dataclass
class Scenario:
    """Один тестовый сценарий."""
    name: str
    description: str
    data: pd.DataFrame
    expectations: list[Expectation] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)


def make_independent(n: int = 1000, seed: int = 42) -> Scenario:
    """Два независимых белых шума. Все метрики должны быть ~0."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"X": rng.normal(size=n), "Y": rng.normal(size=n)})
    expectations = []

    # Undirected метрики: |value| < threshold
    for m in ("correlation_full", "correlation_partial", "h2_full",
              "mutinf_full", "dcor_full", "ordinal_full", "coherence_full"):
        expectations.append(Expectation(
            metric_group=m, pair=(0, 1), check="near_zero",
            threshold=0.15, tolerance=0.15,
            description=f"{m}: независимые ряды → |value| < 0.15",
        ))

    # Directed метрики: |value| < threshold
    for m in ("correlation_directed", "h2_directed", "te_full",
              "dcor_directed", "ordinal_directed"):
        for pair in [(0, 1), (1, 0)]:
            expectations.append(Expectation(
                metric_group=m, pair=pair, check="near_zero",
                threshold=0.15, tolerance=0.15,
                description=f"{m}: независимые → |M[{pair[0]},{pair[1]}]| < 0.15",
            ))

    # P-value методы: p >> 0.05
    for m in ("granger_full",):
        for pair in [(0, 1), (1, 0)]:
            expectations.append(Expectation(
                metric_group=m, pair=pair, check="pvalue_high",
                threshold=0.05,
                description=f"{m}: независимые → p > 0.05",
            ))

    return Scenario(
        name="independent",
        description="Два независимых белых шума (N=1000). Все метрики должны быть ~0.",
        data=df,
        expectations=expectations,
        tags=["basic", "null_hypothesis"],
    )


def make_linear_lagged(
    n: int = 1000,
    lag: int = 3,
    coeff: float = 0.8,
    noise_std: float = 0.2,
    seed: int = 42,
) -> Scenario:
    """Y[t] = coeff * X[t - lag] + noise. X→Y с известным лагом."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(lag, n):
        y[t] = coeff * x[t - lag] + noise_std * rng.normal()
    # Z — независимый контроль
    z = rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    expectations = []

    # Undirected: X-Y связь сильная, X-Z и Y-Z слабые
    for m in ("correlation_full", "h2_full", "mutinf_full",
              "dcor_full", "coherence_full"):
        expectations.append(Expectation(
            metric_group=m, pair=(0, 1), check="significantly_positive",
            threshold=0.2,
            description=f"{m}: X→Y (lag={lag}) → |M[X,Y]| > 0.2",
        ))
        expectations.append(Expectation(
            metric_group=m, pair=(0, 2), check="near_zero",
            threshold=0.15, tolerance=0.15,
            description=f"{m}: X,Z независимы → |M[X,Z]| < 0.15",
        ))

    # Directed: X→Y >> Y→X
    for m in ("correlation_directed", "h2_directed",
              "dcor_directed", "ordinal_directed"):
        expectations.append(Expectation(
            metric_group=m, pair=(0, 1), check="greater_than",
            compare_pair=(1, 0),
            description=f"{m}: X→Y должен быть > Y→X",
        ))

    # TE: X→Y > Y→X
    expectations.append(Expectation(
        metric_group="te_full", pair=(0, 1), check="greater_than",
        compare_pair=(1, 0),
        description="TE: X→Y > Y→X",
    ))

    # Granger: X→Y значим, Y→X нет
    expectations.append(Expectation(
        metric_group="granger_full", pair=(0, 1), check="pvalue_low",
        threshold=0.05,
        description="Granger: X→Y p < 0.05",
    ))
    expectations.append(Expectation(
        metric_group="granger_full", pair=(1, 0), check="pvalue_high",
        threshold=0.05,
        description="Granger: Y→X p > 0.05 (нет обратной причинности)",
    ))

    return Scenario(
        name=f"linear_lagged_L{lag}",
        description=f"Y = {coeff}·X(t-{lag}) + N(0,{noise_std}²). "
                    "X→Y линейная причинность, Z независим.",
        data=df,
        expectations=expectations,
        tags=["basic", "directed", "linear"],
    )


def make_nonlinear_quadratic(n: int = 1000, seed: int = 42) -> Scenario:
    """Y = X² + noise. Нелинейная зависимость, линейная корреляция ~0."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x ** 2 + 0.1 * rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y})

    expectations = []

    # Корреляция должна быть ~0 (X и X² некоррелированы для симметричного X)
    expectations.append(Expectation(
        metric_group="correlation_full", pair=(0, 1), check="near_zero",
        threshold=0.15, tolerance=0.15,
        description="Корреляция: X vs X² ≈ 0 (симметричное распределение)",
    ))

    # MI, dCor, ordinal MI должны обнаружить зависимость
    for m in ("mutinf_full", "dcor_full", "ordinal_full"):
        expectations.append(Expectation(
            metric_group=m, pair=(0, 1), check="significantly_positive",
            threshold=0.15,
            description=f"{m}: X vs X² → нелинейная зависимость обнаружена",
        ))

    return Scenario(
        name="nonlinear_quadratic",
        description="Y = X² + noise. Корреляция ~0, но нелинейные метрики > 0.",
        data=df,
        expectations=expectations,
        tags=["basic", "nonlinear"],
    )


def make_bidirectional_asymmetric(
    n: int = 1000,
    lag_xy: int = 2,
    lag_yx: int = 5,
    coeff_xy: float = 0.7,
    coeff_yx: float = 0.3,
    seed: int = 42,
) -> Scenario:
    """Двунаправленная связь с разными лагами и силами.

    X(t) = coeff_yx * Y(t - lag_yx) + noise
    Y(t) = coeff_xy * X(t - lag_xy) + noise

    X→Y сильнее и быстрее чем Y→X.
    """
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    y = np.zeros(n)
    max_lag = max(lag_xy, lag_yx)
    # Инициализация
    for t in range(max_lag):
        x[t] = rng.normal()
        y[t] = rng.normal()
    for t in range(max_lag, n):
        x[t] = coeff_yx * y[t - lag_yx] + 0.3 * rng.normal()
        y[t] = coeff_xy * x[t - lag_xy] + 0.3 * rng.normal()

    df = pd.DataFrame({"X": x, "Y": y})

    expectations = []

    # При lag=2: X→Y должен быть сильнее
    for m in ("correlation_directed", "dcor_directed"):
        expectations.append(Expectation(
            metric_group=m, pair=(0, 1), check="greater_than",
            compare_pair=(1, 0),
            description=f"{m}: X→Y (coeff={coeff_xy}, lag={lag_xy}) > Y→X (coeff={coeff_yx}, lag={lag_yx})",
        ))

    return Scenario(
        name="bidirectional_asymmetric",
        description=f"X→Y: coeff={coeff_xy}, lag={lag_xy}; Y→X: coeff={coeff_yx}, lag={lag_yx}. "
                    "Тест на асимметрию направленных метрик.",
        data=df,
        expectations=expectations,
        tags=["advanced", "directed", "bidirectional"],
    )


def make_confounded(n: int = 1000, seed: int = 42) -> Scenario:
    """X и Y оба зависят от Z (конфаундер), но не друг от друга.

    full correlation X-Y > 0, но partial correlation X-Y|Z ≈ 0.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    x = 0.8 * z + 0.3 * rng.normal(size=n)
    y = 0.6 * z + 0.3 * rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    expectations = []

    # Full correlation X-Y > 0 (spurious)
    expectations.append(Expectation(
        metric_group="correlation_full", pair=(0, 1), check="significantly_positive",
        threshold=0.3,
        description="Full corr(X,Y) > 0.3 (ложная связь через конфаундер Z)",
    ))

    # Partial correlation X-Y|Z ≈ 0
    expectations.append(Expectation(
        metric_group="correlation_partial", pair=(0, 1), check="near_zero",
        threshold=0.15, tolerance=0.15,
        description="Partial corr(X,Y|Z) ≈ 0 (Z объясняет связь)",
    ))

    return Scenario(
        name="confounded",
        description="X и Y оба зависят от Z. Full corr > 0, partial corr ≈ 0.",
        data=df,
        expectations=expectations,
        tags=["basic", "partial", "confounding"],
    )


def make_known_coherence(n: int = 2000, freq: float = 0.05, seed: int = 42) -> Scenario:
    """X и Y — синусоиды одной частоты с шумом. Когерентность на freq должна быть высокой."""
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    x = np.sin(2 * np.pi * freq * t) + 0.3 * rng.normal(size=n)
    y = 0.7 * np.sin(2 * np.pi * freq * t + 0.5) + 0.3 * rng.normal(size=n)
    df = pd.DataFrame({"X": x, "Y": y})

    expectations = [
        Expectation(
            metric_group="coherence_full", pair=(0, 1),
            check="significantly_positive", threshold=0.3,
            description=f"Когерентность: общая частота {freq} Hz → coherence > 0.3",
        ),
    ]

    return Scenario(
        name="coherent_sinusoids",
        description=f"Два синуса на частоте {freq} Hz + шум. Когерентность > 0.",
        data=df,
        expectations=expectations,
        tags=["basic", "frequency"],
    )


def make_ill_conditioned(n: int = 200, seed: int = 42) -> Scenario:
    """Почти вырожденная матрица: X1 ≈ X2, X3 независим.

    Тест на устойчивость partial correlation при collinearity.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = x1 + 0.01 * rng.normal(size=n)  # почти дубликат
    x3 = rng.normal(size=n)
    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3})

    expectations = [
        # Partial correlation не должна взрываться (|value| < 5)
        Expectation(
            metric_group="correlation_partial", pair=(0, 2),
            check="finite",
            description="Partial corr(X1,X3|X2): конечное значение (не inf/nan) при collinearity",
        ),
        Expectation(
            metric_group="correlation_partial", pair=(1, 2),
            check="finite",
            description="Partial corr(X2,X3|X1): конечное значение при collinearity",
        ),
    ]

    return Scenario(
        name="ill_conditioned",
        description="X1 ≈ X2 (collinear), X3 независим. Тест устойчивости partial correlation.",
        data=df,
        expectations=expectations,
        tags=["robustness", "partial"],
    )


# ─── Реестр сценариев ────────────────────────────────────────────────

ALL_SCENARIOS = {
    "independent": make_independent,
    "linear_lagged": make_linear_lagged,
    "nonlinear_quadratic": make_nonlinear_quadratic,
    "bidirectional_asymmetric": make_bidirectional_asymmetric,
    "confounded": make_confounded,
    "coherent_sinusoids": make_known_coherence,
    "ill_conditioned": make_ill_conditioned,
}

QUICK_SCENARIOS = ["independent", "linear_lagged", "nonlinear_quadratic", "confounded"]

DEFAULT_SCENARIOS = list(ALL_SCENARIOS.keys())
