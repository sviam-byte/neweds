"""Модуль генерации синтетических временных рядов для тестирования.

Здесь есть два слоя:
1) готовые пресеты (coupled_system, random_walks)
2) генератор по формулам (x(t), y(t,x), z(t,x,y), ...)

Формулы вычисляются через безопасный парсер (AST) с белым списком функций.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from neweds.core.formula_evaluator import FormulaSpec, make_eval_env, safe_eval_vector


def generate_coupled_system(
    n_samples: int = 500,
    coupling_strength: float = 0.8,
    noise_level: float = 0.2,
    *,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Генерирует систему из 4 переменных:

    - X: авторегрессионный процесс (источник).
    - Y: зависит от X (X -> Y) с лагом 1.
    - Z: независимый шум (random walk).
    - S: сезонный компонент (синус).
    """
    rng = np.random.default_rng(seed)

    e_x = rng.normal(0, 1, n_samples)
    e_y = rng.normal(0, 1, n_samples)
    e_z = rng.normal(0, 1, n_samples)

    x = np.zeros(n_samples)
    y = np.zeros(n_samples)

    for t in range(1, n_samples):
        x[t] = 0.5 * x[t - 1] + noise_level * e_x[t]
        y[t] = 0.5 * y[t - 1] + coupling_strength * x[t - 1] + noise_level * e_y[t]

    z = np.cumsum(e_z * noise_level)

    t_idx = np.arange(n_samples)
    s = np.sin(2 * np.pi * t_idx / 50) + rng.normal(0, 0.1, n_samples)

    df = pd.DataFrame(
        {
            "Source (X)": x,
            "Target (Y)": y,
            "Noise (Z)": z,
            "Season (S)": s,
        }
    )

    return df.iloc[50:].reset_index(drop=True)


def generate_random_walks(
    n_vars: int = 5,
    n_samples: int = 500,
    *,
    seed: int | None = None,
) -> pd.DataFrame:
    """Генерирует N случайных блужданий (часто дают ложные корреляции)."""
    rng = np.random.default_rng(seed)
    data = {}
    for i in range(n_vars):
        data[f"RW_{i + 1}"] = np.cumsum(rng.normal(0, 1, n_samples))
    return pd.DataFrame(data)


def generate_independent_ar1(
    n_vars: int = 3,
    n_samples: int = 500,
    *,
    phi: float = 0.7,
    noise_level: float = 0.5,
    seed: int | None = 42,
) -> pd.DataFrame:
    """Генерирует независимые AR(1)-процессы для sanity-check метрик связности."""
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {}
    p = float(phi)

    for i in range(int(n_vars)):
        e = rng.normal(0.0, float(noise_level), size=int(n_samples))
        x = np.zeros(int(n_samples), dtype=float)
        for t in range(1, int(n_samples)):
            x[t] = p * x[t - 1] + e[t]
        data[f"AR1_{i + 1}"] = x

    return pd.DataFrame(data)


def generate_chain_system_4d(
    n_samples: int = 500,
    coupling_strength: float = 0.8,
    noise_level: float = 0.2,
    season_period: int = 50,
) -> pd.DataFrame:
    """Генерирует цепочку 4D: X1→X2→X3→X4 с лагом 1 и слабой сезонностью в X1."""
    rng = np.random.default_rng(42)
    n = int(n_samples)

    x1 = np.zeros(n, dtype=float)
    x2 = np.zeros(n, dtype=float)
    x3 = np.zeros(n, dtype=float)
    x4 = np.zeros(n, dtype=float)

    e1 = rng.normal(0.0, 1.0, size=n)
    e2 = rng.normal(0.0, 1.0, size=n)
    e3 = rng.normal(0.0, 1.0, size=n)
    e4 = rng.normal(0.0, 1.0, size=n)

    t_idx = np.arange(n, dtype=float)
    season = np.sin(2 * np.pi * t_idx / float(max(2, int(season_period))))

    for t in range(1, n):
        x1[t] = 0.6 * x1[t - 1] + float(noise_level) * e1[t] + 0.3 * season[t]
        x2[t] = 0.6 * x2[t - 1] + float(coupling_strength) * x1[t - 1] + float(noise_level) * e2[t]
        x3[t] = 0.6 * x3[t - 1] + float(coupling_strength) * x2[t - 1] + float(noise_level) * e3[t]
        x4[t] = 0.6 * x4[t - 1] + float(coupling_strength) * x3[t - 1] + float(noise_level) * e4[t]

    df = pd.DataFrame({"X1": x1, "X2": x2, "X3": x3, "X4": x4})
    # Отбрасываем разогрев, чтобы уменьшить влияние начальных нулевых условий.
    return df.iloc[50:].reset_index(drop=True)


# =========================
# Генерация по формулам
# =========================


def generate_formula_dataset(
    *,
    n_samples: int = 500,
    dt: float = 1.0,
    seed: int | None = 42,
    specs: Iterable[FormulaSpec] | None = None,
) -> pd.DataFrame:
    """Генерирует датасет по формулам.

    Семантика зависимостей:
    - первый ряд может использовать только t
    - следующий может использовать t и ранее вычисленные ряды (x, y, ...)

    Пример:
      X: sin(2*pi*t/50) + 0.2*randn()
      Y: 0.8*X + 0.3*randn()
      Z: rw(0.5)

    Переменные в формулах:
      t — массив времени длины N
      X, Y, Z ... — ранее вычисленные ряды (регистр важен: используйте имена рядов)

    Разрешённые функции:
      sin, cos, tan, exp, log, sqrt, abs, clip, where, minimum, maximum,
      randn(scale=1), randu(scale=1), rw(scale=1), ar1(phi=0.7, scale=1)
    """

    if specs is None:
        specs = [
            FormulaSpec("X", "sin(2*pi*t/50) + 0.2*randn()"),
            FormulaSpec("Y", "0.8*X + 0.3*randn()"),
            FormulaSpec("Z", "rw(0.5)"),
        ]

    n = int(n_samples)
    if n < 5:
        raise ValueError("n_samples должно быть >= 5")

    t = np.arange(n, dtype=float) * float(dt)
    rng = np.random.default_rng(seed)
    env = make_eval_env(n=n, rng=rng)

    names: dict[str, Any] = {"t": t}
    out: dict[str, np.ndarray] = {}

    for spec in specs:
        if not spec.name or not spec.expr:
            continue
        vec = safe_eval_vector(spec.expr, env=env, names={**names, **out})
        out[str(spec.name)] = vec

    if not out:
        raise ValueError("Не удалось сгенерировать ни одного ряда")

    return pd.DataFrame(out)
