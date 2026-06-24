"""Тесты корректности метрик на синтетических данных с известным ground truth.

Допуски достаточно широки для численного шума, но ловят смену знака,
потерянный лаг и случайное транспонирование матрицы.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neweds.core.metric_runner import compute_metric


def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _independent(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})


def _coupled_var1(n: int = 600, alpha: float = 0.7, seed: int = 1) -> pd.DataFrame:
    """Y(t) = alpha * X(t-1) + small noise. X is white noise."""

    rng = _rng(seed)
    x = rng.normal(size=n)
    y = np.zeros(n)
    y[0] = rng.normal()
    for t in range(1, n):
        y[t] = alpha * x[t - 1] + 0.2 * rng.normal()
    return pd.DataFrame({"x": x, "y": y})


def _lagged_copy(n: int = 600, seed: int = 2) -> pd.DataFrame:
    rng = _rng(seed)
    x = rng.normal(size=n)
    y = np.concatenate([[0.0], x[:-1]])  # y(t) = x(t-1)
    return pd.DataFrame({"x": x, "y": y})


def test_correlation_independent_is_near_zero() -> None:
    matrix = compute_metric(_independent(), "correlation_full", lag=1)
    assert matrix.shape == (2, 2)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)
    assert abs(matrix[0, 1]) < 0.1


def test_correlation_strong_when_signals_share_a_lag() -> None:
    matrix = compute_metric(_lagged_copy(), "correlation_full", lag=1)
    # Pearson для полного лага не выявит связь y(t)=x(t-1) — нужна directed-версия.
    matrix = compute_metric(_lagged_copy(), "correlation_directed", lag=1)
    assert matrix[0, 1] > 0.8, f"x -> y correlation expected high, got {matrix[0, 1]}"


def test_directed_correlation_recovers_var1_direction() -> None:
    df = _coupled_var1(alpha=0.7)
    matrix = compute_metric(df, "correlation_directed", lag=1)
    # x ведёт y с лагом 1 → (x→y) должно быть больше (y→x).
    assert matrix[0, 1] > matrix[1, 0], (
        f"expected x->y > y->x, got {matrix[0, 1]:.3f} vs {matrix[1, 0]:.3f}"
    )
    assert matrix[0, 1] > 0.4


def test_dcor_independent_near_zero() -> None:
    matrix = compute_metric(_independent(n=400), "dcor_full", lag=1)
    assert matrix.shape == (2, 2)
    assert 0 <= matrix[0, 1] <= 1
    assert matrix[0, 1] < 0.25


def test_dcor_lagged_copy_is_high() -> None:
    matrix = compute_metric(_lagged_copy(n=400), "dcor_directed", lag=1)
    assert matrix[0, 1] > 0.5


def test_ordinal_independent_is_low() -> None:
    matrix = compute_metric(_independent(n=400), "ordinal_full", lag=1)
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] < 0.3


def test_wavelet_registry_metric_recovers_multiscale_copy() -> None:
    rng = _rng(7)
    t = np.linspace(0.0, 10.0 * np.pi, 512)
    x = np.sin(t) + 0.4 * np.sin(6.0 * t)
    df = pd.DataFrame(
        {
            "x": x + 0.03 * rng.normal(size=t.size),
            "y": x + 0.03 * rng.normal(size=t.size),
            "noise": rng.normal(size=t.size),
        }
    )

    matrix = compute_metric(df, "wavelet_full", lag=1)

    assert matrix.shape == (3, 3)
    assert matrix[0, 1] > 0.8
    assert matrix[0, 1] > matrix[0, 2]


@pytest.mark.parametrize(
    "variant",
    [
        "correlation_full",
        "correlation_spearman",
        "correlation_kendall",
        "dcor_full",
        "wavelet_full",
    ],
)
def test_metric_matrix_is_symmetric_for_undirected_variants(variant: str) -> None:
    matrix = compute_metric(_independent(n=300), variant, lag=1)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-8)
