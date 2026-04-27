"""Юнит-тесты реестра метрик связности."""

import numpy as np
import pandas as pd

from neweds.metrics.connectivity import (
    correlation_matrix,
    granger_matrix,
    kendall_matrix,
    spearman_matrix,
)
from neweds.metrics.registry import get_metric_func


def test_correlation_identity() -> None:
    """Корреляция одинаковых столбцов должна быть 1 для всех элементов."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    matrix = correlation_matrix(df)
    assert np.allclose(matrix, [[1.0, 1.0], [1.0, 1.0]], equal_nan=True)


def test_correlation_nan_handling() -> None:
    """Попарный NaN не должен ломать корреляцию на доступных точках."""
    df = pd.DataFrame({"a": [1, 2, np.nan, 4], "b": [1, 2, 3, 4]})
    matrix = correlation_matrix(df)
    assert np.isfinite(matrix[0, 1])
    assert matrix[0, 1] > 0.99


def test_granger_empty_safe() -> None:
    """Матрица Грейнджера не должна падать на слишком коротких рядах."""
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [1.0, 2.0]})
    matrix = granger_matrix(df, lag=5)
    assert matrix.shape == (2, 2)


def test_registry_lookup() -> None:
    """Реестр должен возвращать вызываемую метрику по каноническому имени."""
    func = get_metric_func("correlation_full")
    out = func(pd.DataFrame({"x": [0, 1], "y": [0, 1]}), lag=1, control=None)
    assert out.shape == (2, 2)


def test_spearman_monotonic_detected() -> None:
    """Спирмен должен показывать почти идеальную связь на строго монотонных рядах."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    matrix = spearman_matrix(df)
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] > 0.99


def test_kendall_monotonic_detected() -> None:
    """Kendall tau-b должен показывать почти идеальную связь на монотонных рядах."""
    df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
    matrix = kendall_matrix(df)
    assert matrix.shape == (2, 2)
    assert matrix[0, 1] > 0.99
