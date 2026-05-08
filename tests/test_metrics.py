"""Юнит-тесты реестра метрик связности."""

import numpy as np
import pandas as pd
import pytest

from neweds.metrics.connectivity import (
    compute_partial_AH_matrix,
    correlation_matrix,
    granger_matrix,
    kendall_matrix,
    mutual_info_matrix,
    mutual_info_matrix_partial,
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
    with pytest.warns(UserWarning, match="Granger skipped"):
        matrix = granger_matrix(df, lag=5)
    assert matrix.shape == (2, 2)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.isnan(matrix[0, 1])
    assert np.isnan(matrix[1, 0])


def test_mutual_info_short_series_returns_nan() -> None:
    """Недостаток наблюдений для MI должен быть видимым NaN, а не нулевой связью."""
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]})
    with pytest.warns(UserWarning, match="Mutual information skipped"):
        matrix = mutual_info_matrix(df, k=5)
    assert matrix.shape == (2, 2)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.isnan(matrix[0, 1])


def test_mutual_info_partial_short_series_returns_nan() -> None:
    """Partial MI не должен маскировать нехватку данных нулём."""
    df = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0],
            "b": [1.0, 2.0, 3.0],
            "c": [0.0, 1.0, 0.0],
        }
    )
    with pytest.warns(UserWarning, match="Conditional mutual information skipped"):
        matrix = mutual_info_matrix_partial(df, k=5)
    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.isnan(matrix[0, 1])


def test_partial_ah_var_failure_returns_nan_matrix(monkeypatch) -> None:
    """Провал VAR residualization не должен подменяться расчётом AH на raw data."""
    import statsmodels.tsa.vector_ar.var_model as var_model

    class FailingVAR:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def fit(self, *_args, **_kwargs):
            raise ValueError("synthetic VAR failure")

    monkeypatch.setattr(var_model, "VAR", FailingVAR)
    df = pd.DataFrame(
        {
            "a": np.arange(20, dtype=float),
            "b": np.arange(20, dtype=float) + 1.0,
        }
    )
    matrix = compute_partial_AH_matrix(df, lag=1)
    assert matrix.shape == (2, 2)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.isnan(matrix[0, 1])


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
