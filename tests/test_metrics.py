"""Юнит-тесты реестра метрик связности."""

import numpy as np
import pandas as pd
import pytest

from neweds.metrics.connectivity import (
    compute_partial_AH_matrix,
    correlation_matrix,
    dcor_matrix,
    dcor_matrix_directed,
    granger_matrix,
    kendall_matrix,
    mutual_info_matrix,
    mutual_info_matrix_partial,
    ordinal_matrix,
    spearman_matrix,
    transfer_entropy_matrix,
    wavelet_matrix,
    wavelet_matrix_partial,
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


def test_heavy_undirected_metric_guardrail_fails_fast() -> None:
    df = pd.DataFrame(np.arange(90, dtype=float).reshape(30, 3), columns=["a", "b", "c"])

    with pytest.raises(ValueError, match="dcor_full.*max_pairwise_pairs"):
        dcor_matrix(df, max_pairwise_pairs=1)


def test_heavy_directed_metric_guardrail_counts_directed_pairs() -> None:
    df = pd.DataFrame(np.arange(90, dtype=float).reshape(30, 3), columns=["a", "b", "c"])

    with pytest.raises(ValueError, match="dcor_directed.*max_pairwise_pairs"):
        dcor_matrix_directed(df, max_pairwise_pairs=5)


def test_guardrail_allows_small_explicit_pairs_on_large_matrix() -> None:
    df = pd.DataFrame(
        np.arange(120, dtype=float).reshape(30, 4),
        columns=["a", "b", "c", "d"],
    )

    matrix = dcor_matrix(df, pairs=[(0, 1)], max_pairwise_pairs=1)

    assert matrix.shape == (4, 4)
    assert matrix[0, 1] >= 0.0


def test_guardrail_can_be_disabled() -> None:
    df = pd.DataFrame(np.arange(90, dtype=float).reshape(30, 3), columns=["a", "b", "c"])

    matrix = dcor_matrix(df, max_pairwise_pairs=1, performance_guardrails=False)

    assert matrix.shape == (3, 3)


def test_cached_ordinal_and_te_full_return_valid_shapes() -> None:
    x = np.linspace(0.0, 1.0, 40)
    df = pd.DataFrame({"a": x, "b": x[::-1], "c": np.sin(x * np.pi)})

    ordinal = ordinal_matrix(df)
    te = transfer_entropy_matrix(df, lag=1)

    assert ordinal.shape == (3, 3)
    assert te.shape == (3, 3)
    assert np.allclose(np.diag(ordinal), 0.0)
    assert np.allclose(np.diag(te), 0.0)
    assert np.isfinite(ordinal).all()
    assert np.isfinite(te).all()


def test_wavelet_identical_multiscale_signals_are_maximally_coupled() -> None:
    t = np.linspace(0.0, 8.0 * np.pi, 512)
    x = np.sin(t) + 0.35 * np.sin(4.0 * t) + 0.1 * np.cos(13.0 * t)
    df = pd.DataFrame({"a": x, "b": x.copy()})

    matrix = wavelet_matrix(df)

    assert matrix.shape == (2, 2)
    assert np.allclose(np.diag(matrix), 1.0)
    assert matrix[0, 1] > 0.999


def test_wavelet_detects_shared_structure_better_than_noise() -> None:
    rng = np.random.default_rng(123)
    t = np.linspace(0.0, 12.0 * np.pi, 512)
    shared = np.sin(t) + 0.45 * np.sin(5.0 * t)
    df = pd.DataFrame(
        {
            "x": shared + 0.05 * rng.normal(size=t.size),
            "y": shared + 0.05 * rng.normal(size=t.size),
            "noise": rng.normal(size=t.size),
        }
    )

    matrix = wavelet_matrix(df)

    assert 0.0 <= matrix[0, 1] <= 1.0
    assert matrix[0, 1] > 0.7
    assert matrix[0, 1] > matrix[0, 2]


def test_wavelet_partial_removes_shared_linear_control() -> None:
    rng = np.random.default_rng(456)
    control = rng.normal(size=512)
    df = pd.DataFrame(
        {
            "x": control + 0.2 * rng.normal(size=512),
            "y": control + 0.2 * rng.normal(size=512),
        }
    )

    full = wavelet_matrix(df)
    partial = wavelet_matrix_partial(df, control_matrix=control[:, None])

    assert np.isfinite(partial[0, 1])
    assert partial[0, 1] < full[0, 1]


def test_wavelet_guardrail_fails_before_large_pairwise_run() -> None:
    df = pd.DataFrame(np.arange(120, dtype=float).reshape(30, 4))

    with pytest.raises(ValueError, match="wavelet_full.*max_pairwise_pairs"):
        wavelet_matrix(df, max_pairwise_pairs=2)


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
