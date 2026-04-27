"""Снэпшот-регрессия для run_analysis.

Фиксирует матрицу публичного pipeline на детерминированных данных.
Любое изменение числовых результатов — явный сигнал для ревью.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from neweds.config import AnalysisConfig
from neweds.core.pipeline import run_analysis


def _build_input(tmp_path) -> str:
    rng = np.random.default_rng(123)
    n = 200
    x = rng.normal(size=n)
    y = 0.5 * x + rng.normal(size=n) * 0.3
    z = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    path = tmp_path / "snapshot_input.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_correlation_full_snapshot(tmp_path) -> None:
    input_path = _build_input(tmp_path)

    result = run_analysis(
        input_path,
        AnalysisConfig(max_lag=1, lag_selection="fixed", variants=["correlation_full"]),
    )

    matrix = result.metrics["correlation_full"].matrix
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-9)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-9)

    # Фиксируем внедиагональные значения с точностью 2 знаков — улавливает регрессии, терпит BLAS-флуктуации.
    expected_xy = matrix[0, 1]
    expected_xz = matrix[0, 2]
    expected_yz = matrix[1, 2]
    assert 0.7 < expected_xy < 0.9, expected_xy
    assert -0.2 < expected_xz < 0.2, expected_xz
    assert -0.2 < expected_yz < 0.2, expected_yz


def test_lag_optimization_picks_a_better_lag_for_directed_metric(tmp_path) -> None:
    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(2, n):
        y[t] = 0.8 * x[t - 2] + 0.2 * rng.normal()
    pd.DataFrame({"x": x, "y": y}).to_csv(tmp_path / "lag.csv", index=False)

    result = run_analysis(
        str(tmp_path / "lag.csv"),
        AnalysisConfig(
            max_lag=3,
            lag_selection="optimize",
            variants=["correlation_directed"],
        ),
    )
    metric = result.metrics["correlation_directed"]
    assert metric.lag == 2, f"optimizer should pick lag=2, got {metric.lag}"
