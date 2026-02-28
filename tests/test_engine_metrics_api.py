"""Regression tests for legacy metric helpers exported by src.core.engine.

These tests protect against accidental large edits where helper functions could
be removed or renamed while refactoring kNN internals.
"""

import numpy as np
import pandas as pd

from src.core import engine


LEGACY_API_NAMES = [
    "correlation_matrix",
    "partial_correlation_matrix",
    "partial_h2_matrix",
    "lagged_directed_correlation",
    "h2_matrix",
    "lagged_directed_h2",
    "coherence_matrix",
    "mutual_info_matrix",
    "mutual_info_matrix_partial",
    "compute_granger_matrix",
    "TE_matrix",
    "TE_matrix_partial",
    "granger_dict",
    "granger_matrix",
    "granger_matrix_partial",
]


def test_engine_keeps_legacy_metric_api_symbols() -> None:
    """Important legacy helpers should stay available for compatibility."""
    missing = [name for name in LEGACY_API_NAMES if not hasattr(engine, name)]
    assert not missing, f"Missing engine API symbols: {missing}"


def test_engine_knn_based_mi_helpers_smoke() -> None:
    """kNN MI helpers should still return stable NxN outputs."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=64),
            "y": rng.normal(size=64),
            "z": rng.normal(size=64),
        }
    )

    mi = engine.mutual_info_matrix(df, k=3)
    pmi = engine.mutual_info_matrix_partial(df, k=3)

    assert mi.shape == (3, 3)
    assert pmi.shape == (3, 3)
    assert np.isfinite(np.diag(mi)).all()
    assert np.isfinite(np.diag(pmi)).all()
