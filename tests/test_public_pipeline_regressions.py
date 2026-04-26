from __future__ import annotations

import numpy as np
import pandas as pd

from neweds.config import AnalysisConfig
from neweds.core.pipeline import run_analysis
from neweds.core.variant_presets import expand_variants
from neweds.metrics.connectivity import lagged_directed_correlation


def test_lagged_directed_correlation_recovers_shifted_signal() -> None:
    x = np.arange(20, dtype=float)
    y = np.concatenate(([np.nan], x[:-1]))

    df = pd.DataFrame({"x": x, "y": y})
    mat = lagged_directed_correlation(df, lag=1)

    assert mat[0, 1] > 0.99


def test_run_analysis_respects_lag_for_directed_metrics(tmp_path) -> None:
    input_path = tmp_path / "lagged.csv"
    x = np.arange(20, dtype=float)
    y = np.roll(x, -3)
    y[-3:] = np.nan
    pd.DataFrame({"x": x, "y": y}).to_csv(input_path, index=False)

    result = run_analysis(
        str(input_path),
        AnalysisConfig(
            max_lag=3,
            lag_selection="fixed",
            variants=["dcor_directed"],
        ),
    )

    assert result.metrics["dcor_directed"].lag == 3
    assert result.metrics["dcor_directed"].contract is not None
    assert result.metrics["dcor_directed"].contract.directed_lag == 3


def test_public_presets_do_not_advertise_legacy_ah_metrics() -> None:
    variants, _ = expand_variants(["causal", "full", "all"])

    assert "ah_full" not in variants
    assert "ah_partial" not in variants
    assert "ah_directed" not in variants
