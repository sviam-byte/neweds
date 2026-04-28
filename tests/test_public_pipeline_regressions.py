from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from neweds.config import AnalysisConfig
import neweds.core.pipeline as public_pipeline
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


def test_public_presets_do_not_advertise_opt_in_ah_metrics() -> None:
    variants, _ = expand_variants(["causal", "full", "all"])

    assert "ah_full" not in variants
    assert "ah_partial" not in variants
    assert "ah_directed" not in variants


def test_run_analysis_excludes_explicit_controls_from_output_matrix(monkeypatch) -> None:
    def _fake_loader(*args, **kwargs):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [1.0, 1.5, 2.5, 3.5],
                "motion": [0.1, 0.2, 0.1, 0.3],
            }
        )
        return df, type("Report", (), {"steps_global": []})()

    monkeypatch.setattr(public_pipeline, "load_or_generate", _fake_loader)

    result = run_analysis(
        "input.csv",
        AnalysisConfig(variants=["correlation_partial"], controls=["motion"]),
    )

    metric = result.metrics["correlation_partial"]
    assert metric.matrix.shape == (2, 2)
    assert metric.metadata["signal_columns"] == ["x", "y"]
    assert metric.metadata["control_columns"] == ["motion"]
    assert metric.metadata["matrix_columns"] == ["x", "y"]
    assert metric.contract is not None
    assert metric.contract.input_channels == 2


def test_run_analysis_full_metrics_also_exclude_controls(monkeypatch) -> None:
    def _fake_loader(*args, **kwargs):
        df = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0, 4.0],
                "y": [4.0, 3.0, 2.0, 1.0],
                "motion": [0.1, 0.2, 0.1, 0.3],
            }
        )
        return df, type("Report", (), {"steps_global": []})()

    monkeypatch.setattr(public_pipeline, "load_or_generate", _fake_loader)

    result = run_analysis(
        "input.csv",
        AnalysisConfig(variants=["correlation_full"], controls=["motion"]),
    )

    metric = result.metrics["correlation_full"]
    assert metric.matrix.shape == (2, 2)
    assert metric.metadata["matrix_columns"] == ["x", "y"]
    assert result.input_info["control_columns"] == ["motion"]


def test_run_analysis_rejects_missing_and_nonnumeric_controls(monkeypatch) -> None:
    def _fake_loader(*args, **kwargs):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "label": ["a", "b", "c"]})
        return df, type("Report", (), {"steps_global": []})()

    monkeypatch.setattr(public_pipeline, "load_or_generate", _fake_loader)

    with pytest.raises(ValueError, match="Unknown control"):
        run_analysis("input.csv", AnalysisConfig(variants=["correlation_full"], controls=["missing"]))

    with pytest.raises(ValueError, match="numeric"):
        run_analysis("input.csv", AnalysisConfig(variants=["correlation_full"], controls=["label"]))


def test_analysis_config_loader_plumbing_and_auto_difference(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_loader(*args, **kwargs):
        captured.update(kwargs)
        df = pd.DataFrame({"x": [1.0, 3.0, 6.0], "y": [2.0, 5.0, 9.0]})
        return df, type("Report", (), {"steps_global": ["load"]})()

    monkeypatch.setattr(public_pipeline, "load_or_generate", _fake_loader)

    result = run_analysis(
        "input.h5",
        AnalysisConfig(
            variants=["correlation_full"],
            auto_difference=True,
            spatial_bin_size=4,
            spatial_grid_size=7,
            spatial_grid_method="median",
            lazy_spatial_bin=True,
            time_chunk=11,
        ),
    )

    assert captured["h5_spatial_bin"] == 4
    assert captured["spatial_grid_size"] == 7
    assert captured["spatial_grid_method"] == "median"
    assert captured["lazy_spatial_bin"] is True
    assert captured["time_chunk"] == 11
    assert result.input_info["shape"] == [2, 2]
    assert "auto_difference" in result.metrics["correlation_full"].metadata["preprocess_steps"]


def test_pvalue_correction_and_windows_are_exposed(monkeypatch) -> None:
    def _fake_loader(*args, **kwargs):
        df = pd.DataFrame({"x": np.arange(12, dtype=float), "y": np.arange(12, dtype=float)})
        return df, type("Report", (), {"steps_global": []})()

    def _fake_compute(data, variant, *, lag=1, control=None, **params):
        return np.array([[0.0, 0.03], [0.04, 0.0]], dtype=float)

    monkeypatch.setattr(public_pipeline, "load_or_generate", _fake_loader)
    monkeypatch.setattr(public_pipeline, "compute_metric", _fake_compute)

    result = run_analysis(
        "input.csv",
        AnalysisConfig(
            variants=["granger_full"],
            pvalue_correction="bonferroni",
            window_sizes=[4],
            window_stride=4,
        ),
    )

    matrix = result.metrics["granger_full"].matrix
    assert np.isclose(matrix[0, 1], 0.06)
    assert np.isclose(matrix[1, 0], 0.08)
    assert "granger_full" in result.windows
