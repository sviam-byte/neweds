"""Tests for deterministic connectivity behavior in group pipeline."""

from pathlib import Path

import numpy as np
import pandas as pd

import src.core.group_pipeline as gp
from src.core.group_pipeline import (
    _correlation_matrix_fast,
    _fdr_bh,
    _mannwhitneyu_vectorized,
    _point_biserial_binary,
    build_missing_bin_qc_table,
    extract_upper_triangle,
    load_subject,
)


def test_correlation_matrix_fast_all_nan_column_becomes_zero_correlations() -> None:
    """All-NaN columns should not propagate NaN into connectivity matrix."""
    df = pd.DataFrame(
        {
            "bin_with_signal": [1.0, 2.0, 3.0],
            "missing_bin": [np.nan, np.nan, np.nan],
            "bin_with_signal_2": [3.0, 1.0, 2.0],
        }
    )

    matrix = _correlation_matrix_fast(df)

    assert np.isfinite(matrix).all()
    assert np.allclose(matrix[1, :], 0.0)
    assert np.allclose(matrix[:, 1], 0.0)


def test_load_subject_disables_subjectwise_preprocessing(monkeypatch, tmp_path: Path) -> None:
    """Loader must not apply per-subject preprocessing before canonical alignment."""
    captured: dict[str, object] = {}

    def _fake_load_or_generate(filepath: str, **kwargs):
        captured["filepath"] = filepath
        captured.update(kwargs)
        return pd.DataFrame({"bin_0_0_0": [1.0, 2.0, 3.0]})

    monkeypatch.setattr(gp, "load_or_generate", _fake_load_or_generate)

    fp = tmp_path / "subject.csv"
    fp.write_text("x,y,z,t0\n0,0,0,1\n", encoding="utf-8")

    _ = load_subject(fp)

    assert captured["preprocess"] is False
    assert captured["fill_missing"] is False
    assert captured["normalize"] is False


def test_missing_bin_qc_table_and_diag_correlation() -> None:
    """QC table should expose missing-bin counts and binary correlation helper should be finite."""
    dfs = {
        "schiz::a": pd.DataFrame({"b1": [1.0, 2.0], "b2": [np.nan, np.nan]}),
        "healthy::b": pd.DataFrame({"b1": [1.0, 2.0], "b2": [1.0, 2.0]}),
    }

    qc = build_missing_bin_qc_table(dfs)
    qc = qc.sort_values("subject_id").reset_index(drop=True)

    assert list(qc["n_missing_bins"]) == [0, 1]
    assert np.isclose(float(qc.loc[0, "missing_bin_fraction"]), 0.0)
    assert np.isclose(float(qc.loc[1, "missing_bin_fraction"]), 0.5)

    labels = np.array([1.0, 0.0])
    corr = _point_biserial_binary(labels, qc["n_missing_bins"].to_numpy())
    assert np.isfinite(corr)


def test_missing_bins_do_not_break_mannwhitney_and_fdr() -> None:
    """All-missing bins must not produce NaN features/p-values in MW/FDR stage."""
    subj_a = pd.DataFrame({
        "bin_1": [1.0, 2.0, 3.0, 4.0],
        "bin_missing": [np.nan, np.nan, np.nan, np.nan],
        "bin_2": [4.0, 3.0, 2.0, 1.0],
    })
    subj_b = pd.DataFrame({
        "bin_1": [1.5, 2.5, 3.5, 4.5],
        "bin_missing": [np.nan, np.nan, np.nan, np.nan],
        "bin_2": [3.5, 2.5, 1.5, 0.5],
    })

    feat_a = extract_upper_triangle(_correlation_matrix_fast(subj_a))[None, :]
    feat_b = extract_upper_triangle(_correlation_matrix_fast(subj_b))[None, :]

    assert np.isfinite(feat_a).all()
    assert np.isfinite(feat_b).all()

    _, p = _mannwhitneyu_vectorized(feat_a, feat_b)
    p_fdr, _ = _fdr_bh(p)

    assert np.isfinite(p).all()
    assert np.isfinite(p_fdr).all()
