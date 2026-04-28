"""Тесты детерминированности group pipeline."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import neweds.cli_group as cli_group
import neweds.core.group_pipeline as gp
from neweds.core.group_pipeline import (
    _correlation_matrix_fast,
    _fdr_bh,
    _mannwhitneyu_vectorized,
    _point_biserial_binary,
    build_missing_bin_qc_table,
    extract_upper_triangle,
    filter_features_by_bin_coverage,
    group_comparison,
    load_group,
    load_subject,
    run_group_pipeline,
)


def test_correlation_matrix_fast_all_nan_column_becomes_zero_correlations() -> None:
    """Столбцы из одних NaN не должны распространять NaN в матрицу связности."""
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
    """Загрузчик не должен применять препроцессинг до canonical alignment."""
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
    """QC-таблица должна содержать счётчики пропущенных бинов, корреляция — конечная."""
    dfs = {
        "case::a": pd.DataFrame({"b1": [1.0, 2.0], "b2": [np.nan, np.nan]}),
        "control::b": pd.DataFrame({"b1": [1.0, 2.0], "b2": [1.0, 2.0]}),
    }

    qc = build_missing_bin_qc_table(dfs)
    qc = qc.sort_values("subject_id").reset_index(drop=True)

    assert list(qc["n_missing_bins"]) == [1, 0]
    assert np.isclose(float(qc.loc[0, "missing_bin_fraction"]), 0.5)
    assert np.isclose(float(qc.loc[1, "missing_bin_fraction"]), 0.0)

    labels = np.array([1.0, 0.0])
    corr = _point_biserial_binary(labels, qc["n_missing_bins"].to_numpy())
    assert np.isfinite(corr)


def test_missing_bins_do_not_break_mannwhitney_and_fdr() -> None:
    """Полностью пустые бины не должны давать NaN в стадии MW/FDR."""
    subj_a = pd.DataFrame(
        {
            "bin_1": [1.0, 2.0, 3.0, 4.0],
            "bin_missing": [np.nan, np.nan, np.nan, np.nan],
            "bin_2": [4.0, 3.0, 2.0, 1.0],
        }
    )
    subj_b = pd.DataFrame(
        {
            "bin_1": [1.5, 2.5, 3.5, 4.5],
            "bin_missing": [np.nan, np.nan, np.nan, np.nan],
            "bin_2": [3.5, 2.5, 1.5, 0.5],
        }
    )

    feat_a = extract_upper_triangle(_correlation_matrix_fast(subj_a))[None, :]
    feat_b = extract_upper_triangle(_correlation_matrix_fast(subj_b))[None, :]

    assert np.isfinite(feat_a).all()
    assert np.isfinite(feat_b).all()

    _, p = _mannwhitneyu_vectorized(feat_a, feat_b)
    p_fdr, _ = _fdr_bh(p)

    assert np.isfinite(p).all()
    assert np.isfinite(p_fdr).all()


def test_filter_features_by_bin_coverage_masks_pairs_with_sparse_bins() -> None:
    """Фильтр покрытия оставляет только пары бинов с достаточным покрытием."""
    bin_ids = ["b0", "b1", "b2"]
    # Признаки соответствуют парам: (0,1), (0,2), (1,2)
    feat_a = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]], dtype=float)
    feat_b = np.array([[0.5, 0.6, 0.7]], dtype=float)

    # b2 отсутствует у одного субъекта => покрытие b2 = 2/3 < 0.8.
    dfs_a = {
        "a1": pd.DataFrame({"b0": [1.0], "b1": [2.0], "b2": [3.0]}),
        "a2": pd.DataFrame({"b0": [1.0], "b1": [2.0], "b2": [np.nan]}),
    }
    dfs_b = {"b1": pd.DataFrame({"b0": [1.0], "b1": [2.0], "b2": [3.0]})}

    out_a, out_b, mask = filter_features_by_bin_coverage(
        feat_a,
        feat_b,
        bin_ids=bin_ids,
        dfs_a=dfs_a,
        dfs_b=dfs_b,
        min_coverage=0.8,
    )

    assert mask.tolist() == [True, False, False]
    assert out_a.shape == (2, 1)
    assert out_b.shape == (1, 1)


def test_group_comparison_reports_effect_size_and_respects_pair_mask() -> None:
    """group_comparison must expose effect size and align bin labels with filtered pairs."""
    features_a = np.array([[0.1, 0.9], [0.2, 1.0]], dtype=float)
    features_b = np.array([[0.3, 1.1], [0.4, 1.2]], dtype=float)
    # Для 3 бинов пары: (b0,b1), (b0,b2), (b1,b2)
    bin_ids = ["b0", "b1", "b2"]
    pair_mask = np.array([True, False, True], dtype=bool)

    df = group_comparison(
        features_a,
        features_b,
        bin_ids=bin_ids,
        alpha=0.05,
        pair_mask=pair_mask,
    )

    assert list(df.columns) == [
        "bin_i",
        "bin_j",
        "u_stat",
        "p_raw",
        "p_fdr",
        "effect_size_r",
        "significant",
    ]
    assert set(zip(df["bin_i"], df["bin_j"])) == {("b0", "b1"), ("b1", "b2")}


def test_load_group_fail_fast_by_default_and_allows_explicit_skip(
    monkeypatch, tmp_path: Path
) -> None:
    (tmp_path / "ok.csv").write_text("x,y,z,t0\n0,0,0,1\n", encoding="utf-8")
    (tmp_path / "bad.csv").write_text("x,y,z,t0\n0,0,0,1\n", encoding="utf-8")

    def _fake_load_subject(path: Path, **kwargs):
        if path.name == "bad.csv":
            raise ValueError("boom")
        return pd.DataFrame({"bin_0_0_0": [1.0, 2.0]})

    monkeypatch.setattr(gp, "load_subject", _fake_load_subject)
    monkeypatch.setattr(gp, "_validate_subject_schema", lambda path, columns: {"file": path.name})
    monkeypatch.setattr(gp, "_peek_columns", lambda path: ["x", "y", "z", "t0"])
    monkeypatch.setattr(gp, "_cross_validate_group_schemas", lambda schemas, group_label: None)

    with pytest.raises(RuntimeError, match="failed to load"):
        load_group(tmp_path, "case")

    loaded = load_group(tmp_path, "case", allow_skip=True)
    assert list(loaded) == ["case::ok"]
    assert loaded.skipped_subjects == [{"file": "bad.csv", "error": "boom"}]


def test_group_pipeline_uses_case_control_canonical_references(monkeypatch, tmp_path: Path) -> None:
    refs: list[str] = []

    def _fake_load_group(directory, group_label, **kwargs):
        return {f"{group_label}::s1": pd.DataFrame({"b0": [1.0, 2.0], "b1": [2.0, 3.0]})}

    def _fake_fit(ref_dfs, strategy):
        refs.append(next(iter(ref_dfs)).split("::", 1)[0])
        return SimpleNamespace(
            n_voxels=2,
            voxel_ids=["b0", "b1"],
            source_info={},
            save=lambda path: None,
        )

    monkeypatch.setattr(gp, "load_group", _fake_load_group)
    monkeypatch.setattr(gp, "fit_canonical_space", _fake_fit)
    monkeypatch.setattr(gp, "align_all", lambda dfs, space: dfs)
    monkeypatch.setattr(
        gp,
        "build_missing_bin_qc_table",
        lambda dfs: pd.DataFrame(
            {"subject_id": list(dfs), "n_missing_bins": [0], "missing_bin_fraction": [0.0]}
        ),
    )
    monkeypatch.setattr(
        gp, "build_feature_matrix", lambda dfs, method: (np.array([[0.1]]), list(dfs))
    )
    monkeypatch.setattr(
        gp,
        "group_comparison",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "bin_i": ["b0"],
                "bin_j": ["b1"],
                "u_stat": [0.0],
                "p_raw": [1.0],
                "p_fdr": [1.0],
                "effect_size_r": [0.0],
                "significant": [False],
            }
        ),
    )

    summary_case = run_group_pipeline(
        tmp_path,
        tmp_path,
        tmp_path / "out_case",
        canonical_reference="case",
        min_bin_coverage=0.0,
        save_canonical_space=False,
        save_feature_matrix=False,
    )
    summary_control = run_group_pipeline(
        tmp_path,
        tmp_path,
        tmp_path / "out_control",
        canonical_reference="control",
        min_bin_coverage=0.0,
        save_canonical_space=False,
        save_feature_matrix=False,
    )

    assert summary_case["canonical_reference"] == "case"
    assert summary_control["canonical_reference"] == "control"
    assert refs[:2] == ["case", "control"]


def test_cli_group_accepts_public_canonical_reference_names() -> None:
    parser = cli_group.build_parser()
    for ref in ["case", "control", "all"]:
        args = parser.parse_args(
            [
                "--case-dir",
                "case_dir",
                "--control-dir",
                "control_dir",
                "--output-dir",
                "out",
                "--canonical-reference",
                ref,
            ]
        )
        assert args.canonical_reference == ref
