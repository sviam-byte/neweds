"""Tests for CSV loading behavior with and without headers."""

from pathlib import Path

from src.core import data_loader
from src.core.data_loader import read_input_table


def test_read_input_table_with_header(tmp_path: Path) -> None:
    """Header row should be detected in auto mode for typical CSV files."""
    csv_path = tmp_path / "with_header.csv"
    csv_path.write_text("time,a,b\n1,10,20\n2,11,21\n", encoding="utf-8")
    df = read_input_table(str(csv_path), header="auto")
    assert list(df.columns) == ["time", "a", "b"]
    assert df.shape == (2, 3)


def test_read_input_table_without_header(tmp_path: Path) -> None:
    """No-header mode should auto-generate column names c1..cn."""
    csv_path = tmp_path / "no_header.csv"
    csv_path.write_text("1,10,20\n2,11,21\n", encoding="utf-8")
    df = read_input_table(str(csv_path), header="no")
    assert list(df.columns) == ["c1", "c2", "c3"]
    assert df.shape == (2, 3)


def test_read_input_table_csv_usecols_limits_columns(tmp_path: Path) -> None:
    """CSV reader should honor explicit usecols to avoid loading all columns."""
    csv_path = tmp_path / "wide.csv"
    csv_path.write_text("1,2,3,4\n5,6,7,8\n", encoding="utf-8")

    df = read_input_table(str(csv_path), header="no", usecols=[0, 2])

    assert df.shape == (2, 2)
    assert list(df.columns) == ["c1", "c2"]


def test_load_or_generate_csv_auto_cap_by_feature_limit(tmp_path: Path) -> None:
    """Wide CSV should be capped before full read when feature_limit is configured."""
    csv_path = tmp_path / "very_wide.csv"
    csv_path.write_text("1,2,3,4,5,6\n7,8,9,10,11,12\n", encoding="utf-8")

    df = data_loader.load_or_generate(
        str(csv_path),
        header="no",
        time_col="none",
        transpose="no",
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
        feature_limit=3,
    )

    assert df.shape == (2, 3)


def test_load_mat_numeric_matrix(tmp_path):
    """MAT with top-level numeric matrix should load via common loader path."""
    import numpy as np
    from scipy.io import savemat

    fp = tmp_path / "demo.mat"
    savemat(fp, {"conn": np.arange(12, dtype=float).reshape(3, 4)})
    df = data_loader.load_or_generate(
        str(fp),
        header="no",
        time_col="none",
        transpose="no",
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
    )
    assert df.shape == (3, 4)


def test_read_input_table_mat_nested(tmp_path):
    """Nested MAT structs should be traversed and best numeric matrix selected."""
    import numpy as np
    from scipy.io import savemat

    fp = tmp_path / "nested.mat"
    savemat(fp, {"outer": {"ts": np.arange(6, dtype=float).reshape(2, 3)}})
    df = data_loader.read_input_table(str(fp))
    assert df.shape == (2, 3)


def test_load_h5_defaults_to_spatial_bins(tmp_path):
    """H5 4D inputs should be aggregated deterministically before heavy processing."""
    import h5py
    import numpy as np

    fp = tmp_path / "demo.h5"
    arr = np.zeros((4, 4, 4, 6), dtype=np.float32)
    arr[0:2, 0:2, 0:2, :] = 1.0
    arr[2:4, 2:4, 2:4, :] = np.arange(6, dtype=np.float32)

    with h5py.File(fp, "w") as f:
        f.create_dataset("timeseries", data=arr)

    df = data_loader.load_or_generate(
        str(fp),
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
    )
    assert df.attrs.get("format") == "spatial_bins"
    assert df.shape[0] == 6
    assert df.shape[1] >= 1


def test_load_h5_can_save_and_reuse_aggregated_h5(tmp_path):
    """Loader should persist aggregated H5 and reuse it on subsequent runs."""
    import h5py
    import numpy as np

    src = tmp_path / "subj1.h5"
    arr = np.zeros((4, 4, 4, 5), dtype=np.float32)
    arr[0:2, 0:2, 0:2, :] = 2.0
    arr[2:4, 2:4, 2:4, :] = np.arange(5, dtype=np.float32)

    with h5py.File(src, "w") as f:
        f.create_dataset("timeseries", data=arr)

    out_dir = tmp_path / "results" / "aggregated_h5"

    df_first = data_loader.load_or_generate(
        str(src),
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
        feature_sampling="spatial",
        h5_spatial_bin=2,
        save_aggregated_h5=True,
        reuse_existing_aggregated_h5=False,
        aggregated_h5_dir=str(out_dir),
    )

    agg_path = df_first.attrs.get("aggregated_h5_path")
    assert isinstance(agg_path, str) and agg_path.endswith("subj1.h5")
    assert Path(agg_path).exists()

    df_reused = data_loader.load_or_generate(
        str(src),
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
        feature_sampling="spatial",
        h5_spatial_bin=2,
        save_aggregated_h5=False,
        reuse_existing_aggregated_h5=True,
        aggregated_h5_dir=str(out_dir),
    )

    assert df_reused.attrs.get("source_kind") == "aggregated_h5"
    assert df_reused.attrs.get("format") == "spatial_bins"
    assert df_reused.shape == df_first.shape


def test_streaming_csv_voxel_wide_spatial_binning(tmp_path: Path) -> None:
    """Large-CSV path should support streaming deterministic spatial binning."""
    csv_path = tmp_path / "vox.csv"
    csv_path.write_text(
        "x,y,z,t0,t1\n"
        "55,0,0,1,3\n"
        "56,0,0,2,4\n"
        "57,0,0,6,8\n",
        encoding="utf-8",
    )

    df = data_loader.load_or_generate(
        str(csv_path),
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
        spatial_grid_size=2,
        spatial_grid_method="mean",
        csv_stream_spatial_bin=True,
        csv_chunk_rows=2,
    )

    assert df.attrs.get("source_kind") == "csv_voxel_spatial_stream"
    assert list(df.columns) == ["bin_27_0_0", "bin_28_0_0"]
    assert df.shape == (2, 2)
    assert float(df.iloc[0, 0]) == 1.0
    assert float(df.iloc[0, 1]) == 4.0


def test_streaming_csv_bin_names_are_stable_with_missing_voxels(tmp_path: Path) -> None:
    """Same voxel coordinates should map to the same bin names across files."""
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text(
        "x,y,z,t0,t1\n"
        "55,0,0,1,2\n"
        "56,0,0,3,4\n"
        "57,0,0,5,6\n",
        encoding="utf-8",
    )
    b.write_text(
        "x,y,z,t0,t1\n"
        "56,0,0,7,8\n"
        "57,0,0,9,10\n",
        encoding="utf-8",
    )

    dfa = data_loader.load_or_generate(
        str(a),
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
        spatial_grid_size=2,
        csv_stream_spatial_bin=True,
        csv_chunk_rows=2,
    )
    dfb = data_loader.load_or_generate(
        str(b),
        preprocess=False,
        normalize=False,
        remove_outliers=False,
        fill_missing=False,
        spatial_grid_size=2,
        csv_stream_spatial_bin=True,
        csv_chunk_rows=2,
    )

    assert "bin_28_0_0" in dfa.columns
    assert "bin_28_0_0" in dfb.columns
    assert "bin_27_0_0" in dfa.columns
    assert "bin_27_0_0" not in dfb.columns


def test_streaming_csv_fixed_range_keeps_empty_bins_as_nan(tmp_path: Path) -> None:
    """Fixed range should preserve empty bins so alignment stays deterministic."""
    csv_path = tmp_path / "fixed_range.csv"
    csv_path.write_text(
        "x,y,z,t0,t1\n"
        "0,0,0,1,2\n",
        encoding="utf-8",
    )

    df = data_loader.stream_csv_voxel_wide_to_timeseries(
        str(csv_path),
        spatial_bin_size=1,
        spatial_bin_range=((0, 1), (0, 0), (0, 0)),
        chunksize=1,
    )

    assert list(df.columns) == ["bin_0_0_0", "bin_1_0_0"]
    assert float(df.iloc[0, 0]) == 1.0
    assert float(df.iloc[1, 0]) == 2.0
    assert df["bin_1_0_0"].isna().all()
