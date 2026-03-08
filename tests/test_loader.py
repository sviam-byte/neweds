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
