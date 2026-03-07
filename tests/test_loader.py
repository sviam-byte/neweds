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
