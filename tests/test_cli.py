"""Tests for the public CLI helpers."""

from __future__ import annotations

from pathlib import Path

from neweds.cli import _split_csv
from neweds.core.batch_pipeline import iter_supported_input_files


def test_iter_supported_input_files_filters_extensions(tmp_path: Path) -> None:
    supported = ["sample.csv", "sample.xlsx", "sample.xls", "sample.parquet"]
    unsupported = ["sample.mat", "sample.h5", "sample.hdf5", "sample.txt"]
    for name in supported + unsupported:
        (tmp_path / name).write_text("x", encoding="utf-8")

    files = iter_supported_input_files(str(tmp_path))

    assert [Path(item).name for item in files] == sorted(supported)


def test_split_csv_handles_whitespace_and_empties() -> None:
    assert _split_csv("a, b ,, c") == ["a", "b", "c"]
    assert _split_csv("") == []
    assert _split_csv(None) == []
