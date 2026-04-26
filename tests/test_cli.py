from __future__ import annotations

from pathlib import Path

from interfaces.cli import _iter_supported_input_files


def test_iter_supported_input_files_excludes_unimplemented_formats(tmp_path: Path) -> None:
    supported = [
        "sample.csv",
        "sample.xlsx",
        "sample.xls",
        "sample.parquet",
    ]
    unsupported = [
        "sample.mat",
        "sample.h5",
        "sample.hdf5",
        "sample.txt",
    ]

    for name in supported + unsupported:
        (tmp_path / name).write_text("x", encoding="utf-8")

    files = _iter_supported_input_files(str(tmp_path))

    assert [Path(item).name for item in files] == sorted(supported)
