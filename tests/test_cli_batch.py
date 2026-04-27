"""Тесты пакетного запуска (batch_pipeline)."""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from neweds.core.batch_pipeline import (
    iter_supported_input_files,
    make_batch_zip,
    write_batch_manifest,
)


def test_iter_supported_input_files_recursive(tmp_path: Path) -> None:
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "b.xlsx").write_text("fake", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.parquet").write_text("fake", encoding="utf-8")
    (nested / "ignore.txt").write_text("nope", encoding="utf-8")

    flat = iter_supported_input_files(str(tmp_path), recursive=False)
    rec = iter_supported_input_files(str(tmp_path), recursive=True)

    assert [Path(item).name for item in flat] == ["a.csv", "b.xlsx"]
    assert [Path(item).name for item in rec] == ["a.csv", "b.xlsx", "c.parquet"]


def test_write_manifest_and_zip(tmp_path: Path) -> None:
    out_root = tmp_path / "analysis_results"
    run_dir = out_root / "sample"
    run_dir.mkdir(parents=True)
    (run_dir / "result.txt").write_text("ok", encoding="utf-8")

    manifest = out_root / "batch_manifest.csv"
    write_batch_manifest(
        [
            {
                "input_file": "input.csv",
                "status": "ok",
                "run_dir": str(run_dir),
                "excel_path": "",
                "html_path": "",
                "error": "",
            }
        ],
        str(manifest),
    )

    with open(manifest, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"

    zip_path = make_batch_zip(str(out_root))
    assert Path(zip_path).exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert any(name.endswith("analysis_results/sample/result.txt") for name in names)
    assert any(name.endswith("analysis_results/batch_manifest.csv") for name in names)
