from __future__ import annotations

import csv
import zipfile
from pathlib import Path

from interfaces import cli


def test_iter_supported_input_files_recursive(tmp_path: Path):
    """CLI должен находить поддерживаемые типы файлов в flat/recursive режимах."""
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (tmp_path / "b.xlsx").write_text("fake", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "c.mat").write_text("fake", encoding="utf-8")
    (nested / "ignore.txt").write_text("nope", encoding="utf-8")

    flat = cli._iter_supported_input_files(str(tmp_path), recursive=False)
    rec = cli._iter_supported_input_files(str(tmp_path), recursive=True)

    assert [Path(x).name for x in flat] == ["a.csv", "b.xlsx"]
    assert [Path(x).name for x in rec] == ["a.csv", "b.xlsx", "c.mat"]


def test_write_manifest_and_zip(tmp_path: Path):
    """Проверка записи batch manifest и упаковки результатов в ZIP."""
    out_root = tmp_path / "analysis_results"
    run_dir = out_root / "sample"
    run_dir.mkdir(parents=True)
    sample_file = run_dir / "result.txt"
    sample_file.write_text("ok", encoding="utf-8")

    manifest = out_root / "batch_manifest.csv"
    cli._write_batch_manifest(
        [
            {
                "input_file": "input.csv",
                "status": "ok",
                "run_dir": str(run_dir),
                "excel_path": "",
                "html_path": "",
                "series_path": "",
                "error": "",
            }
        ],
        str(manifest),
    )

    with open(manifest, "r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "ok"

    zip_path = cli._make_batch_zip(str(out_root))
    assert Path(zip_path).exists()
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = set(zf.namelist())
    assert any(name.endswith("analysis_results/sample/result.txt") for name in names)
    assert any(name.endswith("analysis_results/batch_manifest.csv") for name in names)
