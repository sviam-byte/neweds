from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from scripts.write_result_provenance import write_result_provenance


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_write_result_provenance_creates_complete_result_card(tmp_path: Path) -> None:
    result_dir = tmp_path / "new_results" / "example-result"
    output = result_dir / "tables" / "summary.csv"
    output.parent.mkdir(parents=True)
    output.write_text("metric,value\nexample,1\n", encoding="utf-8")

    code_file = tmp_path / "analysis.py"
    code_file.write_text("print('analysis')\n", encoding="utf-8")

    metadata = write_result_provenance(
        result_dir=result_dir,
        result_id="example-result",
        title="Пример результата",
        result_type="test",
        status="completed",
        execution_mode="fresh_run",
        summary="Проверена запись полной карточки результата.",
        meaning="Карточка делает запуск проверяемым и воспроизводимым.",
        command="python analysis.py",
        inputs=["input/example.csv"],
        code_files=[code_file],
        repository=None,
        findings=["Создан тестовый результат."],
        limitations=["Это только модульный тест."],
        source_time_start="2026-06-23T10:00:00+03:00",
        source_time_end="2026-06-23T10:01:00+03:00",
        notes=["Тестовая запись."],
    )

    about_path = result_dir / "ABOUT_THIS_RESULT.md"
    metadata_path = result_dir / "run_metadata.json"
    inventory_path = result_dir / "output_inventory.csv"

    assert about_path.is_file()
    assert metadata_path.is_file()
    assert inventory_path.is_file()
    assert (result_dir / "code_snapshot" / "files" / "analysis.py").is_file()
    assert "Пример результата" in about_path.read_text(encoding="utf-8")

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata == metadata
    assert saved_metadata["result_id"] == "example-result"
    assert saved_metadata["git"]["available"] is False

    with inventory_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_path = {row["path"]: row for row in rows}

    assert "ABOUT_THIS_RESULT.md" in by_path
    assert "run_metadata.json" in by_path
    assert "tables/summary.csv" in by_path
    assert "code_snapshot/files/analysis.py" in by_path
    assert "output_inventory.csv" not in by_path
    assert len(rows) == saved_metadata["integrity_manifest_file_count"]

    for relative, row in by_path.items():
        assert row["sha256"] == _sha256(result_dir / relative)
