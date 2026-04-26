"""Architecture boundary tests."""

from __future__ import annotations

from pathlib import Path


def read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_metric_runner_does_not_import_legacy_engine() -> None:
    text = read("src/core/metric_runner.py")
    assert "src.core.engine" not in text
    assert "BigMasterTool" not in text


def test_public_pipeline_does_not_import_legacy_engine() -> None:
    text = read("src/core/pipeline.py")
    assert "src.core.engine" not in text
    assert "BigMasterTool" not in text
    assert "legacy_adapter" not in text


def test_metrics_registry_does_not_import_legacy_engine() -> None:
    text = read("src/metrics/registry.py")
    assert "src.core.engine" not in text
    assert "BigMasterTool" not in text


def test_reporting_has_public_wrappers() -> None:
    assert "def write_excel_report" in read("src/reporting/excel_writer.py")
    assert "def write_html_report" in read("src/reporting/html_generator.py")


def test_main_cli_has_modern_pipeline_path() -> None:
    text = read("interfaces/cli.py")
    assert "from src.core.engine import BigMasterTool" not in text
    assert "run_analysis" in text


def test_engine_is_marked_legacy() -> None:
    text = read("src/core/engine.py")
    assert "Legacy analysis engine" in text
