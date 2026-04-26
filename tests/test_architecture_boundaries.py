"""Architecture boundary tests for the public package surface."""

from __future__ import annotations

import importlib

import neweds
from neweds.metrics.registry import list_metrics


def test_top_level_public_api_is_stable() -> None:
    expected = {
        "AnalysisConfig",
        "AnalysisResult",
        "ComputationContract",
        "DEFAULT_EDGE_THRESHOLD",
        "DEFAULT_MAX_LAG",
        "DEFAULT_PVALUE_ALPHA",
        "MetricResult",
        "PYINFORM_AVAILABLE",
        "WindowResult",
        "__version__",
        "run_analysis",
    }
    assert expected.issubset(set(neweds.__all__))


def test_pipeline_does_not_import_removed_legacy_engine() -> None:
    pipeline = importlib.import_module("neweds.core.pipeline")
    text = open(pipeline.__file__, encoding="utf-8").read()
    assert "BigMasterTool" not in text
    assert "from neweds.core.engine" not in text


def test_metric_runner_only_uses_registry() -> None:
    runner_src = open(
        importlib.import_module("neweds.core.metric_runner").__file__,
        encoding="utf-8",
    ).read()
    assert "BigMasterTool" not in runner_src
    assert "from neweds.metrics.registry" in runner_src


def test_metric_registry_contains_known_categories() -> None:
    metrics = list_metrics()
    assert metrics, "registry must not be empty"
    valid_categories = {"correlation", "information", "spectral", "ordinal", "causal"}
    for metric in metrics:
        assert metric.description, f"metric {metric.name!r} has empty description"
        assert metric.category in valid_categories
        assert callable(metric.func)


def test_reporting_exposes_public_writers() -> None:
    excel = importlib.import_module("neweds.reporting.excel_writer")
    html = importlib.import_module("neweds.reporting.html_generator")
    assert callable(excel.write_excel_report)
    assert callable(html.write_html_report)


def test_legacy_modules_have_been_removed() -> None:
    import importlib.util

    legacy_names = (
        "neweds.core.engine",
        "interfaces",
        "interfaces.cli",
        "interfaces.gui",
        "interfaces.web",
        "interfaces.legacy_cli",
    )
    for legacy in legacy_names:
        try:
            spec = importlib.util.find_spec(legacy)
        except ModuleNotFoundError:
            spec = None
        assert spec is None, f"{legacy} is still importable"
