"""Architecture-level tests for the public pipeline boundary."""

from __future__ import annotations

import pandas as pd

from neweds.config import AnalysisConfig
from neweds.core.metric_runner import compute_all_metrics, compute_metric
from neweds.core.pipeline import run_analysis
from neweds.core.results import AnalysisResult
from neweds.reporting.excel_writer import write_excel_report
from neweds.reporting.html_generator import write_html_report


def test_compute_metric_uses_registry_contract() -> None:
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 2, 3, 4]})

    one = compute_metric(df, "correlation_full")
    many = compute_all_metrics(df, ["correlation_full"])

    assert one.shape == (2, 2)
    assert "correlation_full" in many
    assert many["correlation_full"].shape == (2, 2)


def test_run_analysis_returns_structured_result(tmp_path) -> None:
    input_path = tmp_path / "series.csv"
    pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    ).to_csv(input_path, index=False)

    result = run_analysis(
        str(input_path),
        AnalysisConfig(
            max_lag=1,
            lag_selection="fixed",
            variants=["correlation_full"],
        ),
    )

    assert isinstance(result, AnalysisResult)
    assert result.input_info["path"] == str(input_path)
    assert "correlation_full" in result.metrics
    assert result.metrics["correlation_full"].matrix.shape == (3, 3)


def test_excel_writer_accepts_analysis_result(tmp_path) -> None:
    input_path = tmp_path / "series.csv"
    pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 3, 4, 5, 6]}).to_csv(
        input_path,
        index=False,
    )

    result = run_analysis(
        str(input_path),
        AnalysisConfig(
            max_lag=1,
            lag_selection="fixed",
            variants=["correlation_full"],
        ),
    )

    out = write_excel_report(result, str(tmp_path / "reports"))

    assert out.endswith("report.xlsx")
    assert (tmp_path / "reports" / "report.xlsx").exists()


def test_html_writer_accepts_analysis_result(tmp_path) -> None:
    input_path = tmp_path / "series.csv"
    pd.DataFrame({"a": [1, 2, 3, 4, 5, 6], "b": [1, 2, 3, 4, 5, 6]}).to_csv(
        input_path,
        index=False,
    )

    result = run_analysis(
        str(input_path),
        AnalysisConfig(
            max_lag=1,
            lag_selection="fixed",
            variants=["correlation_full"],
        ),
    )

    out = write_html_report(result, str(tmp_path / "reports"))

    assert out.endswith("report.html")
    assert (tmp_path / "reports" / "report.html").exists()
