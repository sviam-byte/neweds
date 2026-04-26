"""End-to-end integration test for the ``neweds`` CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _build_demo_csv(path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame(
        {
            "alpha": rng.normal(size=n),
            "beta": rng.normal(size=n),
            "gamma": rng.normal(size=n),
        }
    )
    df.to_csv(path, index=False)


def test_cli_writes_html_and_excel_reports(tmp_path: Path) -> None:
    csv_path = tmp_path / "demo.csv"
    _build_demo_csv(csv_path)
    out_dir = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "neweds.cli",
            str(csv_path),
            "--variants",
            "correlation_full,dcor_full",
            "--output-dir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr

    html = out_dir / "report.html"
    excel = out_dir / "report.xlsx"
    assert html.exists() and html.stat().st_size > 1024
    assert excel.exists() and excel.stat().st_size > 1024
