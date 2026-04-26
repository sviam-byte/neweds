"""Batch analysis orchestration shared by CLI and other interfaces."""

from __future__ import annotations

import csv
import traceback
import zipfile
from pathlib import Path
from typing import Any

from ..config import AnalysisConfig
from ..reporting.excel_writer import write_excel_report
from ..reporting.html_generator import write_html_report
from .pipeline import run_analysis

SUPPORTED_INPUT_EXTS = (".csv", ".xlsx", ".xls", ".parquet")
MANIFEST_FIELDS = ["input_file", "status", "run_dir", "excel_path", "html_path", "error"]


def iter_supported_input_files(folder: str, recursive: bool = False) -> list[str]:
    """Return supported input files from a directory."""

    root = Path(folder)
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        str(path)
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_INPUT_EXTS
    )


def safe_slug(text: str) -> str:
    """Normalize a path fragment into a safe directory slug."""

    slug = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (text or "item"))
    slug = slug.strip("._")
    return slug or "item"


def write_batch_manifest(rows: list[dict[str, Any]], manifest_csv: str) -> None:
    """Write a stable CSV manifest for batch runs."""

    Path(manifest_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in MANIFEST_FIELDS})


def make_batch_zip(out_root: str) -> str:
    """Create a ZIP archive with the whole batch output directory."""

    out_root_p = Path(out_root)
    zip_path = str(out_root_p.with_suffix(".zip"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(out_root_p.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_root_p.parent))
    return zip_path


def run_one_analysis(
    input_file: str,
    output_dir: str,
    *,
    variants: list[str],
    lags: int = 1,
    lag_selection: str = "fixed",
    graph_threshold: float = 0.2,
    p_alpha: float = 0.05,
    write_html: bool = True,
    write_excel: bool = True,
) -> dict[str, str]:
    """Run one analysis job and return a manifest row."""

    cfg = AnalysisConfig(
        max_lag=int(lags),
        lag_selection=str(lag_selection),
        p_value_alpha=float(p_alpha),
        graph_threshold=float(graph_threshold),
        variants=list(variants),
    )
    result = run_analysis(input_file, cfg)

    html_path = ""
    excel_path = ""
    if write_html:
        html_path = write_html_report(
            result,
            output_dir,
            graph_threshold=float(graph_threshold),
            p_alpha=float(p_alpha),
        )
    if write_excel:
        excel_path = write_excel_report(
            result,
            output_dir,
            threshold=float(graph_threshold),
            p_value_alpha=float(p_alpha),
        )

    return {
        "input_file": input_file,
        "status": "ok",
        "run_dir": output_dir,
        "excel_path": excel_path,
        "html_path": html_path,
        "error": "",
    }


def run_batch(
    input_dir: str,
    output_dir: str,
    *,
    variants: list[str],
    recursive: bool = False,
    lags: int = 1,
    lag_selection: str = "fixed",
    graph_threshold: float = 0.2,
    p_alpha: float = 0.05,
    write_html: bool = True,
    write_excel: bool = True,
    create_zip: bool = False,
) -> tuple[list[dict[str, str]], str, str | None]:
    """Run analysis for every supported file and write a batch manifest."""

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    for input_file in iter_supported_input_files(input_dir, recursive=recursive):
        run_dir = output_root / safe_slug(Path(input_file).stem)
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            rows.append(
                run_one_analysis(
                    input_file,
                    str(run_dir),
                    variants=variants,
                    lags=lags,
                    lag_selection=lag_selection,
                    graph_threshold=graph_threshold,
                    p_alpha=p_alpha,
                    write_html=write_html,
                    write_excel=write_excel,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "input_file": input_file,
                    "status": "error",
                    "run_dir": str(run_dir),
                    "excel_path": "",
                    "html_path": "",
                    "error": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                }
            )

    manifest_path = str(output_root / "batch_manifest.csv")
    write_batch_manifest(rows, manifest_path)
    zip_path = make_batch_zip(str(output_root)) if create_zip else None
    return rows, manifest_path, zip_path
