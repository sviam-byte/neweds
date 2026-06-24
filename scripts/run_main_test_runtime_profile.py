from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from neweds.config import AnalysisConfig
from neweds.core.pipeline import run_analysis
from neweds.metrics.registry import list_metrics
from neweds.reporting.excel_writer import write_excel_report
from neweds.reporting.html_generator import write_html_report


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_clean(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _mean_abs_offdiag(matrix: np.ndarray) -> float | None:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.size == 0:
        return None
    mask = ~np.eye(arr.shape[0], dtype=bool)
    vals = np.abs(arr[mask])
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else None


def _metric_summary(result) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, item in result.metrics.items():
        matrix = np.asarray(item.matrix, dtype=float)
        finite = np.isfinite(matrix)
        out[name] = {
            "shape": list(matrix.shape),
            "directed": bool(item.directed),
            "lag": item.lag,
            "pvalue_based": bool(item.pvalue_based),
            "finite_fraction": float(finite.mean()) if finite.size else 0.0,
            "min": float(np.nanmin(matrix)) if finite.any() else None,
            "max": float(np.nanmax(matrix)) if finite.any() else None,
            "mean_abs_offdiag": _mean_abs_offdiag(matrix),
            "category": item.metadata.get("category"),
            "experimental": item.metadata.get("experimental"),
            "partial_mode": item.metadata.get("partial_mode"),
        }
    return out


def _window_summary(result) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for variant, payload in result.windows.items():
        sizes = payload.get("sizes", {}) if isinstance(payload, dict) else {}
        out[variant] = {
            "policy": payload.get("policy") if isinstance(payload, dict) else None,
            "stride": payload.get("stride") if isinstance(payload, dict) else None,
            "sizes": {},
        }
        for size, info in sizes.items():
            ticks = info.get("ticks", []) if isinstance(info, dict) else []
            best = info.get("best_window", {}) if isinstance(info, dict) else {}
            out[variant]["sizes"][str(size)] = {
                "tick_count": len(ticks),
                "best_window": {
                    "start": best.get("start"),
                    "end": best.get("end"),
                    "metric": best.get("metric"),
                },
                "extremes": info.get("extremes", {}) if isinstance(info, dict) else {},
            }
    return out


def _write_timing(root: Path, runs: list[dict[str, Any]], repeats_requested: int) -> None:
    runtime_dir = root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    with (runtime_dir / "timing_runs.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repeat",
                "status",
                "total_seconds",
                "analysis_seconds",
                "html_seconds",
                "excel_seconds",
                "error",
            ],
        )
        writer.writeheader()
        for row in runs:
            stages = row.get("stages", {})
            writer.writerow(
                {
                    "repeat": row.get("repeat"),
                    "status": row.get("status"),
                    "total_seconds": row.get("total_seconds"),
                    "analysis_seconds": stages.get("analysis"),
                    "html_seconds": stages.get("html"),
                    "excel_seconds": stages.get("excel"),
                    "error": row.get("error", ""),
                }
            )

    with (runtime_dir / "stage_timing.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["repeat", "stage", "seconds"])
        writer.writeheader()
        for row in runs:
            for stage, seconds in row.get("stages", {}).items():
                writer.writerow({"repeat": row.get("repeat"), "stage": stage, "seconds": seconds})

    ok = [float(r["total_seconds"]) for r in runs if r.get("status") == "ok"]
    mean = float(np.mean(ok)) if ok else None
    summary = {
        "repeats_requested": int(repeats_requested),
        "repeats_completed": len(runs),
        "ok_repeats": len(ok),
        "total_seconds": {
            "runs": ok,
            "mean": mean,
            "std": float(np.std(ok, ddof=1)) if len(ok) > 1 else 0.0 if ok else None,
            "min": float(np.min(ok)) if ok else None,
            "max": float(np.max(ok)) if ok else None,
            "spread_abs": float(np.max(ok) - np.min(ok)) if ok else None,
            "spread_pct": float((np.max(ok) - np.min(ok)) / mean * 100.0)
            if ok and mean
            else None,
        },
        "runs": runs,
    }
    _write_json(runtime_dir / "runtime_summary.json", summary)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="examples/demo_timeseries.csv")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--window-sizes", default="60,120")
    parser.add_argument("--window-stride", type=int, default=60)
    parser.add_argument("--max-lag", type=int, default=3)
    parser.add_argument("--lag-selection", choices=["fixed", "optimize"], default="fixed")
    parser.add_argument("--exclude-ah", action="store_true")
    parser.add_argument("--no-reports", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    root = Path(args.out_root)
    root.mkdir(parents=True, exist_ok=True)

    variants = [metric.name for metric in list_metrics()]
    if args.exclude_ah:
        variants = [name for name in variants if not name.startswith("ah_")]
    window_sizes = [int(item.strip()) for item in str(args.window_sizes).split(",") if item.strip()]

    config = AnalysisConfig(
        max_lag=int(args.max_lag),
        lag_selection=str(args.lag_selection),
        variants=variants,
        window_sizes=window_sizes,
        window_stride=int(args.window_stride),
        heavy_window_max_windows=3,
        performance_guardrails=True,
    )
    profile = {
        "profile": "full_limited_runtime",
        "input": str(args.input),
        "variants": variants,
        "window_sizes": window_sizes,
        "window_stride": int(args.window_stride),
        "max_lag": int(args.max_lag),
        "lag_selection": str(args.lag_selection),
        "classification": {"status": "skipped", "reason": "input labels were not supplied"},
        "permutation_p_test": False,
        "reports": not bool(args.no_reports),
    }
    _write_json(root / "profile.json", profile)

    print(f"OUT_ROOT={root}", flush=True)
    print(f"VARIANTS={len(variants)} {' '.join(variants)}", flush=True)

    runs: list[dict[str, Any]] = []
    for repeat in range(1, int(args.repeats) + 1):
        run_dir = root / f"run_{repeat}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"RUN {repeat}/{args.repeats} start", flush=True)

        row: dict[str, Any] = {"repeat": repeat, "status": "error", "stages": {}}
        t_total = time.perf_counter()
        try:
            t0 = time.perf_counter()
            result = run_analysis(str(args.input), config)
            row["stages"]["analysis"] = time.perf_counter() - t0

            html_path = ""
            excel_path = ""
            if not args.no_reports:
                t0 = time.perf_counter()
                html_path = write_html_report(
                    result,
                    str(run_dir),
                    graph_threshold=0.2,
                    p_alpha=0.05,
                )
                row["stages"]["html"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                excel_path = write_excel_report(
                    result,
                    str(run_dir),
                    threshold=0.2,
                    p_value_alpha=0.05,
                )
                row["stages"]["excel"] = time.perf_counter() - t0

            _write_json(
                run_dir / "run_summary.json",
                {
                    "repeat": repeat,
                    "input_info": result.input_info,
                    "metric_summary": _metric_summary(result),
                    "window_summary": _window_summary(result),
                    "logs": result.logs,
                    "html_path": html_path,
                    "excel_path": excel_path,
                    "classification": profile["classification"],
                },
            )
            row["status"] = "ok"
            row["html_path"] = html_path
            row["excel_path"] = excel_path
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
            (run_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"RUN {repeat}/{args.repeats} error: {row['error']}", flush=True)
        finally:
            row["total_seconds"] = time.perf_counter() - t_total
            runs.append(row)
            _write_timing(root, runs, int(args.repeats))
            print(
                f"RUN {repeat}/{args.repeats} done status={row['status']} "
                f"seconds={row['total_seconds']:.3f}",
                flush=True,
            )

    print("DONE", flush=True)
    return 0 if any(row.get("status") == "ok" for row in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
