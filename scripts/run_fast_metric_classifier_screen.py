from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from neweds.core.fmri_stage2_benchmark import (  # noqa: E402
    discover_hcp_inputs,
    load_regional_array,
    preprocess_regional_timeseries,
    preprocessing_branch_configs,
    registry_snapshot,
)
from neweds.core.metric_runner import compute_metric  # noqa: E402
from neweds.metrics.registry import get_metric, list_metrics  # noqa: E402
from scripts.write_result_provenance import write_result_provenance  # noqa: E402


HEAVY_PREFIXES = (
    "mutinf_",
    "dcor_",
    "ah_",
    "granger_",
    "te_",
    "coherence_",
)

FEATURE_COLUMNS = (
    "value_count",
    "finite_fraction",
    "mean",
    "std",
    "median",
    "q05",
    "q25",
    "q75",
    "q95",
    "min",
    "max",
    "mean_abs",
    "std_abs",
    "q95_abs",
    "positive_fraction",
    "negative_fraction",
    "zero_fraction",
    "node_strength_mean",
    "node_strength_std",
    "node_strength_q95",
    "node_strength_max",
    "p_mean",
    "p_median",
    "p_min",
    "p_lt_005_fraction",
    "neglog10p_mean",
    "neglog10p_q95",
)


@dataclass(frozen=True, slots=True)
class SubjectSeries:
    scope: str
    scope_kind: str
    subject_id: str
    group: str
    values_time_node: np.ndarray
    source_path: str
    array_key: str = ""
    note: str = ""


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in str(text).split(",") if item.strip()]


def _window_tasks(window_sizes: list[int], window_starts: list[int]) -> list[tuple[int, int]]:
    tasks: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for size in window_sizes:
        starts = [0] if int(size) <= 0 else window_starts
        for start in starts:
            key = (int(size), int(start))
            if key not in seen:
                tasks.append(key)
                seen.add(key)
    return tasks


def _make_result_dir(root: Path, result_date: str, slug: str) -> Path:
    base = root / f"{result_date}_{slug}"
    if not base.exists():
        base.mkdir(parents=True)
        return base
    stamped = root / f"{result_date}_{slug}_{datetime.now().strftime('%H%M%S')}"
    stamped.mkdir(parents=True)
    return stamped


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    for col in range(arr.shape[1]):
        x = arr[:, col]
        finite = np.isfinite(x)
        if int(finite.sum()) < 4:
            continue
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x))
        if not np.isfinite(std) or std <= 1e-12:
            continue
        out[finite, col] = (x[finite] - mean) / std
    return out


def _sample_subject_paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    paths = sorted(paths, key=lambda p: p.name)
    if limit <= 0 or len(paths) <= limit:
        return paths
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.arange(len(paths)), size=int(limit), replace=False)
    return [paths[int(idx)] for idx in sorted(selected)]


def _sample_manifest(manifests: list[Any], limit_per_group: int, seed: int) -> list[Any]:
    if limit_per_group <= 0:
        return sorted(manifests, key=lambda m: (m.group, m.subject_id, m.representation))
    out: list[Any] = []
    for group in ("HC", "SZ"):
        group_items = [m for m in manifests if str(m.group) == group]
        group_items = sorted(group_items, key=lambda m: str(m.subject_id))
        if len(group_items) > limit_per_group:
            rng = np.random.default_rng(seed + (0 if group == "HC" else 10_000))
            idx = rng.choice(np.arange(len(group_items)), size=int(limit_per_group), replace=False)
            group_items = [group_items[int(i)] for i in sorted(idx)]
        out.extend(group_items)
    return out


def _subject_id_from_aal3(path: Path) -> str:
    return path.name.split("_", 1)[0]


def load_aal3_scope(root: Path, max_subjects_per_group: int, seed: int) -> tuple[list[SubjectSeries], list[dict[str, Any]]]:
    rows: list[SubjectSeries] = []
    inventory: list[dict[str, Any]] = []
    for group, folder in (("HC", "Group_HC"), ("SZ", "Group_SZ")):
        group_dir = root / folder
        paths = sorted(group_dir.glob("*_AAL3_timeseries.csv"))
        selected = _sample_subject_paths(paths, max_subjects_per_group, seed + (1 if group == "HC" else 2))
        inventory.append(
            {
                "scope": "AAL3_whole_roi",
                "group": group,
                "available_subjects": len(paths),
                "selected_subjects": len(selected),
                "status": "ok" if selected else "missing",
                "source": str(group_dir),
            }
        )
        for path in selected:
            data = pd.read_csv(path).to_numpy(dtype=np.float64)
            if data.shape[0] < data.shape[1]:
                data = data.T
            data = _zscore_columns(data)
            rows.append(
                SubjectSeries(
                    scope="AAL3_whole_roi",
                    scope_kind="whole_roi",
                    subject_id=_subject_id_from_aal3(path),
                    group=group,
                    values_time_node=data,
                    source_path=str(path),
                    note="AAL3 file transposed when original shape was ROI x time.",
                )
            )
    return rows, inventory


def load_hcp_scopes(
    hcp_result: Path,
    scopes: set[str],
    max_subjects_per_group: int,
    seed: int,
) -> tuple[list[SubjectSeries], list[dict[str, Any]]]:
    representation_by_scope = {
        "HCP360_GM_active_mean": "gm_active_mean",
        "HCP360_whole_brain": "whole_brain",
    }
    requested_reps = [
        rep for scope, rep in representation_by_scope.items() if scope in scopes
    ]
    if not requested_reps:
        return [], []

    manifests, alignment = discover_hcp_inputs(hcp_result, representations=requested_reps)
    branch = preprocessing_branch_configs(("baseline_without_GSR",))[0]
    rows: list[SubjectSeries] = []
    inventory: list[dict[str, Any]] = []
    for scope, representation in representation_by_scope.items():
        if scope not in scopes:
            continue
        rep_manifests = [m for m in manifests if m.representation == representation]
        selected = _sample_manifest(rep_manifests, max_subjects_per_group, seed + len(scope))
        by_group = Counter(str(m.group) for m in selected)
        available_by_group = Counter(str(m.group) for m in rep_manifests)
        inventory.extend(
            {
                "scope": scope,
                "group": group,
                "available_subjects": int(available_by_group.get(group, 0)),
                "selected_subjects": int(by_group.get(group, 0)),
                "status": "ok" if by_group.get(group, 0) else "missing",
                "source": str(hcp_result),
            }
            for group in ("HC", "SZ")
        )
        for manifest in selected:
            raw_nodes_time = load_regional_array(manifest)
            processed = preprocess_regional_timeseries(raw_nodes_time, branch)
            rows.append(
                SubjectSeries(
                    scope=scope,
                    scope_kind="gm_only" if representation != "whole_brain" else "whole_brain",
                    subject_id=str(manifest.subject_id),
                    group=str(manifest.group),
                    values_time_node=processed,
                    source_path=str(manifest.npz_path),
                    array_key=str(manifest.array_key),
                    note="HCP360 baseline_without_GSR preprocessing from Stage2 utilities.",
                )
            )
    inventory.append(
        {
            "scope": "HCP360_alignment",
            "group": "all",
            "available_subjects": int(len(alignment)),
            "selected_subjects": int(alignment["paired_stage2_eligible"].sum()) if not alignment.empty else 0,
            "status": "ok" if not alignment.empty else "missing",
            "source": str(hcp_result / "subject_status.jsonl"),
        }
    )
    return rows, inventory


def load_tissue_mean_scope(
    tissue_result: Path,
    scopes: set[str],
    max_subjects_per_group: int,
    seed: int,
) -> tuple[list[SubjectSeries], list[dict[str, Any]]]:
    scope = "tissue_mean_GM_WM_CSF"
    if scope not in scopes:
        return [], []
    path = tissue_result / "temporal" / "tissue_mean_timeseries.csv"
    if not path.is_file():
        return [], [
            {
                "scope": scope,
                "group": "all",
                "available_subjects": 0,
                "selected_subjects": 0,
                "status": "missing",
                "source": str(path),
            }
        ]
    table = pd.read_csv(path)
    rows: list[SubjectSeries] = []
    inventory: list[dict[str, Any]] = []
    for group in ("HC", "SZ"):
        subject_ids = sorted(table.loc[table["group"].eq(group), "subject_id"].astype(str).unique())
        if max_subjects_per_group > 0 and len(subject_ids) > max_subjects_per_group:
            rng = np.random.default_rng(seed + (31 if group == "HC" else 32))
            idx = rng.choice(np.arange(len(subject_ids)), size=int(max_subjects_per_group), replace=False)
            subject_ids = [subject_ids[int(i)] for i in sorted(idx)]
        inventory.append(
            {
                "scope": scope,
                "group": group,
                "available_subjects": int(table.loc[table["group"].eq(group), "subject_id"].nunique()),
                "selected_subjects": len(subject_ids),
                "status": "ok" if subject_ids else "missing",
                "source": str(path),
            }
        )
        for subject_id in subject_ids:
            sub = table[
                table["group"].eq(group) & table["subject_id"].astype(str).eq(str(subject_id))
            ].sort_values("time_index")
            columns = [col for col in ("GM", "WM", "CSF") if col in sub.columns]
            values = _zscore_columns(sub[columns].to_numpy(dtype=np.float64))
            rows.append(
                SubjectSeries(
                    scope=scope,
                    scope_kind="tissue_mean",
                    subject_id=str(subject_id),
                    group=group,
                    values_time_node=values,
                    source_path=str(path),
                    array_key=",".join(columns),
                    note="Tissue-mean scope has 3 mean signals, not regional WM connectivity.",
                )
            )
    return rows, inventory


def _valid_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    valid = np.zeros(arr.shape[1], dtype=bool)
    for col in range(arr.shape[1]):
        x = arr[:, col]
        finite = np.isfinite(x)
        valid[col] = bool(int(finite.sum()) >= 8 and float(np.nanstd(x[finite])) > 1e-12)
    return valid


def _window_slice(values: np.ndarray, start: int, size: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if size <= 0:
        return arr
    start = int(max(0, start))
    end = int(min(arr.shape[0], start + size))
    return arr[start:end, :]


def _pair_cache_key(n: int, directed: bool, count: int, seed: int) -> tuple[int, bool, int, int]:
    return int(n), bool(directed), int(count), int(seed)


def _sample_pairs(
    n: int,
    *,
    directed: bool,
    count: int,
    seed: int,
    cache: dict[tuple[int, bool, int, int], list[tuple[int, int]]],
) -> list[tuple[int, int]]:
    key = _pair_cache_key(n, directed, count, seed)
    if key in cache:
        return cache[key]
    if directed:
        all_pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    else:
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if count <= 0 or count >= len(all_pairs):
        pairs = all_pairs
    else:
        rng = np.random.default_rng(seed + n + (100_000 if directed else 0))
        idx = rng.choice(np.arange(len(all_pairs)), size=int(count), replace=False)
        pairs = [all_pairs[int(i)] for i in sorted(idx)]
    cache[key] = pairs
    return pairs


def _values_from_pairs(matrix: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    values = []
    for i, j in pairs:
        if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1]:
            values.append(float(arr[i, j]))
    return np.asarray(values, dtype=np.float64)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return math.nan
    return float(np.nanquantile(finite, q))


def _metric_features(
    values: np.ndarray,
    pairs: list[tuple[int, int]],
    n_nodes: int,
    *,
    pvalue_based: bool,
) -> dict[str, float]:
    out = {name: math.nan for name in FEATURE_COLUMNS}
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(vals)
    finite = vals[finite_mask]
    out["value_count"] = float(vals.size)
    out["finite_fraction"] = float(finite_mask.mean()) if vals.size else 0.0
    if finite.size == 0:
        return out
    out.update(
        {
            "mean": float(np.nanmean(finite)),
            "std": float(np.nanstd(finite)),
            "median": float(np.nanmedian(finite)),
            "q05": _safe_quantile(finite, 0.05),
            "q25": _safe_quantile(finite, 0.25),
            "q75": _safe_quantile(finite, 0.75),
            "q95": _safe_quantile(finite, 0.95),
            "min": float(np.nanmin(finite)),
            "max": float(np.nanmax(finite)),
            "mean_abs": float(np.nanmean(np.abs(finite))),
            "std_abs": float(np.nanstd(np.abs(finite))),
            "q95_abs": _safe_quantile(np.abs(finite), 0.95),
            "positive_fraction": float(np.mean(finite > 0)),
            "negative_fraction": float(np.mean(finite < 0)),
            "zero_fraction": float(np.mean(np.isclose(finite, 0.0))),
        }
    )
    strengths = np.zeros(int(n_nodes), dtype=np.float64)
    counts = np.zeros(int(n_nodes), dtype=np.float64)
    for value, (i, j) in zip(vals, pairs):
        if not np.isfinite(value):
            continue
        abs_value = abs(float(value))
        strengths[i] += abs_value
        counts[i] += 1.0
        strengths[j] += abs_value
        counts[j] += 1.0
    valid_strength = counts > 0
    if valid_strength.any():
        node_values = strengths[valid_strength] / counts[valid_strength]
        out["node_strength_mean"] = float(np.mean(node_values))
        out["node_strength_std"] = float(np.std(node_values))
        out["node_strength_q95"] = _safe_quantile(node_values, 0.95)
        out["node_strength_max"] = float(np.max(node_values))
    if pvalue_based:
        clipped = np.clip(finite, 1e-300, 1.0)
        neglog = -np.log10(clipped)
        out["p_mean"] = float(np.nanmean(clipped))
        out["p_median"] = float(np.nanmedian(clipped))
        out["p_min"] = float(np.nanmin(clipped))
        out["p_lt_005_fraction"] = float(np.mean(clipped < 0.05))
        out["neglog10p_mean"] = float(np.nanmean(neglog))
        out["neglog10p_q95"] = _safe_quantile(neglog, 0.95)
    return out


def _global_control(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=1)
    sums = np.where(finite, arr, 0.0).sum(axis=1)
    mean = np.divide(sums, counts, out=np.zeros(arr.shape[0], dtype=np.float64), where=counts > 0)
    return mean.reshape(-1, 1)


def _ar1(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x)
    if int(finite.sum()) < 4:
        return math.nan
    x = x[finite]
    if np.std(x[:-1]) <= 1e-12 or np.std(x[1:]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def compute_feature_rows(
    subjects: list[SubjectSeries],
    result_dir: Path,
    *,
    metric_names: list[str],
    lags: list[int],
    window_sizes: list[int],
    window_starts: list[int],
    max_pairs: int,
    heavy_max_pairs: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feature_rows: list[dict[str, Any]] = []
    status_rows: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    pair_cache: dict[tuple[int, bool, int, int], list[tuple[int, int]]] = {}
    tasks = _window_tasks(window_sizes, window_starts)
    total = len(subjects) * len(metric_names) * len(lags) * len(tasks)
    done = 0
    started = time.perf_counter()

    for subject in subjects:
        for window_size, start in tasks:
            window = _window_slice(subject.values_time_node, start, window_size)
            valid = _valid_columns(window)
            selected = window[:, valid]
            finite_rows = np.isfinite(selected).all(axis=1) if selected.size else np.zeros(0, dtype=bool)
            selected = selected[finite_rows]
            label = f"w{selected.shape[0]}_start{int(start)}" if window_size > 0 else "full"
            control = _global_control(selected) if selected.size else np.empty((0, 1), dtype=np.float64)
            qc_rows.append(
                {
                    "scope": subject.scope,
                    "subject_id": subject.subject_id,
                    "group": subject.group,
                    "window_label": label,
                    "window_start": int(start),
                    "window_size_requested": int(window_size),
                    "n_timepoints_used": int(selected.shape[0]),
                    "n_nodes_total": int(window.shape[1]) if window.ndim == 2 else 0,
                    "n_nodes_valid": int(selected.shape[1]) if selected.ndim == 2 else 0,
                    "global_std": float(np.nanstd(control[:, 0])) if control.size else math.nan,
                    "global_ar1": _ar1(control[:, 0]) if control.size else math.nan,
                }
            )
            if selected.shape[0] < 16 or selected.shape[1] < 2:
                for metric_name in metric_names:
                    for lag in lags:
                        status_rows.append(
                            {
                                "scope": subject.scope,
                                "subject_id": subject.subject_id,
                                "group": subject.group,
                                "window_label": label,
                                "lag": int(lag),
                                "metric": metric_name,
                                "status": "skipped_invalid_window",
                                "seconds": 0.0,
                                "message": "too few valid timepoints or nodes",
                            }
                        )
                continue

            frame = pd.DataFrame(selected, columns=[f"v{i:03d}" for i in range(selected.shape[1])])
            for lag in lags:
                for metric_name in metric_names:
                    done += 1
                    metric = get_metric(metric_name)
                    pair_limit = heavy_max_pairs if metric_name.startswith(HEAVY_PREFIXES) else max_pairs
                    pairs = _sample_pairs(
                        frame.shape[1],
                        directed=bool(metric.directed),
                        count=int(pair_limit),
                        seed=seed + int(lag) * 1009 + len(metric_name),
                        cache=pair_cache,
                    )
                    t0 = time.perf_counter()
                    status = "ok"
                    message = ""
                    features = {name: math.nan for name in FEATURE_COLUMNS}
                    try:
                        params: dict[str, Any] = {
                            "pairs": pairs,
                            "max_pairwise_pairs": max(1, len(pairs)),
                            "performance_guardrails": True,
                        }
                        if metric.supports_control:
                            params["control_matrix"] = control
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            matrix = compute_metric(frame, metric_name, lag=int(lag), **params)
                        values = _values_from_pairs(np.asarray(matrix, dtype=np.float64), pairs)
                        features = _metric_features(
                            values,
                            pairs,
                            frame.shape[1],
                            pvalue_based=bool(metric.pvalue_based),
                        )
                    except Exception as exc:  # noqa: BLE001 - scientific audit should ledger failures
                        status = "failed"
                        message = f"{type(exc).__name__}: {exc}"
                    seconds = time.perf_counter() - t0
                    status_rows.append(
                        {
                            "scope": subject.scope,
                            "scope_kind": subject.scope_kind,
                            "subject_id": subject.subject_id,
                            "group": subject.group,
                            "window_label": label,
                            "window_start": int(start),
                            "window_size_requested": int(window_size),
                            "lag": int(lag),
                            "metric": metric_name,
                            "category": metric.category,
                            "directed": bool(metric.directed),
                            "pvalue_based": bool(metric.pvalue_based),
                            "supports_control": bool(metric.supports_control),
                            "partial_control": "global_signal" if metric.supports_control else "",
                            "pairs_requested": int(pair_limit),
                            "pairs_used": int(len(pairs)),
                            "n_nodes": int(frame.shape[1]),
                            "n_timepoints": int(frame.shape[0]),
                            "status": status,
                            "seconds": float(seconds),
                            "message": message,
                        }
                    )
                    if status == "ok":
                        feature_rows.append(
                            {
                                "scope": subject.scope,
                                "scope_kind": subject.scope_kind,
                                "subject_id": subject.subject_id,
                                "group": subject.group,
                                "window_label": label,
                                "window_start": int(start),
                                "window_size_requested": int(window_size),
                                "lag": int(lag),
                                "metric": metric_name,
                                "category": metric.category,
                                "directed": bool(metric.directed),
                                "pvalue_based": bool(metric.pvalue_based),
                                "n_nodes": int(frame.shape[1]),
                                "n_timepoints": int(frame.shape[0]),
                                "pairs_used": int(len(pairs)),
                                **features,
                            }
                        )
                    if done % 50 == 0:
                        elapsed = time.perf_counter() - started
                        print(
                            f"FEATURES done={done}/{total} elapsed_sec={elapsed:.1f} "
                            f"last_scope={subject.scope} last_metric={metric_name}",
                            flush=True,
                        )
                        pd.DataFrame(status_rows).to_csv(
                            result_dir / "metric_compute_status.partial.csv",
                            index=False,
                            encoding="utf-8-sig",
                        )
    return pd.DataFrame(feature_rows), pd.DataFrame(status_rows), pd.DataFrame(qc_rows)


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, float | int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    try:
        auc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc = math.nan
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "roc_auc": auc,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else math.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_classification(features: pd.DataFrame, *, n_splits: int, repeats: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    group_cols = ["scope", "window_label", "lag", "metric"]
    if features.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for keys, group in features.groupby(group_cols, sort=True):
        scope, window_label, lag, metric_name = keys
        ordered = group.sort_values("subject_id").reset_index(drop=True)
        labels = ordered["group"].map({"HC": 0, "SZ": 1}).to_numpy(dtype=int)
        counts = Counter(int(x) for x in labels)
        local_splits = int(min(n_splits, min(counts.values()) if counts else 0))
        feature_cols = [col for col in FEATURE_COLUMNS if col in ordered.columns]
        X = ordered[feature_cols].to_numpy(dtype=np.float64)
        if len(counts) < 2 or local_splits < 2 or X.shape[0] < 6:
            summary_rows.append(
                {
                    "scope": scope,
                    "window_label": window_label,
                    "lag": int(lag),
                    "metric": metric_name,
                    "status": "skipped_too_few_subjects",
                    "n_subjects": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                }
            )
            continue

        cv = RepeatedStratifiedKFold(
            n_splits=local_splits,
            n_repeats=int(max(1, repeats)),
            random_state=int(seed),
        )
        fold_metric_rows = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, labels), start=1):
            model = Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            C=0.5,
                            class_weight="balanced",
                            max_iter=1000,
                            solver="liblinear",
                            random_state=int(seed + fold_idx),
                        ),
                    ),
                ]
            )
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    model.fit(X[train_idx], labels[train_idx])
                    pred_test = model.predict(X[test_idx]).astype(int)
                    pred_train = model.predict(X[train_idx]).astype(int)
                    if hasattr(model, "decision_function"):
                        score_test = model.decision_function(X[test_idx]).astype(float)
                    else:
                        score_test = model.predict_proba(X[test_idx])[:, 1].astype(float)
                test_metrics = _classification_metrics(labels[test_idx], pred_test, score_test)
                train_bal = float(balanced_accuracy_score(labels[train_idx], pred_train))
                row = {
                    "scope": scope,
                    "window_label": window_label,
                    "lag": int(lag),
                    "metric": metric_name,
                    "fold": int(fold_idx),
                    "status": "ok",
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_balanced_accuracy": train_bal,
                    "train_test_gap": train_bal - float(test_metrics["balanced_accuracy"]),
                    **test_metrics,
                }
                fold_rows.append(row)
                fold_metric_rows.append(row)
                for subject_id, true_label, pred_label, score in zip(
                    ordered.loc[test_idx, "subject_id"].astype(str),
                    labels[test_idx],
                    pred_test,
                    score_test,
                ):
                    prediction_rows.append(
                        {
                            "scope": scope,
                            "window_label": window_label,
                            "lag": int(lag),
                            "metric": metric_name,
                            "fold": int(fold_idx),
                            "subject_id": subject_id,
                            "true_group": "SZ" if int(true_label) else "HC",
                            "predicted_group": "SZ" if int(pred_label) else "HC",
                            "score_SZ": float(score),
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                fold_rows.append(
                    {
                        "scope": scope,
                        "window_label": window_label,
                        "lag": int(lag),
                        "metric": metric_name,
                        "fold": int(fold_idx),
                        "status": "failed",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                )
        fold_df = pd.DataFrame(fold_metric_rows)
        if fold_df.empty:
            summary_rows.append(
                {
                    "scope": scope,
                    "window_label": window_label,
                    "lag": int(lag),
                    "metric": metric_name,
                    "status": "failed",
                    "n_subjects": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                }
            )
            continue
        summary: dict[str, Any] = {
            "scope": scope,
            "window_label": window_label,
            "lag": int(lag),
            "metric": metric_name,
            "status": "ok",
            "n_subjects": int(X.shape[0]),
            "n_hc": int(counts.get(0, 0)),
            "n_sz": int(counts.get(1, 0)),
            "n_features": int(X.shape[1]),
            "n_folds": int(len(fold_df)),
        }
        for col in (
            "balanced_accuracy",
            "roc_auc",
            "accuracy",
            "f1",
            "sensitivity",
            "specificity",
            "train_balanced_accuracy",
            "train_test_gap",
        ):
            summary[f"{col}_mean"] = float(fold_df[col].mean())
            summary[f"{col}_std"] = float(fold_df[col].std(ddof=1)) if len(fold_df) > 1 else 0.0
        summary_rows.append(summary)
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows), pd.DataFrame(prediction_rows)


def _rank_outputs(summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    ok = summary[summary["status"].eq("ok")].copy()
    if ok.empty:
        return ok, ok
    ok = ok.sort_values(
        ["scope", "balanced_accuracy_mean", "roc_auc_mean", "train_test_gap_mean"],
        ascending=[True, False, False, True],
    )
    ok["rank_within_scope_window_lag"] = (
        ok.groupby(["scope", "window_label", "lag"]).cumcount() + 1
    )
    best = (
        ok.sort_values(
            ["scope", "metric", "balanced_accuracy_mean", "roc_auc_mean", "train_test_gap_mean"],
            ascending=[True, True, False, False, True],
        )
        .groupby(["scope", "metric"], as_index=False)
        .head(1)
        .copy()
    )
    best = best.sort_values(
        ["scope", "balanced_accuracy_mean", "roc_auc_mean", "train_test_gap_mean"],
        ascending=[True, False, False, True],
    )
    best["rank_within_scope"] = best.groupby("scope").cumcount() + 1
    return ok, best


def _overall_ranking(best_by_scope: pd.DataFrame) -> pd.DataFrame:
    if best_by_scope.empty:
        return pd.DataFrame()
    rows = []
    for metric_name, group in best_by_scope.groupby("metric", sort=True):
        rows.append(
            {
                "metric": metric_name,
                "n_scopes": int(group["scope"].nunique()),
                "mean_best_balanced_accuracy": float(group["balanced_accuracy_mean"].mean()),
                "mean_best_roc_auc": float(group["roc_auc_mean"].mean()),
                "mean_rank_within_scope": float(group["rank_within_scope"].mean()),
                "best_scope": str(
                    group.sort_values(
                        ["balanced_accuracy_mean", "roc_auc_mean"],
                        ascending=[False, False],
                    ).iloc[0]["scope"]
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        ["mean_best_balanced_accuracy", "mean_best_roc_auc", "mean_rank_within_scope"],
        ascending=[False, False, True],
    )
    out["overall_rank"] = np.arange(1, len(out) + 1)
    return out


def _write_report(path: Path, ranking: pd.DataFrame, best: pd.DataFrame, overall: pd.DataFrame, limitations: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Fast metric classifier screening",
        "",
        "Exploratory, approximate HC/SZ metric ranking. This run uses pair sketches and compact metric summaries, not full edge-level matrices.",
        "",
        "## Overall ranking",
        "",
        _markdown_table(overall.head(20)),
        "",
        "## Best metric per scope",
        "",
        _markdown_table(best.head(40)),
        "",
        "## Best scope-window-lag rows",
        "",
        _markdown_table(ranking.head(40)),
        "",
        "## Limitations",
        "",
        _markdown_table(limitations),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    keep = [
        col
        for col in (
            "overall_rank",
            "rank_within_scope",
            "scope",
            "metric",
            "window_label",
            "lag",
            "balanced_accuracy_mean",
            "roc_auc_mean",
            "f1_mean",
            "train_test_gap_mean",
            "n_subjects",
            "mean_best_balanced_accuracy",
            "mean_best_roc_auc",
            "best_scope",
            "status",
            "limitation",
        )
        if col in view.columns
    ]
    if keep:
        view = view[keep]
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
        else:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = [str(col) for col in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in view.columns) + " |")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-results", required=True)
    parser.add_argument("--aal3-root", default="")
    parser.add_argument("--hcp-result", default="")
    parser.add_argument("--tissue-result", default="")
    parser.add_argument("--result-date", default=date.today().isoformat())
    parser.add_argument("--slug", default="metric-classifier-fast-screening")
    parser.add_argument(
        "--scopes",
        default="AAL3_whole_roi,HCP360_GM_active_mean,HCP360_whole_brain,tissue_mean_GM_WM_CSF",
    )
    parser.add_argument("--max-subjects-per-group", type=int, default=4)
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--heavy-max-pairs", type=int, default=12)
    parser.add_argument("--lags", default="1,3")
    parser.add_argument("--window-size", type=int, default=300)
    parser.add_argument("--window-sizes", default="")
    parser.add_argument("--window-starts", default="0,150")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--cv-repeats", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--metrics", default="all")
    parser.add_argument("--skip-provenance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    started_wall = datetime.now().astimezone()
    started = time.perf_counter()
    root = Path(args.new_results)
    result_dir = _make_result_dir(root, args.result_date, args.slug)
    print(f"OUT_DIR={result_dir}", flush=True)

    scopes = {item.strip() for item in str(args.scopes).split(",") if item.strip()}
    lags = _parse_ints(args.lags)
    starts = _parse_ints(args.window_starts)
    window_sizes = _parse_ints(args.window_sizes) if str(args.window_sizes).strip() else [int(args.window_size)]
    metric_names = [m.name for m in list_metrics()] if str(args.metrics).lower() == "all" else [
        item.strip() for item in str(args.metrics).split(",") if item.strip()
    ]

    config = {
        "started_at": started_wall.isoformat(),
        "new_results": str(root),
        "result_dir": str(result_dir),
        "scopes": sorted(scopes),
        "max_subjects_per_group": int(args.max_subjects_per_group),
        "max_pairs": int(args.max_pairs),
        "heavy_max_pairs": int(args.heavy_max_pairs),
        "heavy_prefixes": list(HEAVY_PREFIXES),
        "lags": lags,
        "window_sizes": window_sizes,
        "window_starts": starts,
        "window_tasks": [
            {"window_size": int(size), "window_start": int(start)}
            for size, start in _window_tasks(window_sizes, starts)
        ],
        "n_splits": int(args.n_splits),
        "cv_repeats": int(args.cv_repeats),
        "random_seed": int(args.random_seed),
        "metrics": metric_names,
        "classification_model": "median-impute + standardize + balanced logistic regression C=0.5",
        "partial_control": "global_signal for metrics marked supports_control",
        "evidence_level": "fast approximate screening",
    }
    _write_json(result_dir / "run_config.json", config)
    registry_snapshot().to_csv(result_dir / "metric_registry.csv", index=False, encoding="utf-8-sig")

    subjects: list[SubjectSeries] = []
    inventory_rows: list[dict[str, Any]] = []
    if "AAL3_whole_roi" in scopes and args.aal3_root:
        loaded, inventory = load_aal3_scope(Path(args.aal3_root), int(args.max_subjects_per_group), int(args.random_seed))
        subjects.extend(loaded)
        inventory_rows.extend(inventory)
    if ({"HCP360_GM_active_mean", "HCP360_whole_brain"} & scopes) and args.hcp_result:
        loaded, inventory = load_hcp_scopes(
            Path(args.hcp_result),
            scopes,
            int(args.max_subjects_per_group),
            int(args.random_seed),
        )
        subjects.extend(loaded)
        inventory_rows.extend(inventory)
    if "tissue_mean_GM_WM_CSF" in scopes and args.tissue_result:
        loaded, inventory = load_tissue_mean_scope(
            Path(args.tissue_result),
            scopes,
            int(args.max_subjects_per_group),
            int(args.random_seed),
        )
        subjects.extend(loaded)
        inventory_rows.extend(inventory)

    limitations = pd.DataFrame(
        [
            {
                "scope": "HCP360_WM_regional",
                "status": "not_computed",
                "limitation": "No ready regional WM HCP360 signal set was found; tissue_mean_GM_WM_CSF is only a 3-signal tissue-mean proxy.",
            },
            {
                "scope": "all",
                "status": "exploratory",
                "limitation": "Pair-sketch features rank metrics approximately and do not replace full edge-level nested validation.",
            },
            {
                "scope": "partial_metrics",
                "status": "exploratory",
                "limitation": "Partial variants use a subject-local global-signal control to avoid infeasible all-node controls in the fast run.",
            },
        ]
    )
    pd.DataFrame(inventory_rows).to_csv(result_dir / "scope_inventory.csv", index=False, encoding="utf-8-sig")
    limitations.to_csv(result_dir / "scope_limitations.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "scope": item.scope,
                "scope_kind": item.scope_kind,
                "subject_id": item.subject_id,
                "group": item.group,
                "n_timepoints": int(item.values_time_node.shape[0]),
                "n_nodes": int(item.values_time_node.shape[1]),
                "source_path": item.source_path,
                "array_key": item.array_key,
                "note": item.note,
            }
            for item in subjects
        ]
    ).to_csv(result_dir / "subject_input_manifest.csv", index=False, encoding="utf-8-sig")

    if not subjects:
        _write_json(result_dir / "runtime_summary.json", {"status": "failed_no_subjects"})
        return 1
    print(
        f"INVENTORY subjects={len(subjects)} scopes={sorted({s.scope for s in subjects})} "
        f"metrics={len(metric_names)}",
        flush=True,
    )

    try:
        features, status, qc = compute_feature_rows(
            subjects,
            result_dir,
            metric_names=metric_names,
            lags=lags,
            window_sizes=window_sizes,
            window_starts=starts,
            max_pairs=int(args.max_pairs),
            heavy_max_pairs=int(args.heavy_max_pairs),
            seed=int(args.random_seed),
        )
        features.to_csv(result_dir / "subject_metric_features.csv", index=False, encoding="utf-8-sig")
        status.to_csv(result_dir / "metric_compute_status.csv", index=False, encoding="utf-8-sig")
        qc.to_csv(result_dir / "subject_preprocessing_qc.csv", index=False, encoding="utf-8-sig")
        print(f"FEATURE_TABLE rows={len(features)} status_rows={len(status)}", flush=True)

        folds, summary, predictions = run_classification(
            features,
            n_splits=int(args.n_splits),
            repeats=int(args.cv_repeats),
            seed=int(args.random_seed),
        )
        folds.to_csv(result_dir / "classification_fold_scores.csv", index=False, encoding="utf-8-sig")
        predictions.to_csv(result_dir / "classification_predictions.csv", index=False, encoding="utf-8-sig")
        summary.to_csv(result_dir / "classification_summary.csv", index=False, encoding="utf-8-sig")
        ranking, best = _rank_outputs(summary)
        overall = _overall_ranking(best)
        ranking.to_csv(result_dir / "metric_ranking_by_scope_window_lag.csv", index=False, encoding="utf-8-sig")
        best.to_csv(result_dir / "metric_best_by_scope.csv", index=False, encoding="utf-8-sig")
        overall.to_csv(result_dir / "overall_metric_ranking.csv", index=False, encoding="utf-8-sig")
        _write_report(result_dir / "reports" / "fast_screening_summary.md", ranking, best, overall, limitations)

        status_text = "completed"
        finding_rows = []
        if not overall.empty:
            top = overall.iloc[0]
            finding_rows.append(
                f"Top approximate metric overall: {top['metric']} "
                f"(mean best balanced accuracy={float(top['mean_best_balanced_accuracy']):.3f})."
            )
        for _, row in best.groupby("scope", sort=True).head(1).iterrows() if not best.empty else []:
            finding_rows.append(
                f"Top {row['scope']}: {row['metric']} "
                f"(balanced accuracy={float(row['balanced_accuracy_mean']):.3f}, "
                f"AUC={float(row['roc_auc_mean']):.3f})."
            )
    except Exception as exc:  # noqa: BLE001
        status_text = "failed"
        finding_rows = []
        (result_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"ERROR {type(exc).__name__}: {exc}", flush=True)
        raise
    finally:
        elapsed = time.perf_counter() - started
        _write_json(
            result_dir / "runtime_summary.json",
            {
                "status": locals().get("status_text", "failed"),
                "started_at": started_wall.isoformat(),
                "finished_at": datetime.now().astimezone().isoformat(),
                "elapsed_seconds": elapsed,
            },
        )

    if not args.skip_provenance:
        write_result_provenance(
            result_dir=result_dir,
            result_id=args.slug,
            title="Fast HC/SZ metric classifier screening",
            result_type="model_evaluation_fast_screening",
            status=status_text,
            execution_mode="fresh_run",
            summary="Fast approximate comparison of all registered metrics for HC/SZ binary classification using pair sketches and compact features.",
            meaning="This narrows which metric families are promising before running expensive full edge-level validation.",
            command="python scripts/run_fast_metric_classifier_screen.py with the arguments recorded in run_config.json",
            inputs=[
                str(Path(args.aal3_root)) if args.aal3_root else "",
                str(Path(args.hcp_result)) if args.hcp_result else "",
                str(Path(args.tissue_result)) if args.tissue_result else "",
            ],
            code_files=[
                REPO / "scripts/run_fast_metric_classifier_screen.py",
                REPO / "neweds/core/metric_runner.py",
                REPO / "neweds/metrics/registry.py",
                REPO / "scripts/write_result_provenance.py",
            ],
            repository=REPO,
            findings=finding_rows or ["No completed ranking rows were produced."],
            limitations=limitations["limitation"].astype(str).tolist(),
        )
    print(f"DONE status={status_text} elapsed_sec={time.perf_counter() - started:.1f}", flush=True)
    return 0 if status_text == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
