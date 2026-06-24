"""HCP-first Stage 2 fMRI benchmark utilities.

This module turns validated HCP360 regional time series into auditable
connectivity matrices, subject-level feature ledgers, and leakage-safe HC/SZ
out-of-fold classification summaries.  It is deliberately conservative:
invalid nodes produce NaN edges, blocked subjects remain blocked, and every
cross-subject transformation is fitted inside the training fold.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from neweds.core.group_pipeline import _fdr_bh
from neweds.core.metric_runner import compute_metric
from neweds.metrics.registry import get_metric, list_metrics

EXPECTED_STAGE2_METRICS = (
    "correlation_full",
    "correlation_spearman",
    "correlation_kendall",
    "correlation_partial",
    "correlation_directed",
    "h2_full",
    "h2_partial",
    "h2_directed",
    "coherence_full",
    "coherence_partial",
    "wavelet_full",
    "wavelet_partial",
    "ordinal_full",
    "ordinal_directed",
    "mutinf_full",
    "mutinf_partial",
    "dcor_full",
    "dcor_partial",
    "dcor_directed",
    "ah_full",
    "ah_partial",
    "ah_directed",
    "granger_full",
    "granger_partial",
    "te_full",
    "te_partial",
)

GM_REPRESENTATION_KEYS = {
    "gm_active_mean": "active_mean_z",
    "gm_pca_pc1_oriented": "pca_pc1_oriented_z",
    "gm_ica_1_oriented": "ica_1_oriented_z",
    "gm_correlation_core": "correlation_core_z",
}
WHOLE_BRAIN_REPRESENTATION = "whole_brain"
DEFAULT_REPRESENTATIONS = (*GM_REPRESENTATION_KEYS, WHOLE_BRAIN_REPRESENTATION)

PRIMARY_BRANCHES = (
    "baseline_without_GSR",
    "AR1_residualized_without_GSR",
)
SENSITIVITY_BRANCHES = (
    "detrended_without_GSR",
    "AR1_plus_detrended_without_GSR",
    "baseline_with_GSR",
    "AR1_residualized_with_GSR",
    "detrended_with_GSR",
    "AR1_plus_detrended_with_GSR",
)
DEFAULT_BRANCHES = (*PRIMARY_BRANCHES, *SENSITIVITY_BRANCHES)


@dataclass(frozen=True, slots=True)
class RegionalInputManifest:
    subject_id: str
    group: Literal["HC", "SZ"]
    representation: str
    source_kind: Literal["gm_only", "whole_brain"]
    atlas_id: str
    npz_path: str
    array_key: str
    node_order: tuple[int, ...]
    n_nodes: int
    n_timepoints: int
    status: str = "ok"
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["node_order"] = list(self.node_order)
        return data


@dataclass(frozen=True, slots=True)
class PreprocessingBranchConfig:
    name: str
    detrend: bool = False
    ar_order: int = 0
    gsr_mode: Literal["none", "representation_global"] = "none"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MetricMatrixResult:
    subject_id: str
    group: Literal["HC", "SZ"]
    representation: str
    branch: str
    metric: str
    lag: int
    status: str
    matrix_path: str
    feature_path: str
    n_nodes: int
    n_valid_nodes: int
    n_timepoints_used: int
    n_features: int
    finite_fraction: float
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SubjectFeatureLedger:
    representation: str
    branch: str
    metric: str
    lag: int
    n_subjects: int
    n_features: int
    feature_manifest_path: str
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NestedLoocvBenchmarkResult:
    representation: str
    branch: str
    metric: str
    model: str
    n_subjects: int
    roc_auc: float
    pr_auc: float
    balanced_accuracy: float
    mcc: float
    sensitivity: float
    specificity: float
    f1: float
    status: str
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PairedRepresentationComparisonResult:
    metric: str
    branch: str
    model: str
    left_representation: str
    right_representation: str
    n_subjects: int
    roc_auc_delta: float
    balanced_accuracy_delta: float
    mcc_delta: float
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    paired_permutation_p: float
    n_changed_predictions: int
    status: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class FmriStage2Config:
    gm_hcp_result: str
    whole_brain_hcp_inputs: str
    new_results_root: str
    metrics_result_dir: str
    classification_result_dir: str
    representations: tuple[str, ...] = DEFAULT_REPRESENTATIONS
    metrics: tuple[str, ...] = ("all",)
    branches: tuple[str, ...] = DEFAULT_BRANCHES
    primary_lag: int = 1
    permutations: int = 1000
    bootstraps: int = 2000
    smoke: bool = False
    smoke_subjects: tuple[str, ...] = ("1185", "1097")
    smoke_metrics: tuple[str, ...] = (
        "correlation_full",
        "correlation_directed",
        "wavelet_full",
    )
    random_seed: int = 1729

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def preprocessing_branch_configs(names: Sequence[str] = DEFAULT_BRANCHES) -> tuple[PreprocessingBranchConfig, ...]:
    mapping = {
        "baseline_without_GSR": PreprocessingBranchConfig("baseline_without_GSR"),
        "AR1_residualized_without_GSR": PreprocessingBranchConfig(
            "AR1_residualized_without_GSR", ar_order=1
        ),
        "detrended_without_GSR": PreprocessingBranchConfig(
            "detrended_without_GSR", detrend=True
        ),
        "AR1_plus_detrended_without_GSR": PreprocessingBranchConfig(
            "AR1_plus_detrended_without_GSR", detrend=True, ar_order=1
        ),
        "baseline_with_GSR": PreprocessingBranchConfig(
            "baseline_with_GSR", gsr_mode="representation_global"
        ),
        "AR1_residualized_with_GSR": PreprocessingBranchConfig(
            "AR1_residualized_with_GSR", ar_order=1, gsr_mode="representation_global"
        ),
        "detrended_with_GSR": PreprocessingBranchConfig(
            "detrended_with_GSR", detrend=True, gsr_mode="representation_global"
        ),
        "AR1_plus_detrended_with_GSR": PreprocessingBranchConfig(
            "AR1_plus_detrended_with_GSR",
            detrend=True,
            ar_order=1,
            gsr_mode="representation_global",
        ),
    }
    return tuple(mapping[name] for name in names)


def resolve_metrics(tokens: Sequence[str]) -> tuple[str, ...]:
    if any(str(token).lower() == "all" for token in tokens):
        names = tuple(metric.name for metric in list_metrics())
    else:
        names = tuple(str(token).strip() for token in tokens if str(token).strip())
    missing = [name for name in names if name not in EXPECTED_STAGE2_METRICS]
    if missing:
        raise ValueError(f"unknown or out-of-scope metrics: {missing}")
    return names


def registry_snapshot() -> pd.DataFrame:
    rows = []
    for index, metric in enumerate(list_metrics()):
        rows.append(
            {
                "registry_order": index,
                "metric": metric.name,
                "category": metric.category,
                "directed": metric.directed,
                "pvalue_based": metric.pvalue_based,
                "supports_control": metric.supports_control,
                "partial_mode": metric.partial_mode,
                "experimental": metric.experimental,
                "stable": metric.stable,
                "expected_stage2_metric": metric.name in EXPECTED_STAGE2_METRICS,
                "description": metric.description,
            }
        )
    return pd.DataFrame(rows)


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_json(path, lines=True)


def _node_order_from_any(value: Any) -> tuple[int, ...]:
    if isinstance(value, str):
        parsed = json.loads(value)
    else:
        parsed = value
    return tuple(int(item) for item in parsed)


def discover_hcp_inputs(
    gm_hcp_result: str | Path,
    *,
    representations: Sequence[str] = DEFAULT_REPRESENTATIONS,
    subject_filter: Sequence[str] | None = None,
) -> tuple[list[RegionalInputManifest], pd.DataFrame]:
    """Create a subject × representation manifest from a full HCP result."""
    result_dir = Path(gm_hcp_result)
    status = _read_jsonl(result_dir / "subject_status.jsonl")
    paired = _read_jsonl(result_dir / "paired_input_manifest.jsonl")
    if status.empty:
        raise FileNotFoundError(f"{result_dir}: missing or empty subject_status.jsonl")
    if paired.empty:
        raise FileNotFoundError(f"{result_dir}: missing or empty paired_input_manifest.jsonl")
    if "node_order" not in paired.columns:
        raise ValueError("paired input manifest must contain node_order")

    wanted = set(str(subject) for subject in subject_filter or [])
    if wanted:
        status = status[status["subject_id"].astype(str).isin(wanted)].copy()
        paired = paired[paired["subject_id"].astype(str).isin(wanted)].copy()

    paired_by_subject = {
        str(row.subject_id): row
        for row in paired.itertuples(index=False)
        if str(getattr(row, "status", "ok")) == "ok"
    }
    manifests: list[RegionalInputManifest] = []
    alignment_rows: list[dict[str, Any]] = []
    for row in status.itertuples(index=False):
        subject_id = str(row.subject_id)
        group = str(row.group)
        gm_status = str(getattr(row, "status", ""))
        gm_npz = str(getattr(row, "signal_npz", "") or getattr(row, "gm_signal_npz", ""))
        node_order = _node_order_from_any(getattr(row, "node_order", [])) if hasattr(row, "node_order") else ()
        if not node_order and (result_dir / "node_table.csv").is_file():
            node_order = tuple(pd.read_csv(result_dir / "node_table.csv")["region_id"].astype(int))
        paired_row = paired_by_subject.get(subject_id)
        wb_order = _node_order_from_any(paired_row.node_order) if paired_row is not None else ()
        node_order_equal = bool(node_order and wb_order and node_order == wb_order)
        has_gm = gm_status in {"ok", "ok_real_data_smoke"} and Path(gm_npz).is_file()
        has_wb = paired_row is not None and Path(str(paired_row.output_npz)).is_file()
        alignment_rows.append(
            {
                "subject_id": subject_id,
                "group": group,
                "gm_status": gm_status,
                "whole_brain_status": getattr(paired_row, "status", "missing") if paired_row is not None else "missing",
                "gm_signal_npz": gm_npz,
                "whole_brain_npz": str(paired_row.output_npz) if paired_row is not None else "",
                "node_order_equal": node_order_equal,
                "paired_stage2_eligible": bool(has_gm and has_wb and node_order_equal),
            }
        )
        if not (has_gm and has_wb and node_order_equal):
            continue
        n_nodes = len(node_order)
        n_timepoints = 600
        for representation in representations:
            if representation in GM_REPRESENTATION_KEYS:
                manifests.append(
                    RegionalInputManifest(
                        subject_id=subject_id,
                        group=group,  # type: ignore[arg-type]
                        representation=representation,
                        source_kind="gm_only",
                        atlas_id=str(getattr(row, "atlas_id", "HCP-MMP1-360")),
                        npz_path=gm_npz,
                        array_key=GM_REPRESENTATION_KEYS[representation],
                        node_order=node_order,
                        n_nodes=n_nodes,
                        n_timepoints=n_timepoints,
                    )
                )
            elif representation == WHOLE_BRAIN_REPRESENTATION:
                manifests.append(
                    RegionalInputManifest(
                        subject_id=subject_id,
                        group=group,  # type: ignore[arg-type]
                        representation=representation,
                        source_kind="whole_brain",
                        atlas_id=str(getattr(paired_row, "atlas_id", "HCP-MMP1-360")),
                        npz_path=str(paired_row.output_npz),
                        array_key="z",
                        node_order=wb_order,
                        n_nodes=n_nodes,
                        n_timepoints=n_timepoints,
                    )
                )
            else:
                raise ValueError(f"unknown representation: {representation}")
    return manifests, pd.DataFrame(alignment_rows)


def load_regional_array(manifest: RegionalInputManifest) -> np.ndarray:
    with np.load(manifest.npz_path) as arrays:
        if manifest.array_key not in arrays:
            raise KeyError(f"{manifest.npz_path}: missing array {manifest.array_key}")
        values = np.asarray(arrays[manifest.array_key], dtype=np.float64)
    expected = (manifest.n_nodes, manifest.n_timepoints)
    if values.shape != expected:
        raise ValueError(f"{manifest.npz_path}:{manifest.array_key} expected {expected}, got {values.shape}")
    return values


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    out = np.full_like(arr, np.nan)
    for col in range(arr.shape[1]):
        x = arr[:, col]
        finite = np.isfinite(x)
        if finite.sum() < 3:
            continue
        mean = float(np.nanmean(x))
        std = float(np.nanstd(x))
        if std <= 1e-12:
            continue
        out[finite, col] = (x[finite] - mean) / std
    return out


def _ar_residualize_column(values: np.ndarray, order: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full_like(x, np.nan)
    finite = np.isfinite(x)
    if finite.sum() <= order + 3 or np.nanstd(x[finite]) <= 1e-12:
        return out
    filled = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
    if not np.isfinite(filled).all():
        filled = np.where(np.isfinite(filled), filled, np.nanmean(filled))
    y = filled[order:]
    design = np.column_stack([filled[order - lag : -lag] for lag in range(1, order + 1)])
    design = np.column_stack([np.ones(design.shape[0]), design])
    try:
        beta, *_ = np.linalg.lstsq(design, y, rcond=None)
        out[order:] = y - design @ beta
    except np.linalg.LinAlgError:
        out[order:] = y - np.nanmean(y)
    return out


def _residualize_global(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    out = np.full_like(arr, np.nan)
    valid_cols = (np.isfinite(arr).sum(axis=0) >= 4) & (np.nanstd(arr, axis=0) > 1e-12)
    if valid_cols.sum() < 2:
        return out
    selected = arr[:, valid_cols]
    finite_selected = np.isfinite(selected)
    counts = finite_selected.sum(axis=1)
    sums = np.where(finite_selected, selected, 0.0).sum(axis=1)
    global_signal = np.divide(
        sums,
        counts,
        out=np.full(selected.shape[0], np.nan, dtype=float),
        where=counts > 0,
    )
    finite_global = np.isfinite(global_signal)
    if finite_global.sum() < 3 or np.nanstd(global_signal[finite_global]) <= 1e-12:
        return out
    design = np.column_stack([np.ones(arr.shape[0]), global_signal])
    for col in range(arr.shape[1]):
        x = arr[:, col]
        finite = np.isfinite(x) & finite_global
        if finite.sum() < 3 or np.nanstd(x[finite]) <= 1e-12:
            continue
        beta, *_ = np.linalg.lstsq(design[finite], x[finite], rcond=None)
        out[finite, col] = x[finite] - design[finite] @ beta
    return out


def preprocess_regional_timeseries(
    values_nodes_time: np.ndarray,
    branch: PreprocessingBranchConfig,
) -> np.ndarray:
    """Return time × node data for one subject-local branch.

    This function is label-blind.  It preserves the time × node shape and leaves
    invalid nodes/time points as NaN rather than replacing them by zeros.
    """
    arr = np.asarray(values_nodes_time, dtype=np.float64).T.copy()
    if branch.detrend:
        detrended = np.full_like(arr, np.nan)
        for col in range(arr.shape[1]):
            x = arr[:, col]
            finite = np.isfinite(x)
            if finite.sum() < 3 or np.nanstd(x[finite]) <= 1e-12:
                continue
            filled = pd.Series(x).interpolate(limit_direction="both").to_numpy(dtype=np.float64)
            detrended[:, col] = signal.detrend(filled)
        arr = detrended
    if branch.ar_order:
        arr = np.column_stack(
            [_ar_residualize_column(arr[:, col], branch.ar_order) for col in range(arr.shape[1])]
        )
    if branch.gsr_mode == "representation_global":
        arr = _residualize_global(arr)
    return _zscore_columns(arr)


def temporal_qc(values_nodes_time: np.ndarray, processed_time_node: np.ndarray) -> dict[str, float | int]:
    raw = np.asarray(values_nodes_time, dtype=np.float64).T
    processed = np.asarray(processed_time_node, dtype=np.float64)
    valid_raw = _valid_columns_loose(raw, min_finite=4)
    valid_processed = _valid_columns_loose(processed, min_finite=4)
    raw_global = (
        _row_nanmean_no_warning(raw[:, valid_raw]) if valid_raw.any() else np.full(raw.shape[0], np.nan)
    )
    proc_global = (
        _row_nanmean_no_warning(processed[:, valid_processed])
        if valid_processed.any()
        else np.full(processed.shape[0], np.nan)
    )
    return {
        "raw_valid_nodes": int(valid_raw.sum()),
        "processed_valid_nodes": int(valid_processed.sum()),
        "raw_nan_fraction": float(np.isnan(raw).mean()),
        "processed_nan_fraction": float(np.isnan(processed).mean()),
        "raw_global_std": float(np.nanstd(raw_global)),
        "processed_global_std": float(np.nanstd(proc_global)),
        "raw_global_ar1": _lag1_autocorr(raw_global),
        "processed_global_ar1": _lag1_autocorr(proc_global),
    }


def _lag1_autocorr(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    if finite.sum() < 4:
        return math.nan
    x = x[finite]
    if np.std(x[:-1]) <= 1e-12 or np.std(x[1:]) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _valid_node_mask(data_time_node: np.ndarray) -> np.ndarray:
    return _valid_columns_loose(data_time_node, min_finite=8)


def _valid_columns_loose(data_time_node: np.ndarray, *, min_finite: int) -> np.ndarray:
    arr = np.asarray(data_time_node, dtype=np.float64)
    valid = np.zeros(arr.shape[1], dtype=bool)
    for col in range(arr.shape[1]):
        x = arr[:, col]
        finite = np.isfinite(x)
        valid[col] = bool(finite.sum() >= min_finite and np.std(x[finite]) > 1e-12)
    return valid


def _row_nanmean_no_warning(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(arr)
    counts = finite.sum(axis=1)
    sums = np.where(finite, arr, 0.0).sum(axis=1)
    return np.divide(
        sums,
        counts,
        out=np.full(arr.shape[0], np.nan, dtype=float),
        where=counts > 0,
    )


def _feature_ids(node_order: Sequence[int], *, directed: bool) -> list[str]:
    ids: list[str] = []
    n = len(node_order)
    if directed:
        for i in range(n):
            for j in range(n):
                if i != j:
                    ids.append(f"{int(node_order[i])}->{int(node_order[j])}")
    else:
        for i in range(n):
            for j in range(i + 1, n):
                ids.append(f"{int(node_order[i])}--{int(node_order[j])}")
    return ids


def matrix_to_features(matrix: np.ndarray, node_order: Sequence[int], *, directed: bool) -> tuple[np.ndarray, list[str]]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (len(node_order), len(node_order)):
        raise ValueError(f"matrix shape {arr.shape} does not match node count {len(node_order)}")
    values: list[float] = []
    ids: list[str] = []
    if directed:
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if i != j:
                    values.append(float(arr[i, j]))
                    ids.append(f"{int(node_order[i])}->{int(node_order[j])}")
    else:
        for i in range(arr.shape[0]):
            for j in range(i + 1, arr.shape[1]):
                values.append(float(arr[i, j]))
                ids.append(f"{int(node_order[i])}--{int(node_order[j])}")
    return np.asarray(values, dtype=np.float32), ids


def compute_metric_matrix(
    *,
    manifest: RegionalInputManifest,
    processed_time_node: np.ndarray,
    metric_name: str,
    lag: int,
    output_dir: Path,
    branch_name: str = "",
) -> MetricMatrixResult:
    metric = get_metric(metric_name)
    node_order = manifest.node_order
    full = np.full((manifest.n_nodes, manifest.n_nodes), np.nan, dtype=np.float32)
    valid_nodes = _valid_node_mask(processed_time_node)
    status = "ok"
    message = ""
    timepoints_used = 0
    try:
        if valid_nodes.sum() < 2:
            raise ValueError("fewer_than_two_valid_nodes")
        selected = processed_time_node[:, valid_nodes]
        finite_rows = np.isfinite(selected).all(axis=1)
        selected = selected[finite_rows]
        timepoints_used = int(selected.shape[0])
        if selected.shape[0] < max(8, lag + 4):
            raise ValueError("too_few_valid_timepoints")
        columns = [f"node_{idx:03d}" for idx in np.flatnonzero(valid_nodes)]
        frame = pd.DataFrame(selected, columns=columns)
        matrix = np.asarray(compute_metric(frame, metric_name, lag=int(lag)), dtype=np.float32)
        if matrix.shape != (len(columns), len(columns)):
            raise ValueError(f"metric returned shape {matrix.shape}, expected {(len(columns), len(columns))}")
        indices = np.flatnonzero(valid_nodes)
        full[np.ix_(indices, indices)] = matrix
    except Exception as exc:
        status = "failed"
        message = f"{type(exc).__name__}: {exc}"
    feature_values, feature_ids = matrix_to_features(full, node_order, directed=metric.directed)
    branch_token = branch_name or "unbranched"
    stem = f"{manifest.representation}__{branch_token}__{manifest.subject_id}__{metric_name}__lag{int(lag)}"
    matrix_path = (
        output_dir
        / "subject_metric_matrices"
        / branch_token
        / manifest.representation
        / metric_name
        / f"{stem}.npz"
    )
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_path,
        matrix=full,
        features=feature_values,
        feature_ids=np.asarray(feature_ids, dtype=object),
        node_order=np.asarray(node_order, dtype=np.int32),
    )
    return MetricMatrixResult(
        subject_id=manifest.subject_id,
        group=manifest.group,
        representation=manifest.representation,
        branch="",
        metric=metric_name,
        lag=int(lag),
        status=status,
        matrix_path=str(matrix_path),
        feature_path=str(matrix_path),
        n_nodes=manifest.n_nodes,
        n_valid_nodes=int(valid_nodes.sum()),
        n_timepoints_used=timepoints_used,
        n_features=int(feature_values.size),
        finite_fraction=float(np.isfinite(feature_values).mean()) if feature_values.size else math.nan,
        message=message,
    )


def _with_branch(result: MetricMatrixResult, branch: str) -> MetricMatrixResult:
    return MetricMatrixResult(
        **{
            **result.as_dict(),
            "branch": branch,
        }
    )


def run_metric_stage(config: FmriStage2Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_dir = Path(config.metrics_result_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    subject_filter = config.smoke_subjects if config.smoke else None
    metrics = config.smoke_metrics if config.smoke else resolve_metrics(config.metrics)
    manifests, alignment = discover_hcp_inputs(
        config.gm_hcp_result,
        representations=config.representations,
        subject_filter=subject_filter,
    )
    branch_configs = preprocessing_branch_configs(config.branches if not config.smoke else PRIMARY_BRANCHES)
    (metrics_dir / "cohort").mkdir(parents=True, exist_ok=True)
    (metrics_dir / "qc").mkdir(parents=True, exist_ok=True)
    (metrics_dir / "metrics").mkdir(parents=True, exist_ok=True)
    alignment.to_csv(metrics_dir / "cohort" / "subject_alignment.csv", index=False, encoding="utf-8-sig")
    registry_snapshot().to_csv(
        metrics_dir / "metrics" / "registered_metrics_snapshot.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame([branch.as_dict() for branch in branch_configs]).to_csv(
        metrics_dir / "qc" / "preprocessing_branches.csv",
        index=False,
        encoding="utf-8-sig",
    )

    status_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    global_rows: list[dict[str, Any]] = []
    for manifest in manifests:
        raw = load_regional_array(manifest)
        for branch in branch_configs:
            processed = preprocess_regional_timeseries(raw, branch)
            qc = temporal_qc(raw, processed)
            temporal_rows.append(
                {
                    **manifest.as_dict(),
                    "branch": branch.name,
                    **qc,
                }
            )
            global_rows.append(
                {
                    "subject_id": manifest.subject_id,
                    "group": manifest.group,
                    "representation": manifest.representation,
                    "branch": branch.name,
                    "gsr_mode": branch.gsr_mode,
                    "global_signal_std_after": qc["processed_global_std"],
                    "global_signal_ar1_after": qc["processed_global_ar1"],
                }
            )
            for metric_name in metrics:
                result = _with_branch(
                    compute_metric_matrix(
                        manifest=manifest,
                        processed_time_node=processed,
                        metric_name=metric_name,
                        lag=config.primary_lag,
                        output_dir=metrics_dir / "metrics",
                        branch_name=branch.name,
                    ),
                    branch.name,
                )
                status_rows.append(result.as_dict())
                feature_rows.append(
                    {
                        "subject_id": manifest.subject_id,
                        "group": manifest.group,
                        "representation": manifest.representation,
                        "branch": branch.name,
                        "metric": metric_name,
                        "lag": config.primary_lag,
                        "feature_path": result.feature_path,
                        "status": result.status,
                        "n_features": result.n_features,
                    }
                )
    status = pd.DataFrame(status_rows)
    features = pd.DataFrame(feature_rows)
    temporal = pd.DataFrame(temporal_rows)
    global_qc = pd.DataFrame(global_rows)
    status.to_csv(metrics_dir / "metrics" / "metric_execution_status.csv", index=False, encoding="utf-8-sig")
    features.to_parquet(metrics_dir / "metrics" / "feature_manifest.parquet", compression="zstd", index=False)
    temporal.to_csv(metrics_dir / "qc" / "temporal_qc_before_after.csv", index=False, encoding="utf-8-sig")
    global_qc.to_csv(metrics_dir / "qc" / "global_signal_qc.csv", index=False, encoding="utf-8-sig")
    return status, features, temporal


def _load_feature_vector(path: str | Path) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=True) as arrays:
        return np.asarray(arrays["features"], dtype=np.float64), [str(x) for x in arrays["feature_ids"]]


def _assemble_feature_table(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    ordered = group.sort_values("subject_id").reset_index(drop=True)
    vectors = []
    feature_ids: list[str] | None = None
    subjects = ordered["subject_id"].astype(str).tolist()
    labels = ordered["group"].map({"HC": 0, "SZ": 1}).to_numpy(dtype=int)
    for path in ordered["feature_path"]:
        vector, ids = _load_feature_vector(path)
        vectors.append(vector)
        if feature_ids is None:
            feature_ids = ids
        elif feature_ids != ids:
            raise ValueError("feature IDs differ across subjects")
    return np.vstack(vectors), labels, subjects, feature_ids or []


def _finite_metric_value(func, y_true: np.ndarray, values: np.ndarray) -> float:
    try:
        out = float(func(y_true, values))
    except Exception:
        return math.nan
    return out if np.isfinite(out) else math.nan


def performance_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]
    return {
        "roc_auc": _finite_metric_value(roc_auc_score, y_true, y_score),
        "pr_auc": _finite_metric_value(average_precision_score, y_true, y_score),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else math.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else math.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _candidate_k_values(n_features: int, n_train: int) -> list[int | str]:
    candidates: list[int | str] = []
    for value in (25, 100, 500, max(1, n_train * 2)):
        if value < n_features:
            candidates.append(int(value))
    candidates.append("all")
    out: list[int | str] = []
    for value in candidates:
        if value not in out:
            out.append(value)
    return out


def _inner_cv(y_train: np.ndarray) -> StratifiedKFold:
    counts = Counter(int(v) for v in y_train)
    n_splits = max(2, min(5, min(counts.values())))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1729)


def _make_model(model: str, n_features: int, n_train: int, seed: int) -> tuple[Any, dict[str, list[Any]]]:
    if model == "l1_logistic":
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif)),
                (
                    "model",
                    LogisticRegression(
                        penalty="l1",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=int(seed),
                    ),
                ),
            ]
        )
        return pipe, {
            "select__k": _candidate_k_values(n_features, n_train),
            "model__C": [0.01, 0.1, 1.0, 10.0],
        }
    if model == "linear_svm":
        pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif)),
                (
                    "model",
                    LinearSVC(
                        class_weight="balanced",
                        max_iter=8000,
                        random_state=int(seed),
                        dual="auto",
                    ),
                ),
            ]
        )
        return pipe, {
            "select__k": _candidate_k_values(n_features, n_train),
            "model__C": [0.01, 0.1, 1.0, 10.0],
        }
    if model == "dummy_majority":
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", DummyClassifier(strategy="most_frequent"))]), {}
    raise ValueError(f"unknown model: {model}")


def nested_loocv_predictions(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    feature_ids: Sequence[str],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    n = len(y)
    predictions = np.zeros(n, dtype=int)
    scores = np.zeros(n, dtype=float)
    stability_rows: list[dict[str, Any]] = []
    for test_index in range(n):
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_index] = False
        X_train, y_train = X[train_mask], y[train_mask]
        X_test = X[[test_index]]
        estimator, grid = _make_model(model, X.shape[1], len(y_train), seed + test_index)
        if grid:
            search = GridSearchCV(
                estimator,
                grid,
                cv=_inner_cv(y_train),
                scoring="roc_auc",
                error_score=np.nan,
                n_jobs=1,
            )
            with np.errstate(all="ignore"):
                search.fit(X_train, y_train)
            fitted = search.best_estimator_
            best_params = search.best_params_
        else:
            fitted = estimator.fit(X_train, y_train)
            best_params = {}
        predictions[test_index] = int(fitted.predict(X_test)[0])
        if hasattr(fitted, "decision_function"):
            scores[test_index] = float(fitted.decision_function(X_test)[0])
        elif hasattr(fitted, "predict_proba"):
            scores[test_index] = float(fitted.predict_proba(X_test)[0, 1])
        else:
            scores[test_index] = float(predictions[test_index])
        stability_rows.extend(
            _selected_feature_rows(
                fitted,
                feature_ids=feature_ids,
                outer_test_index=test_index,
                best_params=best_params,
                model=model,
            )
        )
    return predictions, scores, stability_rows


def _selected_feature_rows(
    fitted: Any,
    *,
    feature_ids: Sequence[str],
    outer_test_index: int,
    best_params: dict[str, Any],
    model: str,
) -> list[dict[str, Any]]:
    if model == "dummy_majority" or not hasattr(fitted, "named_steps"):
        return []
    selector = fitted.named_steps.get("select")
    model_step = fitted.named_steps.get("model")
    if selector is None or model_step is None or not hasattr(selector, "get_support"):
        return []
    support = np.flatnonzero(selector.get_support())
    if support.size == 0:
        return []
    coefficients = getattr(model_step, "coef_", None)
    if coefficients is None:
        weights = np.ones(support.size, dtype=float)
    else:
        weights = np.ravel(coefficients)
        if weights.size != support.size:
            weights = np.resize(weights, support.size)
    order = np.argsort(np.abs(weights))[::-1][:200]
    rows = []
    for rank, local_index in enumerate(order, start=1):
        global_index = int(support[local_index])
        rows.append(
            {
                "outer_test_index": int(outer_test_index),
                "feature_id": str(feature_ids[global_index]),
                "rank": int(rank),
                "weight": float(weights[local_index]),
                "model": model,
                "best_params": json.dumps(best_params, ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _bootstrap_ci(
    y: np.ndarray,
    pred: np.ndarray,
    score: np.ndarray,
    *,
    metric_name: str,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    if n_boot <= 0:
        return math.nan, math.nan
    rng = np.random.default_rng(seed)
    indices_by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    values = []
    for _ in range(int(n_boot)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in indices_by_class if len(indices)]
        )
        if len(np.unique(y[sampled])) < 2:
            continue
        perf = performance_from_predictions(y[sampled], pred[sampled], score[sampled])
        values.append(float(perf[metric_name]))
    if not values:
        return math.nan, math.nan
    return float(np.nanquantile(values, 0.025)), float(np.nanquantile(values, 0.975))


def _permutation_p_value(
    X: np.ndarray,
    y: np.ndarray,
    observed_auc: float,
    *,
    model: str,
    feature_ids: Sequence[str],
    n_perm: int,
    seed: int,
) -> float:
    if n_perm <= 0 or not np.isfinite(observed_auc):
        return math.nan
    rng = np.random.default_rng(seed)
    count = 0
    valid = 0
    for i in range(int(n_perm)):
        y_perm = rng.permutation(y)
        if len(np.unique(y_perm)) < 2:
            continue
        pred, score, _ = nested_loocv_predictions(
            X,
            y_perm,
            model=model,
            feature_ids=feature_ids,
            seed=seed + 10_000 + i,
        )
        auc = _finite_metric_value(roc_auc_score, y_perm, score)
        if np.isfinite(auc):
            valid += 1
            count += int(auc >= observed_auc)
    return float((count + 1) / (valid + 1)) if valid else math.nan


def run_classification_stage(config: FmriStage2Config, feature_manifest: pd.DataFrame | None = None) -> pd.DataFrame:
    class_dir = Path(config.classification_result_dir)
    metrics_dir = Path(config.metrics_result_dir)
    class_dir.mkdir(parents=True, exist_ok=True)
    (class_dir / "classification").mkdir(parents=True, exist_ok=True)
    (class_dir / "reports").mkdir(parents=True, exist_ok=True)
    if feature_manifest is None:
        feature_manifest = pd.read_parquet(metrics_dir / "metrics" / "feature_manifest.parquet")
    ok = feature_manifest[feature_manifest["status"].eq("ok")].copy()
    models = ("l1_logistic", "linear_svm", "dummy_majority") if not config.smoke else ("l1_logistic", "dummy_majority")
    oof_rows: list[dict[str, Any]] = []
    perf_rows: list[dict[str, Any]] = []
    perm_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []
    for keys, group in ok.groupby(["representation", "branch", "metric", "lag"], sort=True):
        representation, branch, metric_name, lag = keys
        try:
            X, y, subjects, feature_ids = _assemble_feature_table(group)
        except Exception as exc:
            perf_rows.append(
                {
                    "representation": representation,
                    "branch": branch,
                    "metric": metric_name,
                    "lag": lag,
                    "model": "all",
                    "status": "failed_feature_assembly",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if len(subjects) < 6 or len(np.unique(y)) < 2:
            continue
        for model in models:
            pred, score, fold_stability = nested_loocv_predictions(
                X,
                y,
                model=model,
                feature_ids=feature_ids,
                seed=config.random_seed,
            )
            perf = performance_from_predictions(y, pred, score)
            for subject, label, prediction, value in zip(subjects, y, pred, score):
                oof_rows.append(
                    {
                        "subject_id": subject,
                        "true_group": "SZ" if int(label) == 1 else "HC",
                        "representation": representation,
                        "branch": branch,
                        "metric": metric_name,
                        "lag": int(lag),
                        "model": model,
                        "oof_predicted_group": "SZ" if int(prediction) == 1 else "HC",
                        "oof_score_SZ": float(value),
                    }
                )
            perf_rows.append(
                {
                    "representation": representation,
                    "branch": branch,
                    "metric": metric_name,
                    "lag": int(lag),
                    "model": model,
                    "n_subjects": len(subjects),
                    "n_features": X.shape[1],
                    "status": "ok",
                    **perf,
                }
            )
            ci_low, ci_high = _bootstrap_ci(
                y,
                pred,
                score,
                metric_name="roc_auc",
                n_boot=config.bootstraps if not config.smoke else min(config.bootstraps, 20),
                seed=config.random_seed,
            )
            boot_rows.append(
                {
                    "representation": representation,
                    "branch": branch,
                    "metric": metric_name,
                    "lag": int(lag),
                    "model": model,
                    "metric_name": "roc_auc",
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_bootstraps": config.bootstraps if not config.smoke else min(config.bootstraps, 20),
                }
            )
            n_perm = config.permutations if not config.smoke else min(config.permutations, 5)
            p_value = _permutation_p_value(
                X,
                y,
                float(perf["roc_auc"]),
                model=model,
                feature_ids=feature_ids,
                n_perm=n_perm,
                seed=config.random_seed,
            )
            perm_rows.append(
                {
                    "representation": representation,
                    "branch": branch,
                    "metric": metric_name,
                    "lag": int(lag),
                    "model": model,
                    "observed_roc_auc": perf["roc_auc"],
                    "permutation_p_value": p_value,
                    "n_permutations": n_perm,
                }
            )
            for row in fold_stability:
                stability_rows.append(
                    {
                        "representation": representation,
                        "branch": branch,
                        "metric": metric_name,
                        "lag": int(lag),
                        **row,
                    }
                )
    oof = pd.DataFrame(oof_rows)
    performance = pd.DataFrame(perf_rows)
    permutations = pd.DataFrame(perm_rows)
    boot = pd.DataFrame(boot_rows)
    stability = summarize_feature_stability(pd.DataFrame(stability_rows))
    fdr = compute_primary_fdr(permutations)
    paired = paired_representation_comparison(oof, performance, bootstraps=config.bootstraps if not config.smoke else 20, seed=config.random_seed)

    oof.to_csv(class_dir / "classification" / "oof_subject_predictions.csv", index=False, encoding="utf-8-sig")
    performance.to_csv(class_dir / "classification" / "metric_performance.csv", index=False, encoding="utf-8-sig")
    permutations.to_csv(class_dir / "classification" / "permutation_results.csv", index=False, encoding="utf-8-sig")
    boot.to_csv(class_dir / "classification" / "bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    fdr.to_csv(class_dir / "classification" / "fdr_results.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(class_dir / "classification" / "gm_vs_whole_paired_comparison.csv", index=False, encoding="utf-8-sig")
    stability.to_csv(class_dir / "classification" / "feature_stability.csv", index=False, encoding="utf-8-sig")
    write_stage2_report(class_dir / "reports" / "final_method_comparison.md", performance, fdr, paired, config)
    return performance


def summarize_feature_stability(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(
            columns=[
                "representation",
                "branch",
                "metric",
                "lag",
                "model",
                "feature_id",
                "selection_count",
                "median_abs_weight",
            ]
        )
    rows["abs_weight"] = rows["weight"].abs()
    return (
        rows.groupby(["representation", "branch", "metric", "lag", "model", "feature_id"], as_index=False)
        .agg(selection_count=("feature_id", "size"), median_abs_weight=("abs_weight", "median"))
        .sort_values(["selection_count", "median_abs_weight"], ascending=[False, False])
    )


def compute_primary_fdr(permutations: pd.DataFrame) -> pd.DataFrame:
    if permutations.empty:
        return pd.DataFrame()
    primary = permutations[
        permutations["representation"].isin(["gm_active_mean", "whole_brain"])
        & permutations["branch"].eq("baseline_without_GSR")
        & permutations["model"].eq("l1_logistic")
    ].copy()
    if primary.empty:
        return primary
    q, sig = _fdr_bh(primary["permutation_p_value"].fillna(1.0).to_numpy(dtype=float), alpha=0.05)
    primary["fdr_family"] = "26_metrics_x_2_core_representations"
    primary["q_value_BH"] = q
    primary["significant_FDR_0_05"] = sig
    return primary


def paired_representation_comparison(
    oof: pd.DataFrame,
    performance: pd.DataFrame,
    *,
    bootstraps: int,
    seed: int,
) -> pd.DataFrame:
    if oof.empty or performance.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    left = "gm_active_mean"
    right = "whole_brain"
    for (metric_name, branch, model), group in oof.groupby(["metric", "branch", "model"], sort=True):
        left_rows = group[group["representation"].eq(left)].copy()
        right_rows = group[group["representation"].eq(right)].copy()
        joined = left_rows.merge(
            right_rows,
            on=["subject_id", "true_group", "metric", "branch", "model", "lag"],
            suffixes=("_left", "_right"),
        )
        if joined.empty:
            continue
        y = joined["true_group"].map({"HC": 0, "SZ": 1}).to_numpy(dtype=int)
        score_l = joined["oof_score_SZ_left"].to_numpy(dtype=float)
        score_r = joined["oof_score_SZ_right"].to_numpy(dtype=float)
        pred_l = joined["oof_predicted_group_left"].map({"HC": 0, "SZ": 1}).to_numpy(dtype=int)
        pred_r = joined["oof_predicted_group_right"].map({"HC": 0, "SZ": 1}).to_numpy(dtype=int)
        perf_l = performance_from_predictions(y, pred_l, score_l)
        perf_r = performance_from_predictions(y, pred_r, score_r)
        deltas = _paired_bootstrap_auc_delta(y, score_l, score_r, bootstraps=bootstraps, seed=seed)
        rows.append(
            PairedRepresentationComparisonResult(
                metric=str(metric_name),
                branch=str(branch),
                model=str(model),
                left_representation=left,
                right_representation=right,
                n_subjects=int(len(joined)),
                roc_auc_delta=float(perf_l["roc_auc"] - perf_r["roc_auc"]),
                balanced_accuracy_delta=float(perf_l["balanced_accuracy"] - perf_r["balanced_accuracy"]),
                mcc_delta=float(perf_l["mcc"] - perf_r["mcc"]),
                bootstrap_ci_low=float(np.nanquantile(deltas, 0.025)) if len(deltas) else math.nan,
                bootstrap_ci_high=float(np.nanquantile(deltas, 0.975)) if len(deltas) else math.nan,
                paired_permutation_p=_paired_score_swap_p(y, score_l, score_r, seed=seed),
                n_changed_predictions=int(np.sum(pred_l != pred_r)),
                status="ok",
            ).as_dict()
        )
    return pd.DataFrame(rows)


def _paired_bootstrap_auc_delta(
    y: np.ndarray,
    score_l: np.ndarray,
    score_r: np.ndarray,
    *,
    bootstraps: int,
    seed: int,
) -> np.ndarray:
    if bootstraps <= 0:
        return np.asarray([], dtype=float)
    rng = np.random.default_rng(seed)
    indices_by_class = [np.flatnonzero(y == label) for label in (0, 1)]
    deltas = []
    for _ in range(int(bootstraps)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in indices_by_class if len(indices)]
        )
        if len(np.unique(y[sampled])) < 2:
            continue
        deltas.append(float(roc_auc_score(y[sampled], score_l[sampled]) - roc_auc_score(y[sampled], score_r[sampled])))
    return np.asarray(deltas, dtype=float)


def _paired_score_swap_p(y: np.ndarray, score_l: np.ndarray, score_r: np.ndarray, *, seed: int, n_perm: int = 1000) -> float:
    if len(np.unique(y)) < 2:
        return math.nan
    observed = float(roc_auc_score(y, score_l) - roc_auc_score(y, score_r))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        swap = rng.random(len(y)) < 0.5
        left = score_l.copy()
        right = score_r.copy()
        left[swap], right[swap] = right[swap], left[swap]
        delta = float(roc_auc_score(y, left) - roc_auc_score(y, right))
        count += int(abs(delta) >= abs(observed))
    return float((count + 1) / (n_perm + 1))


def write_stage2_report(
    path: Path,
    performance: pd.DataFrame,
    fdr: pd.DataFrame,
    paired: pd.DataFrame,
    config: FmriStage2Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HCP360 Stage 2 HC/SZ benchmark",
        "",
        "This is an exploratory method-comparison benchmark, not a diagnostic validation.",
        "",
        f"- Smoke mode: `{config.smoke}`",
        f"- Representations: `{', '.join(config.representations)}`",
        f"- Metrics requested: `{', '.join(config.metrics)}`",
        f"- Primary lag: `{config.primary_lag}`",
        f"- Permutations: `{config.permutations}`",
        f"- Bootstraps: `{config.bootstraps}`",
        "",
        "## Performance snapshot",
        "",
        _markdown_table(performance.head(50) if not performance.empty else performance),
        "",
        "## Primary FDR family",
        "",
        _markdown_table(fdr.head(80) if not fdr.empty else fdr),
        "",
        "## GM active-mean vs whole-brain paired comparison",
        "",
        _markdown_table(paired.head(80) if not paired.empty else paired),
        "",
        "## Guardrails",
        "",
        "- AAL3v2 remains fail-closed unless exact atlas validation succeeds.",
        "- Blocked subjects are not backfilled.",
        "- NaN/invalid metric values are carried into the ledger and handled inside train folds.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    view = df.copy()
    for col in view.columns:
        view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = [str(col) for col in view.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in view.columns) + " |")
    return "\n".join(lines)


def run_stage2_benchmark(config: FmriStage2Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        _status, features, _temporal = run_metric_stage(config)
        performance = run_classification_stage(config, features)
    return features, performance


__all__ = [
    "DEFAULT_BRANCHES",
    "DEFAULT_REPRESENTATIONS",
    "EXPECTED_STAGE2_METRICS",
    "FmriStage2Config",
    "MetricMatrixResult",
    "NestedLoocvBenchmarkResult",
    "PairedRepresentationComparisonResult",
    "PreprocessingBranchConfig",
    "RegionalInputManifest",
    "SubjectFeatureLedger",
    "compute_metric_matrix",
    "discover_hcp_inputs",
    "matrix_to_features",
    "preprocess_regional_timeseries",
    "registry_snapshot",
    "resolve_metrics",
    "run_classification_stage",
    "run_metric_stage",
    "run_stage2_benchmark",
]
