#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metric_benchmark.py — Predictive-power benchmark for neweds connectivity metrics
==================================================================================
Reads pre-computed connectivity matrices from neweds batch output,
evaluates each metric's ability to separate HC vs SZ.

Design (n=39–43, small sample):
  • LOOCV (leave-one-out) — stable for n<50
  • Feature selection INSIDE CV (SelectKBest, k chosen by inner 5-fold)
  • Two classifiers: LogReg L1 + SVM RBF
  • Permutation test (1000 shuffles) → p-value per metric
  • Bootstrap CI (2000 resamples) on AUC, balanced_accuracy, MCC
  • Final Excel: metric × model × {AUC, bacc, MCC, p_perm} + ranking

Input:
  Root folder with subject subfolders, each containing a data/ directory
  produced by neweds (with *_dense.npy / *_sparse.npz files).

Usage:
  python metric_benchmark.py --root "D:\\path\\time_series_analysis" --output results.xlsx

  # If HC IDs are not auto-detected, pass them explicitly:
  python metric_benchmark.py --root ... --hc_ids 1185 1186 1195 1196 1203 1207 1208 1212 1213 1217 1229 1230 1236 1242 1245
"""

import json
import logging
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse, stats

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    accuracy_score, f1_score,
)
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════
RANDOM_STATE = 42
N_PERM = 1000
N_BOOTSTRAP = 2000
CI_ALPHA = 0.025  # two-sided 95%
N_INNER_FOLDS = 5
K_BEST_CANDIDATES = [5, 10, 20, 50, 100]

# HC subject IDs from the "Здоровые (600 точек)" list
DEFAULT_HC_IDS = {
    1185, 1186, 1195, 1196, 1203, 1207, 1208,
    1212, 1213, 1217, 1229, 1230, 1236, 1242, 1245,
}

DIRECTED_VARIANTS = {
    "correlation_directed", "h2_directed", "granger_full", "granger_partial",
    "te_full", "te_partial", "ah_directed", "dcor_directed", "ordinal_directed",
}

# All known variant names from neweds registry (for file-name matching)
_ALL_VARIANT_NAMES = sorted([
    "correlation_full", "correlation_spearman", "correlation_kendall",
    "correlation_partial", "correlation_directed",
    "h2_full", "h2_partial", "h2_directed",
    "mutinf_full", "mutinf_partial",
    "coherence_full", "coherence_partial",
    "granger_full", "granger_partial",
    "te_full", "te_partial",
    "dcor_full", "dcor_partial", "dcor_directed",
    "ordinal_full", "ordinal_directed",
], key=len, reverse=True)  # longest first for greedy matching


# ═══════════════════════════════════════════════════════════════
# 1. Data loading
# ═══════════════════════════════════════════════════════════════

def parse_subject_id(folder_name: str) -> Optional[int]:
    m = re.match(r"(\d+)", folder_name)
    return int(m.group(1)) if m else None


def discover_subjects(root: str) -> List[dict]:
    import zipfile
    subjects = []
    for entry in sorted(Path(root).iterdir()):
        if not entry.is_dir():
            continue
        data_dir = entry / "data"
        # Auto-extract data.zip if data/ doesn't exist
        if not data_dir.exists():
            data_zip = entry / "data.zip"
            if data_zip.exists():
                log.info(f"Extracting {data_zip}...")
                try:
                    with zipfile.ZipFile(str(data_zip), "r") as zf:
                        zf.extractall(str(entry))
                except Exception as e:
                    log.warning(f"  Failed to extract {data_zip}: {e}")
        if not data_dir.exists():
            continue
        sid = parse_subject_id(entry.name)
        if sid is None:
            continue
        subjects.append({"id": sid, "name": entry.name, "data_dir": str(data_dir)})
    log.info(f"Found {len(subjects)} subjects with data/ directories")
    return subjects


def _extract_variant_from_filename(stem: str, suffix: str) -> Optional[str]:
    """Extract variant name from file stem like '1097_correlation_full_dense'."""
    base = stem.rsplit(f"_{suffix}", 1)[0]  # remove _dense or _sparse
    for vname in _ALL_VARIANT_NAMES:
        if base.endswith(vname):
            return vname
    return None


def discover_variants(subjects: List[dict]) -> List[str]:
    variant_sets = []
    for subj in subjects:
        data_dir = Path(subj["data_dir"])
        variants = set()

        # Try manifest
        for mf in data_dir.glob("*_manifest.json"):
            try:
                m = json.loads(mf.read_text(encoding="utf-8"))
                for v in m.get("variants", {}):
                    variants.add(v)
            except Exception:
                pass

        # Fallback: scan files
        if not variants:
            for f in data_dir.glob("*_dense.npy"):
                v = _extract_variant_from_filename(f.stem, "dense")
                if v:
                    variants.add(v)
            for f in data_dir.glob("*_sparse.npz"):
                v = _extract_variant_from_filename(f.stem, "sparse")
                if v:
                    variants.add(v)

        variant_sets.append(variants)

    if not variant_sets:
        return []
    common = variant_sets[0]
    for vs in variant_sets[1:]:
        common &= vs
    result = sorted(common)
    log.info(f"{len(result)} common variants: {result}")
    return result


def load_matrix(data_dir: str, variant: str) -> Optional[np.ndarray]:
    dp = Path(data_dir)

    for f in dp.glob(f"*_{variant}_dense.npy"):
        return np.nan_to_num(np.load(str(f)), nan=0.0, posinf=0.0, neginf=0.0)

    for f in dp.glob(f"*_{variant}_sparse.npz"):
        return np.nan_to_num(sparse.load_npz(str(f)).toarray(), nan=0.0, posinf=0.0, neginf=0.0)

    for f in dp.glob(f"*_{variant}_dense.csv"):
        df = pd.read_csv(str(f), index_col=0)
        return np.nan_to_num(df.values.astype(float), nan=0.0, posinf=0.0, neginf=0.0)

    return None


def load_node_names(data_dir: str) -> Optional[List[str]]:
    """Load node names from nodes.csv in subject's data/ dir."""
    nodes_csv = Path(data_dir) / "nodes.csv"
    if nodes_csv.exists():
        df = pd.read_csv(str(nodes_csv))
        if "name" in df.columns:
            return df["name"].astype(str).tolist()
    # Fallback: try dense.csv which has named index/columns
    for f in Path(data_dir).glob("*_dense.csv"):
        df = pd.read_csv(str(f), index_col=0, nrows=0)
        return [str(c) for c in df.columns]
    return None


def reindex_matrix(mat: np.ndarray, src_names: List[str],
                   target_names: List[str]) -> np.ndarray:
    """Reindex a matrix from src_names ordering to target_names ordering.
    target_names must be a subset of src_names."""
    src_idx = {name: i for i, name in enumerate(src_names)}
    indices = [src_idx[n] for n in target_names]
    return mat[np.ix_(indices, indices)]


def find_common_nodes(subjects: List[dict]) -> Optional[List[str]]:
    """Find intersection of node names across all subjects, preserving order."""
    all_names = []
    for subj in subjects:
        names = load_node_names(subj["data_dir"])
        if names is None:
            log.warning(f"  No node names found for {subj['name']}")
            return None
        all_names.append(names)

    # Intersection preserving order of first subject
    common = set(all_names[0])
    for names in all_names[1:]:
        common &= set(names)

    # Preserve order from first subject
    ordered = [n for n in all_names[0] if n in common]
    return ordered


def scan_matrix_sizes(subjects: List[dict], test_variant: str) -> Dict[int, int]:
    """Scan all subjects and return {subject_id: matrix_dim}."""
    dims = {}
    for subj in subjects:
        mat = load_matrix(subj["data_dir"], test_variant)
        if mat is not None:
            dims[subj["id"]] = mat.shape[0]
    return dims


def matrix_to_features(mat: np.ndarray, variant: str) -> np.ndarray:
    n = mat.shape[0]
    if variant in DIRECTED_VARIANTS:
        mask = ~np.eye(n, dtype=bool)
        return mat[mask]
    else:
        return mat[np.triu_indices(n, k=1)]


# ═══════════════════════════════════════════════════════════════
# 2. Classification
# ═══════════════════════════════════════════════════════════════

def get_models() -> Dict[str, tuple]:
    return {
        "LogReg_L1": (
            LogisticRegression,
            dict(penalty="l1", solver="saga", max_iter=5000, random_state=RANDOM_STATE),
            "C", [0.001, 0.01, 0.1, 1.0, 10.0],
        ),
        "SVM_RBF": (
            SVC,
            dict(kernel="rbf", probability=True, random_state=RANDOM_STATE),
            "C", [0.01, 0.1, 1.0, 10.0, 100.0],
        ),
    }


def build_pipeline(model_cls, model_params, k_best: int):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("select", SelectKBest(f_classif, k=k_best)),
        ("model", model_cls(**model_params)),
    ])


def loocv_evaluate(X, y, model_cls, model_params, param_name, param_values):
    n = len(y)
    loo = LeaveOneOut()
    oof_pred = np.zeros(n, dtype=int)
    oof_prob = np.zeros(n, dtype=float)

    max_feat = X.shape[1]
    k_cands = [k for k in K_BEST_CANDIDATES if k <= max_feat]
    if not k_cands:
        k_cands = [max_feat]

    for train_idx, test_idx in loo.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        inner_cv = StratifiedKFold(N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        best_score, best_pv, best_k = -1, param_values[0], k_cands[0]

        for pv in param_values:
            for k in k_cands:
                if k > X_tr.shape[1]:
                    continue
                scores = []
                params = {**model_params, param_name: pv}
                for t2, v2 in inner_cv.split(X_tr, y_tr):
                    try:
                        pipe = build_pipeline(model_cls, params, k)
                        pipe.fit(X_tr[t2], y_tr[t2])
                        scores.append(balanced_accuracy_score(y_tr[v2], pipe.predict(X_tr[v2])))
                    except Exception:
                        scores.append(0.0)
                ms = np.mean(scores)
                if ms > best_score:
                    best_score, best_pv, best_k = ms, pv, k

        params = {**model_params, param_name: best_pv}
        pipe = build_pipeline(model_cls, params, best_k)
        pipe.fit(X_tr, y_tr)
        oof_pred[test_idx] = pipe.predict(X_te)
        try:
            oof_prob[test_idx] = pipe.predict_proba(X_te)[:, 1]
        except Exception:
            oof_prob[test_idx] = float(oof_pred[test_idx])

    return _compute_all_metrics(y, oof_pred, oof_prob)


def _compute_all_metrics(y_true, y_pred, y_prob):
    m = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }
    try:
        m["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        m["roc_auc"] = np.nan
    m["_pred"] = y_pred.copy()
    m["_prob"] = y_prob.copy()
    return m


# ═══════════════════════════════════════════════════════════════
# 3. Permutation test
# ═══════════════════════════════════════════════════════════════

def permutation_test(X, y, model_cls, model_params, param_name, param_values,
                     observed_bacc, n_perm=N_PERM):
    rng = np.random.RandomState(RANDOM_STATE)
    count_ge = 0
    for i in range(n_perm):
        if (i + 1) % 200 == 0:
            log.info(f"      perm {i+1}/{n_perm}")
        y_perm = rng.permutation(y)
        try:
            res = loocv_evaluate(X, y_perm, model_cls, model_params, param_name, param_values)
            if res["balanced_accuracy"] >= observed_bacc:
                count_ge += 1
        except Exception:
            pass
    return (count_ge + 1) / (n_perm + 1)


# ═══════════════════════════════════════════════════════════════
# 4. Bootstrap CI
# ═══════════════════════════════════════════════════════════════

def _bootstrap(y_true, y_pred, y_prob, fn, use_prob=False, n_boot=N_BOOTSTRAP):
    rng = np.random.RandomState(RANDOM_STATE)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            val = fn(yt, y_prob[idx]) if use_prob else fn(yt, y_pred[idx])
            scores.append(val)
        except Exception:
            continue
    if not scores:
        return np.nan, np.nan, np.nan
    return np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)


# ═══════════════════════════════════════════════════════════════
# 5. Ensemble
# ═══════════════════════════════════════════════════════════════

def ensemble_evaluate(X_dict, y, top_variants):
    X_concat = np.hstack([X_dict[v] for v in top_variants])
    log.info(f"  Ensemble: {len(top_variants)} metrics → {X_concat.shape[1]} features")
    return loocv_evaluate(
        X_concat, y,
        LogisticRegression,
        dict(penalty="l1", solver="saga", max_iter=5000, random_state=RANDOM_STATE),
        "C", [0.001, 0.01, 0.1, 1.0, 10.0],
    )


# ═══════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Hardcoded config ──
    ROOT = r"D:\Шизофрения (600 точек)\time_series_analysis"

    print("=" * 70)
    print("  NEWEDS METRIC BENCHMARK: HC vs SZ classification")
    print("=" * 70)
    print()
    print("  1 = быстрый режим (без permutation test)")
    print("  0 = полный режим  (с permutation test, ~часы)")
    print()
    mode = input("  Режим [1/0]: ").strip()
    skip_perm = mode != "0"
    n_perm = N_PERM if not skip_perm else 0
    top_k_ensemble = 3
    skip_ensemble = False

    root = Path(ROOT)
    output_path = root / "benchmark_results.xlsx"
    hc_ids = DEFAULT_HC_IDS

    print("=" * 70)
    print("  NEWEDS METRIC BENCHMARK: HC vs SZ classification")
    print("=" * 70)
    print(f"  Root:     {root}")
    print(f"  HC IDs:   {sorted(hc_ids)}")
    print(f"  Perms:    {'SKIP' if skip_perm else n_perm}")
    print("=" * 70)

    # ── Discover subjects ──
    subjects = discover_subjects(str(root))
    if len(subjects) < 10:
        log.error(f"Only {len(subjects)} subjects — check --root")
        sys.exit(1)

    for s in subjects:
        s["label"] = 0 if s["id"] in hc_ids else 1
        s["group"] = "HC" if s["label"] == 0 else "SZ"

    n_hc = sum(1 for s in subjects if s["label"] == 0)
    n_sz = sum(1 for s in subjects if s["label"] == 1)
    y = np.array([s["label"] for s in subjects])
    print(f"\n  Subjects: {len(subjects)} ({n_hc} HC, {n_sz} SZ)\n")

    # ── Discover & load variants ──
    variants = discover_variants(subjects)
    if not variants:
        log.error("No common variants found")
        sys.exit(1)

    # ── Pre-scan: matrix sizes + node intersection ──
    _test_variant = variants[0]
    _dims = scan_matrix_sizes(subjects, _test_variant)
    from collections import Counter

    # Exclude subjects where matrix is missing entirely
    missing_subj = [s for s in subjects if s["id"] not in _dims]
    if missing_subj:
        log.warning(f"Excluding {len(missing_subj)} subjects with no matrices: "
                    f"{[s['name'] for s in missing_subj]}")
        subjects = [s for s in subjects if s["id"] in _dims]
        y = np.array([s["label"] for s in subjects])
        n_hc = sum(1 for s in subjects if s["label"] == 0)
        n_sz = sum(1 for s in subjects if s["label"] == 1)

    if _dims:
        dim_counts = Counter(_dims.values())
        print(f"\n  Matrix size distribution ({_test_variant}):")
        for dim, cnt in dim_counts.most_common():
            ids_with_dim = [sid for sid, d in _dims.items() if d == dim]
            hc_in = sum(1 for sid in ids_with_dim if sid in hc_ids)
            sz_in = len(ids_with_dim) - hc_in
            print(f"    {dim}×{dim}: {cnt} subjects ({hc_in} HC, {sz_in} SZ)")

    # Find common nodes across all subjects (intersection by spatial bin names)
    need_reindex = len(set(_dims.values())) > 1
    common_nodes = None
    subject_nodes = {}  # id → list of node names

    if need_reindex:
        print(f"\n  Different matrix sizes detected → finding common nodes...")
        common_nodes = find_common_nodes(subjects)
        if common_nodes is None:
            log.error("Cannot find node names (no nodes.csv or dense.csv). "
                      "Re-run neweds with fixed bin_range or provide nodes.csv.")
            sys.exit(1)
        # Cache per-subject node names
        for subj in subjects:
            names = load_node_names(subj["data_dir"])
            subject_nodes[subj["id"]] = names

        n_common = len(common_nodes)
        max_dim = max(_dims.values())
        min_dim = min(_dims.values())
        print(f"    Node intersection: {n_common} common nodes "
              f"(from range {min_dim}–{max_dim})")
        print(f"    All {len(subjects)} subjects retained — no exclusions")
    else:
        dim_val = list(_dims.values())[0] if _dims else 0
        print(f"\n  All matrices are {dim_val}×{dim_val} — no reindexing needed")

    X_dict = {}
    variant_info = {}
    for variant in variants:
        log.info(f"Loading {variant}...")
        feats = []
        ok = True
        n_raw = None
        mat_shape = None
        for subj in subjects:
            mat = load_matrix(subj["data_dir"], variant)
            if mat is None:
                log.warning(f"  Missing {variant} for {subj['name']}")
                ok = False
                break
            # Reindex to common nodes if needed
            if need_reindex and common_nodes is not None:
                src_names = subject_nodes.get(subj["id"])
                if src_names is None or not set(common_nodes).issubset(set(src_names)):
                    log.warning(f"  {subj['name']}: node names don't cover common set, skipping variant")
                    ok = False
                    break
                mat = reindex_matrix(mat, src_names, common_nodes)
            mat_shape = mat.shape
            feat = matrix_to_features(mat, variant)
            if n_raw is None:
                n_raw = len(feat)
            elif len(feat) != n_raw:
                log.warning(f"  Dim mismatch {subj['name']}: {len(feat)} vs {n_raw}, skipping variant")
                ok = False
                break
            feats.append(feat)

        if not ok or not feats:
            continue

        X = np.vstack(feats)
        var_mask = X.std(axis=0) > 1e-12
        X = X[:, var_mask]
        if X.shape[1] < 2:
            log.warning(f"  {variant}: <2 features after filtering")
            continue

        X_dict[variant] = X
        variant_info[variant] = {
            "n_raw": n_raw, "n_used": X.shape[1],
            "mat_shape": mat_shape, "directed": variant in DIRECTED_VARIANTS,
        }
        log.info(f"  → {X.shape[1]} features")

    print(f"\n  Loaded {len(X_dict)} variants\n")
    if not X_dict:
        log.error("No variants loaded")
        sys.exit(1)

    # ── Evaluate ──
    models = get_models()
    all_results = []

    for vi, variant in enumerate(sorted(X_dict), 1):
        X = X_dict[variant]
        info = variant_info[variant]
        print(f"{'─'*60}")
        print(f"  [{vi}/{len(X_dict)}] {variant}  ({X.shape[1]} features, "
              f"{'directed' if info['directed'] else 'symmetric'})")

        for model_name, (mcls, mpar, pname, pvals) in models.items():
            t0 = time.time()
            res = loocv_evaluate(X, y, mcls, mpar, pname, pvals)
            elapsed = time.time() - t0

            row = {
                "variant": variant, "model": model_name,
                "n_features": X.shape[1], "directed": info["directed"],
                "accuracy": res["accuracy"],
                "balanced_accuracy": res["balanced_accuracy"],
                "f1": res["f1"], "mcc": res["mcc"], "roc_auc": res["roc_auc"],
            }

            # Bootstrap CIs
            bacc_ci = _bootstrap(y, res["_pred"], res["_prob"], balanced_accuracy_score)
            auc_ci = _bootstrap(y, res["_pred"], res["_prob"], roc_auc_score, use_prob=True)
            mcc_ci = _bootstrap(y, res["_pred"], res["_prob"], matthews_corrcoef)
            row["bacc_ci95"] = f"[{bacc_ci[1]:.3f}, {bacc_ci[2]:.3f}]"
            row["auc_ci95"] = f"[{auc_ci[1]:.3f}, {auc_ci[2]:.3f}]"
            row["mcc_ci95"] = f"[{mcc_ci[1]:.3f}, {mcc_ci[2]:.3f}]"

            # Permutation
            if not skip_perm:
                log.info(f"    Permutation test ({n_perm})...")
                row["p_perm"] = permutation_test(
                    X, y, mcls, mpar, pname, pvals,
                    res["balanced_accuracy"], n_perm,
                )
            else:
                row["p_perm"] = np.nan

            row["time_sec"] = round(elapsed, 1)
            sig = ""
            if not skip_perm:
                p = row["p_perm"]
                sig = f"  p={p:.4f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'ns'}"
            print(f"    {model_name}: bacc={res['balanced_accuracy']:.3f}  "
                  f"AUC={res['roc_auc']:.3f}  MCC={res['mcc']:.3f}{sig}")

            all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # ── Ensemble ──
    ens_rows = []
    if not skip_ensemble and len(X_dict) >= 2:
        print(f"\n{'═'*60}")
        print(f"  ENSEMBLE (top {top_k_ensemble})")
        best_pv = results_df.groupby("variant")["balanced_accuracy"].max()
        top_v = best_pv.nlargest(top_k_ensemble).index.tolist()
        print(f"  Members: {top_v}")
        eres = ensemble_evaluate(X_dict, y, top_v)
        erow = {
            "variant": f"ENSEMBLE_top{top_k_ensemble}",
            "model": "LogReg_L1",
            "n_features": sum(X_dict[v].shape[1] for v in top_v),
            "directed": False,
            "accuracy": eres["accuracy"],
            "balanced_accuracy": eres["balanced_accuracy"],
            "f1": eres["f1"], "mcc": eres["mcc"], "roc_auc": eres["roc_auc"],
        }
        bc = _bootstrap(y, eres["_pred"], eres["_prob"], balanced_accuracy_score)
        ac = _bootstrap(y, eres["_pred"], eres["_prob"], roc_auc_score, use_prob=True)
        mc = _bootstrap(y, eres["_pred"], eres["_prob"], matthews_corrcoef)
        erow["bacc_ci95"] = f"[{bc[1]:.3f}, {bc[2]:.3f}]"
        erow["auc_ci95"] = f"[{ac[1]:.3f}, {ac[2]:.3f}]"
        erow["mcc_ci95"] = f"[{mc[1]:.3f}, {mc[2]:.3f}]"
        erow["p_perm"] = np.nan
        erow["time_sec"] = 0
        erow["ensemble_members"] = ", ".join(top_v)
        print(f"  → bacc={eres['balanced_accuracy']:.3f}  AUC={eres['roc_auc']:.3f}  MCC={eres['mcc']:.3f}")
        ens_rows.append(erow)

    # ── Save Excel ──
    full_df = pd.concat([results_df, pd.DataFrame(ens_rows)], ignore_index=True)
    full_df = full_df.sort_values("balanced_accuracy", ascending=False).reset_index(drop=True)
    full_df.insert(0, "rank", range(1, len(full_df) + 1))

    # Summary: best model per variant
    summary_rows = []
    for variant in sorted(X_dict):
        vdf = results_df[results_df["variant"] == variant]
        best = vdf.loc[vdf["balanced_accuracy"].idxmax()]
        summary_rows.append({
            "variant": variant, "best_model": best["model"],
            "n_features": best["n_features"],
            "balanced_accuracy": best["balanced_accuracy"],
            "roc_auc": best["roc_auc"], "mcc": best["mcc"],
            "p_perm": best["p_perm"],
            "bacc_ci95": best["bacc_ci95"], "auc_ci95": best["auc_ci95"],
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("balanced_accuracy", ascending=False)
    summary_df.insert(0, "rank", range(1, len(summary_df) + 1))

    subj_df = pd.DataFrame([{"id": s["id"], "name": s["name"], "group": s["group"]} for s in subjects])
    n_min = min(n_hc, n_sz)
    d_80 = 2.8 / np.sqrt(n_min)
    power_df = pd.DataFrame([{
        "n_hc": n_hc, "n_sz": n_sz, "n_total": len(subjects),
        "min_detectable_d_80pct": round(d_80, 3),
        "medium_effect_detectable": "YES" if d_80 <= 0.5 else "NO (underpowered for d=0.5)",
        "max_features_conservative": n_min // 10,
        "cv_method": "LOOCV",
        "n_permutations": n_perm if not skip_perm else "SKIPPED",
    }])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(str(output_path), engine="openpyxl") as w:
        summary_df.to_excel(w, sheet_name="Summary", index=False)
        full_df.to_excel(w, sheet_name="All results", index=False)
        power_df.to_excel(w, sheet_name="Power analysis", index=False)
        subj_df.to_excel(w, sheet_name="Subjects", index=False)

    print(f"\n{'═'*60}")
    print(f"  ✓ Saved → {output_path}")
    print(f"\n  TOP-5:")
    for _, r in full_df.head(5).iterrows():
        p = r.get("p_perm", np.nan)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns" if np.isfinite(p) else "?"
        print(f"    {int(r['rank']):2d}. {r['variant']:25s} {r['model']:12s}  "
              f"bacc={r['balanced_accuracy']:.3f}  AUC={r['roc_auc']:.3f}  {sig}")


if __name__ == "__main__":
    main()
