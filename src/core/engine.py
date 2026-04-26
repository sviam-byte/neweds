#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Legacy analysis engine.

This module contains the original BigMasterTool implementation kept for
backward compatibility with the GUI/web interfaces.

New portfolio-facing code should use:
    src.core.pipeline.run_analysis
    src.core.metric_runner.compute_metric
    src.core.results.AnalysisResult

Do not add new public pipeline logic here.
"""
import argparse, logging, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.vector_ar.var_model import VAR

from ..config import (
    DEFAULT_EMBED_DIM, DEFAULT_EMBED_TAU, DEFAULT_MAX_LAG, DEFAULT_PVALUE_ALPHA,
    SAVE_FOLDER, AnalysisConfig,
    is_control_sensitive_method, is_directed_method, is_pvalue_method,
)
from ..analysis import stats as analysis_stats
from ..analysis.dimred import apply_dimred
from ..metrics import connectivity as metrics_connectivity
from ..metrics.registry import METRICS_REGISTRY, get_metric_func, register_metric
from ..reporting import ExcelReportWriter, HTMLReportGenerator
from ..visualization import plots
from .data_loader import load_or_generate, preprocess_timeseries
from .fft_analysis import fft_analysis, frequency_analysis, plot_coherence_vs_frequency
from .pair_resolver import resolve_pairs
from .preprocessing import configure_warnings
from .statistics import (
    apply_pvalue_correction_matrix, as_float64_1d,
    lag_quality, pair_score, select_best_median_worst,
)
from .window_scanner import analyze_sliding_windows

plt = plots.plt

# ── RunLog ───────────────────────────────────────────────────────────────
class RunLog:
    def __init__(self): self.items: List[str] = []
    def add(self, msg):
        try: self.items.append(str(msg))
        except: pass
    def as_text(self): return "\n".join(self.items)

# ── AH metrics ───────────────────────────────────────────────────────────
def _H_ratio_direction(X, Y, m=DEFAULT_EMBED_DIM, tau=DEFAULT_EMBED_TAU):
    n = len(X)
    if len(Y) != n or n < 2: return None
    L = n - (m-1)*tau
    if L < 2: return None
    Xs = np.zeros((L, m)); Ys = np.zeros((L, m))
    for j in range(m):
        Xs[:, j] = X[j*tau:j*tau+L]; Ys[:, j] = Y[j*tau:j*tau+L]
    v = ~np.isnan(Xs).any(1) & ~np.isnan(Ys).any(1)
    if not np.any(v): return None
    Xv, Yv = Xs[v], Ys[v]
    if len(Xv) < 2: return None
    tX = cKDTree(Xv); _, iX = tX.query(Xv, k=2)
    if iX.shape[1] < 2: return None
    nn = iX[:, 1]; dY1 = np.sqrt(np.sum((Yv - Yv[nn])**2, axis=1))
    tY = cKDTree(Yv); dY, _ = tY.query(Yv, k=2)
    dY2 = np.where(dY[:, 1] == 0, 1e-10, dY[:, 1])
    r = dY1/dY2; r = r[np.isfinite(r)]
    return float(np.mean(r)) if len(r) > 0 else None

def AH_matrix(df, embed_dim=DEFAULT_EMBED_DIM, tau=DEFAULT_EMBED_TAU, pairs=None, **_kw):
    from ..metrics.connectivity import _get_effective_pairs, _iter_pairs
    df = df.dropna(axis=0, how='any'); N = df.shape[1]; out = np.zeros((N, N)); arr = df.values
    if pairs is not None: eff = _iter_pairs(N, pairs, directed=True)
    else:
        nf = N*(N-1)
        if nf > 10_000_000:
            mp = min(500_000, N*5); rng = np.random.default_rng(42); es = set()
            bi, bj = rng.integers(0, N, size=mp*3), rng.integers(0, N, size=mp*3)
            for ii, jj in zip(bi, bj):
                if ii != jj: es.add((int(ii), int(jj)))
                if len(es) >= mp: break
            eff = list(es)
        else: eff = [(s,t) for s in range(N) for t in range(N) if s != t]
    def _ah(pair):
        s, t = pair; H = _H_ratio_direction(arr[:,s], arr[:,t], m=embed_dim, tau=tau)
        return (s, t, 0.0) if H is None or H <= 0 else (s, t, min(1.0, 1.0/H))
    if len(eff) > 800:
        try:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=-1, backend="loky")(delayed(_ah)(p) for p in eff)
        except ImportError: res = [_ah(p) for p in eff]
    else: res = [_ah(p) for p in eff]
    for s, t, v in res: out[s, t] = v
    return out

def compute_partial_AH_matrix(data, max_lag=DEFAULT_MAX_LAG, embed_dim=DEFAULT_EMBED_DIM,
                               tau=DEFAULT_EMBED_TAU, control=None, pairs=None):
    df = data.dropna(axis=0, how='any'); N = df.shape[1]
    if N < 2: return np.zeros((N, N))
    if control and len(control) > 0:
        rdf = pd.DataFrame(index=df.index)
        for col in df.columns:
            Xc = df[control]; y = df[col]
            if len(Xc) > 0 and not Xc.isnull().any().any():
                try: mdl = LinearRegression().fit(Xc.values, y.values); rdf[col] = y.values - mdl.predict(Xc.values)
                except: rdf[col] = y
            else: rdf[col] = y
    else:
        try: r = VAR(df.values).fit(max_lag, ic=None); rdf = pd.DataFrame(r.resid, columns=df.columns)
        except: rdf = df
    return AH_matrix(rdf, embed_dim=embed_dim, tau=tau, pairs=pairs)

for _n, _f in {
    "ah_full": lambda d, lag=1, control=None, **kw: AH_matrix(d, pairs=kw.get("pairs")),
    "ah_partial": lambda d, lag=1, control=None, **kw: compute_partial_AH_matrix(d, max_lag=lag, control=control, pairs=kw.get("pairs")),
    "ah_directed": lambda d, lag=1, control=None, **kw: (AH_matrix(d, pairs=kw.get("pairs")) if not control
        else compute_partial_AH_matrix(d, max_lag=lag, control=control, pairs=kw.get("pairs"))),
}.items():
    register_metric(_n, _f)

method_mapping = dict(METRICS_REGISTRY)

# Legacy metric API re-exports kept for compatibility with older tests and callers.
correlation_matrix = metrics_connectivity.correlation_matrix
partial_correlation_matrix = metrics_connectivity.partial_correlation_matrix
partial_h2_matrix = metrics_connectivity.partial_h2_matrix
lagged_directed_correlation = metrics_connectivity.lagged_directed_correlation
coherence_matrix = metrics_connectivity.coherence_matrix
mutual_info_matrix = metrics_connectivity.mutual_info_matrix
mutual_info_matrix_partial = metrics_connectivity.mutual_info_matrix_partial
granger_matrix = metrics_connectivity.granger_matrix
granger_matrix_partial = metrics_connectivity.granger_matrix_partial
compute_granger_matrix = metrics_connectivity.granger_matrix
TE_matrix = metrics_connectivity.transfer_entropy_matrix
TE_matrix_partial = metrics_connectivity.transfer_entropy_matrix_partial

def h2_matrix(df, lag=1, control=None, **kwargs):
    return metrics_connectivity.correlation_matrix(df, lag=lag, control=control, **kwargs) ** 2


def lagged_directed_h2(df, lag=1, control=None, **kwargs):
    return metrics_connectivity.lagged_directed_correlation(df, lag=lag, control=control, **kwargs) ** 2


def granger_dict(df, lag=1, control=None, **kwargs):
    return {"matrix": metrics_connectivity.granger_matrix(df, lag=lag, control=control, **kwargs)}

# ── Method metadata ──────────────────────────────────────────────────────
@dataclass(frozen=True)
class MethodSpec:
    directed: bool; is_p_value: bool; control_dependent: bool; supports_lag: bool; description: str = ""

def get_method_spec(v):
    return MethodSpec(directed=is_directed_method(v), is_p_value=is_pvalue_method(v),
                      control_dependent=is_control_sensitive_method(v), supports_lag=is_directed_method(v))

# ── Connectivity dispatcher ──────────────────────────────────────────────
def compute_connectivity_variant(data, variant, lag=1, control=None, *, pairs=None,
    partial_mode="global", pairwise_policy="others", custom_controls=None,
    control_strategy="none", control_pca_k=0):
    try:
        if control is not None and len(control) == 0: control = None
        control_matrix = None; control_desc = []
        if isinstance(variant, str) and variant.endswith("_partial") and control is None and control_strategy != "none":
            try:
                n = len(data); ctrls = []
                if control_strategy in {"global","global_mean","global_mean_trend","mean_trend"}:
                    ctrls.append(pd.to_numeric(data.mean(axis=1), errors="coerce").to_numpy(dtype=np.float64))
                    control_desc.append("global_mean")
                if control_strategy in {"global_mean_trend","mean_trend","trend"}:
                    t = np.arange(n, dtype=np.float64)
                    t = (t-t.mean())/(t.std()+1e-12) if n > 1 else t
                    ctrls.append(t); control_desc.append("linear_trend")
                k = int(max(0, control_pca_k))
                if k > 0:
                    X = np.nan_to_num(data.to_numpy(dtype=np.float64), nan=0.0)
                    U, S, _ = np.linalg.svd(X, full_matrices=False)
                    for i in range(min(k, U.shape[1])):
                        ctrls.append(U[:,i]*S[i]); control_desc.append(f"pca[{i+1}]")
                if ctrls: control_matrix = np.vstack(ctrls).T
            except: control_matrix = None; control_desc = []
        if variant in method_mapping:
            return get_metric_func(variant)(data, lag=lag, control=control,
                control_matrix=control_matrix, control_desc=control_desc, pairs=pairs)
        from ..metrics import connectivity
        return connectivity.correlation_matrix(data)
    except Exception as e:
        logging.error("[ComputeVariant] %s failed: %s", variant, e)
        return None


# ── BigMasterTool ────────────────────────────────────────────────────────

class BigMasterTool:
    """Main orchestrator for time series connectivity analysis."""

    def __init__(self, data=None, enable_experimental=False, config=None, stage_callback=None):
        self.log = RunLog()
        self.data_raw = pd.DataFrame()
        self.data_preprocessed = pd.DataFrame()
        self.data_after_autodiff = pd.DataFrame()
        self.preprocessing_report = None
        self.autodiff_report = {"enabled": False, "differenced": []}
        self.dimred_report = {"enabled": False, "method": "none"}
        self.dimred_mapping = pd.DataFrame()
        self.data_dimred = pd.DataFrame()
        self.data_dimred_base = None
        self.pairwise_summaries = {}
        self.coords_df = None
        self.qc_raw = self.qc_clean = None
        self.graph_results = {}
        if data is not None:
            data = data.loc[:, (data != data.iloc[0]).any()]
            self.data = data.copy()
            try: self.data.attrs = {}
            except: pass
            for c in list(self.data.columns):
                self.data[c] = pd.to_numeric(self.data[c], errors="coerce")
            if len(self.data.columns) > 0 and isinstance(self.data.columns[0], int):
                self.data.columns = [f"c{i+1}" for i in range(self.data.shape[1])]
        else:
            self.data = pd.DataFrame()
        self.data_normalized = pd.DataFrame()
        self.results = {}
        self.results_meta = {}
        self.variant_lags = {}
        self.window_analysis = {}
        self.config = config or AnalysisConfig(enable_experimental=enable_experimental)
        self._stage_callback = stage_callback
        self.fs = 1.0
        self.undirected_methods = [m for m in method_mapping if not get_method_spec(m).directed]
        self.directed_methods = [m for m in method_mapping if get_method_spec(m).directed]

    def set_stage_callback(self, cb):
        self._stage_callback = cb

    def _stage(self, stage, progress=None, **meta):
        try: logging.info("[Stage] %s", stage)
        except: pass
        cb = getattr(self, "_stage_callback", None)
        if cb:
            try: cb(stage, progress, dict(meta or {}))
            except: pass

    # ── Data loading ─────────────────────────────────────────────────

    def load_data_excel(self, filepath, **kwargs):
        self._stage("Загрузка данных", 0.0)
        qc_enabled = bool(kwargs.pop("qc_enabled", True))
        try:
            self._stage("Загрузка RAW", 0.05)
            self.data_raw = load_or_generate(filepath, preprocess=False, normalize=False,
                remove_outliers=False, fill_missing=False, check_stationarity=False)
        except: self.data_raw = pd.DataFrame()
        self._stage("Загрузка + предобработка", 0.15)
        df_out = load_or_generate(filepath, return_report=True, **kwargs)
        if isinstance(df_out, tuple): self.data, self.preprocessing_report = df_out
        else: self.data, self.preprocessing_report = df_out, None
        self.data_preprocessed = self.data.copy()
        self._stage("Данные загружены", 0.35)
        try:
            notes = (self.preprocessing_report.notes if self.preprocessing_report else {}) or {}
            coords = notes.get("coords")
            if isinstance(coords, list) and coords: self.coords_df = pd.DataFrame(coords)
        except: pass
        try:
            if qc_enabled and (self.coords_df is not None or self.data.shape[1] >= 20):
                self._stage("QC", 0.45)
                if not self.data_raw.empty:
                    self.qc_raw = analysis_stats.voxel_qc(self.data_raw, coords=self.coords_df)
                self.qc_clean = analysis_stats.voxel_qc(self.data, coords=self.coords_df)
        except: pass
        try: self.results_meta.setdefault("__run__", {}).setdefault("qc_enabled", qc_enabled)
        except: pass
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        if self.config.auto_difference: self._apply_auto_diff()
        self.data_after_autodiff = self.data.copy()
        self._stage("Данные готовы", 0.70)
        return self.data

    def _apply_auto_diff(self):
        self.autodiff_report = {"enabled": True, "differenced": []}
        self._stage("Авто-дифференцирование", 0.55)
        cnt = 0
        for col in self.data.columns:
            if not pd.api.types.is_numeric_dtype(self.data[col]): continue
            _, pval = analysis_stats.test_stationarity(self.data[col])
            if pval is None or pval <= 0.05: continue
            s = pd.to_numeric(self.data[col], errors="coerce").astype(float)
            d = s.diff()
            if len(d) > 0: d.iloc[0] = 0.0
            mu = float(d.mean(skipna=True)) if np.isfinite(d.mean(skipna=True)) else 0.0
            d -= mu
            sd = float(d.std(skipna=True)) if np.isfinite(d.std(skipna=True)) else 0.0
            if sd > 1e-12: d /= sd
            self.data[col] = d.fillna(0.0); cnt += 1
            self.autodiff_report["differenced"].append(col)
        if cnt > 0: logging.warning("Differenced %d non-stationary series.", cnt)

    def normalize_data(self):
        if self.data.empty: return
        self._stage("Нормализация", 0.75)
        cols = [c for c in self.data.columns if pd.api.types.is_numeric_dtype(self.data[c])]
        self.data_normalized = self.data.copy()
        arr = self.data_normalized[cols].to_numpy(dtype=np.float64)
        means = np.nanmean(arr, axis=0); stds = np.nanstd(arr, axis=0); stds[stds < 1e-12] = 1.0
        self.data_normalized[cols] = (arr - means) / stds

    def _apply_fdr_correction(self):
        if self.config.pvalue_correction != "fdr_bh": return
        for v, mat in self.results.items():
            if mat is not None and is_pvalue_method(v):
                self.results[v] = apply_pvalue_correction_matrix(mat, directed=get_method_spec(v).directed)

    # ── Run methods ──────────────────────────────────────────────────

    def run_all_methods(self, **kwargs):
        self._stage("Подготовка", 0.72)
        self._maybe_apply_dimred(**kwargs); self._maybe_post_preprocess(**kwargs)
        self.normalize_data()
        if self.data_normalized.empty: return
        prev = dict((self.results_meta or {}).get("__run__", {}) or {})
        self.results = {}; self.results_meta = {"__run__": prev} if prev else {}
        self.variant_lags = {}; self.window_analysis = {}
        variants = list(method_mapping.keys()); nt = max(1, len(variants))
        for i, v in enumerate(variants, 1):
            self._stage(f"Расчёт: {v} ({i}/{nt})", 0.80 + 0.18*(i-1)/nt)
            try:
                mat, meta = self._compute_variant_auto(v, **kwargs)
                self.results[v] = mat; self.results_meta[v] = meta
                if meta.get("chosen_lag"): self.variant_lags[v] = int(meta["chosen_lag"])
                if meta.get("window"): self.window_analysis[v] = meta["window"]
            except Exception as e:
                logging.error("Error %s: %s", v, e)
                self.results[v] = None; self.results_meta[v] = {"error": str(e)}
        self._apply_fdr_correction(); self._stage("Готово", 1.0)

    def run_selected_methods(self, variants, max_lag=5, **kwargs):
        self._stage("Подготовка", 0.72)
        self._maybe_apply_dimred(**kwargs); self._maybe_post_preprocess(**kwargs)
        self.normalize_data()
        prev = dict((self.results_meta or {}).get("__run__", {}) or {})
        self.results = {}; self.results_meta = {"__run__": prev} if prev else {}
        self.window_analysis = {}
        self.config.max_lag = max(self.config.max_lag, int(max_lag))
        method_options = kwargs.get("method_options") or {}
        used_lags: Dict[str, int] = {}; nt = max(1, len(variants))
        for i, v in enumerate(variants, 1):
            self._stage(f"Расчёт: {v} ({i}/{nt})", 0.80 + 0.18*(i-1)/nt)
            if v not in method_mapping: continue
            try:
                vkw = dict(kwargs)
                if isinstance(method_options.get(v), dict): vkw.update(method_options[v])
                mat, meta = self._compute_variant_auto(v, **vkw)
                self.results[v] = mat; self.results_meta[v] = meta
                if meta.get("chosen_lag"): used_lags[v] = int(meta["chosen_lag"])
                if meta.get("window"): self.window_analysis[v] = meta["window"]
            except Exception as e:
                logging.error("Error %s: %s", v, e)
                self.results[v] = None; self.results_meta[v] = {"error": str(e)}
        self.variant_lags = used_lags; self._apply_fdr_correction()
        self._stage("Готово", 1.0); return used_lags

    # ── Core variant computation ─────────────────────────────────────

    def _compute_variant_auto(self, variant, **kwargs):
        df = self.data_normalized
        if df is None or df.empty: return None, {"error": "empty data"}
        meta = {"variant": variant}
        if is_control_sensitive_method(variant):
            meta["partial"] = {"mode": kwargs.get("partial_mode", "pairwise"),
                "pairwise_policy": kwargs.get("pairwise_policy", "others"),
                "control_strategy": kwargs.get("control_strategy", "none")}
        n_cols = df.shape[1]
        pairs_idx, pair_mode, pair_meta = resolve_pairs(
            n_cols, df.columns, getattr(self, "coords_df", None),
            str(kwargs.get("pair_mode") or "auto").lower(),
            int(kwargs.get("pair_auto_threshold") or 500), **kwargs)
        meta.update(pair_meta)
        supports_lag = is_directed_method(variant) or variant.startswith(("granger","te_","ah_"))
        lag_sel = (kwargs.get("lag_selection") or self.config.lag_selection or "optimize").lower()
        max_lag_v = int(max(1, kwargs.get("max_lag") or self.config.max_lag or 1))

        def _at_lag(d, lag):
            return compute_connectivity_variant(d, variant, lag=int(max(1, lag)),
                control=kwargs.get("control"), pairs=pairs_idx,
                partial_mode=kwargs.get("partial_mode", "pairwise"),
                pairwise_policy=kwargs.get("pairwise_policy", "others"),
                custom_controls=kwargs.get("custom_controls"),
                control_strategy=kwargs.get("control_strategy", "none"),
                control_pca_k=int(kwargs.get("control_pca_k", 0) or 0))

        chosen_lag = 1
        if not supports_lag or lag_sel == "fixed":
            chosen_lag = int(max(1, kwargs.get("lag", 1)))
        else:
            best_s, best_l, no_imp = float("-inf"), 1, 0
            for lag in range(1, max_lag_v+1):
                mat = _at_lag(df, lag)
                sc = lag_quality(variant, mat, is_pvalue_method(variant))
                if np.isfinite(sc) and sc > best_s: best_s, best_l, no_imp = sc, lag, 0
                else: no_imp += 1
                if no_imp >= 3 and lag >= 3: break
            chosen_lag = best_l
        meta["chosen_lag"] = chosen_lag

        # Diagnostic scans
        scans = {k: bool(kwargs.get(f"scan_{k}", False)) for k in ("window_pos","window_size","lag","cube")}
        if any(scans.values()):
            self._run_diagnostic_scans(df, variant, chosen_lag, pairs_idx, meta, kwargs,
                                        _at_lag, supports_lag, max_lag_v, scans)

        # Window mode
        window_sizes = kwargs.get("window_sizes") or self.config.window_sizes
        if not window_sizes: return _at_lag(df, chosen_lag), meta
        policy = (kwargs.get("window_policy") or self.config.window_policy or "best").lower()
        window_sizes = [int(w) for w in window_sizes if int(w) >= 2]
        stride_ovr = kwargs.get("window_stride") or self.config.window_stride
        _is_p = is_pvalue_method(variant)
        mats, best, best_q = [], None, float("-inf")
        for w in window_sizes:
            stride = int(stride_ovr) if stride_ovr else max(1, w//5)
            wi = analyze_sliding_windows(df, variant, w, stride,
                compute_variant_func=compute_connectivity_variant,
                is_pvalue=_is_p, lag=chosen_lag, pairs=pairs_idx)
            bw = (wi or {}).get("best_window")
            if bw and bw.get("matrix") is not None:
                mats.append(np.asarray(bw["matrix"]))
                q = float(bw.get("metric", float("nan")))
                if np.isfinite(q) and q > best_q:
                    best_q = q; best = {"window_size": w, "stride": stride,
                                         "best_window": bw, "curve": wi.get("curve")}
        if not mats: return _at_lag(df, chosen_lag), meta
        mat = np.nanmean(np.stack(mats,0),0) if policy == "mean" else (
            np.asarray(best["best_window"]["matrix"]) if best else np.asarray(mats[0]))
        meta["window"] = {"sizes": window_sizes, "policy": policy, "best": best}
        return mat, meta

    def _run_diagnostic_scans(self, df, variant, chosen_lag, pairs_idx, meta, kwargs,
                               _at_lag, supports_lag, max_lag_v, scans):
        _is_p = is_pvalue_method(variant); scan_meta = {"flags": scans}
        ws = [int(w) for w in (kwargs.get("window_sizes_grid") or kwargs.get("window_sizes") or
              self.config.window_sizes or []) if int(w)>=2]
        dw = int(kwargs.get("window_size", ws[0] if ws else min(200, max(10, len(df)//5))))
        dw = max(2, min(dw, len(df)))
        max_win = int(kwargs.get("window_max_windows", 250))
        n_jobs = int(kwargs.get("scan_n_jobs") or 1)
        if scans.get("window_pos"):
            stride = int(kwargs.get("window_stride") or max(1, dw//5))
            info = analyze_sliding_windows(df, variant, dw, stride,
                compute_variant_func=compute_connectivity_variant, is_pvalue=_is_p,
                lag=chosen_lag, pairs=pairs_idx, return_matrices=True,
                max_windows=max_win, n_jobs=n_jobs)
            ticks = [{"id": f"pos_w{dw}_i{i}", **t}
                     for i, t in enumerate((info or {}).get("ticks") or [])]
            scan_meta["window_pos"] = {"window_size": dw, "lag": chosen_lag,
                "curve": (info or {}).get("curve"), "best_window": (info or {}).get("best_window"),
                "ticks": ticks}
        if scans.get("window_size") and ws:
            xs, ys, ticks = [], [], []
            for w in ws:
                stride = int(kwargs.get("window_stride") or max(1, w//5))
                info = analyze_sliding_windows(df, variant, w, stride,
                    compute_variant_func=compute_connectivity_variant, is_pvalue=_is_p,
                    lag=chosen_lag, pairs=pairs_idx, max_windows=max_win, n_jobs=n_jobs)
                bw = (info or {}).get("best_window") or {}; q = bw.get("metric", float("nan"))
                xs.append(w); ys.append(float(q) if np.isfinite(q) else float("nan"))
                ticks.append({"id": f"size_w{w}", "window_size": w, "metric": ys[-1],
                              "matrix": bw.get("matrix")})
            scan_meta["window_size"] = {"lag": chosen_lag, "curve": {"x": xs, "y": ys},
                "ticks": ticks, "extremes": select_best_median_worst(ticks, key="metric")}
        if scans.get("lag") and supports_lag:
            lmin, lmax = max(1, int(kwargs.get("lag_min", 1))), max(1, int(kwargs.get("lag_max", max_lag_v)))
            lstep = max(1, int(kwargs.get("lag_step", 1)))
            xs, ys, ticks = [], [], []
            for lag in range(lmin, lmax+1, lstep):
                m = _at_lag(df, lag); q = lag_quality(variant, m, _is_p)
                xs.append(lag); ys.append(float(q) if np.isfinite(q) else float("nan"))
                ticks.append({"id": f"lag_l{lag}", "lag": lag, "metric": ys[-1], "matrix": m})
            scan_meta["lag"] = {"curve": {"x": xs, "y": ys},
                "grid": list(range(lmin, lmax+1, lstep)), "ticks": ticks,
                "extremes": select_best_median_worst(ticks, key="metric")}
        if scans.get("cube") and ws:
            lag_grid = kwargs.get("lag_grid")
            if lag_grid is None:
                lm2, lx2, ls2 = max(1,int(kwargs.get("lag_min",1))), max(1,int(kwargs.get("lag_max",max_lag_v))), max(1,int(kwargs.get("lag_step",1)))
                lag_grid = list(range(lm2, lx2+1, ls2))
            combos = [(w, lg) for w in ws for lg in lag_grid]
            cl = int(kwargs.get("cube_combo_limit", 500))
            if len(combos) > cl:
                idx = np.linspace(0, len(combos)-1, cl).round().astype(int)
                combos = [combos[i] for i in idx]
            points = []; cn = int(kwargs.get("cube_n_jobs") or 1)
            def _cube(w, lg):
                stride = int(kwargs.get("window_stride") or max(1, w//5))
                info = analyze_sliding_windows(df, variant, w, stride,
                    compute_variant_func=compute_connectivity_variant, is_pvalue=_is_p,
                    lag=lg, pairs=pairs_idx, max_windows=max(1, max_win//max(1,len(combos))), n_jobs=1)
                return [{"id": f"cube_w{w}_l{lg}_s{t.get('start',0)}", "window_size": w, "lag": lg,
                         "start": t.get("start",0), "end": t.get("end",0),
                         "metric": float(t.get("metric", float("nan")))}
                        for t in (info or {}).get("ticks") or [] if np.isfinite(t.get("metric", float("nan")))]
            if cn == 1 or len(combos) <= 1:
                all_pts = [_cube(w, lg) for w, lg in combos]
            else:
                try:
                    from joblib import Parallel, delayed
                    all_pts = Parallel(n_jobs=cn, backend=str(kwargs.get("scan_backend") or "loky"))(
                        delayed(_cube)(w, lg) for w, lg in combos)
                except ImportError: all_pts = [_cube(w, lg) for w, lg in combos]
            for pts in all_pts: points.extend(pts)
            scan_meta["cube"] = {"combos": combos, "lag_grid": lag_grid, "window_sizes": ws,
                "points": points, "extremes": select_best_median_worst(points, key="metric")}
        meta["window_scans"] = scan_meta

    # ── Graph topology ───────────────────────────────────────────────

    def calculate_graph_metrics(self, threshold=0.2):
        from ..analysis.graph import analyze_graph_topology
        self.graph_results = {}; names = list(self.data.columns)
        for v, mat in (self.results or {}).items():
            if mat is None: continue
            d = get_method_spec(v).directed
            if is_pvalue_method(v):
                cm = np.zeros_like(mat, dtype=float)
                mk = (mat < threshold) & (~np.eye(mat.shape[0], dtype=bool))
                cm[mk] = 1.0 - mat[mk]
                self.graph_results[v] = analyze_graph_topology(cm, names, threshold=0.01, directed=d)
            else:
                self.graph_results[v] = analyze_graph_topology(mat, names, threshold=float(threshold), directed=d)

    # ── Export ────────────────────────────────────────────────────────

    def export_html_report(self, output_path, **kwargs):
        return HTMLReportGenerator(self).generate(output_path, **kwargs)

    def export_big_excel(self, save_path, **kwargs):
        return ExcelReportWriter(self).write(save_path, **kwargs)

    def export_series_bundle(self, save_path):
        def _p(d):
            try: return d if d is not None and not getattr(d,'empty',True) else None
            except: return None
        with pd.ExcelWriter(save_path, engine='openpyxl') as w:
            for nm, d in [("RAW", _p(self.data_raw)), ("PREPROCESSED", _p(self.data_preprocessed)),
                          ("AFTER_AUTODIFF", _p(self.data_after_autodiff)), ("NORMALIZED", _p(self.data_normalized))]:
                (d or pd.DataFrame()).to_excel(w, sheet_name=nm, index=False)
            try:
                for nm2, d2 in [("QC_RAW", self.qc_raw), ("QC_CLEAN", self.qc_clean), ("COORDS", self.coords_df)]:
                    if d2 is not None and not getattr(d2,'empty',True): d2.to_excel(w, sheet_name=nm2, index=False)
            except: pass
        return save_path

    def export_connectivity_bundle(self, out_dir, name_prefix="run", dense_n_limit=2000,
                                    topk_per_node=50, min_abs_weight=0.0, include_scan_matrices=True):
        from pathlib import Path
        from ..reporting.connectivity_export import ExportPolicy, export_connectivity_matrix, save_manifest, save_nodes_csv
        data_dir = str(Path(out_dir)/"data"); os.makedirs(data_dir, exist_ok=True)
        try: self.export_dimred_bundle(out_dir, name_prefix=name_prefix)
        except: pass
        names = list(getattr(self,"data",pd.DataFrame()).columns)
        policy = ExportPolicy(dense_n_limit=dense_n_limit, topk_per_node=topk_per_node, min_abs_weight=min_abs_weight)
        manifest = {"name_prefix": name_prefix, "variants": {}}
        if names: save_nodes_csv(data_dir, names)
        for v, mat in (self.results or {}).items():
            try:
                arr = np.asarray(mat)
                if arr.ndim != 2 or arr.shape[0] != arr.shape[1]: continue
                ln = names if names and arr.shape[0]==len(names) else [f"v{i:04d}" for i in range(arr.shape[0])]
                manifest["variants"][v] = export_connectivity_matrix(data_dir, name_prefix, v, arr, ln, policy)
            except Exception as e: manifest["variants"][v] = {"error": str(e)}
        return str(save_manifest(data_dir, name_prefix, manifest))

    # ── Dimred ────────────────────────────────────────────────────────

    def _maybe_apply_dimred(self, **kwargs):
        enabled = bool(kwargs.get("dimred_enabled", False))
        method = str(kwargs.get("dimred_method") or "none").strip().lower()
        target_n = int(kwargs.get("dimred_target", 0) or 0)
        target_var = kwargs.get("dimred_target_var")
        try: target_var = float(target_var) if target_var is not None and str(target_var).strip() != "" else None
        except: target_var = None
        base = (self.data_preprocessed if isinstance(getattr(self,"data_preprocessed",None), pd.DataFrame) and not self.data_preprocessed.empty else getattr(self,"data",None))
        if base is None or getattr(base, "empty", True):
            self.dimred_report = {"enabled": False, "reason": "no_data"}; self.data_dimred = pd.DataFrame(); self.dimred_mapping = pd.DataFrame(); return
        self.data_dimred_base = base; n0 = int(base.shape[1])
        if not enabled or method in ("none","off","disabled"):
            self.dimred_report = {"enabled": False, "method": "none", "k": n0}
            self.data_dimred = base.copy()
            self.dimred_mapping = pd.DataFrame({"source": list(base.columns), "target": list(base.columns), "weight": 1.0})
            self.data = base.copy(); return
        if target_n <= 0 and (target_var is None or not (0.0 < target_var <= 1.0)):
            target_n = int(min(500, n0))
        mtk = kwargs.get("dimred_mapping_topk")
        if mtk is None and n0 >= 5000: mtk = 50
        res = apply_dimred(base, method=method,
            target_n=(int(min(target_n, n0)) if target_n and target_n > 0 else None),
            target_var=target_var, seed=int(kwargs.get("dimred_seed", 0) or 0),
            coords_df=getattr(self,"coords_df",None),
            kmeans_batch=int(kwargs.get("dimred_kmeans_batch", 2048) or 2048),
            spatial_bin=int(kwargs.get("dimred_spatial_bin", 2) or 2),
            pca_priority=str(kwargs.get("dimred_priority") or "auto").strip().lower(),
            pca_solver=str(kwargs.get("dimred_pca_solver") or "full").strip().lower(),
            mapping_topk=(int(mtk) if mtk is not None else None),
            mapping_min_abs=(float(kwargs.get("dimred_mapping_min_abs")) if kwargs.get("dimred_mapping_min_abs") is not None else None))
        self.data_dimred = res.reduced; self.dimred_mapping = res.mapping
        self.dimred_report = {"enabled": True, **(res.meta or {}), "n_before": n0, "n_after": int(res.reduced.shape[1])}
        self.data_preprocessed = self.data_dimred.copy(); self.data = self.data_dimred.copy()
        if bool(kwargs.get("dimred_save_variants", False)) and str(kwargs.get("dimred_variants") or "").strip():
            try:
                parts = [p.strip() for p in str(kwargs["dimred_variants"]).replace(";",",").split(",") if p.strip()]
                self.dimred_report["saved_variants"] = sorted(set(int(float(p)) for p in parts if float(p)>0))
            except Exception as e: self.dimred_report["saved_variants_error"] = str(e)

    def export_dimred_bundle(self, out_dir, name_prefix="run"):
        import json; from pathlib import Path
        data_dir = Path(out_dir)/"data"; data_dir.mkdir(parents=True, exist_ok=True)
        paths: Dict[str, str] = {}; pref = f"{name_prefix}_" if name_prefix else ""
        try:
            df = getattr(self,"data_dimred",None)
            if df is not None and not getattr(df,"empty",True):
                p = data_dir/f"{pref}timeseries_dimred.csv"; df.to_csv(p, index=True); paths["timeseries_dimred_csv"] = str(p)
        except: pass
        try:
            mp = getattr(self,"dimred_mapping",None)
            if mp is not None and not getattr(mp,"empty",True):
                p = data_dir/f"{pref}dimred_mapping.csv"; mp.to_csv(p, index=False); paths["dimred_mapping_csv"] = str(p)
        except: pass
        try:
            rep = getattr(self,"dimred_report",{}) or {}; p = data_dir/f"{pref}dimred_meta.json"
            p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8"); paths["dimred_meta_json"] = str(p)
        except: pass
        try:
            rep2 = getattr(self,"dimred_report",{}) or {}; targets = rep2.get("saved_variants") or []
            m = str(rep2.get("method","none")).strip().lower()
            if targets and m not in ("none","off","disabled"):
                base = getattr(self,"data_dimred_base",None)
                if base is not None and not getattr(base,"empty",True):
                    vroot = data_dir/f"{pref}dimred_variants"; vroot.mkdir(parents=True, exist_ok=True)
                    for t in targets:
                        try:
                            t2 = int(min(int(t), base.shape[1]))
                            rr = apply_dimred(base, method=m, target_n=t2, seed=int(rep2.get("seed",0) or 0), coords_df=getattr(self,"coords_df",None))
                            sub = vroot/f"{m}_n{t2}"; sub.mkdir(parents=True, exist_ok=True)
                            rr.reduced.to_csv(sub/"timeseries_dimred.csv", index=True)
                            rr.mapping.to_csv(sub/"dimred_mapping.csv", index=False)
                            (sub/"dimred_meta.json").write_text(json.dumps(rr.meta, ensure_ascii=False, indent=2), encoding="utf-8")
                        except: continue
                    paths["dimred_variants_dir"] = str(vroot)
        except: pass
        return paths

    def _maybe_post_preprocess(self, **kwargs):
        stage = str(kwargs.get("preprocess_stage", "pre")).strip().lower()
        if stage not in ("post","both"): return
        post = kwargs.get("post_preprocess", {}) or {}
        if not bool(post.get("enabled", False)): return
        try:
            self.data = preprocess_timeseries(self.data, enabled=True,
                log_transform=bool(post.get("log_transform", False)),
                remove_outliers=bool(post.get("remove_outliers", False)),
                normalize=bool(post.get("normalize", True)),
                fill_missing=bool(post.get("fill_missing", True)),
                check_stationarity=False, return_report=False,
                **{k: post[k] for k in post if k not in ("enabled","log_transform","remove_outliers","normalize","fill_missing")})
        except Exception as e: logging.warning("Post-preprocess failed: %s", e)

    # ── Diagnostics ──────────────────────────────────────────────────

    def test_stationarity(self, series): return analysis_stats.test_stationarity(series)

    def get_preprocessing_summary(self):
        rep = {}
        try:
            if self.preprocessing_report is not None:
                pr = self.preprocessing_report
                rep["preprocess"] = {"enabled": bool(getattr(pr,"enabled",True)),
                    "steps_global": list(getattr(pr,"steps_global",[])),
                    "steps_by_column": dict(getattr(pr,"steps_by_column",{})),
                    "dropped_columns": list(getattr(pr,"dropped_columns",[])),
                    "notes": dict(getattr(pr,"notes",{}))}
            else: rep["preprocess"] = {"enabled": None}
        except: rep["preprocess"] = {"enabled": None}
        rep["autodiff"] = dict(self.autodiff_report or {"enabled": False})
        try: rep["dimred"] = dict(getattr(self,"dimred_report",{}) or {})
        except: rep["dimred"] = {}
        return rep

    def get_harmonics(self, top_k=5, fs=None):
        out = {}
        if self.data.empty: return out
        fs0 = float(fs) if fs is not None else float(getattr(self,"fs",1.0))
        for col in self.data.columns:
            out[col] = analysis_stats.fft_peaks(self.data[col], fs=fs0, top_k=int(max(1, top_k)))
        return out

    def get_diagnostics(self):
        diagnostics = {}
        if self.data.empty: return diagnostics
        for col in self.data.columns:
            s = self.data[col]; stat, pv = analysis_stats.test_stationarity(s)
            diagnostics[col] = {"adf_stat": stat, "adf_p": pv,
                "hurst_rs": analysis_stats.compute_hurst_rs(s),
                "sample_entropy": analysis_stats.compute_sample_entropy(s),
                "shannon_entropy": analysis_stats.shannon_entropy(s),
                "seasonality": analysis_stats.detect_seasonality(s),
                "autocorr": analysis_stats.autocorr_summary(s),
                "fft_peaks": analysis_stats.fft_peaks(s, top_k=3)}
        return diagnostics

    def build_pairwise_summaries(self, *, p_alpha=0.05):
        self.pairwise_summaries = {}
        cols = list(self.data.columns) if self.data is not None and not self.data.empty else []
        for variant, mat in (self.results or {}).items():
            if mat is None or not isinstance(mat, np.ndarray) or mat.size == 0: continue
            _ip = is_pvalue_method(variant); thr = float(p_alpha) if _ip else float(getattr(self.config,"graph_threshold",0.2))
            rows = []
            for i, a in enumerate(cols):
                for j, b in enumerate(cols):
                    if i == j: continue
                    v = mat[i, j]
                    if v is None or not np.isfinite(float(v)): continue
                    flag = ("significant" if float(v)<thr else "") if _ip else ("strong" if abs(float(v))>thr else "")
                    rows.append({"src": a, "tgt": b, "value": float(v), "flag": flag})
            if rows: self.pairwise_summaries[variant] = pd.DataFrame(rows)

    # ── Plot helpers (thin wrappers for report generators) ────────────

    def plot_time_series(self, data, title="Time Series"):
        return plots.plot_time_series(data, title=title)

    def plot_single_time_series(self, series, title=""):
        return plots.plot_single_time_series(series, title=title)

    def plot_fft(self, data, title="FFT"):
        return plots.plot_fft(data, title=title)

    def plot_sliding_window_comparison(self, sw_res, legend_text=""):
        return plots.plot_sliding_window_comparison(sw_res, legend_text=legend_text)


# ── CLI ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connectivity analysis for multivariate time series.")
    parser.add_argument("input_file", help="Path to input CSV or Excel file")
    parser.add_argument("--lags", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--pvalue-alpha", type=float, default=DEFAULT_PVALUE_ALPHA)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--no-outliers", action="store_true")
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--no-stationarity-check", action="store_true")
    parser.add_argument("--graph-threshold", type=float, default=0.5)
    parser.add_argument("--output", default=None)
    parser.add_argument("--quiet-warnings", action="store_true")
    parser.add_argument("--experimental", action="store_true")
    parser.add_argument("--no-excel", action="store_true")
    parser.add_argument("--report-html", default=None, dest="report_html")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    configure_warnings(quiet=args.quiet_warnings)
    filepath = os.path.abspath(args.input_file)
    output_path = args.output or os.path.join(SAVE_FOLDER, "AllMethods_Full.xlsx")
    output_dir = os.path.dirname(output_path)
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    tool = BigMasterTool(enable_experimental=args.experimental)
    tool.load_data_excel(filepath, log_transform=args.log, remove_outliers=not args.no_outliers,
        normalize=not args.no_normalize, fill_missing=True, check_stationarity=not args.no_stationarity_check)
    tool.run_all_methods()
    if not args.no_excel:
        tool.export_big_excel(output_path, threshold=args.graph_threshold, p_value_alpha=args.pvalue_alpha)
    if args.report_html:
        rp = os.path.abspath(args.report_html); os.makedirs(os.path.dirname(rp), exist_ok=True)
        tool.export_html_report(rp, graph_threshold=args.graph_threshold, p_value_alpha=args.pvalue_alpha)
    print("Done:", output_path)
