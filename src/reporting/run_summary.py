#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Русские пояснения к запуску.

Важно для архитектуры:
- ядро считает; UI/CLI/HTML показывают.
- этот модуль собирает человеко-понятное описание из отчётов ядра.
"""

from __future__ import annotations

from typing import Optional


def _fmt_bool(x: bool) -> str:
    return "да" if bool(x) else "нет"


def build_run_summary_ru(tool, *, run_dir: Optional[str] = None) -> str:
    """Строит краткое русское пояснение: что загружено/почищено/как считалось."""
    lines: list[str] = []

    # 1) Импорт
    rep = getattr(tool, "preprocessing_report", None)
    notes = getattr(rep, "notes", None) if rep is not None else None
    notes = notes or {}
    fmt = notes.get("format") or notes.get("detected_format") or "—"
    shape = None
    try:
        df = getattr(tool, "data", None)
        if df is not None:
            shape = tuple(df.shape)
    except Exception:
        shape = None

    lines.append("Импорт")
    if shape:
        lines.append(f"- матрица: {shape[0]} точек × {shape[1]} рядов")
    lines.append(f"- формат: {fmt}")
    has_coords = getattr(tool, "coords_df", None) is not None
    lines.append(f"- координаты (x,y,z): {_fmt_bool(has_coords)}")

    # 2) Предобработка
    lines.append("Предобработка")
    if rep is None:
        lines.append("- отчёт предобработки: нет")
    else:
        enabled = bool(getattr(rep, "enabled", True))
        lines.append(f"- включена: {_fmt_bool(enabled)}")
        steps_g = list(getattr(rep, "steps_global", []) or [])
        if steps_g:
            # коротко, без простыней
            preview = steps_g[:10]
            lines.append("- шаги: " + "; ".join(str(x) for x in preview) + ("; …" if len(steps_g) > 10 else ""))
        dropped = list(getattr(rep, "dropped_columns", []) or [])
        if dropped:
            lines.append(f"- удалены константные/плохие колонки: {len(dropped)}")

        # Явная подсказка про автокорреляцию, если она диагностировалась.
        try:
            notes = getattr(rep, "notes", {}) or {}
            ac = notes.get("autocorr") if isinstance(notes, dict) else None
            if isinstance(ac, dict):
                order = ac.get("order")
                b = (ac.get("before") or {}).get("corr_lag1") or {}
                a = (ac.get("after") or {}).get("corr_lag1") or {}
                bmed = b.get("median")
                amed = a.get("median")
                red = ac.get("lag1_reduction_median")
                lines.append("Автокорреляция")
                lines.append(f"- AR(p): p={order}")
                if bmed is not None and amed is not None:
                    try:
                        lines.append(f"- lag=1 corr (медиана): до={float(bmed):.3g}, после={float(amed):.3g}")
                    except Exception:
                        pass
                if red is not None:
                    try:
                        lines.append(f"- снижение lag=1 (медиана): {100.0*float(red):.1f}%")
                    except Exception:
                        pass
        except Exception:
            pass

    # 3) QC
    qc_enabled = True
    try:
        qc_enabled = bool((getattr(tool, "results_meta", {}) or {}).get("__run__", {}).get("qc_enabled", True))
    except Exception:
        qc_enabled = True
    lines.append("QC по вокселям")
    lines.append(f"- считали QC: {_fmt_bool(qc_enabled)}")
    qc_clean = getattr(tool, "qc_clean", None)
    if qc_enabled and qc_clean is not None:
        try:
            n = int(qc_clean.shape[0])
            lines.append(f"- QC строк: {n}")
            # покажем 2–3 ключевых диапазона
            for col in ["std", "drift_slope", "ar1", "spikes_frac"]:
                if col in qc_clean.columns:
                    vmin = float(qc_clean[col].min())
                    vmax = float(qc_clean[col].max())
                    lines.append(f"- {col}: min={vmin:.4g}, max={vmax:.4g}")
        except Exception:
            pass

    # 4) Partial/Directed — по факту рассчитанных методов
    results_meta = getattr(tool, "results_meta", {}) or {}
    partial_methods = []
    directed_methods = []
    for k, meta in results_meta.items():
        if not isinstance(meta, dict) or k.startswith("__"):
            continue
        if "partial" in (k or ""):
            partial_methods.append(k)
        if bool(meta.get("directed")) or ("directed" in (k or "")):
            directed_methods.append(k)

    if partial_methods:
        # берем первую попавшуюся декларацию контроля
        m0 = partial_methods[0]
        p = (results_meta.get(m0) or {}).get("partial") or {}
        ctrl = p.get("control_strategy", "none")
        k_pca = int(p.get("control_pca_k", 0) or 0)
        lines.append("Partial (метрики *_partial)")
        if ctrl == "none":
            lines.append("- контроль: нет (считали на исходных рядах)")
        elif ctrl == "global_mean":
            lines.append("- контроль: вычтен глобальный сигнал (среднее по всем рядам)")
        elif ctrl == "global_mean_trend":
            lines.append("- контроль: вычтен глобальный сигнал + линейный тренд")
        else:
            lines.append("- контроль: вычтен глобальный сигнал + линейный тренд")
            if k_pca > 0:
                lines.append(f"- + PCA компоненты: k={k_pca}")
        lines.append(f"- partial-методов посчитано: {len(partial_methods)}")

    if directed_methods:
        lines.append("Directed (направленные/лаговые методы)")
        # покажем выбранный лаг из первого метода
        m0 = directed_methods[0]
        lag = (results_meta.get(m0) or {}).get("chosen_lag")
        if lag is not None:
            lines.append(f"- выбранный лаг (пример): {m0} -> lag={lag}")
        lines.append(f"- directed-методов посчитано: {len(directed_methods)}")

    if run_dir:
        lines.append("Сохранение")
        lines.append(f"- папка результатов: {run_dir}")

    return "\n".join(lines)
