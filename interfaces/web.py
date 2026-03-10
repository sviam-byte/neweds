"""Веб-интерфейс (Streamlit) для Time Series Analysis Tool (локально)."""

from __future__ import annotations

import gc
import json
import traceback
import os
import sys
import tempfile
import zipfile
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import EXPERIMENTAL_METHODS, SAVE_FOLDER, STABLE_METHODS
from src.core import engine, generator
from src.core.preprocessing import configure_warnings
from src.validation.runner import run_quick_validation, run_full_validation, run_validation
from src.validation.scenarios import ALL_SCENARIOS, QUICK_SCENARIOS

configure_warnings()

PRESET_NAMES = [
    "Fast stable",
    "Default stable",
    "Heavy research",
    "fMRI batch safe",
]


def _preset_payload(name: str) -> dict:
    """Возвращает пресет sane-defaults для типовых запусков."""
    fast_methods = [
        m for m in ["correlation_full", "dcor_full", "ordinal_full"]
        if m in (STABLE_METHODS + EXPERIMENTAL_METHODS)
    ]
    stable_methods = [
        m for m in [
            "correlation_full", "correlation_partial", "coherence_full",
            "dcor_full", "ordinal_full", "granger_full",
        ] if m in (STABLE_METHODS + EXPERIMENTAL_METHODS)
    ]
    all_heavy = [m for m in (STABLE_METHODS + EXPERIMENTAL_METHODS) if m not in {"te_partial"}]

    presets = {
        "Fast stable": {
            "selected_methods": fast_methods, "enable_experimental": False,
            "remove_ar1": True, "remove_ar_order": 2, "ar_diagnostics": True,
            "lag_selection_mode_ui": "fixed", "lag": 1, "max_lag": 3,
            "use_main_windows": False, "window_sizes_text": "128,256", "window_stride_main": 0,
            "window_policy": "best", "include_scans": False,
            "scan_lag": False, "scan_window_pos": False, "scan_window_size": False, "scan_cube": False,
            "output_mode": "html", "include_diagnostics": True, "include_matrix_tables": False,
            "include_fft_plots": False, "save_series_bundle": True, "time_stride": 1,
        },
        "Default stable": {
            "selected_methods": stable_methods, "enable_experimental": False,
            "remove_ar1": True, "remove_ar_order": 2, "ar_diagnostics": True,
            "lag_selection_mode_ui": "fixed", "lag": 1, "max_lag": 5,
            "use_main_windows": False, "window_sizes_text": "128,256", "window_stride_main": 0,
            "window_policy": "best", "include_scans": False,
            "scan_lag": False, "scan_window_pos": False, "scan_window_size": False, "scan_cube": False,
            "output_mode": "both", "include_diagnostics": True, "include_matrix_tables": False,
            "include_fft_plots": False, "save_series_bundle": True, "time_stride": 1,
        },
        "Heavy research": {
            "selected_methods": all_heavy, "enable_experimental": True,
            "remove_ar1": True, "remove_ar_order": 3, "ar_diagnostics": True,
            "lag_selection_mode_ui": "optimize", "lag": 1, "max_lag": 8,
            "use_main_windows": True, "window_sizes_text": "128,256,512", "window_stride_main": 0,
            "window_policy": "best", "include_scans": True,
            "scan_lag": True, "scan_window_pos": True, "scan_window_size": True, "scan_cube": False,
            "output_mode": "both", "include_diagnostics": True, "include_matrix_tables": True,
            "include_fft_plots": True, "save_series_bundle": True, "time_stride": 1,
        },
        "fMRI batch safe": {
            "selected_methods": fast_methods, "enable_experimental": False,
            "remove_ar1": True, "remove_ar_order": 2, "ar_diagnostics": True,
            "lag_selection_mode_ui": "fixed", "lag": 1, "max_lag": 3,
            "use_main_windows": False, "window_sizes_text": "128,256", "window_stride_main": 0,
            "window_policy": "best", "include_scans": False,
            "scan_lag": False, "scan_window_pos": False, "scan_window_size": False, "scan_cube": False,
            "output_mode": "html", "include_diagnostics": True, "include_matrix_tables": False,
            "include_fft_plots": False, "save_series_bundle": True,
            "spatial_grid_size": 12, "spatial_grid_method": "mean", "lazy_spatial_bin": True,
            "time_chunk": 25, "time_stride": 2, "feature_limit": 0, "dimred_enabled": False,
            # Batch-safe дефолты: без рекурсии и с фокусом на HDF5-файлы.
            "batch_recursive": False, "batch_skip_existing": True,
            "batch_allowed_exts": [".h5", ".hdf5"],
        },
    }
    return presets.get(name, presets["Default stable"]).copy()


def _apply_preset_to_session(name: str) -> None:
    """Применяет выбранный пресет к session_state."""
    payload = _preset_payload(name)
    st.session_state["launch_preset"] = name
    for k, v in payload.items():
        st.session_state[k] = v


SUPPORTED_INPUT_EXTS = (".csv", ".xlsx", ".xls", ".parquet", ".mat", ".h5", ".hdf5")

# Устойчивый дефолт для первого реального прогона без экспериментальных метрик.
DEFAULT_STABLE_METHODS = [
    m
    for m in [
        "correlation_full",
        "correlation_partial",
        "coherence_full",
        "dcor_full",
        "ordinal_full",
        "granger_full",
    ]
    if m in STABLE_METHODS
]


def _json_default(obj):
    """JSON-serializer для numpy/pathlib типов в run-артефактах."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _parse_int_list_text(text: str) -> list[int] | None:
    text = (text or "").strip()
    if not text:
        return None
    xs: list[int] = []
    for tok in text.replace("[", "").replace("]", "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            xs.append(int(tok))
        except Exception:
            continue
    return xs or None



def _save_uploaded_file(uploaded, dst_dir: Path) -> Path:
    """Сохраняет UploadedFile в временную директорию и возвращает путь."""
    name = Path(getattr(uploaded, "name", "upload.bin")).name
    out = dst_dir / name
    out.write_bytes(uploaded.getbuffer())
    return out


def _safe_slug(text: str) -> str:
    """Нормализует произвольный текст в безопасный slug для имени папки."""
    safe = "".join(ch for ch in str(text or "item") if ch.isalnum() or ch in "-_. ").strip().replace(" ", "_")
    return safe or "item"


def _zip_tree(src_dir: Path, zip_path: Path) -> Path:
    """Упаковывает дерево src_dir в zip_path, сохраняя относительные пути."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for pp in sorted(src_dir.rglob("*")):
            if pp.is_file():
                zf.write(pp, arcname=str(pp.relative_to(src_dir)))
    return zip_path


def _write_json(path: Path, payload: dict) -> None:
    """Пишет JSON на диск с единым форматированием."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _append_text(path: Path, text: str) -> None:
    """Добавляет строку в лог-файл, создавая директорию при необходимости."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def _normalize_exts(exts: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Нормализует список расширений к виду ('.h5', '.csv').

    Пустой/некорректный ввод приводит к SUPPORTED_INPUT_EXTS.
    """
    if not exts:
        return SUPPORTED_INPUT_EXTS

    normalized: list[str] = []
    for ext in exts:
        value = str(ext or "").strip().lower()
        if not value:
            continue
        if not value.startswith("."):
            value = f".{value}"
        if value in SUPPORTED_INPUT_EXTS and value not in normalized:
            normalized.append(value)
    return tuple(normalized or SUPPORTED_INPUT_EXTS)


def _iter_input_files(
    folder: str,
    recursive: bool = False,
    allowed_exts: list[str] | tuple[str, ...] | None = None,
) -> list[Path]:
    """Возвращает поддерживаемые входные файлы для batch-режима.

    По умолчанию обход НЕ рекурсивный. Через ``allowed_exts`` можно заранее
    ограничить набор форматов (например, только HDF5).
    """
    root = Path(folder).expanduser()
    if not root.exists() or not root.is_dir():
        return []

    exts = _normalize_exts(allowed_exts)
    found: list[Path] = []
    iterator = root.rglob("*") if recursive else root.iterdir()
    for p in iterator:
        if not p.is_file():
            continue
        # Не обрабатываем артефакты уже созданных запусков.
        if "time_series_analysis" in {part.lower() for part in p.parts}:
            continue
        if p.suffix.lower() in exts:
            found.append(p)
    return sorted(found)


def _default_batch_output_root(input_folder: str) -> str:
    """Строит дефолтный путь для batch-результатов рядом с входной папкой."""
    folder = Path(input_folder).expanduser() if input_folder else Path(SAVE_FOLDER)
    if folder.exists() and folder.is_dir():
        return str(folder / "time_series_analysis")
    return str(Path(SAVE_FOLDER) / "runs" / "time_series_analysis")


def _native_pick_path(dialog: str = "file", title: str = "Выбери путь", initialdir: str | None = None) -> str:
    """Открывает нативный диалог выбора файла/папки и возвращает выбранный путь.

    В headless-средах (или если tkinter недоступен) безопасно возвращает пустую строку,
    чтобы UI продолжил работу без падения.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        start_dir = str(Path(initialdir).expanduser()) if initialdir else str(Path.home())

        if dialog == "dir":
            out = filedialog.askdirectory(title=title, initialdir=start_dir, mustexist=False)
        else:
            out = filedialog.askopenfilename(
                title=title,
                initialdir=start_dir,
                filetypes=[("Data", "*.csv *.xlsx *.xls *.parquet *.mat *.h5 *.hdf5"), ("All files", "*.*")],
            )

        root.destroy()
        return str(out or "")
    except Exception:
        return ""


def _show_local_dialog_hint() -> None:
    """Пояснение про нативные диалоги выбора пути и fallback-режим."""
    st.caption(
        "Кнопки выбора открывают системный диалог (tkinter). "
        "Если диалог недоступен (например, headless/remote), просто введи путь вручную."
    )


def _consume_pending_widget_value(widget_key: str) -> None:
    """Переносит отложенное значение в ключ виджета до его создания.

    Это защищает от ошибки Streamlit, когда код пытается менять значение
    уже созданного widget-key в том же rerun.
    """
    pending_key = f"__pending_{widget_key}"
    if pending_key in st.session_state:
        st.session_state[widget_key] = st.session_state.pop(pending_key)


def _make_run_dir(stem: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(ch for ch in (stem or "run") if ch.isalnum() or ch in "-_ ").strip().replace(" ", "_")
    run_dir = Path(SAVE_FOLDER) / "runs" / f"{safe}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _maybe_reset_formula_defaults(preset: str) -> None:
    defaults = {
        "Custom": {"x": "sin(2*pi*t/50) + 0.2*randn()", "y": "0.8*X + 0.3*randn()", "z": "rw(0.5)"},
        "Random": {"x": "randn()", "y": "randn()", "z": "randn()"},
        "Linear + noise": {"x": "0.01*t + 0.2*randn()", "y": "0.5*X + 0.2*randn()", "z": "-0.005*t + 0.2*randn()"},
        "Sin/Cos coupling": {"x": "sin(2*pi*t/50) + 0.1*randn()", "y": "cos(2*pi*t/50) + 0.4*X + 0.1*randn()", "z": "sin(2*pi*t/15) + 0.1*randn()"},
        "AR(1)": {"x": "ar1(phi=0.85, scale=0.5)", "y": "0.7*X + ar1(phi=0.6, scale=0.4)", "z": "ar1(phi=0.3, scale=0.8)"},
    }
    if st.session_state.get("_preset_prev") != preset:
        st.session_state["_preset_prev"] = preset
        d = defaults.get(preset, defaults["Custom"])
        st.session_state["x_expr"] = d["x"]
        st.session_state["y_expr"] = d["y"]
        st.session_state["z_expr"] = d["z"]



def _render_validation_tab() -> None:
    """Вкладка валидации: синтетические сценарии с известным ground truth."""
    from src.metrics.registry import METRICS_REGISTRY

    st.subheader("Валидация метрик связности")
    st.markdown(
        "Синтетические данные с **известной** структурой зависимостей. "
        "Каждый сценарий проверяет, что метрики корректно обнаруживают (или не обнаруживают) связи."
    )

    # --- Выбор режима ---
    mode = st.radio(
        "Режим",
        [
            f"Быстрый ({len(QUICK_SCENARIOS)} сценария × стабильные метрики)",
            f"Полный ({len(ALL_SCENARIOS)} сценариев × все метрики)",
            "Выборочный",
        ],
        index=0,
        horizontal=True,
        key="val_mode",
    )

    lag = st.slider("Лаг для directed-метрик", 1, 10, 3, key="val_lag")

    selected_scenarios = list(ALL_SCENARIOS.keys())
    selected_metrics = list(METRICS_REGISTRY.keys())

    if mode.startswith("Выборочный"):
        c1, c2 = st.columns(2)
        with c1:
            selected_scenarios = st.multiselect(
                "Сценарии",
                list(ALL_SCENARIOS.keys()),
                default=list(ALL_SCENARIOS.keys()),
                key="val_scenarios",
            )
        with c2:
            selected_metrics = st.multiselect(
                "Метрики",
                list(METRICS_REGISTRY.keys()),
                default=list(METRICS_REGISTRY.keys()),
                key="val_metrics",
            )

    # --- Описания сценариев ---
    with st.expander("Описание сценариев", expanded=False):
        for name, factory in ALL_SCENARIOS.items():
            scenario = factory()
            st.markdown(f"**{name}**: {scenario.description}")
            st.caption(f"  Проверок: {len(scenario.expectations)}, данные: {scenario.data.shape}")

    # --- Запуск ---
    if not st.button("Запустить валидацию", type="primary", key="val_run"):
        return

    stage_box = st.empty()
    prog = st.progress(0)

    def _val_progress(stage: str, progress: float):
        try:
            stage_box.markdown(f"**{stage}**")
            if progress is not None:
                prog.progress(int(max(0, min(1, float(progress))) * 100))
        except Exception:
            pass

    with st.spinner("Валидация..."):
        if mode.startswith("Быстрый"):
            report = run_quick_validation(lag=lag, progress_callback=_val_progress)
        elif mode.startswith("Полный"):
            report = run_full_validation(lag=lag, progress_callback=_val_progress)
        else:
            report = run_validation(
                scenario_names=selected_scenarios,
                metric_names=selected_metrics,
                lag=lag,
                progress_callback=_val_progress,
            )

    prog.progress(100)

    # --- Результаты: сводка ---
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Всего проверок", report.n_total)
    with c2:
        st.metric("Пройдено", report.n_passed)
    with c3:
        color = "🟢" if report.n_failed == 0 else "🔴"
        st.metric(f"{color} Провалено", report.n_failed)
    with c4:
        st.metric("Время (сек)", f"{report.elapsed_total_sec:.1f}")

    if report.n_failed == 0:
        st.success("Все проверки пройдены.")
    else:
        st.error(f"{report.n_failed} проверок провалено. Подробности ниже.")

    # --- Таблица провалов ---
    if report.n_failed > 0:
        st.subheader("Провалившиеся проверки")
        fail_df = report.failures_df()
        if not fail_df.empty:
            st.dataframe(fail_df, use_container_width=True, height=min(400, 50 + 35 * len(fail_df)))

    # --- По сценариям ---
    st.subheader("Результаты по сценариям")
    by_scen = report.by_scenario()
    for s_name, info in by_scen.items():
        status = "✅" if info["failed"] == 0 else "❌"
        with st.expander(f"{status} {s_name}: {info['passed']}/{info['total']} пройдено"):
            rows = []
            for c in info["checks"]:
                rows.append({
                    "metric": c.metric,
                    "pair": f"({c.pair[0]},{c.pair[1]})",
                    "check": c.check,
                    "value": f"{c.actual_value:.4f}" if np.isfinite(c.actual_value) else str(c.actual_value),
                    "result": "PASS" if c.passed else "FAIL",
                    "details": c.message,
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # --- По метрикам ---
    st.subheader("Результаты по метрикам")
    by_met = report.by_metric()
    met_rows = []
    for m_name, info in sorted(by_met.items()):
        status = "✅" if info["failed"] == 0 else "❌"
        met_rows.append({
            "metric": f"{status} {m_name}",
            "passed": info["passed"],
            "failed": info["failed"],
            "total": info["total"],
            "rate": f"{info['passed']/max(1,info['total'])*100:.0f}%",
        })
    if met_rows:
        st.dataframe(pd.DataFrame(met_rows), use_container_width=True)

    # --- Ошибки вычислений ---
    errors = [r for r in report.metric_results if r.error is not None]
    if errors:
        with st.expander(f"⚠️ Ошибки вычислений ({len(errors)})", expanded=False):
            for r in errors:
                st.text(f"[{r.scenario}] {r.metric}: {r.error}")

    # --- Полная таблица ---
    with st.expander("Полная таблица проверок", expanded=False):
        full_df = report.summary_df()
        if not full_df.empty:
            st.dataframe(full_df, use_container_width=True)

def main() -> None:
    st.set_page_config(page_title="Анализ Временных Рядов (Локально)", layout="wide")
    st.title("Анализ Связности Временных Рядов")
    st.caption(f"Локальная версия. Результаты сохраняются в папку: {SAVE_FOLDER}")

    with st.expander("🚀 Пресеты запуска", expanded=True):
        cpr1, cpr2 = st.columns([2, 1])
        with cpr1:
            cur = st.session_state.get("launch_preset", "Default stable")
            preset_name = st.selectbox(
                "Готовый профиль",
                PRESET_NAMES,
                index=(PRESET_NAMES.index(cur) if cur in PRESET_NAMES else PRESET_NAMES.index("Default stable")),
                key="launch_preset",
            )
        with cpr2:
            if st.button("Применить пресет", key="apply_launch_preset"):
                _apply_preset_to_session(preset_name)
                st.rerun()

    # Применяем отложенные значения до создания соответствующих виджетов.
    _consume_pending_widget_value("ui_local_file_path")
    _consume_pending_widget_value("ui_batch_input_folder")
    _consume_pending_widget_value("ui_batch_output_root")

    source = st.radio(
        "Источник данных",
        ["Файл (CSV/XLSX/MAT/Parquet)", "Папка (batch, потоково)", "Синтетика (формулы)", "Синтетика (пресеты)", "Валидация метрик"],
        index=0,
        horizontal=True,
    )

    uploaded_file = None
    uploaded_files = []
    synth_df: pd.DataFrame | None = None
    synth_name = "synthetic"
    local_file_path = ""
    batch_input_folder = ""
    batch_output_root = ""
    batch_recursive = False
    batch_skip_existing = True
    batch_allowed_exts = [".h5", ".hdf5"]

    if source.startswith("Файл"):
        _show_local_dialog_hint()
        cf1, cf2 = st.columns([3, 1])
        with cf1:
            local_file_path = st.text_input(
                "Локальный путь к файлу",
                value=st.session_state.get("ui_local_file_path", st.session_state.get("local_file_path", "")),
                key="ui_local_file_path",
                placeholder=r"D:\data\subject01.h5",
            )
        with cf2:
            if st.button("Выбрать файл…", key="pick_local_file"):
                picked = _native_pick_path(
                    "file",
                    "Выбери входной файл",
                    st.session_state.get("ui_local_file_path") or str(Path.home()),
                )
                if picked:
                    st.session_state["__pending_ui_local_file_path"] = picked
                    st.rerun()

        with st.expander("Загрузка через браузер (fallback)", expanded=False):
            uploaded_file = st.file_uploader(
                "Выберите файл",
                type=["csv", "xlsx", "xls", "mat", "parquet", "h5", "hdf5"],
                max_upload_size=1024,  # Дублируем лимит из .streamlit/config.toml для явного поведения UI.
            )
    elif source.startswith("Папка"):
        st.info("Локальный batch-режим: укажи папку с данными и отдельную папку результатов. Файлы будут идти потоково по одному.")
        _show_local_dialog_hint()
        c_batch1, c_batch2 = st.columns([3, 1])
        with c_batch1:
            batch_input_folder = st.text_input(
                "Папка с входными файлами",
                value=st.session_state.get("ui_batch_input_folder", st.session_state.get("batch_input_folder", "")),
                key="ui_batch_input_folder",
                placeholder=r"D:\data\fmri_batch",
            )
        with c_batch2:
            if st.button("Выбрать папку…", key="pick_input_folder"):
                picked = _native_pick_path(
                    "dir",
                    "Выбери папку с входными файлами",
                    st.session_state.get("ui_batch_input_folder") or str(Path.home()),
                )
                if picked:
                    st.session_state["__pending_ui_batch_input_folder"] = picked
                    if not st.session_state.get("ui_batch_output_root"):
                        st.session_state["__pending_ui_batch_output_root"] = _default_batch_output_root(picked)
                    st.rerun()

        c_batch_out1, c_batch_out2 = st.columns(2)
        with c_batch_out1:
            batch_output_root = st.text_input(
                "Папка для результатов",
                value=st.session_state.get(
                    "ui_batch_output_root",
                    st.session_state.get("batch_output_root", _default_batch_output_root(st.session_state.get("batch_input_folder", ""))),
                ),
                key="ui_batch_output_root",
            )
        with c_batch_out2:
            st.caption("Результаты будут сохранены в структуре time_series_analysis")

        c_batch3, c_batch4, c_batch5 = st.columns([1, 1, 2])
        with c_batch3:
            batch_recursive = st.checkbox("Рекурсивно проходить подпапки", value=False, key="batch_recursive")
        with c_batch4:
            batch_skip_existing = st.checkbox("Пропускать уже обработанные файлы", value=True, key="batch_skip_existing")
        with c_batch5:
            batch_allowed_exts = st.multiselect(
                "Какие расширения брать",
                options=list(SUPPORTED_INPUT_EXTS),
                default=st.session_state.get("batch_allowed_exts", [".h5", ".hdf5"]),
                key="batch_allowed_exts",
                help="Если ничего не выбрано, будет использован полный список поддерживаемых расширений.",
            )

        if batch_input_folder.strip():
            preview_files = _iter_input_files(
                batch_input_folder.strip(),
                recursive=bool(st.session_state.get("batch_recursive", False)),
                allowed_exts=st.session_state.get("batch_allowed_exts"),
            )
            st.caption(f"Найдено файлов: {len(preview_files)}")
            if preview_files:
                st.dataframe(
                    pd.DataFrame({"file": [str(p) for p in preview_files[:30]]}),
                    use_container_width=True,
                    height=220,
                )

    elif source.startswith("Синтетика (формулы)"):
        with st.expander("Синтетика: формулы X/Y/Z", expanded=True):
            c0, c1, c2 = st.columns(3)
            with c0:
                preset = st.selectbox(
                    "Шаблон",
                    ["Custom", "Random", "Linear + noise", "Sin/Cos coupling", "AR(1)"],
                    index=2,
                )
                _maybe_reset_formula_defaults(preset)
            with c1:
                n_samples = st.number_input("n_samples", min_value=20, max_value=200000, value=800, step=10, key="n_samples")
                dt = st.number_input("dt", min_value=0.0001, max_value=1000.0, value=1.0, step=0.1, format="%.4f", key="dt")
            with c2:
                seed = st.number_input("seed", min_value=0, max_value=10_000_000, value=42, step=1, key="seed")

            st.caption(
                "Переменные: t (время), X (первый ряд), Y (второй), Z (третий). Функции: sin, cos, exp, log, sqrt, "
                "randn(scale=1), rw(scale=1), ar1(phi=0.7, scale=1)."
            )

            x_expr = st.text_input("X(t) =", key="x_expr")
            y_expr = st.text_input("Y(t, X) =", key="y_expr")
            z_expr = st.text_input("Z(t, X, Y) =", key="z_expr")

            synth_name = st.text_input("Имя набора (для папки/файлов)", value=synth_name)

            if st.button("Сгенерировать preview", type="secondary"):
                try:
                    synth_df = generator.generate_formula_dataset(
                        n_samples=int(n_samples),
                        dt=float(dt),
                        seed=int(seed),
                        specs=[
                            generator.FormulaSpec("X", x_expr),
                            generator.FormulaSpec("Y", y_expr),
                            generator.FormulaSpec("Z", z_expr),
                        ],
                    )
                    st.success(f"OK: shape={synth_df.shape}")
                    with st.expander("Preview рядов", expanded=False):
                        st.line_chart(synth_df)
                        st.dataframe(synth_df.head(200))
                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")

    elif source.startswith("Синтетика (пресеты)"):
        with st.expander("Синтетика: пресеты", expanded=True):
            preset = st.selectbox(
                "Набор",
                ["Coupled system (X→Y, Z noise, S season)", "Random walks"],
                index=0,
                key="preset",
            )
            n_samples = st.number_input("n_samples", min_value=20, max_value=200000, value=800, step=10, key="preset_n_samples")
            seed = st.number_input("seed", min_value=0, max_value=10_000_000, value=42, step=1, key="preset_seed")
            synth_name = st.text_input("Имя набора (для папки/файлов)", value=synth_name)

            if st.button("Сгенерировать preview", type="secondary", key="preset_preview"):
                try:
                    if preset.startswith("Coupled"):
                        synth_df = generator.generate_coupled_system(n_samples=int(n_samples))
                    else:
                        synth_df = generator.generate_random_walks(n_vars=3, n_samples=int(n_samples))
                    st.success(f"OK: shape={synth_df.shape}")
                    with st.expander("Preview рядов", expanded=False):
                        st.line_chart(synth_df)
                        st.dataframe(synth_df.head(200))
                except Exception as e:
                    st.error(f"Ошибка генерации: {e}")

    elif source.startswith("Валидация"):
        _render_validation_tab()
        return

    # Синхронизация UI-ключей с рабочими ключами для совместимости старой логики.
    st.session_state["local_file_path"] = local_file_path
    st.session_state["batch_input_folder"] = batch_input_folder
    st.session_state["batch_output_root"] = batch_output_root

    # === БЛОК 1: ПРЕДОБРАБОТКА (с пояснениями) ===
    with st.expander("🛠️ 1. Подготовка данных (Preprocessing & DimRed)", expanded=False):
        st.info("Настройте, как очистить данные перед анализом.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Очистка сигналов**")
            preprocess = st.checkbox("Включить очистку", value=True)
            fill_missing = st.checkbox("Заполнять пропуски (interp)", value=True)
            if preprocess:
                normalize_mode_label = st.selectbox(
                    "Нормализация (приведение к одному масштабу)",
                    ["z-score", "robust z (median/MAD)", "rank (dense: 1..K)", "rank (percentile: 0..1)", "нет"],
                    index=0,
                )
                normalize = normalize_mode_label != "нет"
                normalize_mode = (
                    "zscore"
                    if normalize_mode_label.startswith("z-score")
                    else (
                        "robust_z"
                        if normalize_mode_label.startswith("robust")
                        else (
                            "rank_dense"
                            if "dense" in normalize_mode_label
                            else ("rank_pct" if "percentile" in normalize_mode_label else "none")
                        )
                    )
                )
                rank_ties = st.selectbox("Rank ties (если rank)", ["average", "min", "max", "dense", "first"], index=0)

                remove_outliers = st.checkbox(
                    "Удалять выбросы (сглаживание)",
                    value=True,
                    help="Заменяет резкие скачки на локальную медиану/маскирование по выбранному правилу.",
                )
                outlier_rule = st.selectbox("Правило выбросов", ["robust_z", "zscore", "iqr", "percentile", "hampel", "jump"], index=0)
                outlier_action = st.selectbox("Что делать с выбросами", ["mask (NaN)", "clip (winsorize)", "median (global)", "local_median"], index=0)
                outlier_z = st.slider("Сила фильтра (Z-score)", 3.0, 10.0, 5.0, help="Меньше = строже фильтр")
                outlier_k = st.number_input("Параметр k (для IQR)", min_value=0.5, max_value=10.0, value=1.5, step=0.1)
                outlier_p_low = st.number_input("Перцентиль low (для percentile/clip)", min_value=0.0, max_value=49.0, value=0.5, step=0.5)
                outlier_p_high = st.number_input("Перцентиль high (для percentile/clip)", min_value=51.0, max_value=100.0, value=99.5, step=0.5)
                outlier_hampel_window = st.number_input("Окно Hampel", min_value=3, max_value=501, value=7, step=2)
                outlier_jump_thr = st.number_input("Порог jump (0=auto)", min_value=0.0, max_value=1e9, value=0.0, step=1.0)
                outlier_local_median_window = st.number_input("Окно local_median", min_value=3, max_value=501, value=7, step=2)
                check_stat = st.checkbox(
                    "Авто-дифференцирование (если ряд нестационарен)",
                    value=False,
                    help="Если тренд меняется, берем разности (производную).",
                )
            else:
                normalize = False
                normalize_mode = "none"
                rank_ties = "average"
                remove_outliers = False
                outlier_rule = "robust_z"
                outlier_action = "mask (NaN)"
                outlier_z = 5.0
                outlier_k = 1.5
                outlier_p_low = 0.5
                outlier_p_high = 99.5
                outlier_hampel_window = 7
                outlier_jump_thr = 0.0
                outlier_local_median_window = 7
                check_stat = False

            # --- Общие настройки (вне if/else preprocess) ---
            enable_experimental = st.checkbox(
                "Экспериментальные методы (TE, AH, directed dCor, ...)",
                value=False,
                help="Включает дополнительные метрики, которые могут быть медленнее или менее стабильны.",
            )
            pvalue_correction = st.selectbox(
                "Поправка на множественные сравнения",
                ["none", "fdr_bh"],
                index=0,
                help="fdr_bh = Benjamini-Hochberg FDR для p-value методов (Granger и т.д.)",
            )

            _out_act = (
                "mask"
                if str(outlier_action).startswith("mask")
                else ("clip" if str(outlier_action).startswith("clip") else ("median" if str(outlier_action).startswith("median") else "local_median"))
            )
            # Пространственная агрегация каналов уменьшает размерность до запуска метрик.
            st.markdown("### Spatial aggregation (channel binning)")
            spatial_bin_size = int(st.number_input(
                "Bin size (channels per bin)",
                min_value=1,
                max_value=500,
                value=8,
                step=1,
            ))
            spatial_bin_method = st.selectbox(
                "Aggregation method",
                ["mean", "median", "sum"],
            )

            log_transform = st.checkbox("Лог-преобразование (только >0)", value=False)
            st.markdown("### AR(p) / prewhitening")
            if "remove_ar1" not in st.session_state:
                st.session_state["remove_ar1"] = True
            if "remove_ar_order" not in st.session_state:
                st.session_state["remove_ar_order"] = 2
            if "ar_diagnostics" not in st.session_state:
                st.session_state["ar_diagnostics"] = True
            remove_ar1 = st.checkbox(
                "Убирать AR(p) перед анализом",
                value=bool(st.session_state.get("remove_ar1", True)),
                key="remove_ar1",
                help="По умолчанию включено. Это один из главных способов убрать ложную связность из-за автокорреляции.",
            )
            remove_ar_order = int(st.number_input(
                "Порядок AR(p)",
                min_value=1, max_value=10,
                value=int(st.session_state.get("remove_ar_order", 2)),
                step=1,
                key="remove_ar_order",
                help="p=1 = AR(1), p=2/3 — более строгий вариант для более длинной памяти.",
            ))
            ar_diagnostics = st.checkbox(
                "Сохранять AR-диагностику до/после очистки",
                value=bool(st.session_state.get("ar_diagnostics", True)),
                key="ar_diagnostics",
            )
            remove_seasonality = st.checkbox("Убрать сезонность (STL)", value=False)
            season_period = st.number_input("Период сезонности (0=авто)", min_value=0, max_value=1000000, value=0, step=1)
            qc_enabled = st.checkbox(
                "QC по каждому ряду/вокселю (mean/std/дрейф/спайки/AR1)",
                value=True,
                help="Помогает быстро увидеть 'битые' ряды и причины ложной связности.",
            )

        with c2:
            st.markdown("**Большие данные (HDF5 / wide CSV)**")
            st.caption("Для нейровизуализации (700K+ вокселей): сколько оставить и как выбрать.")
            feature_limit = int(st.number_input(
                "Макс. вокселей/каналов (0=без дополнительного post-cap)",
                min_value=0, max_value=100000, value=0, step=50,
                help="0 = не делать дополнительный post-cap после безопасной H5-загрузки. Для spatial-режима лучше оставить 0.",
            ))
            feature_sampling = st.selectbox(
                "Метод отбора вокселей",
                ["spatial (фиксированные 3D-блоки)", "variance (самые изменчивые)", "activity (макс. сумма)", "random"],
                index=0,
            )
            _fs_map = {"spatial": "spatial", "variance": "variance", "activity": "activity", "random": "random"}
            feature_sampling_val = _fs_map.get(feature_sampling.split()[0], "variance")
            # Единый контрол: spatial grid для 4D fMRI (H5).
            # grid=5 → ~7900 бинов, grid=10 → ~1100, grid=15 → ~390.
            st.markdown("### Spatial grid (4D fMRI)")
            st.caption(
                "Нарезает объём кубами N³ и усредняет вокселы внутри каждого куба. "
                "grid=10 → ~1100 бинов (рекомендуется). grid=5 → ~7900 (медленно для partial-методов)."
            )
            spatial_grid_size = int(st.slider(
                "Grid size (voxels per side)",
                min_value=3,
                max_value=25,
                value=12,
                help="Размер стороны куба. 12 = более безопасный старт для больших 4D HDF5; при необходимости потом можно уменьшить до 10.",
                # Более консервативный дефолт снижает риск OOM на крупных 4D данных.
            ))
            # h5_spatial_bin наследует spatial_grid_size для единообразия
            h5_spatial_bin = spatial_grid_size
            spatial_grid_method = st.selectbox(
                "Aggregation (4D grid)",
                ["mean", "median", "sum"],
            )

            st.markdown("### Lazy HDF5 processing")
            lazy_spatial_bin = st.checkbox(
                "Lazy loading (чанками по времени, экономит RAM)",
                value=True,
            )
            time_chunk = int(st.slider(
                "Time chunk size",
                min_value=10,
                max_value=200,
                value=25,
            ))
            time_stride = int(st.number_input(
                "Шаг по времени (1=все точки, 2=каждый 2-й, ...)",
                min_value=1, max_value=100, value=1, step=1,
                help="Даунсэмплинг: stride=2 вдвое ускоряет загрузку HDF5.",
            ))

            st.markdown("**Снижение размерности (для больших данных)**")
            dimred_enabled = st.checkbox("Включить DimRed", value=False)
            dimred_method = "variance"
            dimred_target = 50
            dimred_target_var = 0.0
            dimred_priority = "explained_variance"
            dimred_pca_solver = "full"
            if dimred_enabled:
                st.caption("Если у вас 100+ каналов, анализ будет долгим. Выберите метод сжатия:")
                dimred_method = st.selectbox(
                    "Метод",
                    [
                        "variance (оставить самые меняющиеся)",
                        "kmeans (объединить похожие в кластеры)",
                        "spatial (усреднить по соседним вокселям)",
                        "pca_full (PCA: полный SVD)",
                        "pca_randomized (PCA: randomized SVD)",
                        "pca_gram (PCA: грам-матрица XX^T)",
                    ],
                )
                st.caption("Цель: либо K компонент, либо доля объяснённой дисперсии.")
                dimred_target = int(st.number_input("K (сколько компонент/каналов оставить, 0=авто)", min_value=0, max_value=50000, value=50, step=10))
                dimred_target_var = float(st.number_input("Explained variance (0..1, пусто/0=не использовать)", min_value=0.0, max_value=1.0, value=0.0, step=0.05, format="%.3f"))
                dimred_priority = st.selectbox("Приоритет (если заданы и K, и explained variance)", ["explained_variance", "n_components"], index=0)
                dimred_pca_solver = st.selectbox("PCA solver (только для pca_full)", ["full", "randomized", "gram"], index=0, help="Если метод выбран как pca_* — solver берётся из метода. Здесь это для pca_full.")
                if dimred_method.startswith("kmeans"):
                    st.caption("K-Means: Группирует похожие временные ряды в один 'средний' ряд.")
                elif dimred_method.startswith("spatial"):
                    st.caption("Spatial: Требует координаты (x,y,z). Бьет пространство на кубики.")

            st.markdown("**Дополнительные настройки**")
            output_mode = st.selectbox("Режим вывода", ["both", "html", "excel"], index=0)
            include_diagnostics = st.checkbox("HTML: показывать диагностику", value=True)
            include_scans = st.toggle("Включить сканирование", value=False)
            include_matrix_tables = st.checkbox("HTML: показывать таблицу матрицы (текстом)", value=False)
            include_fft_plots = st.checkbox("HTML: FFT-графики", value=True)
            harmonic_top_k = st.number_input("Гармоники: top_k", min_value=1, max_value=20, value=3)
            save_series_bundle = st.checkbox(
                "Сохранять пакет рядов (raw+clean+QC+coords)",
                value=True,
                help="Пишет отдельный *_series.xlsx рядом с отчётами.",
            )

            st.markdown("**Partial-контроль (для *_partial)**")
            control_strategy = st.selectbox(
                "Что вычесть перед *_partial",
                ["нет", "глобальный сигнал", "глобальный + тренд", "глобальный + тренд + PCA"],
                index=1,
                help="Partial считаем на остатках после регрессии на выбранные компоненты контроля.",
            )
            control_pca_k = 0
            if "PCA" in control_strategy:
                control_pca_k = int(st.number_input("PCA k", min_value=1, max_value=50, value=3, step=1))

    # === БЛОК 2: ПАРАМЕТРЫ СВЯЗНОСТИ ===
    with st.expander("⚙️ 2. Параметры связности (Lags & Windows)", expanded=True):
        tabs = st.tabs(["Main estimation", "Diagnostic scans", "Топология графа"])

        with tabs[0]:
            st.write("Настройки, которые влияют на основную итоговую матрицу связности.")
            col_lag, col_thr = st.columns(2)
            with col_lag:
                preset_lag_mode = st.session_state.get("lag_selection_mode_ui", "fixed")
                lag_mode = st.radio(
                    "Режим лага для основного расчёта",
                    ["Автоматически (Optimize)", "Фиксированный"],
                    horizontal=True,
                    index=0 if preset_lag_mode == "optimize" else 1,
                )
                if lag_mode.startswith("Фикс"):
                    lag_selection = "fixed"
                    lag = st.slider("Лаг (точек)", 1, 50, int(st.session_state.get("lag", 1)), key="lag")
                    max_lag = max(1, int(st.session_state.get("max_lag", 12)))
                else:
                    lag_selection = "optimize"
                    max_lag = st.slider("Максимальный лаг для поиска", 1, 20, int(st.session_state.get("max_lag", 3)), key="max_lag", help="Проверим лаги от 1 до N и выберем лучший")
                    lag = 1

                use_main_windows = st.checkbox("Использовать окна в основном расчёте", value=bool(st.session_state.get("use_main_windows", False)), key="use_main_windows")
                window_policy = st.selectbox("Агрегация по окнам", ["best", "mean"], index=0 if str(st.session_state.get("window_policy", "best")) == "best" else 1, key="window_policy")
                window_sizes_text = st.text_input("Размеры окон (через запятую)", value=str(st.session_state.get("window_sizes_text", "128,256")), key="window_sizes_text")
                window_stride_main = st.number_input("Шаг окна (main, 0=auto)", min_value=0, max_value=100000, value=int(st.session_state.get("window_stride_main", 0)), step=1, key="window_stride_main")
                window_cube_level = st.selectbox("Legacy window×lag×position", ["off", "basic", "full"], index=0)
                window_cube_eval_limit = st.number_input("Main-cube eval_limit", min_value=20, max_value=5000, value=120, step=10)

            with col_thr:
                graph_threshold = st.slider(
                    "Порог значимости графа",
                    0.0,
                    1.0,
                    0.30,
                    0.05,
                    help="Связи слабее этого значения будут считаться шумом",
                )
                alpha = st.number_input("P-value alpha (для стат. тестов)", 0.001, 0.1, 0.05, format="%.3f")
                threshold = float(graph_threshold)

        with tabs[1]:
            st.info("Это не основной расчёт, а диагностические sweep/scans по окнам/лагам. Они заметно тяжелее.")
            if "include_scans" not in st.session_state:
                st.session_state["include_scans"] = bool(include_scans)
            include_scans = st.checkbox("Включить диагностические scans", value=bool(st.session_state.get("include_scans", False)), key="include_scans")
            scan_lag = st.checkbox("Scan по лагам", value=bool(st.session_state.get("scan_lag", False)), key="scan_lag")
            scan_window_pos = st.checkbox("Scan по позициям окна", value=bool(st.session_state.get("scan_window_pos", False)), key="scan_window_pos")
            scan_window_size = st.checkbox("Scan по размерам окна", value=bool(st.session_state.get("scan_window_size", False)), key="scan_window_size")
            scan_cube = st.checkbox("Scan cube", value=bool(st.session_state.get("scan_cube", False)), key="scan_cube")
            if include_scans:
                st.markdown("**1. Скользящее окно (динамика во времени)**")
                win_range = st.slider("Диапазон размеров окна", 32, 512, (96, 192), step=32)
                window_min, window_max = win_range
                window_step = st.number_input("window_step", min_value=1, max_value=1000000, value=96, step=1)
                window_size_default = st.number_input("window_size (для scan_window_pos)", min_value=2, max_value=1000000, value=128, step=1)

                st.markdown("**2. Скан по лагам**")
                lag_min = st.number_input("lag_min", min_value=1, max_value=2000, value=1, step=1)
                lag_max = st.number_input("lag_max", min_value=1, max_value=2000, value=min(2, int(max_lag)), step=1)
                lag_step = st.number_input("lag_step", min_value=1, max_value=2000, value=1, step=1)

                st.markdown("**3. 4D Куб (Window × Lag × Time)**")
                window_start_min = st.number_input("window_start_min (0=auto)", min_value=0, max_value=10_000_000, value=0, step=1)
                window_start_max = st.number_input("window_start_max (0=auto)", min_value=0, max_value=10_000_000, value=0, step=1)
                window_stride_scan = st.number_input("window_stride (scan, 0=auto)", min_value=0, max_value=10_000_000, value=0, step=1)
                window_max_windows = st.number_input("window_max_windows", min_value=1, max_value=5000, value=60, step=1)
                cube_combo_limit = st.number_input("cube_combo_limit", min_value=1, max_value=200000, value=9, step=1)
                cube_eval_limit = st.number_input("cube_eval_limit", min_value=1, max_value=2_000_000, value=225, step=5)
                cube_matrix_mode = st.selectbox("cube_matrix_mode", ["all", "selected"], index=0)
                cube_matrix_limit = st.number_input("cube_matrix_limit", min_value=1, max_value=2_000_000, value=225, step=5)
                cube_gallery_mode = st.selectbox("cube_gallery_mode", ["extremes", "topbottom", "quantiles"], index=0)
                cube_gallery_k = st.number_input("cube_gallery_k", min_value=1, max_value=1000, value=1, step=1)
                cube_gallery_limit = st.number_input("cube_gallery_limit", min_value=3, max_value=5000, value=60, step=5)
            else:
                window_min, window_max, window_step, window_size_default = 64, 192, 64, 128
                scan_lag = scan_cube = scan_window_pos = scan_window_size = False
                lag_min, lag_max, lag_step = 1, min(3, int(max_lag)), 1
                window_start_min = window_start_max = window_stride_scan = 0
                window_max_windows = 60
                cube_combo_limit, cube_eval_limit, cube_matrix_limit = 9, 225, 225
                cube_matrix_mode, cube_gallery_mode = "all", "extremes"
                cube_gallery_k, cube_gallery_limit = 1, 60

        with tabs[2]:
            st.markdown("**Network Science**")
            calc_topology = st.checkbox("Рассчитать метрики графа", value=True)
            st.caption("Найдем Хабы (Centrality), Кластеры (Communities) и построим таблицу лидеров.")

        st.markdown("---")
        st.subheader("Метод-специфичные оверрайды (advanced)")
        method_options_text = st.text_area(
            "method_options (JSON, ключ = метод)",
            value="",
            placeholder='Напр.: {"te_directed": {"scan_cube": false, "cube_matrix_mode": "selected"}}',
            height=80,
        )

    # === БЛОК 3: ВЫБОР МЕТОДОВ ===
    st.subheader("3. Выбор методов")


    all_methods = STABLE_METHODS + EXPERIMENTAL_METHODS
    if "selected_methods" not in st.session_state or not st.session_state.get("selected_methods"):
        st.session_state["selected_methods"] = list(DEFAULT_STABLE_METHODS)
    selected_methods = st.multiselect(
        "Выберите методы",
        all_methods,
        default=st.session_state.get("selected_methods", DEFAULT_STABLE_METHODS),
        key="selected_methods",
    )

    with st.expander("Быстрые кнопки", expanded=False):
        cqb1, cqb2, cqb3, cqb4 = st.columns(4)
        with cqb1:
            if st.button("Только fast stable"):
                st.session_state["selected_methods"] = _preset_payload("Fast stable")["selected_methods"]
                st.rerun()
        with cqb2:
            if st.button("Только default stable"):
                st.session_state["selected_methods"] = _preset_payload("Default stable")["selected_methods"]
                st.rerun()
        with cqb3:
            if st.button("Все stable"):
                st.session_state["selected_methods"] = list(STABLE_METHODS)
                st.rerun()
        with cqb4:
            if st.button("Все методы"):
                st.session_state["selected_methods"] = list(all_methods)
                st.rerun()

    run_plan = {
            "preprocess": preprocess,
            "fill_missing": fill_missing,
            "remove_outliers": remove_outliers,
            "outlier_rule": outlier_rule,
            "outlier_action": _out_act,
            "outlier_z": float(outlier_z),
            "outlier_k": float(outlier_k),
            "outlier_p_low": float(outlier_p_low),
            "outlier_p_high": float(outlier_p_high),
            "outlier_hampel_window": int(outlier_hampel_window),
            "outlier_jump_thr": (None if float(outlier_jump_thr)==0.0 else float(outlier_jump_thr)),
            "normalize": normalize,
            "normalize_mode": normalize_mode,
            "rank_ties": rank_ties,
            "remove_ar1": bool(remove_ar1),
            "remove_ar_order": int(remove_ar_order),
            "ar_diagnostics": bool(ar_diagnostics),
            "remove_seasonality": bool(remove_seasonality),
            "season_period": (None if int(season_period)==0 else int(season_period)),
            "qc_enabled": bool(qc_enabled),
            "spatial_bin_size": int(spatial_bin_size),
            "spatial_bin_method": str(spatial_bin_method),
            "spatial_grid_size": int(spatial_grid_size),
            "spatial_grid_method": str(spatial_grid_method),
            "lazy_spatial_bin": bool(lazy_spatial_bin),
            "time_chunk": int(time_chunk),
            "time_stride": int(time_stride),
            "selected_methods": list(selected_methods),
            "lag_selection": str(lag_selection),
            "lag": int(lag),
            "max_lag": int(max_lag),
            "use_main_windows": bool(use_main_windows),
            "lag_selection_mode_ui": ("optimize" if lag_selection == "optimize" else "fixed"),
            "window_sizes_text": str(window_sizes_text),
            "window_policy": str(window_policy),
            "include_scans": bool(include_scans),
            "output_mode": str(output_mode),
            "save_series_bundle": bool(save_series_bundle),
            "launch_preset": st.session_state.get("launch_preset", "Default stable"),
        }

    with st.expander("План запуска (что будет сделано)", expanded=False):
        st.write(run_plan)

    if source.startswith("Папка"):
        st.subheader("Пакетная обработка")
        if not batch_input_folder.strip():
            st.info("Укажи папку с входными файлами.")
            return

        if st.button("Запустить пакетный анализ", type="primary"):
            input_root = Path(batch_input_folder).expanduser()
            if not input_root.exists() or not input_root.is_dir():
                st.error(f"Папка не найдена: {input_root}")
                return

            files = _iter_input_files(
                str(input_root),
                recursive=bool(batch_recursive),
                allowed_exts=st.session_state.get("batch_allowed_exts"),
            )
            if not files:
                st.error("В указанной папке не найдено поддерживаемых файлов.")
                return

            batch_root = Path(batch_output_root.strip() or _default_batch_output_root(str(input_root))).expanduser()
            batch_root.mkdir(parents=True, exist_ok=True)

            manifest_path = batch_root / "batch_manifest.csv"
            manifest_jsonl = batch_root / "batch_manifest.jsonl"
            batch_log = batch_root / "batch_log.txt"
            run_config_path = batch_root / "run_config.json"
            _write_json(run_config_path, run_plan)
            _append_text(batch_log, f"[START] {datetime.now().isoformat()} | input_root={input_root} | files={len(files)}")

            manifest_rows: list[dict] = []
            # Два индикатора прогресса: общий по набору и локальный по текущему файлу.
            overall_box = st.empty()
            current_box = st.empty()
            prog_overall = st.progress(0)
            prog_current = st.progress(0)
            for i, src_path in enumerate(files, start=1):
                overall_box.markdown(f"**Набор данных:** {i}/{len(files)} — `{Path(src_path).name}`")
                current_box.markdown("**Этап текущего файла:** подготовка")
                prog_current.progress(0)
                p = Path(src_path)
                try:
                    rel_parent = p.parent.relative_to(input_root) if input_root in p.parents or p.parent == input_root else Path(".")
                except Exception:
                    rel_parent = Path(".")
                safe_rel = str(rel_parent).replace("..", "_up_").replace(":", "_").replace("\\", "__").replace("/", "__")
                stem = p.stem if safe_rel in {"", "."} else f"{safe_rel}__{p.stem}"
                safe_stem = _safe_slug(stem)
                run_dir = batch_root / safe_stem
                status_json = run_dir / "status.json"

                if bool(batch_skip_existing) and status_json.exists():
                    try:
                        prev = json.loads(status_json.read_text(encoding="utf-8"))
                        prev_status = str(prev.get("status", "")).lower()
                        if prev_status in {"ok", "partial", "skipped"}:
                            row = {
                                "index": i,
                                "input_file": str(p),
                                "relative_parent": str(rel_parent),
                                "run_dir": str(run_dir),
                                "status": "skipped",
                                "excel_path": prev.get("excel_path", ""),
                                "html_path": prev.get("html_path", ""),
                                "series_path": prev.get("series_path", ""),
                                "status_json": str(status_json),
                                "error": "",
                            }
                            manifest_rows.append(row)
                            _append_text(batch_log, f"[SKIP ] {p}")
                            prog_overall.progress(int(100 * i / max(1, len(files))))
                            continue
                    except Exception:
                        pass

                run_dir.mkdir(parents=True, exist_ok=True)
                row = {
                    "index": i,
                    "input_file": str(p),
                    "relative_parent": str(rel_parent),
                    "run_dir": str(run_dir),
                    "status": "error",
                    "excel_path": "",
                    "html_path": "",
                    "series_path": "",
                    "status_json": str(status_json),
                    "error": "",
                }
                tool = None
                try:
                    def _batch_stage_cb(stage: str, progress, meta: dict):
                        """Коллбек движка: обновляет UI-этап и прогресс для текущего файла."""
                        try:
                            current_box.markdown(f"**Этап текущего файла:** {stage}")
                            if progress is not None:
                                prog_current.progress(int(max(0.0, min(1.0, float(progress))) * 100))
                        except Exception:
                            pass

                    cfg = engine.AnalysisConfig(
                        max_lag=int(max_lag),
                        p_value_alpha=float(alpha),
                        graph_threshold=float(threshold),
                        enable_experimental=bool(enable_experimental),
                        auto_difference=bool(check_stat),
                        pvalue_correction=str(pvalue_correction),
                        spatial_bin_size=int(spatial_bin_size),
                        spatial_bin_method=str(spatial_bin_method),
                        spatial_grid_size=int(spatial_grid_size),
                        spatial_grid_method=str(spatial_grid_method),
                        lazy_spatial_bin=bool(lazy_spatial_bin),
                        time_chunk=int(time_chunk),
                    )
                    tool = engine.BigMasterTool(config=cfg, stage_callback=_batch_stage_cb)
                    tool.load_data_excel(
                        str(src_path),
                        preprocess=bool(preprocess),
                        fill_missing=bool(fill_missing),
                        normalize=bool(normalize),
                        normalize_mode=str(normalize_mode),
                        rank_ties=str(rank_ties),
                        remove_outliers=bool(remove_outliers),
                        outlier_rule=str(outlier_rule),
                        outlier_action=str(outlier_action).split()[0],
                        outlier_z=float(outlier_z),
                        outlier_k=float(outlier_k),
                        outlier_p_low=float(outlier_p_low),
                        outlier_p_high=float(outlier_p_high),
                        outlier_hampel_window=int(outlier_hampel_window),
                        outlier_jump_thr=(None if float(outlier_jump_thr) == 0.0 else float(outlier_jump_thr)),
                        outlier_local_median_window=int(outlier_local_median_window),
                        check_stationarity=bool(check_stat),
                        log_transform=bool(log_transform),
                        remove_ar1=bool(remove_ar1),
                        remove_ar_order=int(remove_ar_order),
                        ar_diagnostics=bool(ar_diagnostics),
                        remove_seasonality=bool(remove_seasonality),
                        season_period=(None if int(season_period) == 0 else int(season_period)),
                        qc_enabled=bool(qc_enabled),
                        feature_limit=int(feature_limit),
                        feature_sampling=str(feature_sampling_val),
                        spatial_grid_size=int(spatial_grid_size),
                        spatial_grid_method=str(spatial_grid_method),
                        lazy_spatial_bin=bool(lazy_spatial_bin),
                        time_chunk=int(time_chunk),
                        time_stride=int(time_stride),
                        h5_spatial_bin=int(h5_spatial_bin),
                    )

                    window_sizes_main = _parse_int_list_text(window_sizes_text) if use_main_windows else None
                    run_window_stride = int(window_stride_main) if int(window_stride_main) > 0 else 0
                    method_options = None
                    if method_options_text.strip():
                        try:
                            method_options = json.loads(method_options_text)
                        except Exception:
                            method_options = None
                    w_grid = list(range(int(window_min), int(window_max) + 1, max(1, int(window_step))))

                    tool.run_selected_methods(
                        selected_methods,
                        max_lag=int(max_lag),
                        lag_selection=lag_selection,
                        lag=int(lag),
                        control_strategy=(
                            "none"
                            if control_strategy == "нет"
                            else (
                                "global_mean"
                                if control_strategy == "глобальный сигнал"
                                else ("global_mean_trend_pca" if "PCA" in control_strategy else "global_mean_trend")
                            )
                        ),
                        control_pca_k=int(control_pca_k or 0),
                        window_sizes=window_sizes_main,
                        window_stride=run_window_stride,
                        window_policy=window_policy,
                        window_cube_level=window_cube_level,
                        window_cube_eval_limit=int(window_cube_eval_limit),
                        method_options=method_options,
                        dimred_enabled=bool(dimred_enabled),
                        dimred_method=str(dimred_method).split()[0],
                        dimred_target=int(dimred_target),
                        dimred_target_var=(float(dimred_target_var) if float(dimred_target_var) > 0 else None),
                        dimred_priority=str(dimred_priority),
                        dimred_pca_solver=str(dimred_pca_solver),
                        scan_window_pos=(bool(scan_window_pos) if include_scans else False),
                        scan_window_size=(bool(scan_window_size) if include_scans else False),
                        scan_lag=(bool(scan_lag) if include_scans else False),
                        scan_cube=(bool(scan_cube) if include_scans else False),
                        window_sizes_grid=w_grid,
                        window_min=int(window_min),
                        window_max=int(window_max),
                        window_step=int(window_step),
                        window_size=int(window_size_default),
                        window_start_min=int(window_start_min),
                        window_start_max=int(window_start_max),
                        window_max_windows=int(window_max_windows),
                        lag_min=int(lag_min),
                        lag_max=int(lag_max),
                        lag_step=int(lag_step),
                        cube_combo_limit=int(cube_combo_limit),
                        cube_eval_limit=int(cube_eval_limit),
                        cube_matrix_mode=str(cube_matrix_mode),
                        cube_matrix_limit=int(cube_matrix_limit),
                        cube_gallery_mode=str(cube_gallery_mode),
                        cube_gallery_k=int(cube_gallery_k),
                        cube_gallery_limit=int(cube_gallery_limit),
                    )

                    if calc_topology:
                        try:
                            tool.calculate_graph_metrics(threshold=float(graph_threshold))
                        except Exception as exc:
                            row["error"] = f"graph_metrics: {exc}"

                    series_path = run_dir / f"{safe_stem}_series.xlsx"
                    series_artifact_path = None

                    if bool(save_series_bundle):
                        try:
                            exported = tool.export_series_bundle(str(series_path))
                            if exported:
                                series_artifact_path = Path(exported)
                        except Exception as exc:
                            row["error"] = (row["error"] + " | " if row["error"] else "") + f"series: {exc}"

                    excel_path = run_dir / f"{safe_stem}_full.xlsx"
                    html_path = run_dir / f"{safe_stem}_report.html"
                    if output_mode in {"excel", "both"}:
                        tool.export_big_excel(
                            str(excel_path),
                            threshold=threshold,
                            p_value_alpha=alpha,
                            include_ar_diagnostics=True,
                        )
                        row["excel_path"] = str(excel_path)
                    if output_mode in {"html", "both"}:
                        tool.export_html_report(
                            str(html_path),
                            graph_threshold=threshold,
                            p_alpha=alpha,
                            include_diagnostics=include_diagnostics,
                            include_scans=include_scans,
                            include_ar_diagnostics=True,
                            include_matrix_tables=include_matrix_tables,
                            include_fft_plots=include_fft_plots,
                            harmonic_top_k=int(harmonic_top_k),
                            include_series_files=True,
                        )
                        row["html_path"] = str(html_path)
                    if series_artifact_path is None and series_path.exists():
                        series_artifact_path = series_path

                    row["series_path"] = (
                        str(series_artifact_path)
                        if series_artifact_path is not None and series_artifact_path.exists()
                        else ""
                    )
                    try:
                        tool.export_connectivity_bundle(
                            str(run_dir),
                            name_prefix=safe_stem,
                            include_scan_matrices=bool(include_scans),
                        )
                    except Exception as exc:
                        row["error"] = (row["error"] + " | " if row["error"] else "") + f"bundle: {exc}"

                    row["status"] = "ok" if not row["error"] else "partial"
                    prog_current.progress(100)
                    _append_text(batch_log, f"[OK   ] {p} | status={row['status']}")
                except Exception as exc:
                    row["status"] = "error"
                    row["error"] = str(exc)
                    _append_text(batch_log, f"[ERROR] {p} | {exc}")
                    _append_text(batch_log, traceback.format_exc())
                finally:
                    _write_json(status_json, row)
                    manifest_rows.append(row)
                    try:
                        del tool
                    except Exception:
                        pass
                    gc.collect()
                    prog_overall.progress(int(100 * i / max(1, len(files))))

            manifest_df = pd.DataFrame(manifest_rows)
            manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
            with manifest_jsonl.open("w", encoding="utf-8") as fh:
                for row in manifest_rows:
                    fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")

            _append_text(batch_log, f"[DONE ] {datetime.now().isoformat()} | root={batch_root}")
            st.success("Пакетный потоковый расчёт завершён")
            st.code(str(batch_root))
            st.dataframe(manifest_df, use_container_width=True)
            st.download_button("Скачать manifest.csv", manifest_path.read_bytes(), manifest_path.name)
            if batch_root.exists():
                zip_path = _zip_tree(batch_root, batch_root.with_suffix(".zip"))
                st.download_button("Скачать ZIP результатов", zip_path.read_bytes(), zip_path.name)
            return
        return

    if st.button("Запустить анализ", type="primary"):
        if source.startswith("Файл") and not str(local_file_path).strip() and not uploaded_file:
            st.error("Файл не указан: выбери локальный путь или загрузи файл через fallback uploader.")
            return

        # Готовим run-dir
        if source.startswith("Файл") and str(local_file_path).strip():
            stem = Path(str(local_file_path).strip()).stem or "run"
        else:
            stem = (Path(uploaded_file.name).stem if uploaded_file else synth_name) or "run"
        run_dir = _make_run_dir(stem)
        _write_json(run_dir / "run_config.json", run_plan)

        # Сохраняем входные данные (или синтетические)
        input_path: Path
        try:
            if source.startswith("Файл") and str(local_file_path).strip():
                input_path = Path(str(local_file_path).strip()).expanduser()
                if not input_path.exists() or not input_path.is_file():
                    st.error(f"Файл не найден: {input_path}")
                    return
            elif uploaded_file:
                input_path = run_dir / uploaded_file.name
                input_path.write_bytes(uploaded_file.getbuffer())
            else:
                if source.startswith("Синтетика (пресеты)"):
                    preset = st.session_state.get("preset", "Coupled system (X→Y, Z noise, S season)")
                    n_samples = int(st.session_state.get("preset_n_samples", 800) or 800)
                    seed = int(st.session_state.get("preset_seed", 42) or 42)
                    np.random.seed(seed)
                    if str(preset).startswith("Coupled"):
                        synth_df = generator.generate_coupled_system(n_samples=n_samples)
                    else:
                        synth_df = generator.generate_random_walks(n_vars=3, n_samples=n_samples)
                else:
                    x_expr = st.session_state.get("x_expr", "sin(2*pi*t/50) + 0.2*randn()")
                    y_expr = st.session_state.get("y_expr", "0.8*X + 0.3*randn()")
                    z_expr = st.session_state.get("z_expr", "rw(0.5)")
                    n_samples = int(st.session_state.get("n_samples", 800) or 800)
                    dt = float(st.session_state.get("dt", 1.0) or 1.0)
                    seed = int(st.session_state.get("seed", 42) or 42)
                    synth_df = generator.generate_formula_dataset(
                        n_samples=n_samples,
                        dt=dt,
                        seed=seed,
                        specs=[
                            generator.FormulaSpec("X", x_expr),
                            generator.FormulaSpec("Y", y_expr),
                            generator.FormulaSpec("Z", z_expr),
                        ],
                    )
                input_path = run_dir / f"{stem}_input.csv"
                synth_df.to_csv(input_path, index=False)
        except Exception as e:
            st.error(f"Ошибка подготовки данных: {e}")
            return

        stage_box = st.empty()
        prog = st.progress(0)

        def _stage_cb(stage: str, progress, meta: dict):
            """Показывает этап текущего запуска и процент готовности."""
            try:
                stage_box.markdown(f"**Этап:** {stage}")
                if progress is not None:
                    prog.progress(int(max(0.0, min(1.0, float(progress))) * 100))
            except Exception:
                pass

        cfg = engine.AnalysisConfig(
            max_lag=int(max_lag),
            p_value_alpha=float(alpha),
            graph_threshold=float(threshold),
            enable_experimental=bool(enable_experimental),
            auto_difference=bool(check_stat),
            pvalue_correction=str(pvalue_correction),
            spatial_bin_size=int(spatial_bin_size),
            spatial_bin_method=str(spatial_bin_method),
            spatial_grid_size=int(spatial_grid_size),
            spatial_grid_method=str(spatial_grid_method),
            lazy_spatial_bin=bool(lazy_spatial_bin),
            time_chunk=int(time_chunk),
        )
        tool = engine.BigMasterTool(config=cfg, stage_callback=_stage_cb)

        with st.spinner("Загрузка и расчёт..."):
            try:
                tool.load_data_excel(
                    str(input_path),
                    preprocess=preprocess,
                    normalize=normalize,
                    normalize_mode=normalize_mode,
                    rank_ties=rank_ties,
                    fill_missing=fill_missing,
                    remove_outliers=remove_outliers,
                    outlier_rule=outlier_rule,
                    outlier_action=_out_act,
                    outlier_z=float(outlier_z),
                    outlier_k=float(outlier_k),
                    outlier_p_low=float(outlier_p_low),
                    outlier_p_high=float(outlier_p_high),
                    outlier_hampel_window=int(outlier_hampel_window),
                    outlier_jump_thr=(None if float(outlier_jump_thr)==0.0 else float(outlier_jump_thr)),
                    outlier_local_median_window=int(outlier_local_median_window),
                    log_transform=log_transform,
                    remove_ar1=bool(remove_ar1),
                    remove_ar_order=int(remove_ar_order),
                    ar_diagnostics=bool(ar_diagnostics),
                    remove_seasonality=bool(remove_seasonality),
                    season_period=(None if int(season_period) == 0 else int(season_period)),
                    qc_enabled=bool(qc_enabled),
                    feature_limit=(int(feature_limit) if int(feature_limit) > 0 else None),
                    feature_sampling=str(feature_sampling_val),
                    h5_spatial_bin=int(h5_spatial_bin),
                    spatial_grid_size=int(spatial_grid_size),
                    spatial_grid_method=str(spatial_grid_method),
                    lazy_spatial_bin=bool(lazy_spatial_bin),
                    time_chunk=int(time_chunk),
                    time_stride=int(time_stride),
                    spatial_bin_size=int(spatial_bin_size),
                    spatial_bin_method=str(spatial_bin_method),
                )

                # Явное сообщение об истинной причине, если после импорта/предобработки
                # не осталось валидных рядов для расчётов.
                df_loaded = getattr(tool, "data", None)
                if (
                    df_loaded is None
                    or getattr(df_loaded, "empty", False)
                    or int(getattr(df_loaded, "shape", (0, 0))[1]) == 0
                ):
                    st.error(
                        "После импорта и предобработки не осталось ни одного ряда. "
                        "Для H5 voxel-data обычно причина в слишком агрессивной фильтрации признаков."
                    )
                    st.stop()

                # main windows
                window_sizes_main = None
                if use_main_windows:
                    window_sizes_main = _parse_int_list_text(window_sizes_text)

                # scans/main используют общий параметр window_stride в движке.
                stride_scan = None if int(window_stride_scan) == 0 else int(window_stride_scan)
                stride_main = None if int(window_stride_main) == 0 else int(window_stride_main)
                run_window_stride = stride_scan if stride_scan is not None else stride_main

                # method options
                method_options = None
                if method_options_text.strip():
                    try:
                        method_options = json.loads(method_options_text)
                        if not isinstance(method_options, dict):
                            method_options = None
                    except Exception:
                        method_options = None

                w_grid = list(range(int(window_min), int(window_max) + 1, max(1, int(window_step))))

                tool.run_selected_methods(
                    selected_methods,
                    max_lag=int(max_lag),
                    lag_selection=lag_selection,
                    lag=int(lag),
                    control_strategy=(
                        "none"
                        if control_strategy == "нет"
                        else (
                            "global_mean"
                            if control_strategy == "глобальный сигнал"
                            else ("global_mean_trend_pca" if "PCA" in control_strategy else "global_mean_trend")
                        )
                    ),
                    control_pca_k=int(control_pca_k or 0),
                    window_sizes=window_sizes_main,
                    window_stride=run_window_stride,
                    window_policy=window_policy,
                    window_cube_level=window_cube_level,
                    window_cube_eval_limit=int(window_cube_eval_limit),
                    method_options=method_options,
                    dimred_enabled=bool(dimred_enabled),
                    dimred_method=str(dimred_method).split()[0],
                    dimred_target=int(dimred_target),
                    dimred_target_var=(float(dimred_target_var) if float(dimred_target_var) > 0 else None),
                    dimred_priority=str(dimred_priority),
                    dimred_pca_solver=str(dimred_pca_solver),
                    # scans
                    scan_window_pos=(bool(scan_window_pos) if include_scans else False),
                    scan_window_size=(bool(scan_window_size) if include_scans else False),
                    scan_lag=(bool(scan_lag) if include_scans else False),
                    scan_cube=(bool(scan_cube) if include_scans else False),
                    window_sizes_grid=w_grid,
                    window_min=int(window_min),
                    window_max=int(window_max),
                    window_step=int(window_step),
                    window_size=int(window_size_default),
                    window_start_min=int(window_start_min),
                    window_start_max=int(window_start_max),
                    window_max_windows=int(window_max_windows),
                    lag_min=int(lag_min),
                    lag_max=int(lag_max),
                    lag_step=int(lag_step),
                    cube_combo_limit=int(cube_combo_limit),
                    cube_eval_limit=int(cube_eval_limit),
                    cube_matrix_mode=str(cube_matrix_mode),
                    cube_matrix_limit=int(cube_matrix_limit),
                    cube_gallery_mode=str(cube_gallery_mode),
                    cube_gallery_k=int(cube_gallery_k),
                    cube_gallery_limit=int(cube_gallery_limit),
                )


                if calc_topology:
                    with st.spinner("Анализ топологии графов..."):
                        try:
                            tool.calculate_graph_metrics(threshold=float(graph_threshold))
                            st.success("Топология рассчитана!")
                        except Exception as e:
                            st.warning(f"Ошибка анализа графов: {e}")

                # Сохраняем ряды отдельным артефактом рядом с отчётами.
                # Это может быть как .xlsx, так и директория с CSV.
                series_path = run_dir / f"{stem}_series.xlsx"
                series_artifact_path = None

                if bool(save_series_bundle):
                    try:
                        exported = tool.export_series_bundle(str(series_path))
                        if exported:
                            series_artifact_path = Path(exported)
                    except Exception:
                        pass

                excel_path = run_dir / f"{stem}_full.xlsx"
                html_path = run_dir / f"{stem}_report.html"

                if output_mode in {"excel", "both"}:
                    tool.export_big_excel(
                            str(excel_path),
                            threshold=threshold,
                            p_value_alpha=alpha,
                            include_ar_diagnostics=True,
                        )

                if output_mode in {"html", "both"}:
                    tool.export_html_report(
                        str(html_path),
                        graph_threshold=threshold,
                        p_alpha=alpha,
                        include_diagnostics=include_diagnostics,
                        include_ar_diagnostics=True,
                        include_scans=include_scans,
                        include_matrix_tables=include_matrix_tables,
                        include_fft_plots=include_fft_plots,
                        harmonic_top_k=int(harmonic_top_k),
                        include_series_files=True,
                    )

                st.success("Готово!")
                st.code(str(run_dir))
                try:
                    prog.progress(100)
                except Exception:
                    pass

                try:
                    from src.reporting.run_summary import build_run_summary_ru

                    st.subheader("Что именно сделано")
                    st.text(build_run_summary_ru(tool, run_dir=str(run_dir)))
                except Exception:
                    pass

                c1, c2, c3 = st.columns(3)
                with c1:
                    if output_mode in {"excel", "both"} and excel_path.exists():
                        st.download_button("Скачать Excel", excel_path.read_bytes(), excel_path.name)
                with c2:
                    if output_mode in {"html", "both"} and html_path.exists():
                        st.download_button("Скачать HTML", html_path.read_bytes(), html_path.name)
                with c3:
                    if series_artifact_path is None and series_path.exists():
                        series_artifact_path = series_path

                    if series_artifact_path is not None and series_artifact_path.exists():
                        if series_artifact_path.is_file():
                            st.download_button(
                                "Скачать ряды",
                                series_artifact_path.read_bytes(),
                                series_artifact_path.name,
                            )
                        elif series_artifact_path.is_dir():
                            series_zip_path = _zip_tree(
                                series_artifact_path,
                                series_artifact_path.with_suffix(".zip"),
                            )
                            st.download_button(
                                "Скачать ряды (ZIP)",
                                series_zip_path.read_bytes(),
                                series_zip_path.name,
                            )

                    if series_artifact_path is not None and series_artifact_path.exists():
                        if series_artifact_path.is_file():
                            st.caption(f"Ряды сохранены в файл: {series_artifact_path.name}")
                        elif series_artifact_path.is_dir():
                            st.caption(f"Ряды сохранены в папку: {series_artifact_path.name}")

                status_payload = {
                    "input_file": str(input_path),
                    "run_dir": str(run_dir),
                    "status": "ok",
                    "excel_path": str(excel_path) if excel_path.exists() else "",
                    "html_path": str(html_path) if html_path.exists() else "",
                    "series_path": (
                        str(series_artifact_path)
                        if series_artifact_path is not None and series_artifact_path.exists()
                        else ""
                    ),
                }
                _write_json(run_dir / "status.json", status_payload)

                # Ряды раскрываются только по клику
                with st.expander("Исходные ряды (preview)", expanded=False):
                    try:
                        df_show = tool.data_raw if not tool.data_raw.empty else tool.data
                        st.line_chart(df_show)
                        st.dataframe(df_show.head(200), height=320)
                    except Exception:
                        pass


                if calc_topology and hasattr(tool, "graph_results"):
                    st.subheader("🏆 Лидеры сети (Top Nodes)")
                    for variant, res in tool.graph_results.items():
                        with st.expander(f"Топология: {variant}"):
                            if isinstance(res, dict) and res.get("error"):
                                st.warning(res["error"])
                                continue
                            c_graph1, c_graph2 = st.columns([2, 1])
                            with c_graph1:
                                st.dataframe(res["node_metrics"].head(10), use_container_width=True)
                            with c_graph2:
                                st.write("Глобальные метрики:")
                                st.json(res["global_metrics"])

                st.subheader("Предварительный просмотр матриц")
                from src.visualization import plots

                # много матриц — прячем в прокручиваемый контейнер.
                # Для обратной совместимости со старыми Streamlit делаем fallback.
                try:
                    matrix_container = st.container(height=650)
                except TypeError:
                    matrix_container = nullcontext()
                with matrix_container:
                    for method in selected_methods:
                        mat = tool.results.get(method)
                        if mat is None:
                            continue
                        chosen = None
                        try:
                            chosen = (tool.results_meta.get(method) or {}).get("chosen_lag")
                        except Exception:
                            chosen = None
                        title = f"{method}" + (f" (chosen_lag={chosen})" if chosen is not None else "")
                        buf = plots.plot_heatmap(mat, title)
                        st.image(buf, caption=title)

            except Exception as e:
                st.error(f"Ошибка выполнения: {e}")
                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
