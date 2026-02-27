"""Веб-интерфейс (Streamlit) для Time Series Analysis Tool (локально)."""

from __future__ import annotations

import json
import os
import sys
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

configure_warnings()


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

def main() -> None:
    st.set_page_config(page_title="Анализ Временных Рядов (Локально)", layout="wide")
    st.title("Анализ Связности Временных Рядов")
    st.caption(f"Локальная версия. Результаты сохраняются в папку: {SAVE_FOLDER}")

    source = st.radio(
        "Источник данных",
        ["Файл (CSV/XLSX)", "Синтетика (формулы)", "Синтетика (пресеты)"],
        index=0,
        horizontal=True,
    )

    uploaded_file = None
    synth_df: pd.DataFrame | None = None
    synth_name = "synthetic"

    if source.startswith("Файл"):
        uploaded_file = st.file_uploader("Выберите файл", type=["csv", "xlsx"])
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

    else:
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

            _out_act = (
                "mask"
                if str(outlier_action).startswith("mask")
                else ("clip" if str(outlier_action).startswith("clip") else ("median" if str(outlier_action).startswith("median") else "local_median"))
            )
            log_transform = st.checkbox("Лог-преобразование (только >0)", value=False)
            remove_ar1 = st.checkbox("Убрать AR(1) (прибл. prewhitening)", value=False)
            remove_seasonality = st.checkbox("Убрать сезонность (STL)", value=False)
            season_period = st.number_input("Период сезонности (0=авто)", min_value=0, max_value=1000000, value=0, step=1)
            qc_enabled = st.checkbox(
                "QC по каждому ряду/вокселю (mean/std/дрейф/спайки/AR1)",
                value=True,
                help="Помогает быстро увидеть 'битые' ряды и причины ложной связности.",
            )

        with c2:
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
            include_scans = st.toggle("Включить сканирование", value=True)
            include_matrix_tables = st.checkbox("HTML: показывать таблицу матрицы (текстом)", value=False)
            include_fft_plots = st.checkbox("HTML: FFT-графики", value=True)
            harmonic_top_k = st.number_input("Гармоники: top_k", min_value=1, max_value=20, value=5)
            save_series_bundle = st.checkbox(
                "Сохранять пакет рядов (raw+clean+QC+coords)",
                value=True,
                help="Пишет отдельный *_series.xlsx рядом с отчётами.",
            )

            st.markdown("**Partial-контроль (для *_partial)**")
            control_strategy = st.selectbox(
                "Что вычесть перед *_partial",
                ["нет", "глобальный сигнал", "глобальный + тренд", "глобальный + тренд + PCA"],
                index=2,
                help="Partial считаем на остатках после регрессии на выбранные компоненты контроля.",
            )
            control_pca_k = 0
            if "PCA" in control_strategy:
                control_pca_k = int(st.number_input("PCA k", min_value=1, max_value=50, value=3, step=1))

    # === БЛОК 2: ПАРАМЕТРЫ СВЯЗНОСТИ ===
    with st.expander("⚙️ 2. Параметры связности (Lags & Windows)", expanded=True):
        tabs = st.tabs(["Основное", "Сканирование (Advanced)", "Топология графа"])

        with tabs[0]:
            st.write("Базовые настройки для расчета одной итоговой матрицы.")
            col_lag, col_thr = st.columns(2)
            with col_lag:
                lag_mode = st.radio("Подбор лага (задержки)", ["Автоматически (Optimize)", "Фиксированный"], horizontal=True)
                if lag_mode.startswith("Фикс"):
                    lag_selection = "fixed"
                    lag = st.slider("Лаг (точек)", 1, 50, 1)
                    max_lag = st.slider("max_lag (для сканов/ограничений)", 1, 200, 12)
                else:
                    lag_selection = "optimize"
                    max_lag = st.slider("Максимальный лаг для поиска", 1, 20, 5, help="Проверим лаги от 1 до N и выберем лучший")
                    lag = 1

                use_main_windows = st.checkbox("Использовать окна в основном расчёте", value=False)
                window_policy = st.selectbox("Политика окон (main)", ["best", "mean"], index=0)
                window_sizes_text = st.text_input("main window_sizes", value="256,512")
                window_stride_main = st.number_input("stride (main, 0=auto)", min_value=0, max_value=100000, value=0, step=1)
                window_cube_level = st.selectbox("Main window×lag×position (legacy)", ["off", "basic", "full"], index=0)
                window_cube_eval_limit = st.number_input("Main-cube eval_limit", min_value=20, max_value=5000, value=120, step=10)

            with col_thr:
                graph_threshold = st.slider(
                    "Порог значимости графа",
                    0.0,
                    1.0,
                    0.25,
                    0.05,
                    help="Связи слабее этого значения будут считаться шумом",
                )
                alpha = st.number_input("P-value alpha (для стат. тестов)", 0.001, 0.1, 0.05, format="%.3f")
                threshold = float(graph_threshold)

        with tabs[1]:
            st.info("Сканирование строит графики того, как меняется связь в зависимости от параметров. Это долго, но полезно.")
            if include_scans:
                st.markdown("**1. Скользящее окно (динамика во времени)**")
                win_range = st.slider("Диапазон размеров окна", 32, 512, (64, 192), step=32)
                window_min, window_max = win_range
                window_step = st.number_input("window_step", min_value=1, max_value=1000000, value=64, step=1)
                window_size_default = st.number_input("window_size (для scan_window_pos)", min_value=2, max_value=1000000, value=128, step=1)

                st.markdown("**2. Скан по лагам**")
                scan_lag = st.checkbox("Проверить влияние лага (кривая качества)", value=True)
                lag_min = st.number_input("lag_min", min_value=1, max_value=2000, value=1, step=1)
                lag_max = st.number_input("lag_max", min_value=1, max_value=2000, value=min(3, int(max_lag)), step=1)
                lag_step = st.number_input("lag_step", min_value=1, max_value=2000, value=1, step=1)

                st.markdown("**3. 4D Куб (Window × Lag × Time)**")
                scan_cube = st.checkbox("Построить 3D-карту устойчивости", value=False, help="Очень ресурсоемко!")
                scan_window_pos = st.checkbox("scan_window_pos", value=True)
                scan_window_size = st.checkbox("scan_window_size", value=True)
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
    selected_methods = st.multiselect("Выберите методы", all_methods, default=STABLE_METHODS[:2])

    with st.expander("План запуска (что будет сделано)", expanded=False):
        st.write({
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
            "remove_seasonality": bool(remove_seasonality),
            "season_period": (None if int(season_period)==0 else int(season_period)),
            "qc_enabled": bool(qc_enabled),
        })

    if st.button("Запустить анализ", type="primary"):
        if source.startswith("Файл") and not uploaded_file:
            st.error("Файл не загружен!")
            return

        # Готовим run-dir
        stem = (Path(uploaded_file.name).stem if uploaded_file else synth_name) or "run"
        run_dir = _make_run_dir(stem)

        # Сохраняем входные данные (или синтетические)
        input_path: Path
        try:
            if uploaded_file:
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

        tool = engine.BigMasterTool(stage_callback=_stage_cb)

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
                    remove_seasonality=bool(remove_seasonality),
                    season_period=(None if int(season_period) == 0 else int(season_period)),
                    qc_enabled=bool(qc_enabled),
                )

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

                # Сохраняем ряды отдельным файлом рядом с отчётами (если не выключено).
                series_path = run_dir / f"{stem}_series.xlsx"
                if bool(save_series_bundle):
                    try:
                        tool.export_series_bundle(str(series_path))
                    except Exception:
                        pass

                excel_path = run_dir / f"{stem}_full.xlsx"
                html_path = run_dir / f"{stem}_report.html"

                if output_mode in {"excel", "both"}:
                    tool.export_big_excel(str(excel_path), threshold=threshold, p_value_alpha=alpha)

                if output_mode in {"html", "both"}:
                    tool.export_html_report(
                        str(html_path),
                        graph_threshold=threshold,
                        p_alpha=alpha,
                        include_diagnostics=include_diagnostics,
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
                    if series_path.exists():
                        st.download_button("Скачать ряды (xlsx)", series_path.read_bytes(), series_path.name)

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
                import traceback

                st.text(traceback.format_exc())


if __name__ == "__main__":
    main()
