#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Графический интерфейс (Tkinter) с вкладками: Один файл, Пакетная обработка, Генератор.
"""

import os
import sys
import traceback
import webbrowser
from pathlib import Path
import re
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import json

import pandas as pd

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.engine import BigMasterTool, method_mapping
from src.core import data_loader, generator
from src.config import AnalysisConfig, PYINFORM_AVAILABLE
from src.core.preprocessing import configure_warnings

# Подавляем FutureWarning от statsmodels (verbose deprecated)
configure_warnings()


class App(tk.Tk):
    """Основное GUI-приложение с вкладками для анализа и генерации данных."""

    def __init__(self) -> None:
        super().__init__()
        self.title("TimeSeries Analysis Tool Pro")
        self.geometry("1000x800")

        self.max_lag = tk.IntVar(value=5)
        self.graph_threshold = tk.DoubleVar(value=0.2)
        self.p_alpha = tk.DoubleVar(value=0.05)
        self.auto_diff = tk.BooleanVar(value=False)
        self.p_correction = tk.StringVar(value="none")
        self.method_vars: dict[str, tk.BooleanVar] = {}

        # Вывод/отчёт
        self.output_mode = tk.StringVar(value="both")  # both|html|excel
        self.include_diagnostics = tk.BooleanVar(value=True)
        self.include_fft_plots = tk.BooleanVar(value=False)
        self.include_scans = tk.BooleanVar(value=True)
        self.include_matrix_tables = tk.BooleanVar(value=False)

        # Предобработка (до анализа)
        self.preproc_enabled = tk.BooleanVar(value=True)
        self.preproc_remove_outliers = tk.BooleanVar(value=True)
        self.preproc_remove_ar1 = tk.BooleanVar(value=False)
        self.preproc_ar_order = tk.IntVar(value=1)
        self.preproc_remove_seasonality = tk.BooleanVar(value=False)
        self.preproc_season_period = tk.IntVar(value=0)  # 0 => auto
        self.preproc_log_transform = tk.BooleanVar(value=False)
        self.preproc_normalize = tk.BooleanVar(value=True)
        self.preproc_normalize_mode = tk.StringVar(value="zscore")  # zscore|minmax|robust_z|rank|rank_dense|rank_pct
        self.preproc_rank_ties = tk.StringVar(value="average")
        self.preproc_outlier_rule = tk.StringVar(value="robust_z")  # robust_z|zscore|iqr|percentile|hampel|jump
        self.preproc_outlier_action = tk.StringVar(value="mask")  # mask|clip|median|local_median
        self.preproc_outlier_z = tk.DoubleVar(value=5.0)
        self.preproc_outlier_k = tk.DoubleVar(value=1.5)
        self.preproc_outlier_p_low = tk.DoubleVar(value=0.5)
        self.preproc_outlier_p_high = tk.DoubleVar(value=99.5)
        self.preproc_outlier_hampel_window = tk.IntVar(value=7)
        self.preproc_outlier_jump_thr = tk.DoubleVar(value=0.0)  # 0 => auto
        self.preproc_outlier_local_median_window = tk.IntVar(value=7)
        self.preproc_fill_missing = tk.BooleanVar(value=True)
        self.preprocess_stage = tk.StringVar(value="pre")  # pre|post|both
        # Пост-обработка после dimred (по умолчанию выключена для обратной совместимости).
        self.post_preproc_enabled = tk.BooleanVar(value=False)
        self.post_preproc_normalize = tk.BooleanVar(value=True)
        self.post_preproc_normalize_mode = tk.StringVar(value="zscore")
        self.post_preproc_rank_ties = tk.StringVar(value="average")
        self.post_preproc_fill_missing = tk.BooleanVar(value=True)

        # Уменьшение размерности (очень большие N): опционально до анализа.
        self.dimred_enabled = tk.BooleanVar(value=False)
        self.dimred_method = tk.StringVar(value="variance")
        self.dimred_target = tk.IntVar(value=500)
        self.dimred_target_var = tk.DoubleVar(value=0.0)  # 0 => ignore, else (0, 1]
        self.dimred_spatial_bin = tk.IntVar(value=2)
        self.dimred_kmeans_batch = tk.IntVar(value=2048)
        self.dimred_seed = tk.IntVar(value=0)
        self.dimred_save_variants = tk.BooleanVar(value=False)
        self.dimred_variants = tk.StringVar(value="100,200,500")

        # Скан-параметры (window/lag/position)
        self.scan_window_pos = tk.BooleanVar(value=False)
        self.scan_window_size = tk.BooleanVar(value=False)
        self.scan_lag = tk.BooleanVar(value=False)
        self.scan_cube = tk.BooleanVar(value=False)

        self.window_sizes_text = tk.StringVar(value="64,128,192")
        self.window_min = tk.IntVar(value=64)
        self.window_max = tk.IntVar(value=192)
        self.window_step = tk.IntVar(value=64)
        self.window_size_default = tk.IntVar(value=128)
        self.window_stride = tk.IntVar(value=0)  # 0 => auto
        self.window_start_min = tk.IntVar(value=0)
        self.window_start_max = tk.IntVar(value=0)  # 0 => auto
        self.window_max_windows = tk.IntVar(value=60)

        self.lag_min = tk.IntVar(value=1)
        self.lag_max = tk.IntVar(value=3)
        self.lag_step = tk.IntVar(value=1)
        self.cube_combo_limit = tk.IntVar(value=9)
        self.cube_eval_limit = tk.IntVar(value=225)
        self.cube_matrix_mode = tk.StringVar(value="all")  # selected|all
        self.cube_matrix_limit = tk.IntVar(value=225)

        # Кубики по парам: какие пары строить (для 3–4 переменных)
        self.cube_pairs_all = tk.BooleanVar(value=False)
        self.cube_pairs_text = tk.StringVar(value="")

        # Cube gallery (матрицы в отчёте для выбранных точек куба)
        self.cube_gallery_mode = tk.StringVar(value="extremes")  # extremes|topbottom|quantiles
        self.cube_gallery_k = tk.IntVar(value=1)
        self.cube_gallery_limit = tk.IntVar(value=60)

        # Упрощение для больших N: режим расчёта по парам/соседям
        self.pair_mode = tk.StringVar(value="auto")  # auto|full|pairs|neighbors|random
        self.pair_auto_threshold = tk.IntVar(value=600)  # если N>=threshold, auto -> neighbors
        self.pairs_text = tk.StringVar(value="")
        self.max_pairs = tk.IntVar(value=50000)
        self.neighbor_kind = tk.StringVar(value="26")  # 6|26
        self.neighbor_radius = tk.IntVar(value=1)
        self.screen_metric = tk.StringVar(value="corr")  # reserved (future)
        self.topk_per_node = tk.IntVar(value=10)  # reserved (future)

        self._init_ui()

        # Строка статуса (этапы выполнения).
        self.status_var = tk.StringVar(value="Готово")
        self.status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        self.status_bar.pack(side="bottom", fill="x")
        self.stage_progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self.stage_progress.pack(side="bottom", fill="x")

    # --- безопасные геттеры Tk-переменных ---
    # В Tkinter IntVar/DoubleVar могут кидать TclError во время ввода ("", "0/", "0?9").
    # Это ломало авто-обновление JSON-плана и запуск.
    _FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:[\.,]\d*)?|[\.,]\d+)(?:[eE][-+]?\d+)?")
    _INT_RE = re.compile(r"[-+]?\d+")

    def _var_raw(self, var, default: object = "") -> object:
        """Вернуть "сырой" value переменной, не падая на TclError."""
        try:
            return var.get()
        except Exception:
            try:
                return self.getvar(getattr(var, "_name", ""))
            except Exception:
                return default

    def _as_str(self, var, default: str = "") -> str:
        v = self._var_raw(var, default)
        try:
            return str(v)
        except Exception:
            return default

    def _as_bool(self, var, default: bool = False) -> bool:
        try:
            return bool(self._var_raw(var, default))
        except Exception:
            return bool(default)

    def _as_float(self, var, default: float = 0.0) -> float:
        v = self._var_raw(var, "")
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return float(default)
        s = s.replace(",", ".")
        try:
            return float(s)
        except Exception:
            m = self._FLOAT_RE.search(s)
            if not m:
                return float(default)
            try:
                return float(m.group(0).replace(",", "."))
            except Exception:
                return float(default)

    def _as_int(self, var, default: int = 0) -> int:
        v = self._var_raw(var, "")
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, int):
            return int(v)
        if isinstance(v, float):
            return int(v)
        s = str(v).strip()
        if not s:
            return int(default)
        # допускаем "12.0" и мусорные суффиксы
        m = self._FLOAT_RE.search(s) or self._INT_RE.search(s)
        if not m:
            return int(default)
        try:
            return int(float(m.group(0).replace(",", ".")))
        except Exception:
            return int(default)


    def _set_stage(self, stage: str, progress=None) -> None:
        """Обновляет строку статуса и прогресс-бар безопасно для Tk main-loop."""
        def _apply() -> None:
            try:
                self.status_var.set(stage)
                if progress is None:
                    self.stage_progress.configure(mode="indeterminate")
                    self.stage_progress.start(10)
                else:
                    self.stage_progress.stop()
                    self.stage_progress.configure(mode="determinate")
                    pct = int(max(0.0, min(1.0, float(progress))) * 100)
                    self.stage_progress["value"] = pct
                self.update_idletasks()
            except Exception:
                pass

        try:
            self.after(0, _apply)
        except Exception:
            _apply()

    def _init_ui(self) -> None:
        """Инициализирует структуру вкладок и общую панель настроек."""
        tab_control = ttk.Notebook(self)

        self.tab_single = ttk.Frame(tab_control)
        self.tab_batch = ttk.Frame(tab_control)
        self.tab_gen = ttk.Frame(tab_control)
        self.tab_settings = ttk.Frame(tab_control)

        tab_control.add(self.tab_single, text="Один файл")
        tab_control.add(self.tab_batch, text="Пакетная обработка (Папка)")
        tab_control.add(self.tab_gen, text="Генератор тестов")
        tab_control.add(self.tab_settings, text="Настройки")

        tab_control.pack(expand=1, fill="both")

        self._build_single_tab(self.tab_single)
        self._build_batch_tab(self.tab_batch)
        self._build_gen_tab(self.tab_gen)
        self._build_settings_panel(self.tab_settings)

    def _make_scrollable(self, parent: ttk.Frame) -> ttk.Frame:
        """Создаёт прокручиваемый контейнер и возвращает внутренний Frame."""
        outer = ttk.Frame(parent)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        hbar = ttk.Scrollbar(outer, orient="horizontal", command=canvas.xview)
        canvas.configure(yscrollcommand=vbar.set)
        canvas.configure(xscrollcommand=hbar.set)

        vbar.pack(side="right", fill="y")
        hbar.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            # Не зажимаем внутренний фрейм по ширине: если он шире окна,
            # горизонтальная прокрутка остаётся доступной.
            req = inner.winfo_reqwidth()
            canvas.itemconfig(win_id, width=max(event.width, req))

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Горизонтальная прокрутка: Shift + колесо мыши.
        def _on_shift_mousewheel(event):
            delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
            if delta:
                canvas.xview_scroll(delta, "units")

        canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
        return inner

    def _build_single_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Анализ одного файла", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.file_path = tk.StringVar()

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Entry(row, textvariable=self.file_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать файл...", command=self._browse_file).pack(side="left", padx=5)

        ttk.Button(frame, text="Запустить анализ (HTML + Excel)", command=self._run_single).pack(pady=10)

    def _build_batch_tab(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Обработка всей папки", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.folder_path = tk.StringVar()

        row = ttk.Frame(frame)
        row.pack(fill="x", pady=5)
        ttk.Entry(row, textvariable=self.folder_path).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Выбрать папку...", command=self._browse_folder).pack(side="left", padx=5)

        self.batch_progress = ttk.Progressbar(frame, orient="horizontal", mode="determinate")
        self.batch_progress.pack(fill="x", pady=10)

        ttk.Button(frame, text="Обработать все файлы в папке", command=self._run_batch).pack(pady=5)
        ttk.Label(
            frame,
            text="* Результаты будут сохранены в подпапке 'time_series_analysis' внутри выбранной папки",
            foreground="gray",
        ).pack()

    def _build_gen_tab(self, parent: ttk.Frame) -> None:
        """Строит вкладку генерации синтетических данных по пресетам."""
        frame = ttk.LabelFrame(parent, text="Создание синтетических данных", padding=10)
        frame.pack(fill="x", padx=10, pady=10)

        self.gen_samples = tk.IntVar(value=500)
        self.gen_coupling = tk.DoubleVar(value=0.8)
        self.gen_noise = tk.DoubleVar(value=0.2)
        self.gen_ar_phi = tk.DoubleVar(value=0.7)
        self.gen_season_period = tk.IntVar(value=50)
        self.gen_nvars = tk.IntVar(value=3)
        self.gen_preset = tk.StringVar(value="Система: X→Y + шум + сезон")

        r1 = ttk.Frame(frame)
        r1.pack(fill="x", pady=4)
        ttk.Label(r1, text="Количество точек:").pack(side="left")
        ttk.Entry(r1, textvariable=self.gen_samples, width=10).pack(side="left", padx=5)

        r2 = ttk.Frame(frame)
        r2.pack(fill="x", pady=4)
        ttk.Label(r2, text="Пресет:").pack(side="left")
        ttk.Combobox(
            r2,
            textvariable=self.gen_preset,
            values=[
                "Система: X→Y + шум + сезон",
                "Случайные блуждания",
                "Независимые AR(1)",
                "Цепочка 4D: X1→X2→X3→X4",
            ],
            state="readonly",
            width=32,
        ).pack(side="left", padx=5)

        r3 = ttk.Frame(frame)
        r3.pack(fill="x", pady=4)
        ttk.Label(r3, text="Число переменных (для RW/AR1):").pack(side="left")
        ttk.Spinbox(r3, from_=2, to=8, textvariable=self.gen_nvars, width=5).pack(side="left", padx=5)

        r4 = ttk.Frame(frame)
        r4.pack(fill="x", pady=4)
        ttk.Label(r4, text="Сила связи (coupling):").pack(side="left")
        ttk.Entry(r4, textvariable=self.gen_coupling, width=10).pack(side="left", padx=5)
        ttk.Label(r4, text="Шум (sigma):").pack(side="left", padx=(15, 0))
        ttk.Entry(r4, textvariable=self.gen_noise, width=10).pack(side="left", padx=5)
        ttk.Label(r4, text="AR(1) phi:").pack(side="left", padx=(15, 0))
        ttk.Entry(r4, textvariable=self.gen_ar_phi, width=10).pack(side="left", padx=5)
        ttk.Label(r4, text="Период сезона:").pack(side="left", padx=(15, 0))
        ttk.Entry(r4, textvariable=self.gen_season_period, width=10).pack(side="left", padx=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill="x", pady=10)
        ttk.Button(btn_frame, text="Сгенерировать и сохранить...", command=self._run_gen).pack(side="left", padx=5)
        ttk.Label(
            frame,
            text="Примечание: после сохранения можно сразу открыть файл во вкладке «Один файл».",
            foreground="gray",
        ).pack(anchor="w")

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        inner = self._make_scrollable(parent)
        frame = ttk.LabelFrame(inner, text="Настройки методов анализа", padding=10)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        p_frame = ttk.Frame(frame)
        p_frame.pack(fill="x")
        ttk.Label(p_frame, text="Макс. лаг:").pack(side="left")
        ttk.Spinbox(p_frame, from_=1, to=50, textvariable=self.max_lag, width=5).pack(side="left", padx=5)

        ttk.Label(p_frame, text="Порог графа:").pack(side="left", padx=(15, 0))
        ttk.Entry(p_frame, textvariable=self.graph_threshold, width=5).pack(side="left", padx=5)

        ttk.Checkbutton(p_frame, text="Авто-дифференцирование", variable=self.auto_diff).pack(side="left", padx=15)

        ttk.Label(p_frame, text="Коррекция P-value:").pack(side="left", padx=(15, 5))
        ttk.Combobox(
            p_frame,
            textvariable=self.p_correction,
            values=["none", "fdr_bh"],
            state="readonly",
            width=10,
        ).pack(side="left")

        ttk.Label(p_frame, text="Вывод:").pack(side="left", padx=(15, 5))
        ttk.Combobox(
            p_frame,
            textvariable=self.output_mode,
            values=["both", "html", "excel"],
            state="readonly",
            width=8,
        ).pack(side="left")

        ttk.Checkbutton(p_frame, text="Первичный анализ", variable=self.include_diagnostics).pack(side="left", padx=10)
        ttk.Checkbutton(p_frame, text="FFT графики", variable=self.include_fft_plots).pack(side="left", padx=6)
        ttk.Checkbutton(p_frame, text="Сканы", variable=self.include_scans).pack(side="left", padx=6)
        ttk.Checkbutton(p_frame, text="Таблицы матриц", variable=self.include_matrix_tables).pack(side="left", padx=6)

        pre = ttk.LabelFrame(frame, text="Предобработка (до анализа)", padding=6)
        pre.pack(fill="x", pady=6)
        pr0 = ttk.Frame(pre)
        pr0.pack(fill="x", pady=2)
        ttk.Checkbutton(pr0, text="Включить предобработку", variable=self.preproc_enabled).pack(side="left")
        ttk.Label(pr0, text="Когда:").pack(side="left", padx=(20, 5))
        ttk.Combobox(pr0, textvariable=self.preprocess_stage, values=["pre", "post", "both"], state="readonly", width=6).pack(side="left")
        ttk.Checkbutton(pr0, text="Убирать выбросы", variable=self.preproc_remove_outliers).pack(side="left", padx=10)
        ttk.Checkbutton(pr0, text="Убирать AR(p)", variable=self.preproc_remove_ar1).pack(side="left", padx=10)
        ttk.Label(pr0, text="p:").pack(side="left", padx=(5, 2))
        ttk.Spinbox(pr0, from_=1, to=50, textvariable=self.preproc_ar_order, width=4).pack(side="left")
        ttk.Checkbutton(pr0, text="Убирать сезонность (STL)", variable=self.preproc_remove_seasonality).pack(side="left", padx=10)
        ttk.Label(pr0, text="Период сезонности (0=авто):").pack(side="left", padx=(10, 5))
        ttk.Entry(pr0, textvariable=self.preproc_season_period, width=6).pack(side="left")
        pr1 = ttk.Frame(pre)
        pr1.pack(fill="x", pady=2)
        ttk.Checkbutton(pr1, text="Лог-преобразование (+)", variable=self.preproc_log_transform).pack(side="left")
        ttk.Checkbutton(pr1, text="Нормализация", variable=self.preproc_normalize).pack(side="left", padx=10)
        ttk.Label(pr1, text="режим:").pack(side="left", padx=(0, 5))
        ttk.Combobox(
            pr1,
            textvariable=self.preproc_normalize_mode,
            values=["zscore", "minmax", "robust_z", "rank", "rank_dense", "rank_pct"],
            state="readonly",
            width=10,
        ).pack(side="left")
        ttk.Label(pr1, text="ties:").pack(side="left", padx=(10,5))
        ttk.Combobox(pr1, textvariable=self.preproc_rank_ties, values=["average","min","max","dense","first"], state="readonly", width=8).pack(side="left")
        ttk.Checkbutton(pr1, text="Заполнять пропуски", variable=self.preproc_fill_missing).pack(side="left", padx=10)

        pr2 = ttk.Frame(pre)
        pr2.pack(fill="x", pady=2)
        ttk.Label(pr2, text="Выбросы:").pack(side="left")
        ttk.Combobox(pr2, textvariable=self.preproc_outlier_rule, values=["robust_z","zscore","iqr","percentile","hampel","jump"], state="readonly", width=10).pack(side="left", padx=(5,10))
        ttk.Label(pr2, text="action:").pack(side="left")
        ttk.Combobox(pr2, textvariable=self.preproc_outlier_action, values=["mask","clip","median","local_median"], state="readonly", width=12).pack(side="left", padx=(5,10))
        ttk.Label(pr2, text="z:").pack(side="left")
        ttk.Entry(pr2, textvariable=self.preproc_outlier_z, width=6).pack(side="left", padx=(5,10))
        ttk.Label(pr2, text="IQR k:").pack(side="left")
        ttk.Entry(pr2, textvariable=self.preproc_outlier_k, width=6).pack(side="left", padx=(5,10))
        ttk.Label(pr2, text="pct lo/hi:").pack(side="left")
        ttk.Entry(pr2, textvariable=self.preproc_outlier_p_low, width=5).pack(side="left", padx=(5,2))
        ttk.Entry(pr2, textvariable=self.preproc_outlier_p_high, width=5).pack(side="left", padx=(2,10))
        ttk.Label(pr2, text="Hampel win:").pack(side="left")
        ttk.Entry(pr2, textvariable=self.preproc_outlier_hampel_window, width=5).pack(side="left", padx=(5,10))
        ttk.Label(pr2, text="jump thr(0=auto):").pack(side="left")
        ttk.Entry(pr2, textvariable=self.preproc_outlier_jump_thr, width=7).pack(side="left", padx=(5,10))
        ttk.Label(pr2, text="local win:").pack(side="left")
        ttk.Entry(pr2, textvariable=self.preproc_outlier_local_median_window, width=5).pack(side="left", padx=(5,10))

        dr = ttk.LabelFrame(frame, text="Уменьшение размерности (до анализа, опционально)", padding=6)
        dr.pack(fill="x", pady=6)

        d0 = ttk.Frame(dr)
        d0.pack(fill="x", pady=2)
        ttk.Checkbutton(d0, text="Включить", variable=self.dimred_enabled).pack(side="left")
        ttk.Label(d0, text="Метод:").pack(side="left", padx=(10, 5))
        ttk.Combobox(
            d0,
            textvariable=self.dimred_method,
            values=["variance", "kmeans", "spatial", "random", "pca"],
            state="readonly",
            width=10,
        ).pack(side="left")
        ttk.Label(d0, text="Цель N (сколько рядов оставить/получить):").pack(side="left", padx=(10, 5))
        ttk.Entry(d0, textvariable=self.dimred_target, width=8).pack(side="left")
        ttk.Label(d0, text="или доля объясн. дисперсии (0..1):").pack(side="left", padx=(10, 5))
        ttk.Entry(d0, textvariable=self.dimred_target_var, width=8).pack(side="left")
        ttk.Label(d0, text="Seed:").pack(side="left", padx=(10, 5))
        ttk.Entry(d0, textvariable=self.dimred_seed, width=6).pack(side="left")

        post = ttk.LabelFrame(frame, text="Пост-предобработка (после dimred)", padding=6)
        post.pack(fill="x", pady=6)
        post0 = ttk.Frame(post)
        post0.pack(fill="x", pady=2)
        ttk.Checkbutton(post0, text="Включить", variable=self.post_preproc_enabled).pack(side="left")
        ttk.Checkbutton(post0, text="Нормализация", variable=self.post_preproc_normalize).pack(side="left", padx=10)
        ttk.Label(post0, text="режим:").pack(side="left", padx=(0, 5))
        ttk.Combobox(
            post0,
            textvariable=self.post_preproc_normalize_mode,
            values=["zscore", "minmax", "robust_z", "rank", "rank_dense", "rank_pct"],
            state="readonly",
            width=10,
        ).pack(side="left")
        ttk.Label(post0, text="ties:").pack(side="left", padx=(10, 5))
        ttk.Combobox(post0, textvariable=self.post_preproc_rank_ties, values=["average", "min", "max", "dense", "first"], state="readonly", width=8).pack(side="left")
        ttk.Checkbutton(post0, text="Заполнять пропуски", variable=self.post_preproc_fill_missing).pack(side="left", padx=10)

        d1 = ttk.Frame(dr)
        d1.pack(fill="x", pady=2)
        ttk.Label(d1, text="Параметры метода:").pack(side="left")
        ttk.Label(d1, text="kmeans batch:").pack(side="left", padx=(10, 5))
        ttk.Entry(d1, textvariable=self.dimred_kmeans_batch, width=8).pack(side="left")
        ttk.Label(d1, text="spatial bin (шаг по xyz):").pack(side="left", padx=(10, 5))
        ttk.Entry(d1, textvariable=self.dimred_spatial_bin, width=6).pack(side="left")
        ttk.Checkbutton(d1, text="Сохранить несколько вариантов", variable=self.dimred_save_variants).pack(side="left", padx=(15, 0))
        ttk.Label(d1, text="Варианты N (через запятую):").pack(side="left", padx=(10, 5))
        ttk.Entry(d1, textvariable=self.dimred_variants, width=18).pack(side="left")

        scans = ttk.LabelFrame(frame, text="Сканы window/lag/position (опционально)", padding=6)
        scans.pack(fill="x", pady=6)

        r0 = ttk.Frame(scans)
        r0.pack(fill="x", pady=2)
        ttk.Checkbutton(r0, text="положение окна", variable=self.scan_window_pos).pack(side="left")
        ttk.Checkbutton(r0, text="размер окна", variable=self.scan_window_size).pack(side="left", padx=10)
        ttk.Checkbutton(r0, text="lag", variable=self.scan_lag).pack(side="left", padx=10)
        ttk.Checkbutton(r0, text="куб (3D)", variable=self.scan_cube).pack(side="left", padx=10)

        r1 = ttk.Frame(scans)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Размеры окон:").pack(side="left")
        ttk.Entry(r1, textvariable=self.window_sizes_text, width=24).pack(side="left", padx=5)
        ttk.Label(r1, text="Окно по умолч.:").pack(side="left", padx=(15, 0))
        ttk.Entry(r1, textvariable=self.window_size_default, width=6).pack(side="left", padx=5)
        ttk.Label(r1, text="Шаг (stride):").pack(side="left", padx=(15, 0))
        ttk.Entry(r1, textvariable=self.window_stride, width=6).pack(side="left", padx=5)
        ttk.Label(r1, text="Макс. окон:").pack(side="left", padx=(15, 0))
        ttk.Entry(r1, textvariable=self.window_max_windows, width=6).pack(side="left", padx=5)

        r1b = ttk.Frame(scans)
        r1b.pack(fill="x", pady=2)
        ttk.Label(r1b, text="Окно мин/макс/шаг:").pack(side="left")
        ttk.Entry(r1b, textvariable=self.window_min, width=6).pack(side="left", padx=2)
        ttk.Entry(r1b, textvariable=self.window_max, width=6).pack(side="left", padx=2)
        ttk.Entry(r1b, textvariable=self.window_step, width=6).pack(side="left", padx=2)
        ttk.Label(r1b, text="Лимит точек куба:").pack(side="left", padx=(10, 0))
        ttk.Entry(r1b, textvariable=self.cube_eval_limit, width=8).pack(side="left", padx=5)

        r2 = ttk.Frame(scans)
        r2.pack(fill="x", pady=2)
        ttk.Label(r2, text="start_min:").pack(side="left")
        ttk.Entry(r2, textvariable=self.window_start_min, width=8).pack(side="left", padx=5)
        ttk.Label(r2, text="start_max (0=auto):").pack(side="left", padx=(10, 0))
        ttk.Entry(r2, textvariable=self.window_start_max, width=8).pack(side="left", padx=5)
        ttk.Label(r2, text="Лаг мин/макс/шаг:").pack(side="left", padx=(10, 0))
        ttk.Entry(r2, textvariable=self.lag_min, width=4).pack(side="left", padx=2)
        ttk.Entry(r2, textvariable=self.lag_max, width=4).pack(side="left", padx=2)
        ttk.Entry(r2, textvariable=self.lag_step, width=4).pack(side="left", padx=2)
        ttk.Label(r2, text="Комбо (w×lag):").pack(side="left", padx=(10, 0))
        ttk.Entry(r2, textvariable=self.cube_combo_limit, width=6).pack(side="left", padx=5)
        ttk.Label(r2, text="Матрицы для куба:").pack(side="left", padx=(10, 0))
        ttk.Combobox(r2, textvariable=self.cube_matrix_mode, values=["selected", "all"], state="readonly", width=9).pack(side="left", padx=5)
        ttk.Label(r2, text="Лимит матриц:").pack(side="left", padx=(10, 0))
        ttk.Entry(r2, textvariable=self.cube_matrix_limit, width=8).pack(side="left", padx=5)

        r2b = ttk.Frame(scans)
        r2b.pack(fill="x", pady=2)
        ttk.Checkbutton(r2b, text="Кубики по всем парам (для 3–4 переменных)", variable=self.cube_pairs_all).pack(side="left")
        ttk.Label(r2b, text="Пары для куба (если не все):").pack(side="left", padx=(15, 5))
        ttk.Entry(r2b, textvariable=self.cube_pairs_text, width=40).pack(side="left", padx=5)
        ttk.Label(r2b, text="пример: X-Y; X-Z; Y-Z").pack(side="left", padx=(10, 0))

        r3 = ttk.Frame(scans)
        r3.pack(fill="x", pady=2)
        ttk.Label(r3, text="Галерея куба:").pack(side="left")
        ttk.Combobox(
            r3,
            textvariable=self.cube_gallery_mode,
            values=["extremes", "topbottom", "quantiles"],
            state="readonly",
            width=12,
        ).pack(side="left", padx=5)
        ttk.Label(r3, text="k:").pack(side="left", padx=(10, 0))
        ttk.Entry(r3, textvariable=self.cube_gallery_k, width=4).pack(side="left", padx=5)
        ttk.Label(r3, text="limit:").pack(side="left", padx=(10, 0))
        ttk.Entry(r3, textvariable=self.cube_gallery_limit, width=6).pack(side="left", padx=5)

        # --- Методы анализа (обязательно) ---
        # Вынесены выше JSON-опций, чтобы пользователь сначала выбрал *что*
        # считать, а уже затем при необходимости настраивал тонкие параметры.
        m_frame = ttk.LabelFrame(frame, text="Методы анализа (выбери, что считать)", padding=6)
        m_frame.pack(fill="both", expand=False, pady=6)

        # Кнопки быстрого выбора: помогают быстро включать/сбрасывать наборы
        # без ручного прокликивания длинного списка.
        m_btns = ttk.Frame(m_frame)
        m_btns.pack(fill="x", pady=(0, 4))

        def _set_all_methods(value: bool) -> None:
            """Массово установить состояние для всех чекбоксов методов."""
            for v in self.method_vars.values():
                v.set(value)

        def _set_default_methods() -> None:
            """Включить быстрый дефолт из самых дешевых и часто полезных методов."""
            default = {"correlation_full", "mutinf_full", "granger_full"}
            for name, v in self.method_vars.items():
                v.set(name in default)

        ttk.Button(m_btns, text="Выбрать все", command=lambda: _set_all_methods(True)).pack(side="left")
        ttk.Button(m_btns, text="Снять все", command=lambda: _set_all_methods(False)).pack(side="left", padx=6)
        ttk.Button(m_btns, text="Дефолт (быстро)", command=_set_default_methods).pack(side="left")

        # Список методов может быть длинным, поэтому используется скроллируемый
        # контейнер на Canvas, чтобы не растягивать окно GUI по высоте.
        canvas = tk.Canvas(m_frame, height=220)
        vsb = ttk.Scrollbar(m_frame, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)

        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        methods = sorted(method_mapping.keys())

        # Дефолт включения при первом открытии вкладки.
        default_on = {"correlation_full", "mutinf_full", "granger_full"}

        for i, m in enumerate(methods):
            var = tk.BooleanVar(value=(m in default_on))
            self.method_vars[m] = var

            col = i % 3
            row = i // 3

            label = m
            # Если pyinform недоступен, отмечаем TE-методы как потенциально
            # более медленные/требующие зависимости.
            if (not PYINFORM_AVAILABLE) and m.startswith("te_"):
                label += " (медленно/зависимость pyinform)"
            ttk.Checkbutton(inner, text=label, variable=var).grid(
                row=row, column=col, sticky="w", padx=10, pady=2
            )

        # План операций (JSON): в авто-режиме отражает то, что реально пойдёт
        # в run_selected_methods на основе текущих флагов GUI.
        adv = ttk.LabelFrame(frame, text="План операций (JSON): что именно будет сделано", padding=6)
        adv.pack(fill="both", expand=False, pady=6)

        top_row = ttk.Frame(adv)
        top_row.pack(fill="x")
        self.auto_plan_json = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_row,
            text="Авто-режим (обновлять при выборе опций)",
            variable=self.auto_plan_json,
            command=lambda: self._refresh_plan_json(force=True),
        ).pack(side="left")
        ttk.Label(top_row, text="(сними галку, если хочешь вручную дописать/переопределить)").pack(
            side="left", padx=8
        )

        self.method_options_text = tk.Text(adv, height=10, wrap="none")
        self.method_options_text.pack(fill="both", expand=True, pady=(6, 0))
        self.method_options_text.insert("1.0", "{}")

        xsb = ttk.Scrollbar(adv, orient="horizontal", command=self.method_options_text.xview)
        xsb.pack(fill="x", pady=(2, 0))
        self.method_options_text.configure(xscrollcommand=xsb.set)

        self._wire_plan_auto_refresh()
        self._refresh_plan_json(force=True)

    def _browse_file(self) -> None:
        f = filedialog.askopenfilename(filetypes=[("Data", "*.csv *.xlsx")])
        if f:
            self.file_path.set(f)

    def _browse_folder(self) -> None:
        d = filedialog.askdirectory()
        if d:
            self.folder_path.set(d)

    def _get_config(self) -> AnalysisConfig:
        return AnalysisConfig(
            max_lag=self._as_int(self.max_lag, 5),
            p_value_alpha=self._as_float(self.p_alpha, 0.05),
            graph_threshold=self._as_float(self.graph_threshold, 0.2),
            auto_difference=self._as_bool(self.auto_diff),
            pvalue_correction=self._as_str(self.p_correction, "none"),
        )

    def _get_selected_methods(self) -> list[str]:
        return [m for m, var in self.method_vars.items() if var.get()]

    def _run_tool(self, df: pd.DataFrame, out_dir: str, name_prefix: str) -> str | None:
        """Запуск движка для одного датафрейма."""
        cfg = self._get_config()
        methods = self._get_selected_methods()

        if not methods:
            messagebox.showwarning("Ошибка", "Выберите хотя бы один метод!")
            return None

        def _stage_cb(stage: str, progress, meta: dict):
            """Проксирует этапы движка в статус-бар GUI."""
            self._set_stage(stage, progress)

        self._set_stage("Старт анализа", 0.0)
        tool = BigMasterTool(df, config=cfg, stage_callback=_stage_cb)
        tool.data = df.fillna(df.mean(numeric_only=True))

        if cfg.auto_difference:
            from src.analysis import stats as s_stats

            for c in tool.data.columns:
                if pd.api.types.is_numeric_dtype(tool.data[c]):
                    _, p = s_stats.test_stationarity(tool.data[c])
                    if p is not None and p > 0.05:
                        tool.data[c] = tool.data[c].diff().fillna(0)

        # --- parse window sizes grid ---
        w_sizes = None
        txt = (self.window_sizes_text.get() or "").strip()
        if txt:
            try:
                w_sizes = [int(x.strip()) for x in txt.replace("[", "").replace("]", "").split(",") if x.strip()]
                w_sizes = [w for w in w_sizes if w >= 2]
            except Exception:
                w_sizes = None
        if not w_sizes:
            try:
                wmin = int(self.window_min.get())
                wmax = int(self.window_max.get())
                wstep = int(self.window_step.get())
                if wmin >= 2 and wmax >= wmin and wstep >= 1:
                    w_sizes = list(range(wmin, wmax + 1, wstep))
            except Exception:
                pass

        # --- per-method overrides ---
        # В авто-режиме JSON-поле — это план и не используется как overrides.
        method_options = {}
        if getattr(self, "auto_plan_json", None) is not None and bool(self.auto_plan_json.get()):
            method_options = {}
        else:
            try:
                raw = (self.method_options_text.get("1.0", "end") or "").strip()
                if raw:
                    method_options = json.loads(raw)
            except Exception:
                method_options = {}

        # --- cube pairs (для 3–4 переменных) ---
        cube_pairs = None
        try:
            if bool(self.scan_cube.get()):
                ncols = int(df.shape[1])
                if bool(self.cube_pairs_all.get()) and (3 <= ncols <= 4):
                    cube_pairs = [(i, j) for i in range(ncols) for j in range(i + 1, ncols)]
                else:
                    raw_pairs = (self.cube_pairs_text.get() or "").strip()
                    if raw_pairs:
                        parts: list[str] = []
                        for token in raw_pairs.replace("—", "-").split(";"):
                            for t in token.split(","):
                                tt = t.strip()
                                if tt:
                                    parts.append(tt)
                        cube_pairs = parts if parts else None
        except Exception:
            cube_pairs = None

        # --- scan params ---
        stride = self._as_int(self.window_stride, 0)
        stride = None if stride <= 0 else stride
        start_max = self._as_int(self.window_start_max, 0)
        start_max = None if start_max <= 0 else start_max

        # Обновляем план перед запуском: пользователю видно финальный набор параметров.
        self._refresh_plan_json(force=True)

        tool.run_selected_methods(
            methods,
            max_lag=int(cfg.max_lag),
            preprocess_stage=str(self._as_str(self.preprocess_stage)).strip().lower(),
            post_preprocess={
                "enabled": self._as_bool(self.post_preproc_enabled),
                "normalize": self._as_bool(self.post_preproc_normalize),
                "normalize_mode": str(self._as_str(self.post_preproc_normalize_mode)),
                "rank_ties": str(self._as_str(self.post_preproc_rank_ties)),
                "fill_missing": self._as_bool(self.post_preproc_fill_missing),
            },
            dimred_enabled=self._as_bool(self.dimred_enabled),
            dimred_method=str(self._as_str(self.dimred_method)),
            dimred_target=self._as_int(self.dimred_target, 500),
            dimred_target_var=(
                None if self._as_float(self.dimred_target_var, 0.0) <= 0 else self._as_float(self.dimred_target_var, 0.0)
            ),
            dimred_spatial_bin=self._as_int(self.dimred_spatial_bin, 2),
            dimred_kmeans_batch=self._as_int(self.dimred_kmeans_batch, 2048),
            dimred_seed=self._as_int(self.dimred_seed, 0),
            dimred_save_variants=self._as_bool(self.dimred_save_variants),
            dimred_variants=str(self._as_str(self.dimred_variants) or ""),
            window_sizes_grid=w_sizes,
            window_min=self._as_int(self.window_min, 64),
            window_max=self._as_int(self.window_max, 192),
            window_step=self._as_int(self.window_step, 64),
            window_size=self._as_int(self.window_size_default, 128),
            window_stride=stride,
            window_start_min=self._as_int(self.window_start_min, 0),
            window_start_max=start_max,
            window_max_windows=self._as_int(self.window_max_windows, 60),
            scan_window_pos=self._as_bool(self.scan_window_pos),
            scan_window_size=self._as_bool(self.scan_window_size),
            scan_lag=self._as_bool(self.scan_lag),
            scan_cube=self._as_bool(self.scan_cube),
            lag_min=self._as_int(self.lag_min, 1),
            lag_max=self._as_int(self.lag_max, 3),
            lag_step=self._as_int(self.lag_step, 1),
            cube_combo_limit=self._as_int(self.cube_combo_limit, 9),
            cube_eval_limit=self._as_int(self.cube_eval_limit, 225),
            cube_matrix_mode=str(self._as_str(self.cube_matrix_mode)),
            cube_matrix_limit=self._as_int(self.cube_matrix_limit, 225),
            cube_gallery_mode=str(self._as_str(self.cube_gallery_mode)),
            cube_gallery_k=self._as_int(self.cube_gallery_k, 1),
            cube_gallery_limit=self._as_int(self.cube_gallery_limit, 60),
            method_options=method_options,
            cube_pairs=cube_pairs,
            pair_mode=str(self._as_str(self.pair_mode)),
            pair_auto_threshold=self._as_int(self.pair_auto_threshold, 600),
            pairs_text=str(self._as_str(self.pairs_text) or ""),
            max_pairs=self._as_int(self.max_pairs, 50000),
            neighbor_kind=str(self._as_str(self.neighbor_kind)),
            neighbor_radius=self._as_int(self.neighbor_radius, 1),
            screen_metric=str(self._as_str(self.screen_metric)),
            topk_per_node=self._as_int(self.topk_per_node, 10),
        )

        self._set_stage("Экспорт результатов", 0.98)

        # Экспорт «как данные»: матрицы/графы/edge-list в out_dir/data.
        # Ошибки экспорта не должны ломать основной HTML/Excel пайплайн.
        try:
            tool.export_connectivity_bundle(
                out_dir=out_dir,
                name_prefix=name_prefix,
                dense_n_limit=2000,
                topk_per_node=max(1, self._as_int(self.topk_per_node, 10)),
                # Для edge-list используем graph_threshold как минимальный |weight|.
                min_abs_weight=float(cfg.graph_threshold),
                include_scan_matrices=True,
            )
        except Exception as e:
            print(f"[WARN] export_connectivity_bundle failed: {e}")

        os.makedirs(out_dir, exist_ok=True)
        html_path = os.path.join(out_dir, f"{name_prefix}_report.html")
        excel_path = os.path.join(out_dir, f"{name_prefix}_full.xlsx")

        mode = (self.output_mode.get() or "both").lower()
        do_html = mode in ("both", "html")
        do_excel = mode in ("both", "excel")

        if do_html:
            self._set_stage("Формирование HTML отчёта", 0.99)
            tool.export_html_report(
                html_path,
                graph_threshold=cfg.graph_threshold,
                p_alpha=cfg.p_value_alpha,
                include_diagnostics=self._as_bool(self.include_diagnostics),
                include_fft_plots=self._as_bool(self.include_fft_plots),
                include_scans=self._as_bool(self.include_scans),
                include_matrix_tables=self._as_bool(self.include_matrix_tables),
            )
        if do_excel:
            self._set_stage("Формирование Excel отчёта", 0.99)
            tool.export_big_excel(excel_path, threshold=cfg.graph_threshold, p_value_alpha=cfg.p_value_alpha)
        self._set_stage("Готово", 1.0)
        return html_path if do_html else None

    def _run_single(self) -> None:
        fp = self.file_path.get()
        if not os.path.exists(fp):
            messagebox.showerror("Error", "Файл не найден")
            return

        try:
            df = data_loader.load_or_generate(
                fp,
                preprocess=(self._as_bool(self.preproc_enabled) and str(self._as_str(self.preprocess_stage)).strip().lower() in ("pre", "both")),
                log_transform=self._as_bool(self.preproc_log_transform),
                remove_outliers=self._as_bool(self.preproc_remove_outliers),
                outlier_rule=str(self._as_str(self.preproc_outlier_rule)),
                outlier_action=str(self._as_str(self.preproc_outlier_action)),
                outlier_z=self._as_float(self.preproc_outlier_z, 5.0),
                outlier_k=self._as_float(self.preproc_outlier_k, 1.5),
                outlier_p_low=self._as_float(self.preproc_outlier_p_low, 0.5),
                outlier_p_high=self._as_float(self.preproc_outlier_p_high, 99.5),
                outlier_hampel_window=self._as_int(self.preproc_outlier_hampel_window, 7),
                outlier_jump_thr=(None if self._as_float(self.preproc_outlier_jump_thr, 0.0) == 0.0 else self._as_float(self.preproc_outlier_jump_thr, 0.0)),
                outlier_local_median_window=self._as_int(self.preproc_outlier_local_median_window, 7),
                normalize=self._as_bool(self.preproc_normalize),
                normalize_mode=str(self._as_str(self.preproc_normalize_mode)),
                rank_ties=str(self._as_str(self.preproc_rank_ties)),
                fill_missing=self._as_bool(self.preproc_fill_missing),
                remove_ar1=self._as_bool(self.preproc_remove_ar1),
                remove_ar_order=self._as_int(self.preproc_ar_order, 1) or 1,
                remove_seasonality=self._as_bool(self.preproc_remove_seasonality),
                season_period=(self._as_int(self.preproc_season_period, 0) if self._as_int(self.preproc_season_period, 0) > 0 else None),
            )
            # Аккуратная папка результатов рядом с исходным файлом.
            out_dir = os.path.join(os.path.dirname(fp), "time_series_analysis", Path(fp).stem)
            name = Path(fp).stem
            os.makedirs(out_dir, exist_ok=True)

            report = self._run_tool(df, out_dir, name)
            if report and messagebox.askyesno("Готово", "Открыть отчет?"):
                webbrowser.open(report)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(traceback.format_exc())

    def _run_batch(self) -> None:
        folder = self.folder_path.get()
        if not os.path.exists(folder):
            messagebox.showerror("Error", "Папка не найдена")
            return

        files = [f for f in os.listdir(folder) if f.lower().endswith((".csv", ".xlsx"))]
        if not files:
            messagebox.showinfo("Info", "В папке нет файлов данных.")
            return

        out_root = os.path.join(folder, "time_series_analysis")
        os.makedirs(out_root, exist_ok=True)

        self.batch_progress["maximum"] = len(files)
        self.batch_progress["value"] = 0

        success_count = 0

        for f in files:
            fp = os.path.join(folder, f)
            try:
                df = data_loader.load_or_generate(
                    fp,
                    preprocess=(self._as_bool(self.preproc_enabled) and str(self._as_str(self.preprocess_stage)).strip().lower() in ("pre", "both")),
                    log_transform=self._as_bool(self.preproc_log_transform),
                    remove_outliers=self._as_bool(self.preproc_remove_outliers),
                    outlier_rule=str(self._as_str(self.preproc_outlier_rule)),
                    outlier_action=str(self._as_str(self.preproc_outlier_action)),
                    outlier_z=self._as_float(self.preproc_outlier_z, 5.0),
                    outlier_k=self._as_float(self.preproc_outlier_k, 1.5),
                    outlier_p_low=self._as_float(self.preproc_outlier_p_low, 0.5),
                    outlier_p_high=self._as_float(self.preproc_outlier_p_high, 99.5),
                    outlier_hampel_window=self._as_int(self.preproc_outlier_hampel_window, 7),
                    outlier_jump_thr=(None if self._as_float(self.preproc_outlier_jump_thr, 0.0) == 0.0 else self._as_float(self.preproc_outlier_jump_thr, 0.0)),
                    outlier_local_median_window=self._as_int(self.preproc_outlier_local_median_window, 7),
                    normalize=self._as_bool(self.preproc_normalize),
                    normalize_mode=str(self._as_str(self.preproc_normalize_mode)),
                    rank_ties=str(self._as_str(self.preproc_rank_ties)),
                    fill_missing=self._as_bool(self.preproc_fill_missing),
                    remove_ar1=self._as_bool(self.preproc_remove_ar1),
                    remove_ar_order=self._as_int(self.preproc_ar_order, 1) or 1,
                    remove_seasonality=self._as_bool(self.preproc_remove_seasonality),
                    season_period=(self._as_int(self.preproc_season_period, 0) if self._as_int(self.preproc_season_period, 0) > 0 else None),
                )
                name = Path(f).stem
                file_out_dir = os.path.join(out_root, name)
                self._run_tool(df, file_out_dir, "report")
                success_count += 1
            except Exception as e:
                print(f"Failed to process {f}: {e}")

            self.batch_progress["value"] += 1
            self.update_idletasks()

        messagebox.showinfo("Готово", f"Обработано {success_count} из {len(files)} файлов.\nРезультаты в: {out_root}")

    def _run_gen(self) -> None:
        """Диспетчер генератора пресетов для синтетических наборов данных."""
        try:
            preset = (self.gen_preset.get() or "").strip()
            n = self._as_int(self.gen_samples, 600)
            coupling = self._as_float(self.gen_coupling, 0.7)
            noise = self._as_float(self.gen_noise, 0.2)
            phi = self._as_float(self.gen_ar_phi, 0.8)
            per = self._as_int(self.gen_season_period, 50)
            nvars = self._as_int(self.gen_nvars, 3)

            if preset.startswith("Система"):
                df = generator.generate_coupled_system(
                    n_samples=n,
                    coupling_strength=coupling,
                    noise_level=noise,
                )
                name = "system_xy_noise_season"
            elif preset.startswith("Случайные"):
                df = generator.generate_random_walks(n_vars=nvars, n_samples=n)
                name = f"random_walks_{nvars}v"
            elif preset.startswith("Независимые"):
                df = generator.generate_independent_ar1(
                    n_vars=nvars,
                    n_samples=n,
                    phi=phi,
                    noise_level=noise,
                )
                name = f"ar1_{nvars}v"
            elif preset.startswith("Цепочка 4D"):
                df = generator.generate_chain_system_4d(
                    n_samples=n,
                    coupling_strength=coupling,
                    noise_level=noise,
                    season_period=per,
                )
                name = "chain_4d"
            else:
                messagebox.showerror("Ошибка", f"Неизвестный пресет: {preset}")
                return

            self._save_and_open_gen(df, name)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def _run_gen_system(self) -> None:
        """Совместимость со старой кнопкой генерации системы."""
        self.gen_preset.set("Система: X→Y + шум + сезон")
        self._run_gen()

    def _run_gen_rw(self) -> None:
        """Совместимость со старой кнопкой генерации блужданий."""
        self.gen_preset.set("Случайные блуждания")
        self._run_gen()

    def _save_and_open_gen(self, df: pd.DataFrame, name_base: str) -> None:
        f = filedialog.asksaveasfilename(initialfile=f"{name_base}.csv", filetypes=[("CSV", "*.csv")])
        if f:
            df.to_csv(f, index=False)
            if messagebox.askyesno("Генерация", "Файл сохранен. Открыть его для анализа в первой вкладке?"):
                self.file_path.set(f)

    # -------------------- План (JSON) --------------------
    def _wire_plan_auto_refresh(self) -> None:
        """Подписывается на ключевые переменные GUI, влияющие на план расчёта."""
        vars_to_watch = [
            # Базовые параметры/вывод.
            self.max_lag,
            self.graph_threshold,
            self.auto_diff,
            self.p_correction,
            self.output_mode,
            self.include_diagnostics,
            self.include_fft_plots,
            self.include_scans,
            self.include_matrix_tables,
            # Предобработка.
            self.preproc_enabled,
            self.preproc_remove_outliers,
            self.preproc_remove_ar1,
            self.preproc_ar_order,
            self.preproc_remove_seasonality,
            self.preproc_season_period,
            self.preproc_log_transform,
            self.preproc_normalize,
            self.preproc_normalize_mode,
            self.preproc_rank_ties,
            self.preproc_outlier_rule,
            self.preproc_outlier_action,
            self.preproc_outlier_z,
            self.preproc_outlier_k,
            self.preproc_outlier_p_low,
            self.preproc_outlier_p_high,
            self.preproc_outlier_hampel_window,
            self.preproc_outlier_jump_thr,
            self.preproc_outlier_local_median_window,
            self.preproc_fill_missing,
            self.preprocess_stage,
            self.post_preproc_enabled,
            self.post_preproc_normalize,
            self.post_preproc_normalize_mode,
            self.post_preproc_rank_ties,
            self.post_preproc_fill_missing,
            # Сканирование.
            self.scan_window_pos,
            self.scan_window_size,
            self.scan_lag,
            self.scan_cube,
            self.window_sizes_text,
            self.window_min,
            self.window_max,
            self.window_step,
            self.window_size_default,
            self.window_stride,
            self.window_start_min,
            self.window_start_max,
            self.window_max_windows,
            self.lag_min,
            self.lag_max,
            self.lag_step,
            self.cube_combo_limit,
            self.cube_eval_limit,
            self.cube_matrix_mode,
            self.cube_matrix_limit,
            self.cube_pairs_all,
            self.cube_pairs_text,
            self.cube_gallery_mode,
            self.cube_gallery_k,
            self.cube_gallery_limit,
            # Работа с большими N.
            self.pair_mode,
            self.pair_auto_threshold,
            self.pairs_text,
            self.max_pairs,
            self.neighbor_kind,
            self.neighbor_radius,
            self.screen_metric,
            self.topk_per_node,
            # Dimred.
            self.dimred_enabled,
            self.dimred_method,
            self.dimred_target,
            self.dimred_target_var,
            self.dimred_spatial_bin,
            self.dimred_kmeans_batch,
            self.dimred_seed,
            self.dimred_save_variants,
            self.dimred_variants,
        ]
        vars_to_watch.extend(self.method_vars.values())

        def _cb(*_):
            self._refresh_plan_json()

        for var in vars_to_watch:
            try:
                var.trace_add("write", _cb)
            except Exception:
                pass

    def _refresh_plan_json(self, force: bool = False) -> None:
        """Обновляет JSON-план. При auto-режиме использует debounce для отзывчивости UI."""
        if getattr(self, "auto_plan_json", None) is None:
            return
        if not force and not bool(self.auto_plan_json.get()):
            return

        if not force:
            if getattr(self, "_plan_refresh_after", None):
                try:
                    self.after_cancel(self._plan_refresh_after)
                except Exception:
                    pass
            self._plan_refresh_after = self.after(150, lambda: self._refresh_plan_json(force=True))
            return

        try:
            plan = self._build_plan_dict()
            txt = json.dumps(plan, ensure_ascii=False, indent=2)
        except Exception as e:
            # Ни один ввод в поле не должен валить Tk callback.
            txt = json.dumps({"error": f"plan build failed: {e}"}, ensure_ascii=False, indent=2)
        try:
            self.method_options_text.delete("1.0", "end")
            self.method_options_text.insert("1.0", txt)
        except Exception:
            pass

    def _build_plan_dict(self) -> dict:
        """Формирует структуру плана расчёта для отображения в GUI."""
        methods = self._get_selected_methods()
        preproc = {
            "enabled": self._as_bool(self.preproc_enabled),
            "outliers": {
                "enabled": self._as_bool(self.preproc_remove_outliers),
                "rule": self._as_str(self.preproc_outlier_rule),
                "action": self._as_str(self.preproc_outlier_action),
                "z": self._as_float(self.preproc_outlier_z, 5.0),
                "k": self._as_float(self.preproc_outlier_k, 1.5),
                "p_low": self._as_float(self.preproc_outlier_p_low, 0.5),
                "p_high": self._as_float(self.preproc_outlier_p_high, 99.5),
                "hampel_window": self._as_int(self.preproc_outlier_hampel_window, 7),
                "jump_thr": self._as_float(self.preproc_outlier_jump_thr, 0.0),
                "local_median_window": self._as_int(self.preproc_outlier_local_median_window, 7),
            },
            "remove_ar1": self._as_bool(self.preproc_remove_ar1),
            "remove_ar_order": self._as_int(self.preproc_ar_order, 1),
            "remove_seasonality": {
                "enabled": self._as_bool(self.preproc_remove_seasonality),
                "period": (
                    self._as_int(self.preproc_season_period, 0)
                    if self._as_int(self.preproc_season_period, 0) > 0
                    else "auto"
                ),
            },
            "log_transform": self._as_bool(self.preproc_log_transform),
            "normalize": {"enabled": self._as_bool(self.preproc_normalize), "mode": self._as_str(self.preproc_normalize_mode)},
            "fill_missing": self._as_bool(self.preproc_fill_missing),
            "rank_ties": self._as_str(self.preproc_rank_ties),
        }
        scans = {
            "window_pos": self._as_bool(self.scan_window_pos),
            "window_size": self._as_bool(self.scan_window_size),
            "lag": self._as_bool(self.scan_lag),
            "cube": self._as_bool(self.scan_cube),
            "window_sizes_text": self._as_str(self.window_sizes_text),
            "window_minmaxstep": [
                self._as_int(self.window_min, 64),
                self._as_int(self.window_max, 192),
                self._as_int(self.window_step, 64),
            ],
            "window_default": self._as_int(self.window_size_default, 128),
            "window_stride": self._as_int(self.window_stride, 0),
            "window_start_min": self._as_int(self.window_start_min, 0),
            "window_start_max": self._as_int(self.window_start_max, 0),
            "window_max_windows": self._as_int(self.window_max_windows, 60),
            "lag_minmaxstep": [
                self._as_int(self.lag_min, 1),
                self._as_int(self.lag_max, 3),
                self._as_int(self.lag_step, 1),
            ],
            "cube_combo_limit": self._as_int(self.cube_combo_limit, 9),
            "cube_eval_limit": self._as_int(self.cube_eval_limit, 225),
            "cube_matrix": {"mode": self._as_str(self.cube_matrix_mode), "limit": self._as_int(self.cube_matrix_limit, 225)},
            "cube_pairs_all": self._as_bool(self.cube_pairs_all),
            "cube_pairs": self._as_str(self.cube_pairs_text),
            "cube_gallery": {
                "mode": self._as_str(self.cube_gallery_mode),
                "k": self._as_int(self.cube_gallery_k, 1),
                "limit": self._as_int(self.cube_gallery_limit, 60),
            },
        }
        dimred = {
            "enabled": self._as_bool(self.dimred_enabled),
            "method": self._as_str(self.dimred_method),
            "target": self._as_int(self.dimred_target, 500),
            "target_var": self._as_float(self.dimred_target_var, 0.0),
            "spatial_bin": self._as_int(self.dimred_spatial_bin, 2),
            "kmeans_batch": self._as_int(self.dimred_kmeans_batch, 2048),
            "seed": self._as_int(self.dimred_seed, 0),
            "save_variants": self._as_bool(self.dimred_save_variants),
            "variants": self._as_str(self.dimred_variants),
        }
        common = {
            "max_lag": self._as_int(self.max_lag, 5),
            "graph_threshold": self._as_float(self.graph_threshold, 0.2),
            "auto_difference": self._as_bool(self.auto_diff),
            "pvalue_correction": self._as_str(self.p_correction),
            "output_mode": self._as_str(self.output_mode),
            "report_flags": {
                "diagnostics": self._as_bool(self.include_diagnostics),
                "fft_plots": self._as_bool(self.include_fft_plots),
                "scans": self._as_bool(self.include_scans),
                "matrix_tables": self._as_bool(self.include_matrix_tables),
            },
        }

        return {
            "selected_methods": methods,
            "common": common,
            "preprocessing": preproc,
            "preprocess_stage": self.preprocess_stage.get(),
            "post_preprocessing": {
                "enabled": self._as_bool(self.post_preproc_enabled),
                "normalize": self._as_bool(self.post_preproc_normalize),
                "normalize_mode": self._as_str(self.post_preproc_normalize_mode),
                "rank_ties": self._as_str(self.post_preproc_rank_ties),
                "fill_missing": self._as_bool(self.post_preproc_fill_missing),
            },
            "dim_reduction": dimred,
            "scans": scans,
            "pairing": {
                "pair_mode": self._as_str(self.pair_mode),
                "pair_auto_threshold": self._as_int(self.pair_auto_threshold, 600),
                "pairs_text": self._as_str(self.pairs_text),
                "max_pairs": self._as_int(self.max_pairs, 50000),
                "neighbor_kind": self._as_str(self.neighbor_kind),
                "neighbor_radius": self._as_int(self.neighbor_radius, 1),
                "screen_metric": self._as_str(self.screen_metric),
                "topk_per_node": self._as_int(self.topk_per_node, 10),
            },
            "method_overrides": "(ручной режим: выключи авто-режим и впиши JSON overrides)",
        }


if __name__ == "__main__":
    app = App()
    app.mainloop()
