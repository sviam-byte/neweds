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
        self.preproc_remove_seasonality = tk.BooleanVar(value=False)
        self.preproc_season_period = tk.IntVar(value=0)  # 0 => auto
        self.preproc_log_transform = tk.BooleanVar(value=False)
        self.preproc_normalize = tk.BooleanVar(value=True)
        self.preproc_fill_missing = tk.BooleanVar(value=True)

        # Уменьшение размерности (очень большие N): опционально до анализа.
        self.dimred_enabled = tk.BooleanVar(value=False)
        self.dimred_method = tk.StringVar(value="variance")
        self.dimred_target = tk.IntVar(value=500)
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
        canvas.configure(yscrollcommand=vbar.set)

        vbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = ttk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(win_id, width=event.width)

        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(event):
            delta = int(-1 * (event.delta / 120)) if getattr(event, "delta", 0) else 0
            if delta:
                canvas.yview_scroll(delta, "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)
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
        ttk.Checkbutton(pr0, text="Убирать выбросы", variable=self.preproc_remove_outliers).pack(side="left", padx=10)
        ttk.Checkbutton(pr0, text="Убирать AR(1)", variable=self.preproc_remove_ar1).pack(side="left", padx=10)
        ttk.Checkbutton(pr0, text="Убирать сезонность (STL)", variable=self.preproc_remove_seasonality).pack(side="left", padx=10)
        ttk.Label(pr0, text="Период сезонности (0=авто):").pack(side="left", padx=(10, 5))
        ttk.Entry(pr0, textvariable=self.preproc_season_period, width=6).pack(side="left")
        pr1 = ttk.Frame(pre)
        pr1.pack(fill="x", pady=2)
        ttk.Checkbutton(pr1, text="Лог-преобразование (+)", variable=self.preproc_log_transform).pack(side="left")
        ttk.Checkbutton(pr1, text="Нормализация (z-score)", variable=self.preproc_normalize).pack(side="left", padx=10)
        ttk.Checkbutton(pr1, text="Заполнять пропуски", variable=self.preproc_fill_missing).pack(side="left", padx=10)

        dr = ttk.LabelFrame(frame, text="Уменьшение размерности (до анализа, опционально)", padding=6)
        dr.pack(fill="x", pady=6)

        d0 = ttk.Frame(dr)
        d0.pack(fill="x", pady=2)
        ttk.Checkbutton(d0, text="Включить", variable=self.dimred_enabled).pack(side="left")
        ttk.Label(d0, text="Метод:").pack(side="left", padx=(10, 5))
        ttk.Combobox(
            d0,
            textvariable=self.dimred_method,
            values=["variance", "kmeans", "spatial", "random"],
            state="readonly",
            width=10,
        ).pack(side="left")
        ttk.Label(d0, text="Цель N (сколько рядов оставить/получить):").pack(side="left", padx=(10, 5))
        ttk.Entry(d0, textvariable=self.dimred_target, width=8).pack(side="left")
        ttk.Label(d0, text="Seed:").pack(side="left", padx=(10, 5))
        ttk.Entry(d0, textvariable=self.dimred_seed, width=6).pack(side="left")

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

        adv = ttk.LabelFrame(frame, text="Доп. опции по методам (JSON, необязательно)", padding=6)
        adv.pack(fill="both", expand=False, pady=6)
        self.method_options_text = tk.Text(adv, height=4)
        self.method_options_text.pack(fill="both", expand=True)
        self.method_options_text.insert(
            "1.0",
            '{\n  "correlation_full": {"scan_cube": true},\n  "granger_full": {"scan_lag": true, "lag_min": 1, "lag_max": 10}\n}',
        )

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
            max_lag=self.max_lag.get(),
            p_value_alpha=self.p_alpha.get(),
            graph_threshold=self.graph_threshold.get(),
            auto_difference=self.auto_diff.get(),
            pvalue_correction=self.p_correction.get(),
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

        tool = BigMasterTool(df, config=cfg)
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
        method_options = {}
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
        stride = int(self.window_stride.get())
        stride = None if stride <= 0 else stride
        start_max = int(self.window_start_max.get())
        start_max = None if start_max <= 0 else start_max

        tool.run_selected_methods(
            methods,
            max_lag=cfg.max_lag,
            dimred_enabled=bool(self.dimred_enabled.get()),
            dimred_method=str(self.dimred_method.get()),
            dimred_target=int(self.dimred_target.get()),
            dimred_spatial_bin=int(self.dimred_spatial_bin.get()),
            dimred_kmeans_batch=int(self.dimred_kmeans_batch.get()),
            dimred_seed=int(self.dimred_seed.get()),
            dimred_save_variants=bool(self.dimred_save_variants.get()),
            dimred_variants=str(self.dimred_variants.get() or ""),
            window_sizes_grid=w_sizes,
            window_min=int(self.window_min.get()),
            window_max=int(self.window_max.get()),
            window_step=int(self.window_step.get()),
            window_size=int(self.window_size_default.get()),
            window_stride=stride,
            window_start_min=int(self.window_start_min.get()),
            window_start_max=start_max,
            window_max_windows=int(self.window_max_windows.get()),
            scan_window_pos=bool(self.scan_window_pos.get()),
            scan_window_size=bool(self.scan_window_size.get()),
            scan_lag=bool(self.scan_lag.get()),
            scan_cube=bool(self.scan_cube.get()),
            lag_min=int(self.lag_min.get()),
            lag_max=int(self.lag_max.get()),
            lag_step=int(self.lag_step.get()),
            cube_combo_limit=int(self.cube_combo_limit.get()),
            cube_eval_limit=int(self.cube_eval_limit.get()),
            cube_matrix_mode=str(self.cube_matrix_mode.get()),
            cube_matrix_limit=int(self.cube_matrix_limit.get()),
            cube_gallery_mode=str(self.cube_gallery_mode.get()),
            cube_gallery_k=int(self.cube_gallery_k.get()),
            cube_gallery_limit=int(self.cube_gallery_limit.get()),
            method_options=method_options,
            cube_pairs=cube_pairs,
            pair_mode=str(self.pair_mode.get()),
            pair_auto_threshold=int(self.pair_auto_threshold.get()),
            pairs_text=str(self.pairs_text.get() or ""),
            max_pairs=int(self.max_pairs.get()),
            neighbor_kind=str(self.neighbor_kind.get()),
            neighbor_radius=int(self.neighbor_radius.get()),
            screen_metric=str(self.screen_metric.get()),
            topk_per_node=int(self.topk_per_node.get()),
        )

        # Экспорт «как данные»: матрицы/графы/edge-list в out_dir/data.
        # Ошибки экспорта не должны ломать основной HTML/Excel пайплайн.
        try:
            tool.export_connectivity_bundle(
                out_dir=out_dir,
                name_prefix=name_prefix,
                dense_n_limit=2000,
                topk_per_node=max(1, int(self.topk_per_node.get())),
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
            tool.export_html_report(
                html_path,
                graph_threshold=cfg.graph_threshold,
                p_alpha=cfg.p_value_alpha,
                include_diagnostics=bool(self.include_diagnostics.get()),
                include_fft_plots=bool(self.include_fft_plots.get()),
                include_scans=bool(self.include_scans.get()),
                include_matrix_tables=bool(self.include_matrix_tables.get()),
            )
        if do_excel:
            tool.export_big_excel(excel_path, threshold=cfg.graph_threshold, p_value_alpha=cfg.p_value_alpha)
        return html_path if do_html else None

    def _run_single(self) -> None:
        fp = self.file_path.get()
        if not os.path.exists(fp):
            messagebox.showerror("Error", "Файл не найден")
            return

        try:
            df = data_loader.load_or_generate(
                fp,
                preprocess=bool(self.preproc_enabled.get()),
                log_transform=bool(self.preproc_log_transform.get()),
                remove_outliers=bool(self.preproc_remove_outliers.get()),
                normalize=bool(self.preproc_normalize.get()),
                fill_missing=bool(self.preproc_fill_missing.get()),
                remove_ar1=bool(self.preproc_remove_ar1.get()),
                remove_seasonality=bool(self.preproc_remove_seasonality.get()),
                season_period=(int(self.preproc_season_period.get()) if int(self.preproc_season_period.get()) > 0 else None),
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
                    preprocess=bool(self.preproc_enabled.get()),
                    log_transform=bool(self.preproc_log_transform.get()),
                    remove_outliers=bool(self.preproc_remove_outliers.get()),
                    normalize=bool(self.preproc_normalize.get()),
                    fill_missing=bool(self.preproc_fill_missing.get()),
                    remove_ar1=bool(self.preproc_remove_ar1.get()),
                    remove_seasonality=bool(self.preproc_remove_seasonality.get()),
                    season_period=(int(self.preproc_season_period.get()) if int(self.preproc_season_period.get()) > 0 else None),
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
            n = int(self.gen_samples.get())
            coupling = float(self.gen_coupling.get())
            noise = float(self.gen_noise.get())
            phi = float(self.gen_ar_phi.get())
            per = int(self.gen_season_period.get())
            nvars = int(self.gen_nvars.get())

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


if __name__ == "__main__":
    app = App()
    app.mainloop()
