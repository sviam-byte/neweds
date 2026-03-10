#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ядро системы - загрузка данных, генерация и основной движок.
"""

from .data_loader import load_or_generate, read_input_table, tidy_timeseries_table
from .preprocessing import configure_warnings, additional_preprocessing
from .generator import generate_coupled_system, generate_random_walks
from .voxel_space import CanonicalVoxelSpace, VoxelStrategy, align_subjects
from .group_pipeline import run_group_pipeline, load_group, fit_canonical_space

__all__ = [
    "load_or_generate",
    "read_input_table",
    "tidy_timeseries_table",
    "configure_warnings",
    "additional_preprocessing",
    "generate_coupled_system",
    "generate_random_walks",
    "CanonicalVoxelSpace",
    "VoxelStrategy",
    "align_subjects",
    "run_group_pipeline",
    "load_group",
    "fit_canonical_space",
]
