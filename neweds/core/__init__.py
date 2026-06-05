#!/usr/bin/env python3

"""Ядро: загрузка данных, препроцессинг, генератор и пайплайны анализа."""

from .data_loader import load_or_generate, read_input_table, tidy_timeseries_table
from .fmri_roi_audit import FmriRoiAuditResult, run_fmri_roi_audit
from .generator import generate_coupled_system, generate_random_walks
from .group_pipeline import fit_canonical_space, load_group, run_group_pipeline
from .preprocessing import additional_preprocessing, configure_warnings
from .voxel_space import CanonicalVoxelSpace, VoxelStrategy, align_subjects

__all__ = [
    "load_or_generate",
    "read_input_table",
    "tidy_timeseries_table",
    "configure_warnings",
    "additional_preprocessing",
    "generate_coupled_system",
    "generate_random_walks",
    "FmriRoiAuditResult",
    "run_fmri_roi_audit",
    "CanonicalVoxelSpace",
    "VoxelStrategy",
    "align_subjects",
    "run_group_pipeline",
    "load_group",
    "fit_canonical_space",
]
