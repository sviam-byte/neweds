"""Shared defaults for NewEDS core modules.

This module intentionally has no imports from the rest of ``neweds`` so it can
be used by config, metrics, and registry code without creating import cycles.
"""

from __future__ import annotations

import importlib.util

DEFAULT_MAX_LAG = 5
DEFAULT_K_MI = 5
DEFAULT_BINS = 8
DEFAULT_OUTLIER_Z = 5
DEFAULT_REGULARIZATION = 1e-8
DEFAULT_EMBED_DIM = 3
DEFAULT_EMBED_TAU = 1
DEFAULT_PVALUE_ALPHA = 0.05
DEFAULT_EDGE_THRESHOLD = 0.2

REG_ALPHA = 1e-5

PYINFORM_AVAILABLE = importlib.util.find_spec("pyinform") is not None
