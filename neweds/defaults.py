"""Общие константы по умолчанию для core-модулей NewEDS.

В этом модуле сознательно нет импортов из остального ``neweds`` — так его можно
использовать из ``config``, ``metrics`` и ``registry`` без циклических импортов.
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
