"""Совместимый прокси-модуль: реальные реализации живут в категорийных подмодулях.

Старые импорты вида ``from neweds.metrics.connectivity import correlation_matrix``
продолжают работать через реэкспорт из:

- :mod:`neweds.metrics.correlation`
- :mod:`neweds.metrics.spectral`
- :mod:`neweds.metrics.causal`
- :mod:`neweds.metrics.information`
- :mod:`neweds.metrics.ordinal`

Новый код лучше импортировать напрямую из конкретного подмодуля.
"""

from __future__ import annotations

from ..defaults import (  # noqa: F401  (исторический реэкспорт констант)
    DEFAULT_BINS,
    DEFAULT_EMBED_DIM,
    DEFAULT_EMBED_TAU,
    DEFAULT_K_MI,
    DEFAULT_MAX_LAG,
    PYINFORM_AVAILABLE,
)
from ._shared import set_module_seed  # noqa: F401
from .causal import (  # noqa: F401
    compute_te_jitter,
    granger_matrix,
    granger_matrix_partial,
    transfer_entropy_matrix,
    transfer_entropy_matrix_partial,
)
from .correlation import (  # noqa: F401
    correlation_matrix,
    kendall_correlation_matrix,
    kendall_matrix,
    lagged_directed_correlation,
    partial_correlation_matrix,
    partial_h2_matrix,
    spearman_correlation_matrix,
    spearman_matrix,
)
from .information import (  # noqa: F401
    AH_matrix,
    AH_matrix_directed,
    compute_partial_AH_matrix,
    dcor_matrix,
    dcor_matrix_directed,
    dcor_matrix_partial,
    mutual_info_matrix,
    mutual_info_matrix_partial,
)
from .ordinal import (  # noqa: F401
    ordinal_matrix,
    ordinal_matrix_directed,
)
from .spectral import (  # noqa: F401
    coherence_matrix,
    coherence_matrix_partial,
)

__all__ = [
    "AH_matrix",
    "AH_matrix_directed",
    "DEFAULT_BINS",
    "DEFAULT_EMBED_DIM",
    "DEFAULT_EMBED_TAU",
    "DEFAULT_K_MI",
    "DEFAULT_MAX_LAG",
    "PYINFORM_AVAILABLE",
    "coherence_matrix",
    "coherence_matrix_partial",
    "compute_partial_AH_matrix",
    "compute_te_jitter",
    "correlation_matrix",
    "dcor_matrix",
    "dcor_matrix_directed",
    "dcor_matrix_partial",
    "granger_matrix",
    "granger_matrix_partial",
    "kendall_correlation_matrix",
    "kendall_matrix",
    "lagged_directed_correlation",
    "mutual_info_matrix",
    "mutual_info_matrix_partial",
    "ordinal_matrix",
    "ordinal_matrix_directed",
    "partial_correlation_matrix",
    "partial_h2_matrix",
    "set_module_seed",
    "spearman_correlation_matrix",
    "spearman_matrix",
    "transfer_entropy_matrix",
    "transfer_entropy_matrix_partial",
]
