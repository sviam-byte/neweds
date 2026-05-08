"""Import-time sanity: ``import neweds`` остаётся лёгким.

Делает три вещи:
1. ``import neweds`` не должен тянуть statsmodels (тяжёлая зависимость, нужна только
   для granger/VAR — должна импортироваться лениво внутри метрик).
2. ``ensure_builtins()`` идемпотентен.
3. Все встроенные метрики имеют валидные имена и категорию.
"""

from __future__ import annotations

import subprocess
import sys

VALID_CATEGORIES = {"correlation", "information", "spectral", "ordinal", "causal"}
VALID_PARTIAL_MODES = {"none", "precision_matrix", "explicit_controls_residualization"}


def test_import_neweds_does_not_load_statsmodels() -> None:
    """``import neweds`` не должен импортировать statsmodels."""
    code = (
        "import sys\n"
        "import neweds\n"
        "assert 'statsmodels' not in sys.modules, "
        "    'statsmodels was loaded eagerly: ' + ','.join(\n"
        "        m for m in sys.modules if m.startswith('statsmodels')\n"
        "    )\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import sanity failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip().endswith("ok")


def test_ensure_builtins_is_idempotent() -> None:
    from neweds.metrics import registry

    registry.ensure_builtins()
    n1 = len(registry.list_metrics())
    registry.ensure_builtins()
    n2 = len(registry.list_metrics())
    assert n1 == n2 and n1 > 0


def test_registry_metadata_consistency() -> None:
    from neweds.metrics import list_metrics

    metrics = list_metrics()
    assert metrics, "registry is empty"
    seen_names: set[str] = set()
    for m in metrics:
        assert m.name not in seen_names, f"duplicate metric name: {m.name}"
        seen_names.add(m.name)
        assert m.category in VALID_CATEGORIES, f"{m.name}: invalid category {m.category}"
        partial_mode = getattr(m, "partial_mode", "none")
        assert partial_mode in VALID_PARTIAL_MODES, f"{m.name}: invalid partial_mode {partial_mode}"
        if m.supports_control:
            assert partial_mode != "none", (
                f"{m.name}: supports_control=True but partial_mode='none'"
            )
