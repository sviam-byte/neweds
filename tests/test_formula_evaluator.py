import numpy as np
import pytest

from neweds.core.formula_evaluator import (
    FormulaSpec,
    UnsafeFormulaError,
    make_eval_env,
    safe_eval_vector,
)
from neweds.core.generator import generate_formula_dataset


def _eval(expr: str) -> np.ndarray:
    t = np.arange(8, dtype=float)
    env = make_eval_env(n=len(t), rng=np.random.default_rng(123))
    return safe_eval_vector(expr, env=env, names={"t": t})


def test_formula_evaluator_rejects_imports() -> None:
    with pytest.raises((SyntaxError, ValueError, UnsafeFormulaError)):
        _eval("__import__('os')")


def test_formula_evaluator_rejects_attribute_access() -> None:
    with pytest.raises(UnsafeFormulaError):
        _eval("sin.__globals__")


def test_formula_evaluator_rejects_subscripts() -> None:
    with pytest.raises(UnsafeFormulaError):
        _eval("t[0]")


def test_formula_evaluator_rejects_lambda() -> None:
    with pytest.raises(UnsafeFormulaError):
        _eval("(lambda x: x)(t)")


def test_formula_evaluator_rejects_comprehensions() -> None:
    with pytest.raises(UnsafeFormulaError):
        _eval("sum([x for x in t])")


def test_formula_evaluator_allows_numeric_numpy_expressions() -> None:
    out = _eval("sin(2*pi*t/8) + 0.5")

    assert out.shape == (8,)
    assert np.isfinite(out).all()
    assert np.isclose(out[0], 0.5)


def test_generate_formula_dataset_uses_restricted_evaluator() -> None:
    df = generate_formula_dataset(
        n_samples=12,
        specs=[
            FormulaSpec("signal_a", "sin(2*pi*t/6)"),
            FormulaSpec("signal_b", "0.5*signal_a + 0.1"),
        ],
    )

    assert list(df.columns) == ["signal_a", "signal_b"]
    assert df.shape == (12, 2)
