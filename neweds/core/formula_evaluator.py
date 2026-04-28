"""Restricted formula evaluator for synthetic signal generation.

The evaluator validates Python AST nodes against a small numeric-expression
whitelist and disables builtins before calling ``eval``. It is intended only
for vectorized numeric expressions over explicitly allowed names and functions.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FormulaSpec:
    """One generated time-series definition."""

    name: str
    expr: str


class UnsafeFormulaError(ValueError):
    """Formula contains a disallowed construct."""


_ALLOWED_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_ALLOWED_UNARYOPS = (ast.UAdd, ast.USub)


def _validate_ast(node: ast.AST, allowed_names: set[str], allowed_funcs: set[str]) -> None:
    for n in ast.walk(node):
        if isinstance(n, ast.Expression):
            continue
        if isinstance(n, ast.BinOp):
            if not isinstance(n.op, _ALLOWED_BINOPS):
                raise UnsafeFormulaError(f"Disallowed operator: {type(n.op).__name__}")
            continue
        if isinstance(n, ast.UnaryOp):
            if not isinstance(n.op, _ALLOWED_UNARYOPS):
                raise UnsafeFormulaError(f"Disallowed unary operator: {type(n.op).__name__}")
            continue
        if isinstance(n, ast.Call):
            if isinstance(n.func, ast.Name):
                fn = n.func.id
                if fn not in allowed_funcs:
                    raise UnsafeFormulaError(f"Disallowed function: {fn}")
            else:
                raise UnsafeFormulaError("Attribute, lambda and subscript calls are disallowed")
            continue
        if isinstance(n, ast.Name):
            if n.id not in allowed_names and n.id not in allowed_funcs:
                raise UnsafeFormulaError(f"Disallowed name: {n.id}")
            continue
        if isinstance(n, ast.Constant):
            if isinstance(n.value, (int, float)) or n.value is None:
                continue
            raise UnsafeFormulaError("Only numeric constants are allowed")
        if isinstance(n, ast.Tuple | ast.keyword | ast.Load):
            continue
        if isinstance(
            n,
            (
                ast.Attribute,
                ast.Subscript,
                ast.Compare,
                ast.BoolOp,
                ast.IfExp,
                ast.Dict,
                ast.List,
                ast.Set,
                ast.Lambda,
                ast.ListComp,
                ast.DictComp,
                ast.SetComp,
                ast.GeneratorExp,
                ast.Await,
                ast.Yield,
                ast.YieldFrom,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
                ast.With,
                ast.Try,
                ast.While,
                ast.For,
                ast.Assign,
                ast.AnnAssign,
                ast.AugAssign,
                ast.FunctionDef,
                ast.ClassDef,
                ast.Return,
            ),
        ):
            raise UnsafeFormulaError(f"Disallowed construct: {type(n).__name__}")


def make_eval_env(
    *,
    n: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Build the allowed function environment for formulas."""

    def randn(scale: float = 1.0) -> np.ndarray:
        return rng.normal(0.0, float(scale), size=n)

    def randu(scale: float = 1.0) -> np.ndarray:
        return rng.uniform(-float(scale), float(scale), size=n)

    def rw(scale: float = 1.0) -> np.ndarray:
        return np.cumsum(randn(scale))

    def ar1(phi: float = 0.7, scale: float = 1.0) -> np.ndarray:
        e = randn(scale)
        x = np.zeros(n, dtype=float)
        p = float(phi)
        for i in range(1, n):
            x[i] = p * x[i - 1] + e[i]
        return x

    return {
        "pi": float(np.pi),
        "e": float(np.e),
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "clip": np.clip,
        "where": np.where,
        "minimum": np.minimum,
        "maximum": np.maximum,
        "randn": randn,
        "randu": randu,
        "rw": rw,
        "ar1": ar1,
    }


def safe_eval_vector(expr: str, *, env: Mapping[str, Any], names: Mapping[str, Any]) -> np.ndarray:
    """Evaluate one validated formula as a numeric vector."""
    expr = (expr or "").strip()
    if not expr:
        raise ValueError("Formula is empty")

    allowed_funcs = {k for k, v in env.items() if callable(v)}
    allowed_names = set(names.keys()) | {k for k, v in env.items() if not callable(v)}

    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Formula syntax error: {exc}") from exc

    _validate_ast(node, allowed_names=allowed_names, allowed_funcs=allowed_funcs)

    code = compile(node, "<formula>", "eval")
    out = eval(code, {"__builtins__": {}}, {**env, **names})  # noqa: S307

    arr = np.asarray(out, dtype=float)
    if arr.shape == ():
        arr = np.full((int(len(names["t"])),), float(arr), dtype=float)

    if arr.shape[0] != len(names["t"]):
        raise ValueError(f"Formula returned length {arr.shape[0]}, expected {len(names['t'])}")
    return arr


__all__ = ["FormulaSpec", "UnsafeFormulaError", "make_eval_env", "safe_eval_vector"]
