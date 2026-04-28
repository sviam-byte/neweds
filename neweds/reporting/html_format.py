"""Small HTML formatting helpers shared by report renderers."""

from __future__ import annotations

import html

import numpy as np


def fmt_num(x, digits: int = 3) -> str:
    try:
        if x is None:
            return "-"
        value = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{value:.{int(digits)}g}" if np.isfinite(value) else "-"


def fmt_frac(x) -> str:
    try:
        value = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{100.0 * value:.1f}%" if np.isfinite(value) else "-"


def html_table(df, *, max_rows: int = 20) -> str:
    if df is None or getattr(df, "empty", True):
        return ""
    view = df.head(int(max_rows))
    return view.to_html(index=False, border=0, escape=True)


def fmt_cell(value) -> str:
    try:
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        f = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return f"{f:.4g}" if np.isfinite(f) else "NaN"
