"""Section-level helpers for the HTML report.

The generator still owns plotting and report context access; these helpers keep
section assembly names explicit while the large legacy report is split further.
"""

from __future__ import annotations


def _value_or_call(value):
    return value() if callable(value) else value


def render_preprocessing_section(value) -> str:
    return _value_or_call(value)


def render_harmonics_section(value):
    return _value_or_call(value)


def render_diagnostics_section(value) -> str:
    return _value_or_call(value)


def render_series_preview_section(value) -> str:
    return _value_or_call(value)
