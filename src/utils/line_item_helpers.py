"""
Line-item helper utilities
===========================
The ``LineItem`` model stores financial data in *long format*:

    LineItem(line_item="revenue", value=1234567.0, ...)

Many analysis functions need to look up a specific metric by name or
collect a time-series across multiple periods.  These helpers centralise
that logic so every consumer file doesn't need to reinvent it.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.models import LineItem


# ── Single-value lookup ──────────────────────────────────────────────────────

def get_metric(line_items: list[LineItem], metric_name: str) -> float | None:
    """
    Return the **latest** (first match, since results are sorted newest-first)
    value for *metric_name* from a list of LineItem objects.

    Returns ``None`` when no matching item is found or the value is
    None / NaN / Inf.

    >>> get_metric(items, "revenue")
    123456789.0
    """
    for item in line_items:
        if getattr(item, "line_item", None) == metric_name:
            v = getattr(item, "value", None)
            if v is None:
                continue
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    continue
                return f
            except (TypeError, ValueError):
                continue
    return None


# ── Multi-period series ──────────────────────────────────────────────────────

def get_metric_series(
    line_items: list[LineItem],
    metric_name: str,
) -> list[float]:
    """
    Return **all** non-None values for *metric_name*, newest period first
    (preserving the sort order of *line_items*).

    Useful for computing growth rates, CAGR, trend analysis, etc.

    >>> get_metric_series(items, "net_income")
    [500.0, 450.0, 400.0]
    """
    values: list[float] = []
    for item in line_items:
        if getattr(item, "line_item", None) == metric_name:
            v = getattr(item, "value", None)
            if v is None:
                continue
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    continue
                values.append(f)
            except (TypeError, ValueError):
                continue
    return values
