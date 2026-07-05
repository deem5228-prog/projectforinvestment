# Data normalization and cleaning
"""
Data Normalizer
===============
แปลงข้อมูลดิบจาก data layer ให้เป็น structured dict ที่พร้อมส่งให้ agents

หน้าที่หลัก:
  1. คำนวณ metrics ที่ yfinance ไม่ได้ให้มาตรงๆ
     (ROIC, FCF yield, interest_coverage, net_cash, working_capital)
  2. เติม fields ที่ขาดใน FinancialMetrics โดยคำนวณจาก line_items
  3. คำนวณ growth rates เมื่อมีข้อมูลหลาย period
  4. สร้าง NormalizedData dict ที่ agents ทุกตัวใช้ร่วมกัน
  5. Validate & clamp ค่าที่ผิดปกติ (inf, NaN, outlier)

Usage:
    from src.normalizer.normalizer import normalize
    data = normalize(ticker, end_date)
    # data["metrics"]    → list[FinancialMetrics]  (enriched)
    # data["line_items"] → list[LineItem]
    # data["prices_df"]  → pd.DataFrame
    # data["computed"]   → dict of derived scalars
"""

import logging
import math
from datetime import datetime, timedelta

import pandas as pd

from src.utils.line_item_helpers import get_metric, get_metric_series

from src.data.api import (
    get_financial_metrics,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)
from src.data.models import FinancialMetrics, LineItem

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

EFFECTIVE_TAX_RATE = 0.21       # US corporate tax proxy
COST_OF_EQUITY     = 0.10       # WACC proxy for ROIC hurdle
LOOKBACK_DAYS      = 730        # 2 years of price history


# ── Public entry point ────────────────────────────────────────────────────────

def normalize(ticker: str, end_date: str) -> dict:
    """
    Fetch raw data, compute derived metrics, return unified NormalizedData dict.

    Returns:
        {
            "ticker":     str,
            "end_date":   str,
            "metrics":    list[FinancialMetrics],   # enriched in-place
            "line_items": list[LineItem],
            "prices_df":  pd.DataFrame,
            "market_cap": float | None,
            "computed":   dict,                     # derived scalars
        }
    """
    logger.info("[normalizer] %s — fetching raw data", ticker)

    start_date = (
        datetime.fromisoformat(end_date) - timedelta(days=LOOKBACK_DAYS)
    ).date().isoformat()

    # ── Raw fetches ───────────────────────────────────────────────────────
    metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)

    line_items = search_line_items(
        ticker,
        [
            # Income statement
            "revenue", "gross_profit", "operating_income", "net_income",
            "interest_expense", "research_and_development",
            "selling_general_and_admin", "depreciation_and_amortization",
            # Balance sheet
            "total_assets", "total_liabilities", "total_equity",
            "cash_and_equivalents", "short_term_investments",
            "total_debt", "long_term_debt", "short_term_debt",
            "accounts_receivable", "inventory", "goodwill_and_intangibles",
            # Cash flow
            "operating_cash_flow", "capital_expenditure", "free_cash_flow",
            "stock_repurchases", "dividends_paid",
            # Share data
            "shares_outstanding", "shares_diluted",
        ],
        end_date,
        period="ttm",
        limit=10,
    )

    prices = get_prices(ticker, start_date, end_date)
    prices_df = prices_to_df(prices) if prices else pd.DataFrame()
    market_cap = get_market_cap(ticker, end_date)

    # ── Compute derived metrics ───────────────────────────────────────────
    logger.info("[normalizer] %s — computing derived metrics", ticker)

    computed = _compute_all(metrics, line_items, prices_df, market_cap)

    # ── Enrich FinancialMetrics objects with computed values ──────────────
    if metrics:
        _enrich_metrics(metrics[0], computed)

    logger.info(
        "[normalizer] %s — done. computed fields: %d",
        ticker, len([v for v in computed.values() if v is not None]),
    )

    return {
        "ticker":     ticker,
        "end_date":   end_date,
        "metrics":    metrics,
        "line_items": line_items,
        "prices_df":  prices_df,
        "market_cap": market_cap,
        "computed":   computed,
    }


# ── Master compute function ───────────────────────────────────────────────────

def _compute_all(
    metrics: list[FinancialMetrics],
    line_items: list[LineItem],
    prices_df: pd.DataFrame,
    market_cap: float | None,
) -> dict:
    """Compute every derived scalar. Returns flat dict, all values safe floats."""

    m    = metrics[0]    if metrics    else None

    c: dict = {}

    # ── 1. Income statement derived ───────────────────────────────────────
    c["revenue"]          = get_metric(line_items, "revenue")
    c["gross_profit"]     = get_metric(line_items, "gross_profit")
    c["operating_income"] = get_metric(line_items, "operating_income")
    c["net_income"]       = get_metric(line_items, "net_income")
    c["interest_expense"] = get_metric(line_items, "interest_expense")
    c["da"]               = get_metric(line_items, "depreciation_and_amortization")
    c["ebitda"]           = _compute_ebitda(line_items)

    # ── 2. Balance sheet derived ──────────────────────────────────────────
    c["total_assets"]     = get_metric(line_items, "total_assets")
    c["total_liabilities"]= get_metric(line_items, "total_liabilities")
    c["total_equity"]     = get_metric(line_items, "total_equity")
    c["total_debt"]       = get_metric(line_items, "total_debt")
    c["cash"]             = get_metric(line_items, "cash_and_equivalents")
    c["net_cash"]         = _compute_net_cash(line_items)
    c["invested_capital"] = _compute_invested_capital(line_items)
    c["working_capital"]  = _compute_working_capital(line_items, m)
    c["tangible_equity"]  = _compute_tangible_equity(line_items)
    c["book_value_per_share"] = _compute_bvps(line_items)

    # ── 3. Cash flow derived ──────────────────────────────────────────────
    c["operating_cash_flow"] = get_metric(line_items, "operating_cash_flow")
    c["capital_expenditure"] = get_metric(line_items, "capital_expenditure")
    c["free_cash_flow"]      = _compute_fcf(line_items)
    c["owner_earnings"]      = _compute_owner_earnings(line_items)
    c["fcf_yield"]           = _div(c["free_cash_flow"], market_cap)
    c["fcf_margin"]          = _div(c["free_cash_flow"], c["revenue"])

    # ── 4. Profitability ratios ───────────────────────────────────────────
    c["gross_margin"]      = _div(c["gross_profit"], c["revenue"]) or _g(m, "gross_margin")
    c["operating_margin"]  = _div(c["operating_income"], c["revenue"]) or _g(m, "operating_margin")
    c["net_margin"]        = _div(c["net_income"], c["revenue"]) or _g(m, "net_margin")
    c["ebitda_margin"]     = _div(c["ebitda"], c["revenue"])

    # ── 5. Return metrics ─────────────────────────────────────────────────
    c["roa"]  = _div(c["net_income"], c["total_assets"]) or _g(m, "return_on_assets")
    c["roe"]  = _div(c["net_income"], c["total_equity"]) or _g(m, "return_on_equity")
    c["roic"] = _compute_roic(line_items, m)
    c["roce"] = _compute_roce(line_items)   # Return on Capital Employed

    # ── 6. Leverage & coverage ────────────────────────────────────────────
    c["debt_to_equity"]   = _div(c["total_debt"], c["total_equity"]) or _g(m, "debt_to_equity")
    c["debt_to_assets"]   = _div(c["total_debt"], c["total_assets"]) or _g(m, "debt_to_assets")
    c["net_debt"]         = _safe(-c["net_cash"]) if c["net_cash"] is not None else None
    c["net_debt_ebitda"]  = _div(c["net_debt"], c["ebitda"])
    c["interest_coverage"]= _compute_interest_coverage(line_items, m)
    c["current_ratio"]    = _g(m, "current_ratio")

    # ── 7. Valuation multiples ────────────────────────────────────────────
    c["market_cap"]       = market_cap
    c["ev"]               = _compute_ev(line_items, market_cap)
    c["pe"]               = _g(m, "price_to_earnings_ratio")
    c["pb"]               = _g(m, "price_to_book_ratio")
    c["ps"]               = _g(m, "price_to_sales_ratio")
    c["ev_ebitda"]        = _div(c["ev"], c["ebitda"]) or _g(m, "ev_to_ebitda")
    c["ev_revenue"]       = _div(c["ev"], c["revenue"]) or _g(m, "ev_to_revenue")
    c["ev_fcf"]           = _div(c["ev"], c["free_cash_flow"])
    c["earnings_yield"]   = _div(1, c["pe"])          # Greenblatt earnings yield
    c["peg"]              = _compute_peg(m)

    # ── 8. Growth rates (multi-period) ───────────────────────────────────
    c["revenue_growth"]   = _growth_rate(line_items, "revenue")   or _g(m, "revenue_growth")
    c["ni_growth"]        = _growth_rate(line_items, "net_income") or _g(m, "earnings_growth")
    c["fcf_growth"]       = _growth_fcf(line_items)
    c["op_income_growth"] = _growth_rate(line_items, "operating_income")
    c["rev_cagr_3y"]      = _cagr(line_items, "revenue", years=3)
    c["ni_cagr_3y"]       = _cagr(line_items, "net_income", years=3)

    # ── 9. Efficiency ─────────────────────────────────────────────────────
    c["asset_turnover"]   = _div(c["revenue"], c["total_assets"]) or _g(m, "asset_turnover")
    c["capex_intensity"]  = _div(abs(c["capital_expenditure"] or 0), c["revenue"])
    c["rd_intensity"]     = _div(abs(get_metric(line_items, "research_and_development") or 0), c["revenue"])
    c["reinvestment_rate"]= _compute_reinvestment_rate(line_items)

    # ── 10. Price-based ───────────────────────────────────────────────────
    if not prices_df.empty:
        c.update(_compute_price_metrics(prices_df))
    else:
        for k in ["ann_vol", "ann_return", "max_drawdown", "sharpe", "calmar",
                  "beta_proxy", "price_52w_high", "price_52w_low", "price_52w_pct"]:
            c[k] = None

    # ── 11. Composite scores ──────────────────────────────────────────────
    c["piotroski_f"] = _piotroski(metrics, line_items)
    c["magic_formula_score"] = _magic_formula(c)

    # ── Clamp & sanitize all values ────────────────────────────────────────
    c = {k: _clamp(v) for k, v in c.items()}

    return c


# ── Enrich FinancialMetrics in-place ─────────────────────────────────────────

def _enrich_metrics(m: FinancialMetrics, c: dict) -> None:
    """
    Fill in None fields on the FinancialMetrics object
    using computed values where available.
    """
    fill_map = {
        "return_on_invested_capital": "roic",
        "return_on_assets":           "roa",
        "return_on_equity":           "roe",
        "interest_coverage":          "interest_coverage",
        "debt_to_equity":             "debt_to_equity",
        "debt_to_assets":             "debt_to_assets",
        "gross_margin":               "gross_margin",
        "operating_margin":           "operating_margin",
        "net_margin":                 "net_margin",
        "free_cash_flow_yield":       "fcf_yield",
        "ev_to_ebitda":               "ev_ebitda",
        "ev_to_revenue":              "ev_revenue",
        "asset_turnover":             "asset_turnover",
        "peg_ratio":                  "peg",
        "revenue_growth":             "revenue_growth",
        "earnings_growth":            "ni_growth",
        "free_cash_flow_growth":      "fcf_growth",
    }
    for field, key in fill_map.items():
        if getattr(m, field, None) is None and c.get(key) is not None:
            try:
                setattr(m, field, c[key])
            except Exception:
                pass   # model may have validators; skip silently


###############################################################################
# Computation helpers
###############################################################################

def _g(obj, field: str, default=None):
    """Safe getattr returning None (not AttributeError) for missing fields."""
    if obj is None:
        return default
    val = getattr(obj, field, default)
    return _safe(val)


def _safe(v, default=None):
    """Convert to float; return default for None/NaN/inf."""
    if v is None:
        return default
    try:
        f = float(v)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return default


def _div(a, b, default=None):
    """Safe division; returns default if denominator is zero or None."""
    a, b = _safe(a), _safe(b)
    if a is None or b is None or b == 0:
        return default
    result = a / b
    return _safe(result, default)


def _clamp(v, lo=-1e9, hi=1e9):
    """Clamp extreme values that would skew agent scoring."""
    if v is None:
        return v
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return max(lo, min(hi, f))
    except (TypeError, ValueError):
        return v   # leave non-numeric values alone


# ── Derived calculations ──────────────────────────────────────────────────────

def _compute_ebitda(line_items: list) -> float | None:
    op   = get_metric(line_items, "operating_income")
    da   = get_metric(line_items, "depreciation_and_amortization")
    if op is None:
        return None
    return op + (da or 0)


def _compute_net_cash(line_items: list) -> float | None:
    cash = get_metric(line_items, "cash_and_equivalents")
    debt = get_metric(line_items, "total_debt")
    if cash is None:
        return None
    return cash - (debt or 0)


def _compute_invested_capital(line_items: list) -> float | None:
    """Invested Capital = Total Equity + Total Debt - Cash"""
    equity = get_metric(line_items, "total_equity")
    debt   = get_metric(line_items, "total_debt")
    cash   = get_metric(line_items, "cash_and_equivalents")
    if equity is None:
        return None
    ic = equity + (debt or 0) - (cash or 0)
    return ic if ic > 0 else None


def _compute_working_capital(line_items: list, m) -> float | None:
    """WC = Current Assets - Current Liabilities (proxy via current_ratio if available)."""
    cr = _g(m, "current_ratio")
    if cr and cr > 0:
        # WC = (CR - 1) * current liabilities; use total_debt as proxy for CL
        cl = get_metric(line_items, "short_term_debt")
        if cl:
            return (cr - 1) * cl
    return None


def _compute_tangible_equity(line_items: list) -> float | None:
    equity      = get_metric(line_items, "total_equity")
    intangibles = get_metric(line_items, "goodwill_and_intangibles")
    if equity is None:
        return None
    return equity - (intangibles or 0)


def _compute_bvps(line_items: list) -> float | None:
    equity = get_metric(line_items, "total_equity")
    shares = get_metric(line_items, "shares_outstanding")
    return _div(equity, shares)


def _compute_fcf(line_items: list) -> float | None:
    """FCF = operating_cash_flow + capital_expenditure (capex is usually negative)."""
    fcf = get_metric(line_items, "free_cash_flow")
    if fcf is not None:
        return fcf
    ocf   = get_metric(line_items, "operating_cash_flow")
    capex = get_metric(line_items, "capital_expenditure")
    if ocf is None:
        return None
    return ocf + (capex or 0)


def _compute_owner_earnings(line_items: list) -> float | None:
    """
    Buffett's Owner Earnings:
    Net Income + D&A - Maintenance CapEx (≈ 85% of total capex)
    """
    ni    = get_metric(line_items, "net_income")
    da    = get_metric(line_items, "depreciation_and_amortization")
    capex = get_metric(line_items, "capital_expenditure")
    if ni is None:
        return None
    maintenance_capex = abs(capex or 0) * 0.85
    return ni + (da or 0) - maintenance_capex


def _compute_roic(line_items: list, m) -> float | None:
    """ROIC = NOPAT / Invested Capital; NOPAT = Operating Income * (1 - tax)."""
    roic = _g(m, "return_on_invested_capital")
    if roic is not None:
        return roic
    op_income = get_metric(line_items, "operating_income")
    ic        = _compute_invested_capital(line_items)
    if op_income is None or ic is None or ic <= 0:
        return None
    nopat = op_income * (1 - EFFECTIVE_TAX_RATE)
    return nopat / ic


def _compute_roce(line_items: list) -> float | None:
    """ROCE = EBIT / Capital Employed; Capital Employed = Total Assets - Current Liabilities."""
    ebit   = get_metric(line_items, "operating_income")
    assets = get_metric(line_items, "total_assets")
    cl     = get_metric(line_items, "short_term_debt")
    if ebit is None or assets is None:
        return None
    capital_employed = assets - (cl or 0)
    return _div(ebit, capital_employed)


def _compute_interest_coverage(line_items: list, m) -> float | None:
    """Interest Coverage = EBIT / Interest Expense."""
    ic = _g(m, "interest_coverage")
    if ic is not None:
        return ic
    ebit     = get_metric(line_items, "operating_income")
    interest = get_metric(line_items, "interest_expense")
    if ebit is None or interest is None or interest == 0:
        return None
    return abs(ebit) / abs(interest)


def _compute_ev(line_items: list, market_cap: float | None) -> float | None:
    """EV = Market Cap + Total Debt - Cash."""
    if market_cap is None:
        return None
    debt = get_metric(line_items, "total_debt") or 0
    cash = get_metric(line_items, "cash_and_equivalents") or 0
    return market_cap + debt - cash


def _compute_peg(m) -> float | None:
    """PEG = P/E / (Earnings Growth * 100)."""
    peg = _g(m, "peg_ratio")
    if peg is not None:
        return peg
    pe = _g(m, "price_to_earnings_ratio")
    eg = _g(m, "earnings_growth")
    if pe and eg and eg > 0:
        return pe / (eg * 100)
    return None


def _compute_reinvestment_rate(line_items: list) -> float | None:
    """Reinvestment Rate = (CapEx + ΔWC - D&A) / NOPAT."""
    capex = get_metric(line_items, "capital_expenditure")
    da    = get_metric(line_items, "depreciation_and_amortization")
    op    = get_metric(line_items, "operating_income")
    if capex is None or op is None:
        return None
    nopat = op * (1 - EFFECTIVE_TAX_RATE)
    net_investment = abs(capex or 0) - (da or 0)
    return _div(net_investment, nopat)


# ── Growth rate helpers ───────────────────────────────────────────────────────

def _field_series(line_items: list, field: str) -> list[float]:
    """Extract ordered list of non-None values for a field (most recent first)."""
    return get_metric_series(line_items, field)


def _growth_rate(line_items: list, field: str) -> float | None:
    """YoY growth: (latest - prior) / |prior|."""
    vals = _field_series(line_items, field)
    if len(vals) < 2 or vals[1] == 0:
        return None
    return (vals[0] - vals[1]) / abs(vals[1])


def _cagr(line_items: list, field: str, years: int = 3) -> float | None:
    """CAGR over `years` periods."""
    vals = _field_series(line_items, field)
    if len(vals) < years + 1:
        return None
    oldest = vals[years]
    latest = vals[0]
    if oldest <= 0 or latest <= 0:
        return None
    return (latest / oldest) ** (1 / years) - 1


def _growth_fcf(line_items: list) -> float | None:
    """FCF growth: try free_cash_flow first, then compute from OCF-CapEx."""
    vals = _field_series(line_items, "free_cash_flow")
    if len(vals) < 2:
        # compute from components
        ocf_series   = get_metric_series(line_items, "operating_cash_flow")
        capex_series = get_metric_series(line_items, "capital_expenditure")
        vals = [ocf + capex for ocf, capex in zip(ocf_series, capex_series)]

    if len(vals) < 2 or vals[1] == 0:
        return None
    return (vals[0] - vals[1]) / abs(vals[1])


# ── Price metrics ─────────────────────────────────────────────────────────────

def _compute_price_metrics(prices_df: pd.DataFrame) -> dict:
    """Compute annualized return, volatility, max drawdown, Sharpe, Calmar."""
    closes = prices_df["close"].dropna()
    if len(closes) < 20:
        return {}

    rets = closes.pct_change().dropna()
    n    = len(closes)

    ann_return = float((closes.iloc[-1] / closes.iloc[0]) ** (252 / n) - 1) if n > 1 else 0.0
    ann_vol    = float(rets.std() * (252 ** 0.5))

    cum = (1 + rets).cumprod()
    running_max = cum.cummax()
    dd  = (cum - running_max) / running_max
    max_dd = float(dd.min())

    sharpe = (ann_return - 0.045) / ann_vol if ann_vol > 0 else 0.0
    calmar = ann_return / abs(max_dd) if max_dd < 0 else None

    # 52-week range
    window = min(252, len(closes))
    high_52w = float(closes.iloc[-window:].max())
    low_52w  = float(closes.iloc[-window:].min())
    current  = float(closes.iloc[-1])
    rng      = high_52w - low_52w
    pct_in_range = (current - low_52w) / rng if rng > 0 else 0.5

    return {
        "ann_return":    ann_return,
        "ann_vol":       ann_vol,
        "max_drawdown":  max_dd,
        "sharpe":        sharpe,
        "calmar":        calmar,
        "price_52w_high": high_52w,
        "price_52w_low":  low_52w,
        "price_52w_pct":  pct_in_range,
    }


# ── Composite scores ──────────────────────────────────────────────────────────

def _piotroski(metrics: list, line_items: list) -> int | None:
    """
    Piotroski F-Score (0-9).
    Profitability(4) + Leverage(3) + Efficiency(2).
    """
    if not metrics or not line_items:
        return None

    m0 = metrics[0]
    m1 = metrics[1] if len(metrics) > 1 else None

    score = 0

    # Profitability
    roa = _safe(_g(m0, "return_on_assets"))
    if roa and roa > 0: score += 1

    fcf0 = _compute_fcf(line_items)
    if fcf0 and fcf0 > 0: score += 1

    roa1 = _safe(_g(m1, "return_on_assets")) if m1 else None
    if roa and roa1 and roa > roa1: score += 1

    ni0  = _safe(get_metric(line_items, "net_income"))
    if fcf0 and ni0 and fcf0 > ni0: score += 1

    # Leverage
    de0 = _safe(_g(m0, "debt_to_equity"))
    de1 = _safe(_g(m1, "debt_to_equity")) if m1 else None
    if de0 is not None and de1 is not None and de0 < de1: score += 1

    cr0 = _safe(_g(m0, "current_ratio"))
    cr1 = _safe(_g(m1, "current_ratio")) if m1 else None
    if cr0 and cr1 and cr0 > cr1: score += 1

    _sh_series = get_metric_series(line_items, "shares_outstanding")
    sh0 = _safe(_sh_series[0]) if len(_sh_series) > 0 else None
    sh1 = _safe(_sh_series[1]) if len(_sh_series) > 1 else None
    if sh0 and sh1 and sh0 <= sh1 * 1.02: score += 1

    # Efficiency
    gm0 = _safe(_g(m0, "gross_margin"))
    gm1 = _safe(_g(m1, "gross_margin")) if m1 else None
    if gm0 and gm1 and gm0 > gm1: score += 1

    at0 = _safe(_g(m0, "asset_turnover"))
    at1 = _safe(_g(m1, "asset_turnover")) if m1 else None
    if at0 and at1 and at0 > at1: score += 1

    return score


def _magic_formula(c: dict) -> float | None:
    """
    Greenblatt Magic Formula = Earnings Yield + ROIC.
    Higher is better.
    """
    ey   = c.get("earnings_yield")
    roic = c.get("roic")
    if ey is None or roic is None:
        return None
    return ey + roic


# ── Summary helper ────────────────────────────────────────────────────────────

def summary(data: dict) -> str:
    """Return a human-readable one-paragraph summary of computed metrics."""
    c = data["computed"]
    t = data["ticker"]

    lines = [f"=== {t} Normalized Summary ==="]

    def _fmt(label: str, key: str, pct: bool = False, x: bool = False):
        v = c.get(key)
        if v is None:
            return
        fmt = f"{v:.1%}" if pct else (f"{v:.2f}x" if x else f"{v:.2f}")
        lines.append(f"  {label}: {fmt}")

    _fmt("Market Cap ($)", "market_cap")
    _fmt("EV ($)",         "ev")
    _fmt("Revenue ($)",    "revenue")
    _fmt("EBITDA ($)",     "ebitda")
    _fmt("FCF ($)",        "free_cash_flow")
    _fmt("Net Cash ($)",   "net_cash")
    print()
    _fmt("Gross Margin",    "gross_margin",    pct=True)
    _fmt("EBITDA Margin",   "ebitda_margin",   pct=True)
    _fmt("FCF Margin",      "fcf_margin",       pct=True)
    _fmt("ROE",             "roe",              pct=True)
    _fmt("ROIC",            "roic",             pct=True)
    _fmt("Interest Coverage","interest_coverage", x=True)
    _fmt("D/E",             "debt_to_equity",   x=True)
    print()
    _fmt("EV/EBITDA",       "ev_ebitda",        x=True)
    _fmt("P/E",             "pe",               x=True)
    _fmt("FCF Yield",       "fcf_yield",        pct=True)
    _fmt("PEG",             "peg")
    print()
    _fmt("Revenue Growth (YoY)", "revenue_growth",  pct=True)
    _fmt("NI Growth (YoY)",      "ni_growth",       pct=True)
    _fmt("Rev CAGR 3Y",          "rev_cagr_3y",     pct=True)
    print()
    _fmt("Ann. Return",     "ann_return",   pct=True)
    _fmt("Ann. Volatility", "ann_vol",      pct=True)
    _fmt("Max Drawdown",    "max_drawdown", pct=True)
    _fmt("Sharpe",          "sharpe")
    _fmt("52w Position",    "price_52w_pct", pct=True)

    f = c.get("piotroski_f")
    if f is not None:
        lines.append(f"  Piotroski F-Score: {f}/9")

    mf = c.get("magic_formula_score")
    if mf is not None:
        lines.append(f"  Magic Formula Score: {mf:.3f}")

    return "\n".join(lines)