# ── dcf_agent.py ────────────────────────────────────────────────────────────
# DCF (Discounted Cash Flow) Fair Value Calculator
# คำนวณราคาที่เหมาะสมของหุ้นจากกระแสเงินสดอิสระ (FCF)
# ใช้คณิตศาสตร์ล้วนๆ — ไม่เรียก LLM / ไม่เปลือง API Quota
# ────────────────────────────────────────────────────────────────────────────

import logging
import math

from src.data.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)
from src.utils.line_item_helpers import get_metric, get_metric_series

logger = logging.getLogger(__name__)


# ── DCF Parameters ───────────────────────────────────────────────────────────

DISCOUNT_RATE = 0.10         # 10% WACC (weighted average cost of capital)
TERMINAL_GROWTH = 0.025      # 2.5% perpetual growth rate
SAFETY_MARGIN = 0.15         # 15% haircut for model uncertainty
MAX_GROWTH_RATE = 0.25       # Cap growth at 25% per year
GROWTH_DECAY = 0.50          # Stage 2 growth = Stage 1 × 0.50
STAGE1_YEARS = 5             # High growth period
STAGE2_YEARS = 5             # Decay period


# ── Main entry point ─────────────────────────────────────────────────────────

def calculate_dcf_fair_value(
    ticker: str,
    end_date: str,
    normalized_data: dict | None = None,
) -> dict:
    """
    Calculate intrinsic (fair) value per share using 3-stage DCF model.

    This function does NOT participate in the voting system.
    It provides a standalone fair-price estimate for display purposes.

    Args:
        ticker:          Stock ticker symbol
        end_date:        ISO date string
        normalized_data: Pre-fetched data from normalizer (optional)

    Returns:
        dict with keys:
          - fair_value_per_share: float  (intrinsic value per share)
          - current_price:        float  (latest market price per share)
          - upside_pct:           float  (upside/downside percentage)
          - verdict:              str    ("Undervalued" / "Fair Value" / "Overvalued")
          - details:              dict   (breakdown of DCF components)
          - error:                str|None (error message if calculation failed)
    """
    logger.info("[dcf] calculating fair value for %s", ticker)

    try:
        # ── 1. Extract financial data ────────────────────────────────────
        if normalized_data:
            line_items = normalized_data.get("line_items", [])
            metrics = normalized_data.get("metrics", [])
            market_cap = normalized_data.get("market_cap")
            computed = normalized_data.get("computed", {})
        else:
            line_items = search_line_items(
                ticker,
                ["free_cash_flow", "revenue", "net_income", "shares_outstanding",
                 "total_debt", "cash_and_equivalents", "operating_cash_flow",
                 "capital_expenditure"],
                end_date,
                period="annual",
                limit=10,
            )
            metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)
            market_cap = get_market_cap(ticker, end_date)
            computed = {}

        # ── 2. Get key values ────────────────────────────────────────────
        fcf = get_metric(line_items, "free_cash_flow")
        revenue = get_metric(line_items, "revenue")
        shares = get_metric(line_items, "shares_outstanding")
        total_debt = get_metric(line_items, "total_debt") or 0
        cash = get_metric(line_items, "cash_and_equivalents") or 0

        # If FCF is not available, try to compute from operating cash flow - capex
        if fcf is None:
            ocf = get_metric(line_items, "operating_cash_flow")
            capex = get_metric(line_items, "capital_expenditure")
            if ocf is not None and capex is not None:
                fcf = ocf - abs(capex)
                logger.info("[dcf] computed FCF from OCF - CapEx: %.0f", fcf)

        # Validate essential data
        if fcf is None or fcf <= 0:
            return _error_result("FCF ไม่มีข้อมูลหรือติดลบ — ไม่สามารถคำนวณ DCF ได้")

        if shares is None or shares <= 0:
            return _error_result("จำนวนหุ้นไม่มีข้อมูล — ไม่สามารถคำนวณราคาต่อหุ้นได้")

        # ── 3. Estimate growth rate ──────────────────────────────────────
        fcf_series = get_metric_series(line_items, "free_cash_flow")
        rev_series = get_metric_series(line_items, "revenue")

        # Try FCF CAGR first, fall back to revenue growth
        growth_rate = _estimate_growth_rate(fcf_series, rev_series, computed)

        # ── 4. Run 3-Stage DCF ───────────────────────────────────────────
        dcf_result = _three_stage_dcf(
            base_fcf=fcf,
            growth_rate=growth_rate,
        )

        # ── 5. Calculate equity value ────────────────────────────────────
        enterprise_value = dcf_result["enterprise_value"]
        equity_value = enterprise_value - total_debt + cash
        equity_value_safe = equity_value * (1 - SAFETY_MARGIN)

        fair_value_per_share = equity_value_safe / shares

        # ── 6. Get current price and compute upside ──────────────────────
        current_price = None
        if market_cap and shares:
            current_price = market_cap / shares

        upside_pct = None
        verdict = "N/A"
        if current_price and current_price > 0:
            upside_pct = ((fair_value_per_share / current_price) - 1) * 100
            if upside_pct >= 20:
                verdict = "Undervalued"
            elif upside_pct <= -20:
                verdict = "Overvalued"
            else:
                verdict = "Fair Value"

        logger.info(
            "[dcf] %s → Fair Value: %.2f, Current: %s, Upside: %s%%",
            ticker,
            fair_value_per_share,
            f"{current_price:.2f}" if current_price else "N/A",
            f"{upside_pct:.1f}" if upside_pct is not None else "N/A",
        )

        return {
            "fair_value_per_share": round(fair_value_per_share, 2),
            "current_price": round(current_price, 2) if current_price else None,
            "upside_pct": round(upside_pct, 1) if upside_pct is not None else None,
            "verdict": verdict,
            "details": {
                "base_fcf": round(fcf, 0),
                "growth_rate_stage1": round(growth_rate * 100, 1),
                "growth_rate_stage2": round(growth_rate * GROWTH_DECAY * 100, 1),
                "terminal_growth": round(TERMINAL_GROWTH * 100, 1),
                "discount_rate": round(DISCOUNT_RATE * 100, 1),
                "safety_margin": round(SAFETY_MARGIN * 100, 1),
                "enterprise_value": round(enterprise_value, 0),
                "equity_value": round(equity_value_safe, 0),
                "total_debt": round(total_debt, 0),
                "cash": round(cash, 0),
                "shares_outstanding": round(shares, 0),
                "pv_stage1": round(dcf_result["pv_stage1"], 0),
                "pv_stage2": round(dcf_result["pv_stage2"], 0),
                "pv_terminal": round(dcf_result["pv_terminal"], 0),
            },
            "error": None,
        }

    except Exception as e:
        logger.error("[dcf] calculation failed for %s: %s", ticker, e, exc_info=True)
        return _error_result(f"DCF calculation error: {str(e)}")


# ── 3-Stage DCF Model ─────────────────────────────────────────────────────────

def _three_stage_dcf(base_fcf: float, growth_rate: float) -> dict:
    """
    3-stage DCF:
      Stage 1 (years 1-5):  High growth (capped at MAX_GROWTH_RATE)
      Stage 2 (years 6-10): Decaying growth (stage1 × GROWTH_DECAY)
      Terminal:              Perpetuity at TERMINAL_GROWTH

    Returns:
        dict with enterprise_value and PV breakdown per stage.
    """
    g1 = min(growth_rate, MAX_GROWTH_RATE)
    g2 = g1 * GROWTH_DECAY
    r = DISCOUNT_RATE

    # Stage 1: Years 1-5
    pv_stage1 = 0
    projected_fcf = base_fcf
    for year in range(1, STAGE1_YEARS + 1):
        projected_fcf *= (1 + g1)
        pv_stage1 += projected_fcf / ((1 + r) ** year)

    # Stage 2: Years 6-10
    pv_stage2 = 0
    for year in range(STAGE1_YEARS + 1, STAGE1_YEARS + STAGE2_YEARS + 1):
        projected_fcf *= (1 + g2)
        pv_stage2 += projected_fcf / ((1 + r) ** year)

    # Terminal value (Gordon Growth Model)
    terminal_fcf = projected_fcf * (1 + TERMINAL_GROWTH)
    terminal_value = terminal_fcf / (r - TERMINAL_GROWTH)
    pv_terminal = terminal_value / ((1 + r) ** (STAGE1_YEARS + STAGE2_YEARS))

    enterprise_value = pv_stage1 + pv_stage2 + pv_terminal

    return {
        "enterprise_value": enterprise_value,
        "pv_stage1": pv_stage1,
        "pv_stage2": pv_stage2,
        "pv_terminal": pv_terminal,
    }


# ── Growth Rate Estimation ────────────────────────────────────────────────────

def _estimate_growth_rate(
    fcf_series: list[float],
    rev_series: list[float],
    computed: dict,
) -> float:
    """
    Estimate forward growth rate from historical data.
    Priority: FCF CAGR → Revenue CAGR → computed fields → fallback 5%.
    Uses 75% of historical rate (conservative estimate).
    """
    # Try FCF CAGR (need at least 3 years)
    fcf_cagr = _calc_cagr(fcf_series, min_years=3)
    if fcf_cagr is not None and fcf_cagr > 0:
        rate = fcf_cagr * 0.75
        logger.info("[dcf] using FCF CAGR: %.1f%% → projected %.1f%%", fcf_cagr * 100, rate * 100)
        return min(rate, MAX_GROWTH_RATE)

    # Try Revenue CAGR
    rev_cagr = _calc_cagr(rev_series, min_years=3)
    if rev_cagr is not None and rev_cagr > 0:
        rate = rev_cagr * 0.75
        logger.info("[dcf] using Revenue CAGR: %.1f%% → projected %.1f%%", rev_cagr * 100, rate * 100)
        return min(rate, MAX_GROWTH_RATE)

    # Try pre-computed fields from normalizer
    for key in ["rev_cagr_3y", "ni_cagr_3y", "revenue_growth"]:
        val = computed.get(key)
        if val is not None and val > 0:
            rate = val * 0.75
            logger.info("[dcf] using computed %s: %.1f%% → projected %.1f%%", key, val * 100, rate * 100)
            return min(rate, MAX_GROWTH_RATE)

    # Fallback: conservative 5%
    logger.info("[dcf] no historical growth data, using fallback 5%%")
    return 0.05


def _calc_cagr(series: list[float], min_years: int = 3) -> float | None:
    """
    Calculate Compound Annual Growth Rate from a time series.
    Series is assumed newest-first (index 0 = latest).
    Returns None if insufficient data or invalid values.
    """
    # Filter out None/zero/negative values
    valid = [v for v in series if v is not None and v > 0]

    if len(valid) < min_years:
        return None

    newest = valid[0]
    oldest = valid[-1]

    if oldest <= 0 or newest <= 0:
        return None

    years = len(valid) - 1
    if years <= 0:
        return None

    try:
        cagr = (newest / oldest) ** (1 / years) - 1
        return cagr
    except (ValueError, ZeroDivisionError):
        return None


# ── Error helper ──────────────────────────────────────────────────────────────

def _error_result(message: str) -> dict:
    """Return a standard error result dict."""
    logger.warning("[dcf] %s", message)
    return {
        "fair_value_per_share": None,
        "current_price": None,
        "upside_pct": None,
        "verdict": "N/A",
        "details": {},
        "error": message,
    }
