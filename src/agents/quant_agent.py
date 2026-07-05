# Quantitative analysis and technical indicators agent
"""
Quant Agent
===========
แนวคิด: วิเคราะห์เชิงปริมาณล้วนๆ ไม่มีความเห็นเชิงคุณภาพ
        ใช้ DCF, relative multiples, factor scoring (value/quality/momentum/low-vol),
        mean reversion z-score, และ statistical anomaly detection
"""

import json
import logging
import math
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing_extensions import Literal
import numpy as np
import pandas as pd

from src.llm import call_llm_json

from src.data.api import (
    get_financial_metrics,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)
from src.utils.line_item_helpers import get_metric, get_metric_series

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class QuantSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


# ── Entry point ───────────────────────────────────────────────────────────────

def quant_agent(ticker: str, end_date: str, normalized_data: dict | None = None) -> dict:
    """
    Analyze a single ticker using quantitative multi-factor framework.
    Returns {"signal": ..., "confidence": ..., "reasoning": ...}
    """
    logger.info("[quant] analyzing %s", ticker)

    start_date_2y = (datetime.fromisoformat(end_date) - timedelta(days=730)).date().isoformat()
    start_date_1y = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    # ── Fetch data ────────────────────────────────────────────────────────
    # Use pre-normalized data if provided by judge_agent, otherwise fetch directly.
    if normalized_data:
        metrics    = normalized_data["metrics"]
        line_items = normalized_data["line_items"]
        prices_df  = normalized_data["prices_df"]
        market_cap = normalized_data["market_cap"]
    else:
        prices = get_prices(ticker, start_date_2y, end_date)
        prices_df = prices_to_df(prices) if prices else pd.DataFrame()
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)
        line_items = search_line_items(
            ticker,
            [
                "revenue", "net_income", "operating_income", "gross_profit",
                "free_cash_flow", "operating_cash_flow", "capital_expenditure",
                "total_debt", "cash_and_equivalents", "total_assets", "total_equity",
                "shares_outstanding", "depreciation_and_amortization",
                "research_and_development", "interest_expense",
            ],
            end_date,
            period="ttm",
            limit=10,
        )
        market_cap = get_market_cap(ticker, end_date)

    # ── Run sub-analyses ──────────────────────────────────────────────────
    dcf             = analyze_dcf_valuation(line_items, market_cap)
    multiples       = analyze_multiples(metrics, line_items, market_cap)
    factor_score    = analyze_multifactor(metrics, line_items, prices_df)
    mean_reversion  = analyze_mean_reversion(prices_df)
    quality_screen  = analyze_quality_screen(metrics, line_items)
    statistical_edge = analyze_statistical_edge(prices_df)
    capital_efficiency = analyze_capital_efficiency(line_items, metrics)

    total = (
        dcf["score"] + multiples["score"] + factor_score["score"]
        + mean_reversion["score"] + quality_screen["score"]
        + statistical_edge["score"] + capital_efficiency["score"]
    )
    max_total = (
        dcf["max_score"] + multiples["max_score"] + factor_score["max_score"]
        + mean_reversion["max_score"] + quality_screen["max_score"]
        + statistical_edge["max_score"] + capital_efficiency["max_score"]
    )

    analysis_data = {
        "ticker": ticker,
        "score": total,
        "max_score": max_total,
        "dcf": dcf,
        "multiples": multiples,
        "factor_score": factor_score,
        "mean_reversion": mean_reversion,
        "quality_screen": quality_screen,
        "statistical_edge": statistical_edge,
        "capital_efficiency": capital_efficiency,
        "market_cap": market_cap,
    }

    output = _generate_llm_output(ticker, analysis_data)
    logger.info("[quant] %s → %s (%d%%)", ticker, output.signal, output.confidence)
    return {"signal": output.signal, "confidence": output.confidence, "reasoning": output.reasoning}


# ── LLM call ──────────────────────────────────────────────────────────────────

def _generate_llm_output(ticker: str, analysis_data: dict) -> QuantSignal:
    score_pct = (
        analysis_data["score"] / analysis_data["max_score"] * 100
        if analysis_data["max_score"] > 0 else 50
    )

    facts = {
        "score": analysis_data["score"],
        "max_score": analysis_data["max_score"],
        "score_pct": round(score_pct, 1),
        "dcf": analysis_data["dcf"]["details"],
        "multiples": analysis_data["multiples"]["details"],
        "factor_score": analysis_data["factor_score"]["details"],
        "mean_reversion": analysis_data["mean_reversion"]["details"],
        "quality_screen": analysis_data["quality_screen"]["details"],
        "statistical_edge": analysis_data["statistical_edge"]["details"],
        "capital_efficiency": analysis_data["capital_efficiency"]["details"],
        "market_cap": analysis_data["market_cap"],
    }

    system_prompt = (
        "You are a quantitative analyst. Use only the provided numerical facts to decide "
        "bullish, bearish, or neutral. No qualitative opinion. Numbers only.\n\n"
        "Factor weights (what matters most):\n"
        "1. DCF — intrinsic value vs market cap (highest weight)\n"
        "2. Multiples — EV/EBITDA, FCF yield, P/E vs growth\n"
        "3. Multi-factor score — value + quality + momentum + low-vol\n"
        "4. Quality screen — ROIC, FCF conversion, balance sheet strength\n"
        "5. Capital efficiency — asset turnover, ROIC trend\n"
        "6. Mean reversion z-score — statistical over/undervaluation\n"
        "7. Statistical edge — price anomalies, vol patterns\n\n"
        "Signal rules:\n"
        "- Bullish: score_pct > 60% AND DCF shows undervaluation AND quality screen passes\n"
        "- Bearish: score_pct < 40% OR DCF shows >30% overvaluation OR quality screen fails\n"
        "- Neutral: 40-60% range or conflicting signals\n\n"
        "Confidence = score_pct rounded to nearest 5. "
        "Cap at 90 if any single factor is missing data. "
        "Keep reasoning under 140 characters. "
        "Use quant vocabulary: z-score, factor, alpha, ROIC, FCF yield, mean reversion. "
        "Return JSON only."
    )

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Facts:\n{json.dumps(facts, separators=(',', ':'), ensure_ascii=False)}\n\n"
        'Return exactly:\n{\n  "signal": "bullish" | "bearish" | "neutral",\n'
        '  "confidence": int,\n  "reasoning": "short justification"\n}'
    )

    fallback = {"signal": "neutral", "confidence": 50, "reasoning": "Insufficient data"}
    data = call_llm_json(system_prompt, user_prompt, fallback=fallback, max_tokens=256)
    try:
        return QuantSignal(**data)
    except Exception as e:
        logger.warning("[quant] LLM output parse failed: %s", e)
        return QuantSignal(**fallback)


###############################################################################
# Helper
###############################################################################

def _safe(value, default=0.0):
    """Safely convert to float, returning default for None/NaN/inf."""
    try:
        v = float(value)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _get_line_value(line_items: list, field: str):
    """Extract the latest non-None value of a field from line_items."""
    return get_metric(line_items, field)


###############################################################################
# Sub-analysis functions
###############################################################################

def analyze_dcf_valuation(line_items: list, market_cap) -> dict:
    """
    Three-stage DCF based on free cash flow.
    Stage 1 (5yr): growth derived from historical FCF CAGR, capped at 20%.
    Stage 2 (5yr): fade to half of stage 1.
    Terminal: 2.5% perpetuity.
    Discount rate: WACC proxy = 10%.

    Score reflects margin of safety (upside to IV).
    """
    if not line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "max_score": 12, "details": "Insufficient data for DCF",
                "intrinsic_value": None, "upside": None}

    # Pull FCF history (most recent first)
    fcf_vals = get_metric_series(line_items, "free_cash_flow")

    # Fallback: use operating_cash_flow - capex
    if not fcf_vals:
        ocf_series = get_metric_series(line_items, "operating_cash_flow")
        cx_series = get_metric_series(line_items, "capital_expenditure")
        for ocf, cx in zip(ocf_series, cx_series):
            fcf_vals.append(ocf + cx)  # capex usually negative

    if not fcf_vals or fcf_vals[0] is None:
        return {"score": 0, "max_score": 12, "details": "No FCF data for DCF",
                "intrinsic_value": None, "upside": None}

    base_fcf = _safe(fcf_vals[0])
    if base_fcf <= 0:
        return {"score": 0, "max_score": 12,
                "details": f"Negative base FCF (${base_fcf:,.0f}) — DCF not meaningful",
                "intrinsic_value": None, "upside": None}

    # Historical FCF CAGR
    if len(fcf_vals) >= 3 and _safe(fcf_vals[-1]) > 0:
        years = len(fcf_vals) - 1
        cagr = (_safe(fcf_vals[0]) / _safe(fcf_vals[-1])) ** (1 / years) - 1
        g1 = max(-0.05, min(cagr * 0.75, 0.20))   # 75% of historical, cap 20%
    else:
        g1 = 0.05   # default conservative 5%

    g2 = g1 * 0.5                      # Stage 2: fade
    g3 = 0.025                          # Terminal: GDP proxy
    r  = 0.10                           # Discount rate (WACC proxy)

    # PV stages
    pv = 0.0
    cf = base_fcf
    for yr in range(1, 6):
        cf *= (1 + g1)
        pv += cf / (1 + r) ** yr

    cf_s2 = base_fcf * (1 + g1) ** 5
    for yr in range(1, 6):
        cf_s2 *= (1 + g2)
        pv += cf_s2 / (1 + r) ** (5 + yr)

    terminal_cf = cf_s2 * (1 + g3)
    terminal_pv = (terminal_cf / (r - g3)) / (1 + r) ** 10

    iv = (pv + terminal_pv) * 0.90     # 10% haircut for model uncertainty
    upside = (iv - market_cap) / market_cap

    # Scoring
    score = 0
    if upside > 0.50:
        score = 12
        detail = f"DCF deeply undervalued: IV ${iv:,.0f} vs mkt ${market_cap:,.0f} ({upside:+.1%} upside)"
    elif upside > 0.25:
        score = 9
        detail = f"DCF undervalued: IV ${iv:,.0f} ({upside:+.1%} upside, g1={g1:.1%})"
    elif upside > 0.05:
        score = 6
        detail = f"DCF slight upside: IV ${iv:,.0f} ({upside:+.1%})"
    elif upside > -0.10:
        score = 4
        detail = f"DCF fairly valued ({upside:+.1%} to IV)"
    elif upside > -0.30:
        score = 2
        detail = f"DCF moderately overvalued ({upside:+.1%})"
    else:
        score = 0
        detail = f"DCF significantly overvalued: IV ${iv:,.0f} ({upside:+.1%})"

    return {
        "score": score, "max_score": 12,
        "details": detail,
        "intrinsic_value": iv, "upside": upside,
    }


def analyze_multiples(metrics: list, line_items: list, market_cap) -> dict:
    """
    Relative multiple scoring: EV/EBITDA, P/E, FCF yield, EV/Revenue, P/B.
    Each multiple scored vs historical norms; total gives valuation signal.
    """
    if not metrics:
        return {"score": 0, "max_score": 10, "details": "No metrics for multiple analysis"}

    score = 0
    reasoning = []
    m = metrics[0]

    # ── EV/EBITDA ─────────────────────────────────────────────────────────
    ev_ebitda = getattr(m, "ev_to_ebitda", None)
    if ev_ebitda is not None:
        if ev_ebitda < 8:
            score += 3; reasoning.append(f"EV/EBITDA {ev_ebitda:.1f}x — deep value")
        elif ev_ebitda < 14:
            score += 2; reasoning.append(f"EV/EBITDA {ev_ebitda:.1f}x — fair")
        elif ev_ebitda < 22:
            score += 1; reasoning.append(f"EV/EBITDA {ev_ebitda:.1f}x — slightly rich")
        else:
            reasoning.append(f"EV/EBITDA {ev_ebitda:.1f}x — expensive")
    else:
        reasoning.append("EV/EBITDA unavailable")

    # ── P/E adjusted for growth (PEG) ─────────────────────────────────────
    pe = getattr(m, "price_to_earnings_ratio", None)
    eg = getattr(m, "earnings_growth", None)
    peg = getattr(m, "peg_ratio", None)

    if peg is None and pe and eg and eg > 0:
        peg = pe / (eg * 100)

    if peg is not None and peg > 0:
        if peg < 0.75:
            score += 3; reasoning.append(f"PEG {peg:.2f} — growth at deep discount")
        elif peg < 1.2:
            score += 2; reasoning.append(f"PEG {peg:.2f} — reasonable for growth")
        elif peg < 2.0:
            score += 1; reasoning.append(f"PEG {peg:.2f} — slightly expensive for growth")
        else:
            reasoning.append(f"PEG {peg:.2f} — overpriced for growth rate")
    elif pe is not None:
        if pe < 12:
            score += 2; reasoning.append(f"P/E {pe:.1f}x — cheap on absolute basis")
        elif pe < 20:
            score += 1; reasoning.append(f"P/E {pe:.1f}x — reasonable")
        else:
            reasoning.append(f"P/E {pe:.1f}x — elevated")
    else:
        reasoning.append("P/E / PEG unavailable")

    # ── FCF yield ─────────────────────────────────────────────────────────
    fcf_yield = None
    if market_cap and market_cap > 0:
        fcf = _get_line_value(line_items, "free_cash_flow")
        if fcf:
            fcf_yield = fcf / market_cap
    if fcf_yield is None:
        fcf_yield = getattr(m, "free_cash_flow_yield", None)

    if fcf_yield is not None:
        if fcf_yield > 0.08:
            score += 3; reasoning.append(f"FCF yield {fcf_yield:.1%} — excellent")
        elif fcf_yield > 0.04:
            score += 2; reasoning.append(f"FCF yield {fcf_yield:.1%} — solid")
        elif fcf_yield > 0.01:
            score += 1; reasoning.append(f"FCF yield {fcf_yield:.1%} — low but positive")
        else:
            reasoning.append(f"FCF yield {fcf_yield:.1%} — negative/negligible")
    else:
        reasoning.append("FCF yield unavailable")

    # ── P/B ───────────────────────────────────────────────────────────────
    pb = getattr(m, "price_to_book_ratio", None)
    roe = getattr(m, "return_on_equity", None)
    if pb is not None and roe is not None:
        # Justified P/B = ROE / cost of equity; use ROE / 0.10 as proxy
        justified_pb = roe / 0.10
        pb_discount = (justified_pb - pb) / justified_pb if justified_pb > 0 else 0
        if pb_discount > 0.30:
            score += 1; reasoning.append(f"P/B {pb:.1f}x cheap vs justified {justified_pb:.1f}x")
        elif pb_discount < -0.30:
            reasoning.append(f"P/B {pb:.1f}x premium to justified {justified_pb:.1f}x")
    elif pb is not None:
        if pb < 1.5:
            score += 1; reasoning.append(f"P/B {pb:.1f}x — asset-cheap")

    return {"score": min(score, 10), "max_score": 10, "details": "; ".join(reasoning)}


def analyze_multifactor(metrics: list, line_items: list, prices_df: pd.DataFrame) -> dict:
    """
    Classic 4-factor model: Value + Quality + Momentum + Low-Volatility.
    Each factor scored 0-3, total max 12.
    """
    score = 0
    reasoning = []
    m = metrics[0] if metrics else None

    # ── Factor 1: Value (low multiple = cheap) ────────────────────────────
    value_score = 0
    if m:
        ev_rev = getattr(m, "ev_to_revenue", None)
        if ev_rev is not None:
            if ev_rev < 2:    value_score += 1
            elif ev_rev > 8:  value_score -= 1
        ps = getattr(m, "price_to_sales_ratio", None)
        if ps is not None:
            if ps < 2:   value_score += 1
            elif ps > 10: value_score -= 1
        pe = getattr(m, "price_to_earnings_ratio", None)
        if pe and 0 < pe < 15: value_score += 1

    value_score = max(0, min(value_score, 3))
    score += value_score
    reasoning.append(f"Value factor {value_score}/3")

    # ── Factor 2: Quality (margins + ROE + low debt) ──────────────────────
    quality_score = 0
    if m:
        roe = getattr(m, "return_on_equity", None)
        if roe and roe > 0.15: quality_score += 1

        gm = getattr(m, "gross_margin", None)
        if gm and gm > 0.35: quality_score += 1

        de = getattr(m, "debt_to_equity", None)
        if de is not None and de < 0.5: quality_score += 1

    quality_score = min(quality_score, 3)
    score += quality_score
    reasoning.append(f"Quality factor {quality_score}/3")

    # ── Factor 3: Momentum (12-1 month return) ────────────────────────────
    momentum_score = 0
    if not prices_df.empty and len(prices_df) >= 63:
        closes = prices_df["close"]
        ret_12m = _safe(closes.iloc[-1] / closes.iloc[0] - 1) if len(closes) >= 252 else None
        ret_1m  = _safe(closes.iloc[-1] / closes.iloc[-21] - 1) if len(closes) >= 21 else None
        ret_3m  = _safe(closes.iloc[-1] / closes.iloc[-63] - 1)

        # 12-1 momentum (skip last month to avoid reversal)
        if ret_12m is not None and ret_1m is not None:
            mom = ret_12m - ret_1m
        else:
            mom = ret_3m

        if mom > 0.15:
            momentum_score = 3
        elif mom > 0.05:
            momentum_score = 2
        elif mom > 0:
            momentum_score = 1

    score += momentum_score
    reasoning.append(f"Momentum factor {momentum_score}/3")

    # ── Factor 4: Low Volatility (lower vol = higher score) ───────────────
    low_vol_score = 0
    if not prices_df.empty and len(prices_df) >= 63:
        rets = prices_df["close"].pct_change().dropna()
        ann_vol = _safe(rets.iloc[-63:].std()) * math.sqrt(252)
        if ann_vol < 0.18:
            low_vol_score = 3
        elif ann_vol < 0.28:
            low_vol_score = 2
        elif ann_vol < 0.40:
            low_vol_score = 1

    score += low_vol_score
    reasoning.append(f"Low-vol factor {low_vol_score}/3 (ann vol {ann_vol:.0%})" if low_vol_score or not prices_df.empty else f"Low-vol factor {low_vol_score}/3")

    return {"score": min(score, 12), "max_score": 12, "details": "; ".join(reasoning)}


def analyze_mean_reversion(prices_df: pd.DataFrame) -> dict:
    """
    Z-score of current price vs rolling 252-day mean (in log-return space).
    Negative z-score → statistically cheap → bullish mean reversion signal.
    Also checks Bollinger Band position.
    """
    if prices_df.empty or len(prices_df) < 63:
        return {"score": 0, "max_score": 8, "details": "Insufficient price data for mean reversion"}

    score = 0
    reasoning = []
    closes = prices_df["close"]

    # ── Z-score vs rolling mean ───────────────────────────────────────────
    window = min(252, len(closes))
    roll_mean = closes.rolling(window).mean()
    roll_std  = closes.rolling(window).std()

    current = _safe(closes.iloc[-1])
    mean_val = _safe(roll_mean.iloc[-1])
    std_val  = _safe(roll_std.iloc[-1])

    if std_val > 0:
        z = (current - mean_val) / std_val
        if z < -1.5:
            score += 4
            reasoning.append(f"Z-score {z:.2f} — statistically cheap, mean reversion upside")
        elif z < -0.5:
            score += 3
            reasoning.append(f"Z-score {z:.2f} — below average, mild mean reversion signal")
        elif z < 0.5:
            score += 2
            reasoning.append(f"Z-score {z:.2f} — near mean, neutral")
        elif z < 1.5:
            score += 1
            reasoning.append(f"Z-score {z:.2f} — above average, slight reversion risk")
        else:
            reasoning.append(f"Z-score {z:.2f} — statistically expensive, reversion risk")
    else:
        z = 0.0
        reasoning.append("Insufficient variance for z-score")

    # ── Bollinger Band position (20-day, 2σ) ──────────────────────────────
    if len(closes) >= 20:
        bb_mean = _safe(closes.rolling(20).mean().iloc[-1])
        bb_std  = _safe(closes.rolling(20).std().iloc[-1])
        upper   = bb_mean + 2 * bb_std
        lower   = bb_mean - 2 * bb_std

        if bb_std > 0:
            bb_pos = (current - lower) / (upper - lower)  # 0 = lower band, 1 = upper
            if bb_pos < 0.15:
                score += 3
                reasoning.append(f"Near lower Bollinger band ({bb_pos:.0%} position) — oversold")
            elif bb_pos < 0.35:
                score += 2
                reasoning.append(f"Below BB midline ({bb_pos:.0%}) — reversion opportunity")
            elif bb_pos < 0.65:
                score += 1
                reasoning.append(f"Near BB midline ({bb_pos:.0%}) — neutral")
            elif bb_pos < 0.85:
                reasoning.append(f"Above BB midline ({bb_pos:.0%}) — slight reversion risk")
            else:
                reasoning.append(f"Near upper Bollinger band ({bb_pos:.0%}) — overbought")
        else:
            reasoning.append("Insufficient BB variance")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning)}


def analyze_quality_screen(metrics: list, line_items: list) -> dict:
    """
    Piotroski F-Score inspired quality screen (9 binary signals → 0-9).
    Maps to 0-10 score for consistency with other modules.

    Signals:
      Profitability (4): positive ROA, positive FCF, ROA improving, FCF > net income
      Leverage (3): D/E falling, current ratio rising, no share dilution
      Efficiency (2): gross margin improving, asset turnover improving
    """
    if not metrics:
        return {"score": 0, "max_score": 10, "details": "No data for quality screen"}

    signals = []
    score_bits = 0
    m0 = metrics[0]
    m1 = metrics[1] if len(metrics) > 1 else None

    # ── Profitability signals ──────────────────────────────────────────────
    roa = getattr(m0, "return_on_assets", None)
    if roa is not None:
        ok = roa > 0
        signals.append(("ROA positive", ok))
        if ok: score_bits += 1

    fcf0 = get_metric(line_items, "free_cash_flow")
    ni0  = get_metric(line_items, "net_income")
    if fcf0 is not None:
        ok = fcf0 > 0
        signals.append(("FCF positive", ok))
        if ok: score_bits += 1

    roa1 = getattr(m1, "return_on_assets", None) if m1 else None
    if roa is not None and roa1 is not None:
        ok = roa > roa1
        signals.append(("ROA improving", ok))
        if ok: score_bits += 1

    if fcf0 is not None and ni0 is not None and ni0 != 0:
        ok = fcf0 > ni0
        signals.append(("FCF > Net income (accrual check)", ok))
        if ok: score_bits += 1

    # ── Leverage signals ───────────────────────────────────────────────────
    de0 = getattr(m0, "debt_to_equity", None)
    de1 = getattr(m1, "debt_to_equity", None) if m1 else None
    if de0 is not None and de1 is not None:
        ok = de0 < de1
        signals.append(("D/E falling", ok))
        if ok: score_bits += 1

    cr0 = getattr(m0, "current_ratio", None)
    cr1 = getattr(m1, "current_ratio", None) if m1 else None
    if cr0 is not None and cr1 is not None:
        ok = cr0 > cr1
        signals.append(("Current ratio rising", ok))
        if ok: score_bits += 1

    _sh_series = get_metric_series(line_items, "shares_outstanding")
    sh0 = _sh_series[0] if len(_sh_series) > 0 else None
    sh1 = _sh_series[1] if len(_sh_series) > 1 else None
    if sh0 is not None and sh1 is not None:
        ok = sh0 <= sh1 * 1.02   # allow 2% tolerance
        signals.append(("No dilution", ok))
        if ok: score_bits += 1

    # ── Efficiency signals ────────────────────────────────────────────────
    gm0 = getattr(m0, "gross_margin", None)
    gm1 = getattr(m1, "gross_margin", None) if m1 else None
    if gm0 is not None and gm1 is not None:
        ok = gm0 > gm1
        signals.append(("Gross margin improving", ok))
        if ok: score_bits += 1

    at0 = getattr(m0, "asset_turnover", None)
    at1 = getattr(m1, "asset_turnover", None) if m1 else None
    if at0 is not None and at1 is not None:
        ok = at0 > at1
        signals.append(("Asset turnover improving", ok))
        if ok: score_bits += 1

    total_signals = len(signals)
    passed = score_bits
    pct = passed / total_signals if total_signals > 0 else 0

    # Map to 0-10
    score = round(pct * 10)
    detail_parts = [f"{name}: {'✓' if ok else '✗'}" for name, ok in signals]
    detail = f"F-Score {passed}/{total_signals} signals passed | " + "; ".join(detail_parts)

    return {"score": score, "max_score": 10, "details": detail}


def analyze_statistical_edge(prices_df: pd.DataFrame) -> dict:
    """
    Statistical anomalies in price data:
    - Autocorrelation (momentum persistence)
    - Skewness of return distribution
    - Calmar ratio (return / max drawdown)
    - Volume-price divergence
    """
    if prices_df.empty or len(prices_df) < 40:
        return {"score": 0, "max_score": 8, "details": "Insufficient data for statistical edge"}

    score = 0
    reasoning = []

    closes = prices_df["close"]
    rets = closes.pct_change().dropna()

    # ── Autocorrelation (lag-1): positive = momentum, negative = mean reversion ──
    if len(rets) >= 20:
        ac1 = _safe(rets.autocorr(lag=1))
        if ac1 > 0.10:
            score += 2
            reasoning.append(f"Positive autocorrelation ({ac1:.3f}) — momentum persistence")
        elif ac1 > 0:
            score += 1
            reasoning.append(f"Slight positive autocorrelation ({ac1:.3f})")
        elif ac1 < -0.10:
            score += 1
            reasoning.append(f"Negative autocorrelation ({ac1:.3f}) — mean reversion tendency")
        else:
            reasoning.append(f"Near-zero autocorrelation ({ac1:.3f}) — random walk")

    # ── Return skewness ───────────────────────────────────────────────────
    if len(rets) >= 30:
        skew = _safe(pd.Series(rets).skew())
        if skew > 0.5:
            score += 2
            reasoning.append(f"Positive skew ({skew:.2f}) — fat right tail favors longs")
        elif skew > 0:
            score += 1
            reasoning.append(f"Slight positive skew ({skew:.2f})")
        elif skew < -1.0:
            reasoning.append(f"Strongly negative skew ({skew:.2f}) — crash risk")
        else:
            reasoning.append(f"Slight negative skew ({skew:.2f})")

    # ── Calmar ratio (annualized return / max drawdown) ───────────────────
    if len(closes) >= 63:
        ann_ret = _safe((closes.iloc[-1] / closes.iloc[0]) ** (252 / len(closes)) - 1)
        cum = (1 + rets).cumprod()
        max_dd = _safe(((cum - cum.cummax()) / cum.cummax()).min())
        if max_dd < 0:
            calmar = ann_ret / abs(max_dd)
            if calmar > 2.0:
                score += 2
                reasoning.append(f"Calmar ratio {calmar:.2f} — excellent risk-adjusted return")
            elif calmar > 0.8:
                score += 1
                reasoning.append(f"Calmar ratio {calmar:.2f} — acceptable")
            else:
                reasoning.append(f"Calmar ratio {calmar:.2f} — poor risk/return")
        else:
            score += 1
            reasoning.append("No drawdown detected — limited history")

    # ── Volume-price divergence (price up, vol down = weak move) ─────────
    if "volume" in prices_df.columns and len(prices_df) >= 20:
        recent_ret = _safe(closes.iloc[-1] / closes.iloc[-20] - 1)
        vol_recent = _safe(prices_df["volume"].iloc[-5:].mean())
        vol_baseline = _safe(prices_df["volume"].iloc[-20:].mean())
        vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1.0

        if recent_ret > 0.05 and vol_ratio > 1.2:
            score += 2
            reasoning.append(f"Price rise confirmed by volume ({vol_ratio:.1f}x) — strong signal")
        elif recent_ret > 0.05 and vol_ratio < 0.8:
            reasoning.append(f"Price rise on low volume ({vol_ratio:.1f}x) — weak breakout")
        elif recent_ret < -0.05 and vol_ratio > 1.5:
            reasoning.append(f"Selloff on high volume ({vol_ratio:.1f}x) — distribution signal")
        else:
            score += 1
            reasoning.append(f"Normal volume pattern ({vol_ratio:.1f}x)")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning)}


def analyze_capital_efficiency(line_items: list, metrics: list) -> dict:
    """
    Capital efficiency: ROIC, reinvestment rate, capital intensity,
    and Greenblatt earnings yield + return on capital (Magic Formula proxy).
    """
    if not line_items or not metrics:
        return {"score": 0, "max_score": 8, "details": "Insufficient data for capital efficiency"}

    score = 0
    reasoning = []
    m = metrics[0]

    # ── ROIC ─────────────────────────────────────────────────────────────
    roic = getattr(m, "return_on_invested_capital", None)
    if roic is None:
        # Estimate: NOPAT / Invested Capital = op_income*(1-tax) / (equity + debt - cash)
        op_income = get_metric(line_items, "operating_income")
        equity    = get_metric(line_items, "total_equity")
        debt      = get_metric(line_items, "total_debt")
        cash      = get_metric(line_items, "cash_and_equivalents")
        if op_income and equity:
            nopat = op_income * 0.79   # assume 21% effective tax
            ic    = _safe(equity) + _safe(debt) - _safe(cash)
            roic  = nopat / ic if ic > 0 else None

    if roic is not None:
        if roic > 0.20:
            score += 3
            reasoning.append(f"Excellent ROIC {roic:.1%} — value-creating machine")
        elif roic > 0.12:
            score += 2
            reasoning.append(f"Good ROIC {roic:.1%} — above cost of capital")
        elif roic > 0.07:
            score += 1
            reasoning.append(f"Marginal ROIC {roic:.1%} — near cost of capital")
        else:
            reasoning.append(f"Poor ROIC {roic:.1%} — value destruction")
    else:
        reasoning.append("ROIC not available")

    # ── Capital intensity (capex / revenue) ──────────────────────────────
    capex = get_metric(line_items, "capital_expenditure")
    rev   = get_metric(line_items, "revenue")
    if capex is not None and rev and rev > 0:
        intensity = abs(_safe(capex)) / _safe(rev)
        if intensity < 0.03:
            score += 2
            reasoning.append(f"Low capex intensity ({intensity:.1%}) — asset-light model")
        elif intensity < 0.08:
            score += 1
            reasoning.append(f"Moderate capex intensity ({intensity:.1%})")
        else:
            reasoning.append(f"High capex intensity ({intensity:.1%}) — capital-hungry")
    else:
        reasoning.append("Capex intensity data unavailable")

    # ── Greenblatt Magic Formula proxy: earnings yield + ROC ─────────────
    ev_ebitda = getattr(m, "ev_to_ebitda", None)
    ey = 1 / ev_ebitda if ev_ebitda and ev_ebitda > 0 else None  # earnings yield proxy

    if ey is not None and roic is not None:
        mf_score = ey + roic   # simple additive Magic Formula proxy
        if mf_score > 0.35:
            score += 3
            reasoning.append(f"Magic Formula score {mf_score:.2f} (EY {ey:.1%} + ROIC {roic:.1%}) — top decile")
        elif mf_score > 0.20:
            score += 2
            reasoning.append(f"Magic Formula score {mf_score:.2f} — above average")
        elif mf_score > 0.10:
            score += 1
            reasoning.append(f"Magic Formula score {mf_score:.2f} — average")
        else:
            reasoning.append(f"Magic Formula score {mf_score:.2f} — unattractive")
    else:
        reasoning.append("Magic Formula proxy unavailable")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning)}