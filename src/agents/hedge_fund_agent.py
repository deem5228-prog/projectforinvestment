# Hedge fund strategy and portfolio allocation agent
"""
Hedge Fund Agent
================
แนวคิด: มองหา catalyst + momentum + relative value + short squeeze opportunity
        วิเคราะห์แบบ multi-strategy hedge fund (L/S equity, event-driven, macro overlay)
"""

import json
import logging
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing_extensions import Literal
import numpy as np
import pandas as pd

from src.llm import call_llm_json

from src.data.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)

from src.utils.line_item_helpers import get_metric, get_metric_series

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class HedgeFundSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


# ── Entry point ───────────────────────────────────────────────────────────────

def hedge_fund_agent(ticker: str, end_date: str, normalized_data: dict | None = None) -> dict:
    """
    Analyze a single ticker using hedge fund multi-strategy framework.
    Returns {"signal": ..., "confidence": ..., "reasoning": ...}
    """
    logger.info("[hedge_fund] analyzing %s", ticker)

    start_date_1y = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()
    start_date_3m = (datetime.fromisoformat(end_date) - timedelta(days=90)).date().isoformat()

    # ── Fetch data ────────────────────────────────────────────────────────────
    # Use pre-normalized data if provided by judge_agent, otherwise fetch directly.
    if normalized_data:
        metrics    = normalized_data["metrics"]
        line_items = normalized_data["line_items"]
        prices_df  = normalized_data["prices_df"]
        market_cap = normalized_data["market_cap"]
    else:
        prices = get_prices(ticker, start_date_1y, end_date)
        prices_df = prices_to_df(prices) if prices else pd.DataFrame()
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=8)
        line_items = search_line_items(
            ticker,
            [
                "revenue", "net_income", "operating_income", "gross_profit",
                "free_cash_flow", "total_debt", "cash_and_equivalents",
                "total_assets", "total_equity", "capital_expenditure",
                "shares_outstanding", "research_and_development",
            ],
            end_date,
            period="ttm",
            limit=8,
        )
        market_cap = get_market_cap(ticker, end_date)

    # News and insider trades are not provided by normalizer — always fetch.
    news = get_company_news(ticker, end_date=end_date, start_date=start_date_3m, limit=50)
    insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date_1y)

    # Enrich news with LLM-based sentiment (yfinance doesn't provide it).
    try:
        from src.utils.sentiment import enrich_news_sentiment
        news = enrich_news_sentiment(news)
    except Exception as e:
        logger.warning("[hedge_fund] sentiment enrichment failed, using raw news: %s", e)

    # ── Run sub-analyses ──────────────────────────────────────────────────────
    momentum        = analyze_momentum(prices_df)
    catalyst        = analyze_catalyst(news, line_items, metrics)
    relative_value  = analyze_relative_value(metrics, line_items, market_cap)
    short_squeeze   = analyze_short_squeeze_potential(prices_df, metrics)
    earnings_quality = analyze_earnings_quality(line_items, metrics)
    flow_signal     = analyze_flow_signal(insider_trades, news)
    macro_overlay   = analyze_macro_overlay(prices_df, metrics)

    total = (
        momentum["score"] + catalyst["score"] + relative_value["score"]
        + short_squeeze["score"] + earnings_quality["score"]
        + flow_signal["score"] + macro_overlay["score"]
    )
    max_total = (
        momentum["max_score"] + catalyst["max_score"] + relative_value["max_score"]
        + short_squeeze["max_score"] + earnings_quality["max_score"]
        + flow_signal["max_score"] + macro_overlay["max_score"]
    )

    analysis_data = {
        "ticker": ticker,
        "score": total,
        "max_score": max_total,
        "momentum": momentum,
        "catalyst": catalyst,
        "relative_value": relative_value,
        "short_squeeze": short_squeeze,
        "earnings_quality": earnings_quality,
        "flow_signal": flow_signal,
        "macro_overlay": macro_overlay,
        "market_cap": market_cap,
    }

    output = _generate_llm_output(ticker, analysis_data)
    logger.info("[hedge_fund] %s → %s (%d%%)", ticker, output.signal, output.confidence)
    return {"signal": output.signal, "confidence": output.confidence, "reasoning": output.reasoning}


# ── LLM call ──────────────────────────────────────────────────────────────────

def _generate_llm_output(ticker: str, analysis_data: dict) -> HedgeFundSignal:
    facts = {
        "score": analysis_data["score"],
        "max_score": analysis_data["max_score"],
        "momentum": analysis_data["momentum"]["details"],
        "catalyst": analysis_data["catalyst"]["details"],
        "relative_value": analysis_data["relative_value"]["details"],
        "short_squeeze": analysis_data["short_squeeze"]["details"],
        "earnings_quality": analysis_data["earnings_quality"]["details"],
        "flow_signal": analysis_data["flow_signal"]["details"],
        "macro_overlay": analysis_data["macro_overlay"]["details"],
        "market_cap": analysis_data["market_cap"],
    }

    system_prompt = (
        "You are a seasoned long/short equity hedge fund PM. "
        "Decide bullish (long), bearish (short), or neutral using only the provided facts.\n\n"
        "Strategy lenses to weigh:\n"
        "1. Momentum — price trend, 52-week position, moving average alignment\n"
        "2. Catalyst — upcoming earnings beats, product launches, M&A, news sentiment\n"
        "3. Relative Value — EV/EBITDA, FCF yield, PEG vs peers\n"
        "4. Short Squeeze — high short interest, low float, potential covering rally\n"
        "5. Earnings Quality — FCF conversion, accruals, revenue sustainability\n"
        "6. Flow Signal — insider buying, institutional accumulation signals\n"
        "7. Macro Overlay — beta, vol regime, sector rotation signal\n\n"
        "Signal rules:\n"
        "- Bullish: strong momentum + positive catalyst + not overvalued, OR short squeeze setup\n"
        "- Bearish: broken momentum + negative catalyst OR deteriorating earnings quality\n"
        "- Neutral: conflicting signals or no clear edge\n\n"
        "Confidence scale:\n"
        "- 85-100%: High-conviction multi-factor alignment\n"
        "- 65-84%: 2-3 factors aligned, manageable risk\n"
        "- 45-64%: Weak or conflicting signals\n"
        "- 10-44%: Against thesis or insufficient data\n\n"
        "Keep reasoning under 150 characters. Use hedge fund vocabulary: "
        "catalyst, convexity, squeeze, re-rate, alpha, flow. "
        "Do not invent data. Return JSON only."
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
        return HedgeFundSignal(**data)
    except Exception as e:
        logger.warning("[hedge_fund] LLM output parse failed: %s", e)
        return HedgeFundSignal(**fallback)


###############################################################################
# Sub-analysis functions
###############################################################################

def analyze_momentum(prices_df: pd.DataFrame) -> dict:
    """
    Price momentum: trend alignment across 20/50/200-day MAs,
    52-week position, RSI, and recent 3-month return.
    """
    if prices_df.empty or len(prices_df) < 20:
        return {"score": 0, "max_score": 10, "details": "Insufficient price data for momentum"}

    score = 0
    reasoning = []
    closes = prices_df["close"]

    # ── Moving average alignment ───────────────────────────────────────────
    ma20  = closes.rolling(20).mean().iloc[-1]  if len(closes) >= 20  else None
    ma50  = closes.rolling(50).mean().iloc[-1]  if len(closes) >= 50  else None
    ma200 = closes.rolling(200).mean().iloc[-1] if len(closes) >= 200 else None
    current = float(closes.iloc[-1])

    aligned = 0
    if ma20  and current > ma20:  aligned += 1
    if ma50  and current > ma50:  aligned += 1
    if ma200 and current > ma200: aligned += 1

    if aligned == 3:
        score += 4
        reasoning.append("Price above 20/50/200 MA — strong bullish alignment")
    elif aligned == 2:
        score += 2
        reasoning.append(f"Price above {aligned}/3 MAs — moderate momentum")
    elif aligned == 1:
        score += 1
        reasoning.append("Weak momentum — price below most MAs")
    else:
        reasoning.append("Price below all MAs — bearish momentum")

    # ── 52-week position ───────────────────────────────────────────────────
    if len(closes) >= 252:
        high_52w = float(closes.iloc[-252:].max())
        low_52w  = float(closes.iloc[-252:].min())
    else:
        high_52w = float(closes.max())
        low_52w  = float(closes.min())

    pct_from_high = (current - high_52w) / high_52w if high_52w > 0 else 0
    range_52w = high_52w - low_52w
    position_in_range = (current - low_52w) / range_52w if range_52w > 0 else 0.5

    if position_in_range > 0.80:
        score += 2
        reasoning.append(f"Near 52-week high ({position_in_range:.0%} of range) — breakout potential")
    elif position_in_range > 0.50:
        score += 1
        reasoning.append(f"Upper half of 52-week range ({position_in_range:.0%})")
    else:
        reasoning.append(f"Lower half of 52-week range ({position_in_range:.0%}) — no breakout")

    # ── 3-month momentum return ────────────────────────────────────────────
    if len(closes) >= 63:
        ret_3m = (current / float(closes.iloc[-63]) - 1)
        if ret_3m > 0.15:
            score += 2
            reasoning.append(f"Strong 3M return ({ret_3m:.1%})")
        elif ret_3m > 0.05:
            score += 1
            reasoning.append(f"Positive 3M return ({ret_3m:.1%})")
        elif ret_3m < -0.15:
            reasoning.append(f"Weak 3M return ({ret_3m:.1%}) — negative momentum")
        else:
            reasoning.append(f"Flat 3M return ({ret_3m:.1%})")

    # ── RSI (14-day) ───────────────────────────────────────────────────────
    if len(closes) >= 15:
        delta = closes.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = float((100 - 100 / (1 + rs)).iloc[-1])

        if 40 <= rsi <= 65:
            score += 2
            reasoning.append(f"RSI {rsi:.0f} — healthy momentum zone")
        elif 65 < rsi <= 75:
            score += 1
            reasoning.append(f"RSI {rsi:.0f} — strong but watch overbought")
        elif rsi > 75:
            reasoning.append(f"RSI {rsi:.0f} — overbought, potential reversal")
        elif rsi < 30:
            score += 1
            reasoning.append(f"RSI {rsi:.0f} — oversold, potential bounce")
        else:
            reasoning.append(f"RSI {rsi:.0f} — weak/neutral zone")

    return {"score": min(score, 10), "max_score": 10, "details": "; ".join(reasoning)}


def analyze_catalyst(news: list, line_items: list, metrics: list) -> dict:
    """
    Catalyst analysis: news sentiment trend, revenue acceleration,
    earnings surprise proxy, and FCF inflection.
    """
    score = 0
    reasoning = []

    # ── News sentiment ────────────────────────────────────────────────────
    if news:
        total = len(news)
        pos = sum(1 for n in news if n.sentiment and n.sentiment.lower() in ["positive", "bullish"])
        neg = sum(1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"])
        pos_ratio = pos / total
        neg_ratio = neg / total

        if pos_ratio > 0.5:
            score += 3
            reasoning.append(f"Strong positive news flow ({pos_ratio:.0%} positive, {total} articles)")
        elif pos_ratio > 0.3:
            score += 2
            reasoning.append(f"Moderate positive sentiment ({pos_ratio:.0%} positive)")
        elif neg_ratio > 0.5:
            reasoning.append(f"Negative news dominating ({neg_ratio:.0%} negative) — headwind")
        else:
            score += 1
            reasoning.append(f"Mixed/neutral news flow ({total} articles)")
    else:
        reasoning.append("No recent news data")

    # ── Revenue acceleration ──────────────────────────────────────────────
    rev_vals = get_metric_series(line_items, "revenue")

    if not rev_vals:
        rev_vals = [m.price_to_sales_ratio for m in metrics if m.price_to_sales_ratio]

    # Try growth from metrics
    rev_growths = [m.revenue_growth for m in metrics if m.revenue_growth is not None]
    if rev_growths:
        latest_growth = rev_growths[0]
        if latest_growth > 0.20:
            score += 3
            reasoning.append(f"Revenue accelerating ({latest_growth:.1%} YoY) — strong catalyst")
        elif latest_growth > 0.08:
            score += 2
            reasoning.append(f"Healthy revenue growth ({latest_growth:.1%} YoY)")
        elif latest_growth > 0:
            score += 1
            reasoning.append(f"Slow revenue growth ({latest_growth:.1%} YoY)")
        else:
            reasoning.append(f"Revenue declining ({latest_growth:.1%} YoY) — negative catalyst")

    # ── FCF inflection ────────────────────────────────────────────────────
    fcf_vals = get_metric_series(line_items, "free_cash_flow")
    if len(fcf_vals) >= 2:
        if fcf_vals[0] > 0 and fcf_vals[1] < 0:
            score += 2
            reasoning.append("FCF turned positive — inflection catalyst")
        elif fcf_vals[0] > fcf_vals[1] * 1.20:
            score += 1
            reasoning.append(f"FCF improving >20% — positive catalyst")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning)}


def analyze_relative_value(metrics: list, line_items: list, market_cap) -> dict:
    """
    Relative value: EV/EBITDA, FCF yield, PEG ratio, P/S vs growth.
    Lower is cheaper → higher score.
    """
    if not metrics:
        return {"score": 0, "max_score": 8, "details": "No metrics for relative value"}

    score = 0
    reasoning = []
    m = metrics[0]

    # ── EV/EBITDA ─────────────────────────────────────────────────────────
    ev_ebitda = getattr(m, "ev_to_ebitda", None)
    if ev_ebitda is not None:
        if ev_ebitda < 10:
            score += 3
            reasoning.append(f"Cheap EV/EBITDA {ev_ebitda:.1f}x — value opportunity")
        elif ev_ebitda < 18:
            score += 2
            reasoning.append(f"Fair EV/EBITDA {ev_ebitda:.1f}x")
        elif ev_ebitda < 30:
            score += 1
            reasoning.append(f"Slightly rich EV/EBITDA {ev_ebitda:.1f}x")
        else:
            reasoning.append(f"Expensive EV/EBITDA {ev_ebitda:.1f}x — valuation risk")
    else:
        reasoning.append("EV/EBITDA not available")

    # ── FCF yield ─────────────────────────────────────────────────────────
    fcf_yield = None
    if market_cap and market_cap > 0:
        fcf = get_metric(line_items, "free_cash_flow")
        if fcf:
            fcf_yield = fcf / market_cap

    if fcf_yield is None:
        fcf_yield = getattr(m, "free_cash_flow_yield", None)

    if fcf_yield is not None:
        if fcf_yield > 0.08:
            score += 3
            reasoning.append(f"High FCF yield {fcf_yield:.1%} — re-rate potential")
        elif fcf_yield > 0.04:
            score += 2
            reasoning.append(f"Decent FCF yield {fcf_yield:.1%}")
        elif fcf_yield > 0:
            score += 1
            reasoning.append(f"Low FCF yield {fcf_yield:.1%}")
        else:
            reasoning.append(f"Negative FCF yield {fcf_yield:.1%} — cash burning")
    else:
        reasoning.append("FCF yield not available")

    # ── PEG ratio ─────────────────────────────────────────────────────────
    peg = getattr(m, "peg_ratio", None)
    if peg is not None:
        if 0 < peg < 1.0:
            score += 2
            reasoning.append(f"PEG {peg:.2f} — growth at reasonable price")
        elif peg < 1.5:
            score += 1
            reasoning.append(f"PEG {peg:.2f} — fairly valued for growth")
        else:
            reasoning.append(f"PEG {peg:.2f} — expensive relative to growth")
    else:
        # Estimate from P/E and earnings growth
        pe = getattr(m, "price_to_earnings_ratio", None)
        eg = getattr(m, "earnings_growth", None)
        if pe and eg and eg > 0:
            est_peg = pe / (eg * 100)
            if est_peg < 1.0:
                score += 1
                reasoning.append(f"Estimated PEG {est_peg:.2f} — reasonable")
            else:
                reasoning.append(f"Estimated PEG {est_peg:.2f} — stretched")
        else:
            reasoning.append("PEG not available")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning)}


def analyze_short_squeeze_potential(prices_df: pd.DataFrame, metrics: list) -> dict:
    """
    Short squeeze setup: rapid price recovery from lows + low float signals
    + beta spike (proxy for high short interest environment).
    """
    score = 0
    reasoning = []

    if prices_df.empty or len(prices_df) < 20:
        return {"score": 0, "max_score": 6, "details": "Insufficient price data for squeeze analysis"}

    closes = prices_df["close"]
    returns = closes.pct_change().dropna()

    # ── Recovery from recent low ──────────────────────────────────────────
    recent_low = float(closes.iloc[-20:].min())
    current = float(closes.iloc[-1])
    recovery = (current - recent_low) / recent_low if recent_low > 0 else 0

    if recovery > 0.20:
        score += 2
        reasoning.append(f"Strong recovery from recent low (+{recovery:.1%}) — squeeze possible")
    elif recovery > 0.10:
        score += 1
        reasoning.append(f"Moderate recovery from recent low (+{recovery:.1%})")
    else:
        reasoning.append(f"No meaningful recovery from recent low ({recovery:.1%})")

    # ── Volatility spike (proxy for squeeze environment) ───────────────────
    if len(returns) >= 30:
        vol_recent = float(returns.iloc[-10:].std())
        vol_baseline = float(returns.iloc[-30:].std())
        vol_ratio = vol_recent / vol_baseline if vol_baseline > 0 else 1.0

        if vol_ratio > 1.8:
            score += 2
            reasoning.append(f"Volatility spike ({vol_ratio:.1f}x baseline) — squeeze environment")
        elif vol_ratio > 1.3:
            score += 1
            reasoning.append(f"Elevated volatility ({vol_ratio:.1f}x) — potential squeeze setup")
        else:
            reasoning.append(f"Normal volatility ({vol_ratio:.1f}x) — low squeeze risk")

    # ── Beta as high-short-interest proxy ────────────────────────────────
    m = metrics[0] if metrics else None
    beta = getattr(m, "beta", None) if m else None
    if beta is not None:
        if beta > 1.5:
            score += 2
            reasoning.append(f"High beta ({beta:.2f}) — amplified move if squeeze triggers")
        elif beta > 1.0:
            score += 1
            reasoning.append(f"Above-market beta ({beta:.2f}) — leveraged upside on squeeze")
        else:
            reasoning.append(f"Low beta ({beta:.2f}) — unlikely squeeze candidate")
    else:
        reasoning.append("Beta data not available")

    return {"score": min(score, 6), "max_score": 6, "details": "; ".join(reasoning)}


def analyze_earnings_quality(line_items: list, metrics: list) -> dict:
    """
    Earnings quality: FCF-to-net-income conversion, accruals ratio,
    revenue vs income growth divergence.
    """
    if not line_items:
        return {"score": 0, "max_score": 8, "details": "No line items for earnings quality"}

    score = 0
    reasoning = []
    # ── FCF / Net income conversion ───────────────────────────────────────
    net_income = get_metric(line_items, "net_income")
    fcf = get_metric(line_items, "free_cash_flow")

    if net_income and fcf and net_income != 0:
        conversion = fcf / net_income
        if conversion > 1.1:
            score += 3
            reasoning.append(f"Excellent FCF conversion ({conversion:.1f}x net income) — high quality earnings")
        elif conversion > 0.8:
            score += 2
            reasoning.append(f"Good FCF conversion ({conversion:.1f}x)")
        elif conversion > 0.5:
            score += 1
            reasoning.append(f"Moderate FCF conversion ({conversion:.1f}x)")
        else:
            reasoning.append(f"Poor FCF conversion ({conversion:.1f}x) — accrual-heavy earnings")
    else:
        reasoning.append("FCF/net income data not available")

    # ── Accruals ratio (net income - FCF) / total assets ─────────────────
    total_assets = get_metric(line_items, "total_assets")
    if net_income and fcf and total_assets and total_assets > 0:
        accruals_ratio = (net_income - fcf) / total_assets
        if abs(accruals_ratio) < 0.05:
            score += 2
            reasoning.append(f"Low accruals ratio ({accruals_ratio:.3f}) — clean earnings")
        elif abs(accruals_ratio) < 0.10:
            score += 1
            reasoning.append(f"Moderate accruals ({accruals_ratio:.3f})")
        else:
            reasoning.append(f"High accruals ({accruals_ratio:.3f}) — earnings quality concern")
    else:
        reasoning.append("Accruals analysis unavailable")

    # ── Revenue vs net income growth divergence ───────────────────────────
    if metrics:
        rev_growth = getattr(metrics[0], "revenue_growth", None)
        earn_growth = getattr(metrics[0], "earnings_growth", None)
        if rev_growth is not None and earn_growth is not None:
            if earn_growth > rev_growth + 0.05:
                score += 2
                reasoning.append(
                    f"Earnings growing faster than revenue ({earn_growth:.1%} vs {rev_growth:.1%}) — operating leverage"
                )
            elif earn_growth > rev_growth:
                score += 1
                reasoning.append(f"Earnings growth slightly ahead of revenue")
            elif earn_growth < rev_growth - 0.10:
                reasoning.append(
                    f"Earnings lagging revenue ({earn_growth:.1%} vs {rev_growth:.1%}) — margin compression"
                )
            else:
                reasoning.append(f"Revenue and earnings growth in line")

    # ── Gross margin trend ────────────────────────────────────────────────
    gm_vals = [m.gross_margin for m in metrics if m.gross_margin is not None]
    if len(gm_vals) >= 2:
        gm_trend = gm_vals[0] - gm_vals[-1]
        if gm_trend > 0.03:
            score += 1
            reasoning.append(f"Gross margin expanding ({gm_vals[0]:.1%}) — pricing power intact")
        elif gm_trend < -0.03:
            reasoning.append(f"Gross margin compressing ({gm_vals[0]:.1%}) — cost pressure")

    return {"score": min(score, 8), "max_score": 8, "details": "; ".join(reasoning)}


def analyze_flow_signal(insider_trades: list, news: list) -> dict:
    """
    Flow signals: insider net buying direction + buying urgency
    (large single transactions = high conviction).
    """
    score = 0
    reasoning = []

    # ── Insider flow ──────────────────────────────────────────────────────
    if not insider_trades:
        score += 1
        reasoning.append("No insider trade data — neutral assumption")
    else:
        total_trades = len(insider_trades)
        buy_trades = [t for t in insider_trades if (t.shares or 0) > 0]
        sell_trades = [t for t in insider_trades if (t.shares or 0) < 0]

        shares_bought = sum(t.shares or 0 for t in buy_trades)
        shares_sold = abs(sum(t.shares or 0 for t in sell_trades))
        net_shares = shares_bought - shares_sold

        # Net direction score
        if net_shares > 0:
            buy_sell_ratio = shares_bought / max(shares_sold, 1)
            if buy_sell_ratio > 3:
                score += 4
                reasoning.append(
                    f"Strong insider buying ({buy_sell_ratio:.1f}x buys vs sells) — high conviction"
                )
            elif buy_sell_ratio > 1.5:
                score += 3
                reasoning.append(f"Insider accumulation (ratio {buy_sell_ratio:.1f}x)")
            else:
                score += 2
                reasoning.append(f"Net insider buying ({net_shares:,} shares)")
        else:
            if len(sell_trades) > len(buy_trades) * 2:
                reasoning.append("Heavy insider selling — negative flow signal")
            else:
                score += 1
                reasoning.append("Slight insider selling — may be routine diversification")

        # Urgency: any large single buy (>0.5% of company implied by price)
        big_buys = [t for t in buy_trades if (t.price_per_share or 0) * (t.shares or 0) > 500_000]
        if big_buys:
            score += 1
            reasoning.append(f"{len(big_buys)} large insider buy transaction(s) — conviction signal")

    # ── News volume as proxy for institutional attention ───────────────────
    if news:
        if len(news) > 30:
            score += 1
            reasoning.append(f"High news volume ({len(news)} articles) — institutional attention")
        elif len(news) > 10:
            reasoning.append(f"Normal news flow ({len(news)} articles)")
        else:
            reasoning.append(f"Low news coverage ({len(news)} articles) — under the radar")

    return {"score": min(score, 6), "max_score": 6, "details": "; ".join(reasoning)}


def analyze_macro_overlay(prices_df: pd.DataFrame, metrics: list) -> dict:
    """
    Macro overlay: beta regime, correlation stability, sector rotation proxy
    via momentum relative to vol. High score = favorable macro setup.
    """
    score = 0
    reasoning = []

    if prices_df.empty or len(prices_df) < 30:
        return {"score": 1, "max_score": 6, "details": "Insufficient data for macro overlay"}

    closes = prices_df["close"]
    returns = closes.pct_change().dropna()

    # ── Trend stability (% of last 60 days price was above 20-day MA) ────
    if len(closes) >= 60:
        ma20 = closes.rolling(20).mean()
        above_ma = (closes.iloc[-60:] > ma20.iloc[-60:]).sum()
        pct_above = above_ma / 60

        if pct_above > 0.75:
            score += 2
            reasoning.append(f"Price above 20MA {pct_above:.0%} of past 60 days — strong trend")
        elif pct_above > 0.50:
            score += 1
            reasoning.append(f"Price above 20MA {pct_above:.0%} of past 60 days — moderate trend")
        else:
            reasoning.append(f"Price below 20MA most of past 60 days ({pct_above:.0%}) — downtrend")

    # ── Vol-adjusted momentum (Sharpe proxy over 63 days) ────────────────
    if len(returns) >= 63:
        ret_63 = returns.iloc[-63:]
        mean_ret = float(ret_63.mean())
        std_ret = float(ret_63.std())
        sharpe_proxy = (mean_ret / std_ret * (252 ** 0.5)) if std_ret > 0 else 0

        if sharpe_proxy > 1.0:
            score += 2
            reasoning.append(f"Strong vol-adjusted momentum (Sharpe proxy {sharpe_proxy:.2f})")
        elif sharpe_proxy > 0.3:
            score += 1
            reasoning.append(f"Positive risk-adjusted return ({sharpe_proxy:.2f})")
        elif sharpe_proxy < -0.5:
            reasoning.append(f"Negative risk-adjusted return ({sharpe_proxy:.2f}) — unfavorable macro")
        else:
            reasoning.append(f"Flat risk-adjusted momentum ({sharpe_proxy:.2f})")

    # ── Beta signal from metrics ──────────────────────────────────────────
    m = metrics[0] if metrics else None
    beta = getattr(m, "beta", None) if m else None
    if beta is not None:
        if 0.8 <= beta <= 1.3:
            score += 1
            reasoning.append(f"Market-aligned beta ({beta:.2f}) — good for risk-on regime")
        elif beta > 1.8:
            reasoning.append(f"Very high beta ({beta:.2f}) — macro risk amplifier")
        elif beta < 0.5:
            reasoning.append(f"Defensive beta ({beta:.2f}) — limited upside in risk-on")
        else:
            score += 1
            reasoning.append(f"Beta {beta:.2f} — acceptable macro exposure")

    # ── Drawdown recovery signal ──────────────────────────────────────────
    if len(closes) >= 60:
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        dd = float(((cum - running_max) / running_max).iloc[-1])
        if dd > -0.05:
            score += 1
            reasoning.append(f"Near all-time high (drawdown {dd:.1%}) — macro tailwind")
        elif dd < -0.25:
            reasoning.append(f"Deep drawdown ({dd:.1%}) — macro headwind or sector out of favor")

    return {"score": min(score, 6), "max_score": 6, "details": "; ".join(reasoning)}