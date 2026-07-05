# Warren Buffett style investing agent
import json
import logging
from pydantic import BaseModel, Field
from typing_extensions import Literal

from src.llm import call_llm_json
from src.utils.line_item_helpers import get_metric, get_metric_series

from src.data.api import (
    get_financial_metrics,
    get_market_cap,
    search_line_items,
)

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class WarrenBuffettSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


# ── Entry point ───────────────────────────────────────────────────────────────

def warren_buffett_agent(ticker: str, end_date: str, normalized_data: dict | None = None) -> dict:
    """
    Analyze a single ticker using Buffett's principles.
    Returns {"signal": ..., "confidence": ..., "reasoning": ...}
    """
    logger.info("[buffett] analyzing %s", ticker)

    # Use pre-normalized data if provided by judge_agent, otherwise fetch directly.
    if normalized_data:
        metrics    = normalized_data["metrics"]
        line_items = normalized_data["line_items"]
        market_cap = normalized_data["market_cap"]
    else:
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)
        line_items = search_line_items(
            ticker,
            [
                "capital_expenditure",
                "depreciation_and_amortization",
                "net_income",
                "shares_outstanding",
                "total_assets",
                "total_liabilities",
                "total_equity",
                "gross_profit",
                "revenue",
                "free_cash_flow",
                "stock_repurchases",
                "dividends_paid",
            ],
            end_date,
            period="ttm",
            limit=10,
        )
        market_cap = get_market_cap(ticker, end_date)

    fundamental_analysis   = analyze_fundamentals(metrics)
    consistency_analysis   = analyze_consistency(line_items)
    moat_analysis          = analyze_moat(metrics)
    pricing_power_analysis = analyze_pricing_power(line_items, metrics)
    book_value_analysis    = analyze_book_value_growth(line_items)
    mgmt_analysis          = analyze_management_quality(line_items)
    intrinsic_value_analysis = calculate_intrinsic_value(line_items)

    total_score = (
        fundamental_analysis["score"]
        + consistency_analysis["score"]
        + moat_analysis["score"]
        + mgmt_analysis["score"]
        + pricing_power_analysis["score"]
        + book_value_analysis["score"]
    )
    max_possible_score = 10 + moat_analysis["max_score"] + mgmt_analysis["max_score"] + 5 + 5

    margin_of_safety = None
    iv = intrinsic_value_analysis.get("intrinsic_value")
    if iv and market_cap:
        margin_of_safety = (iv - market_cap) / market_cap

    analysis_data = {
        "ticker": ticker,
        "score": total_score,
        "max_score": max_possible_score,
        "fundamental_analysis": fundamental_analysis,
        "consistency_analysis": consistency_analysis,
        "moat_analysis": moat_analysis,
        "pricing_power_analysis": pricing_power_analysis,
        "book_value_analysis": book_value_analysis,
        "management_analysis": mgmt_analysis,
        "intrinsic_value_analysis": intrinsic_value_analysis,
        "market_cap": market_cap,
        "margin_of_safety": margin_of_safety,
    }

    output = _generate_llm_output(ticker, analysis_data)
    logger.info("[buffett] %s → %s (%d%%)", ticker, output.signal, output.confidence)
    return {"signal": output.signal, "confidence": output.confidence, "reasoning": output.reasoning}


# ── LLM call (Google Gemini) ─────────────────────────────────────────────────

def _generate_llm_output(ticker: str, analysis_data: dict) -> WarrenBuffettSignal:
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "fundamentals": analysis_data.get("fundamental_analysis", {}).get("details"),
        "consistency": analysis_data.get("consistency_analysis", {}).get("details"),
        "moat": analysis_data.get("moat_analysis", {}).get("details"),
        "pricing_power": analysis_data.get("pricing_power_analysis", {}).get("details"),
        "book_value": analysis_data.get("book_value_analysis", {}).get("details"),
        "management": analysis_data.get("management_analysis", {}).get("details"),
        "intrinsic_value": analysis_data.get("intrinsic_value_analysis", {}).get("intrinsic_value"),
        "market_cap": analysis_data.get("market_cap"),
        "margin_of_safety": analysis_data.get("margin_of_safety"),
    }

    system_prompt = (
        "You are Warren Buffett. Decide bullish, bearish, or neutral using only the provided facts.\n\n"
        "Checklist:\n"
        "- Circle of competence\n"
        "- Competitive moat\n"
        "- Management quality\n"
        "- Financial strength\n"
        "- Valuation vs intrinsic value\n"
        "- Long-term prospects\n\n"
        "Signal rules:\n"
        "- Bullish: strong business AND margin_of_safety > 0.\n"
        "- Bearish: poor business OR clearly overvalued.\n"
        "- Neutral: good business but margin_of_safety <= 0, or mixed evidence.\n\n"
        "Confidence scale:\n"
        "- 90-100%: Exceptional business, attractive price\n"
        "- 70-89%: Good business, fair valuation\n"
        "- 50-69%: Mixed signals\n"
        "- 30-49%: Concerning fundamentals\n"
        "- 10-29%: Poor business or overvalued\n\n"
        "Keep reasoning under 120 characters. Do not invent data. Return JSON only."
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
        return WarrenBuffettSignal(**data)
    except Exception as e:
        logger.warning("[buffett] LLM output parse failed: %s", e)
        return WarrenBuffettSignal(**fallback)


###############################################################################
# Analysis functions (ไม่แตะ — logic เดิมทั้งหมด)
###############################################################################

def analyze_fundamentals(metrics: list) -> dict:
    if not metrics:
        return {"score": 0, "details": "Insufficient fundamental data"}
    m = metrics[0]
    score = 0
    reasoning = []

    if m.return_on_equity and m.return_on_equity > 0.15:
        score += 2
        reasoning.append(f"Strong ROE of {m.return_on_equity:.1%}")
    elif m.return_on_equity:
        reasoning.append(f"Weak ROE of {m.return_on_equity:.1%}")
    else:
        reasoning.append("ROE data not available")

    if m.debt_to_equity and m.debt_to_equity < 0.5:
        score += 2
        reasoning.append("Conservative debt levels")
    elif m.debt_to_equity:
        reasoning.append(f"High D/E ratio of {m.debt_to_equity:.1f}")
    else:
        reasoning.append("Debt to equity data not available")

    if m.operating_margin and m.operating_margin > 0.15:
        score += 2
        reasoning.append("Strong operating margins")
    elif m.operating_margin:
        reasoning.append(f"Weak operating margin of {m.operating_margin:.1%}")
    else:
        reasoning.append("Operating margin data not available")

    if m.current_ratio and m.current_ratio > 1.5:
        score += 1
        reasoning.append("Good liquidity position")
    elif m.current_ratio:
        reasoning.append(f"Weak liquidity (current ratio {m.current_ratio:.1f})")
    else:
        reasoning.append("Current ratio data not available")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_consistency(line_items: list) -> dict:
    if len(line_items) < 4:
        return {"score": 0, "details": "Insufficient historical data"}
    score = 0
    reasoning = []

    earnings = [getattr(i, "value", None) for i in line_items if getattr(i, "line_item", None) == "net_income" and getattr(i, "value", None) is not None]
    # fallback: use model extra fields
    if not earnings:
        earnings = get_metric_series(line_items, "net_income")

    if len(earnings) >= 4:
        if all(earnings[i] > earnings[i + 1] for i in range(len(earnings) - 1)):
            score += 3
            reasoning.append("Consistent earnings growth over past periods")
        else:
            reasoning.append("Inconsistent earnings growth pattern")
        if earnings[-1] != 0:
            growth_rate = (earnings[0] - earnings[-1]) / abs(earnings[-1])
            reasoning.append(f"Total earnings growth {growth_rate:.1%} over {len(earnings)} periods")
    else:
        reasoning.append("Insufficient earnings data for trend analysis")

    return {"score": score, "details": "; ".join(reasoning)}


def analyze_moat(metrics: list) -> dict:
    if not metrics or len(metrics) < 5:
        return {"score": 0, "max_score": 5, "details": "Insufficient data for moat analysis"}
    score = 0
    reasoning = []

    roes = [m.return_on_equity for m in metrics if m.return_on_equity is not None]
    if len(roes) >= 5:
        high = sum(1 for r in roes if r > 0.15)
        ratio = high / len(roes)
        if ratio >= 0.8:
            score += 2
            reasoning.append(f"Excellent ROE consistency: {high}/{len(roes)} periods >15%")
        elif ratio >= 0.6:
            score += 1
            reasoning.append(f"Good ROE: {high}/{len(roes)} periods >15%")
        else:
            reasoning.append(f"Inconsistent ROE: only {high}/{len(roes)} periods >15%")

    margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(margins) >= 5:
        avg = sum(margins) / len(margins)
        recent_avg = sum(margins[:3]) / 3
        older_avg = sum(margins[-3:]) / 3
        if avg > 0.2 and recent_avg >= older_avg:
            score += 1
            reasoning.append(f"Strong stable margins (avg {avg:.1%}) — pricing power moat")
        elif avg > 0.15:
            reasoning.append(f"Decent margins (avg {avg:.1%})")
        else:
            reasoning.append(f"Low margins (avg {avg:.1%})")

    if len(roes) >= 5 and len(margins) >= 5:
        roe_avg = sum(roes) / len(roes)
        roe_std = (sum((r - roe_avg) ** 2 for r in roes) / len(roes)) ** 0.5
        m_avg = sum(margins) / len(margins)
        m_std = (sum((m - m_avg) ** 2 for m in margins) / len(margins)) ** 0.5
        stability = 1 - ((roe_std / roe_avg if roe_avg > 0 else 1) + (m_std / m_avg if m_avg > 0 else 1)) / 2
        if stability > 0.7:
            score += 1
            reasoning.append(f"High performance stability ({stability:.1%})")

    turnovers = [m.asset_turnover for m in metrics if m.asset_turnover is not None]
    if turnovers and any(t > 1.0 for t in turnovers):
        score += 1
        reasoning.append("Efficient asset utilization")

    return {"score": min(score, 5), "max_score": 5, "details": "; ".join(reasoning)}


def analyze_management_quality(line_items: list) -> dict:
    if not line_items:
        return {"score": 0, "max_score": 2, "details": "Insufficient data"}
    score = 0
    reasoning = []
    buybacks = get_metric(line_items, "stock_repurchases")
    if buybacks and buybacks < 0:
        score += 1
        reasoning.append("Company repurchasing shares (shareholder-friendly)")
    else:
        reasoning.append("No significant buybacks")

    dividends = get_metric(line_items, "dividends_paid")
    if dividends and dividends < 0:
        score += 1
        reasoning.append("Company pays dividends")
    else:
        reasoning.append("No or minimal dividends")

    return {"score": score, "max_score": 2, "details": "; ".join(reasoning)}


def analyze_pricing_power(line_items: list, metrics: list) -> dict:
    if not line_items or not metrics:
        return {"score": 0, "details": "Insufficient data"}
    score = 0
    reasoning = []

    gross_margins = [m.gross_margin for m in metrics if m.gross_margin is not None]
    if len(gross_margins) >= 3:
        recent_avg = sum(gross_margins[:2]) / 2
        older_avg = sum(gross_margins[-2:]) / 2
        if recent_avg > older_avg + 0.02:
            score += 3
            reasoning.append("Expanding gross margins — strong pricing power")
        elif recent_avg > older_avg:
            score += 2
            reasoning.append("Improving gross margins — good pricing power")
        elif abs(recent_avg - older_avg) < 0.01:
            score += 1
            reasoning.append("Stable gross margins")
        else:
            reasoning.append("Declining gross margins — pricing pressure")

        avg = sum(gross_margins) / len(gross_margins)
        if avg > 0.5:
            score += 2
            reasoning.append(f"Consistently high gross margins ({avg:.1%})")
        elif avg > 0.3:
            score += 1
            reasoning.append(f"Good gross margins ({avg:.1%})")

    return {"score": score, "details": "; ".join(reasoning) or "Limited pricing power data"}


def analyze_book_value_growth(line_items: list) -> dict:
    if len(line_items) < 3:
        return {"score": 0, "details": "Insufficient data"}

    equity_series = get_metric_series(line_items, "total_equity")
    shares_series = get_metric_series(line_items, "shares_outstanding")
    bvs = [e / s for e, s in zip(equity_series, shares_series) if s and s > 0]

    if len(bvs) < 3:
        return {"score": 0, "details": "Insufficient book value data"}

    score = 0
    reasoning = []
    growth_periods = sum(1 for i in range(len(bvs) - 1) if bvs[i] > bvs[i + 1])
    rate = growth_periods / (len(bvs) - 1)

    if rate >= 0.8:
        score += 3
        reasoning.append("Consistent book value per share growth")
    elif rate >= 0.6:
        score += 2
        reasoning.append("Good book value growth pattern")
    elif rate >= 0.4:
        score += 1
        reasoning.append("Moderate book value growth")
    else:
        reasoning.append("Inconsistent book value growth")

    if len(bvs) >= 2 and bvs[-1] > 0 and bvs[0] > 0:
        cagr = (bvs[0] / bvs[-1]) ** (1 / (len(bvs) - 1)) - 1
        if cagr > 0.15:
            score += 2
            reasoning.append(f"Excellent book value CAGR {cagr:.1%}")
        elif cagr > 0.1:
            score += 1
            reasoning.append(f"Good book value CAGR {cagr:.1%}")
        else:
            reasoning.append(f"Book value CAGR {cagr:.1%}")

    return {"score": score, "details": "; ".join(reasoning)}


def calculate_intrinsic_value(line_items: list) -> dict:
    if not line_items or len(line_items) < 3:
        return {"intrinsic_value": None, "details": ["Insufficient data"]}

    net_income = get_metric(line_items, "net_income")
    depreciation = get_metric(line_items, "depreciation_and_amortization")
    capex = get_metric(line_items, "capital_expenditure")

    if not all([net_income, depreciation, capex]):
        return {"intrinsic_value": None, "details": ["Missing DCF components"]}

    maintenance_capex = max(abs(capex) * 0.85, depreciation)
    owner_earnings = net_income + depreciation - maintenance_capex

    hist = get_metric_series(line_items, "net_income")
    if len(hist) >= 3 and hist[-1] > 0:
        raw_growth = (hist[0] / hist[-1]) ** (1 / (len(hist) - 1)) - 1
        g1 = min(max(raw_growth * 0.7, -0.05), 0.08)
    else:
        g1 = 0.03

    g2 = min(g1 * 0.5, 0.04)
    g3 = 0.025
    r = 0.10

    pv = 0
    for yr in range(1, 6):
        pv += owner_earnings * (1 + g1) ** yr / (1 + r) ** yr
    base = owner_earnings * (1 + g1) ** 5
    for yr in range(1, 6):
        pv += base * (1 + g2) ** yr / (1 + r) ** (5 + yr)
    final = base * (1 + g2) ** 5
    terminal = final * (1 + g3) / (r - g3) / (1 + r) ** 10

    intrinsic = (pv + terminal) * 0.85
    return {
        "intrinsic_value": intrinsic,
        "owner_earnings": owner_earnings,
        "details": [f"DCF IV: ${intrinsic:,.0f} (g1={g1:.1%}, r={r:.0%})"],
    }