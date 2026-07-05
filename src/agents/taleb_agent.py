# Nassim Taleb style risk management/antifragility agent
import json
import logging
import math
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
from typing_extensions import Literal
import numpy as np
import pandas as pd

from src.llm import call_llm_json
from src.utils.line_item_helpers import get_metric, get_metric_series

from src.data.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    get_prices,
    prices_to_df,
    search_line_items,
)

logger = logging.getLogger(__name__)


# ── Output schema ─────────────────────────────────────────────────────────────

class NassimTalebSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")


# ── Entry point ───────────────────────────────────────────────────────────────

def nassim_taleb_agent(ticker: str, end_date: str, normalized_data: dict | None = None) -> dict:
    """
    Analyze a single ticker using Taleb's antifragility principles.
    Returns {"signal": ..., "confidence": ..., "reasoning": ...}
    """
    logger.info("[taleb] analyzing %s", ticker)

    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    # Use pre-normalized data if provided by judge_agent, otherwise fetch directly.
    if normalized_data:
        metrics    = normalized_data["metrics"]
        line_items = normalized_data["line_items"]
        prices_df  = normalized_data["prices_df"]
        market_cap = normalized_data["market_cap"]
    else:
        prices = get_prices(ticker, start_date, end_date)
        prices_df = prices_to_df(prices) if prices else pd.DataFrame()
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=10)
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow", "net_income", "total_debt",
                "cash_and_equivalents", "total_assets", "total_liabilities",
                "revenue", "operating_income", "research_and_development",
                "capital_expenditure", "shares_outstanding",
            ],
            end_date,
            period="ttm",
            limit=5,
        )
        market_cap = get_market_cap(ticker, end_date)

    # Insider trades and news are not provided by normalizer — always fetch.
    insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)
    news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=100)

    tail_risk      = analyze_tail_risk(prices_df)
    antifragility  = analyze_antifragility(metrics, line_items, market_cap)
    convexity      = analyze_convexity(metrics, line_items, prices_df, market_cap)
    fragility      = analyze_fragility(metrics, line_items)
    skin_in_game   = analyze_skin_in_game(insider_trades)
    vol_regime     = analyze_volatility_regime(prices_df)
    black_swan     = analyze_black_swan_sentinel(news, prices_df)

    total = (tail_risk["score"] + antifragility["score"] + convexity["score"]
             + fragility["score"] + skin_in_game["score"] + vol_regime["score"] + black_swan["score"])
    max_total = (tail_risk["max_score"] + antifragility["max_score"] + convexity["max_score"]
                 + fragility["max_score"] + skin_in_game["max_score"] + vol_regime["max_score"] + black_swan["max_score"])

    analysis_data = {
        "ticker": ticker,
        "score": total,
        "max_score": max_total,
        "tail_risk_analysis": tail_risk,
        "antifragility_analysis": antifragility,
        "convexity_analysis": convexity,
        "fragility_analysis": fragility,
        "skin_in_game_analysis": skin_in_game,
        "volatility_regime_analysis": vol_regime,
        "black_swan_analysis": black_swan,
        "market_cap": market_cap,
    }

    output = _generate_llm_output(ticker, analysis_data)
    logger.info("[taleb] %s → %s (%d%%)", ticker, output.signal, output.confidence)
    return {"signal": output.signal, "confidence": output.confidence, "reasoning": output.reasoning}


# ── LLM call (Google Gemini) ─────────────────────────────────────────────────

def _generate_llm_output(ticker: str, analysis_data: dict) -> NassimTalebSignal:
    facts = {
        "score": analysis_data.get("score"),
        "max_score": analysis_data.get("max_score"),
        "tail_risk": analysis_data.get("tail_risk_analysis", {}).get("details"),
        "antifragility": analysis_data.get("antifragility_analysis", {}).get("details"),
        "convexity": analysis_data.get("convexity_analysis", {}).get("details"),
        "fragility": analysis_data.get("fragility_analysis", {}).get("details"),
        "skin_in_game": analysis_data.get("skin_in_game_analysis", {}).get("details"),
        "volatility_regime": analysis_data.get("volatility_regime_analysis", {}).get("details"),
        "black_swan": analysis_data.get("black_swan_analysis", {}).get("details"),
        "market_cap": analysis_data.get("market_cap"),
    }

    system_prompt = (
        "You are Nassim Taleb. Decide bullish, bearish, or neutral using only the provided facts.\n\n"
        "Checklist:\n"
        "- Antifragility (benefits from disorder)\n"
        "- Tail risk profile (fat tails, skewness)\n"
        "- Convexity (asymmetric payoff potential)\n"
        "- Fragility via negativa (avoid the fragile)\n"
        "- Skin in the game (insider alignment)\n"
        "- Volatility regime (low vol = danger)\n\n"
        "Signal rules:\n"
        "- Bullish: antifragile business with convex payoff AND not fragile.\n"
        "- Bearish: fragile business (high leverage, thin margins, volatile earnings) OR no skin in the game.\n"
        "- Neutral: mixed signals, or insufficient data.\n\n"
        "Confidence scale:\n"
        "- 90-100%: Truly antifragile with strong convexity and skin in the game\n"
        "- 70-89%: Low fragility with decent optionality\n"
        "- 50-69%: Mixed fragility signals\n"
        "- 30-49%: Some fragility, weak insider alignment\n"
        "- 10-29%: Clearly fragile or dangerous vol regime\n\n"
        "Use Taleb vocabulary: antifragile, convexity, skin in the game, via negativa, barbell, turkey problem.\n"
        "Keep reasoning under 150 characters. Do not invent data. Return JSON only."
    )

    user_prompt = (
        f"Ticker: {ticker}\n"
        f"Facts:\n{json.dumps(facts, separators=(',', ':'), ensure_ascii=False)}\n\n"
        'Return exactly:\n{\n  "signal": "bullish" | "bearish" | "neutral",\n'
        '  "confidence": int,\n  "reasoning": "short justification"\n}'
    )

    fallback = {"signal": "neutral", "confidence": 50, "reasoning": "Insufficient data"}
    data = call_llm_json(system_prompt, user_prompt, fallback=fallback, max_tokens=256)
    return NassimTalebSignal(**data)


###############################################################################
# Helper
###############################################################################

def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or np.isnan(value):
            return default
        return float(value)
    except (ValueError, TypeError, OverflowError):
        return default


###############################################################################
# Analysis functions (ไม่แตะ — logic เดิมทั้งหมด)
###############################################################################

def analyze_tail_risk(prices_df: pd.DataFrame) -> dict:
    if prices_df.empty or len(prices_df) < 20:
        return {"score": 0, "max_score": 8, "details": "Insufficient price data"}

    score = 0
    reasoning = []
    returns = prices_df["close"].pct_change().dropna()

    kurt = safe_float(returns.rolling(63).kurt().iloc[-1] if len(returns) >= 63 else returns.kurt())
    if kurt > 5:
        score += 2; reasoning.append(f"Extremely fat tails (kurtosis {kurt:.1f})")
    elif kurt > 2:
        score += 1; reasoning.append(f"Moderate fat tails (kurtosis {kurt:.1f})")
    else:
        reasoning.append(f"Near-Gaussian tails (kurtosis {kurt:.1f})")

    skew = safe_float(returns.rolling(63).skew().iloc[-1] if len(returns) >= 63 else returns.skew())
    if skew > 0.5:
        score += 2; reasoning.append(f"Positive skew ({skew:.2f}) favors long convexity")
    elif skew > -0.5:
        score += 1; reasoning.append(f"Symmetric distribution (skew {skew:.2f})")
    else:
        reasoning.append(f"Negative skew ({skew:.2f}) — crash-prone")

    pos = returns[returns > 0]
    neg = returns[returns < 0]
    if len(pos) > 20 and len(neg) > 20:
        right = np.percentile(pos, 95)
        left = abs(np.percentile(neg, 5))
        tr = right / left if left > 0 else 1.0
        if tr > 1.2:
            score += 2; reasoning.append(f"Asymmetric upside (tail ratio {tr:.2f})")
        elif tr > 0.8:
            score += 1; reasoning.append(f"Balanced tails (tail ratio {tr:.2f})")
        else:
            reasoning.append(f"Asymmetric downside (tail ratio {tr:.2f})")

    cum = (1 + returns).cumprod()
    max_dd = safe_float(((cum - cum.cummax()) / cum.cummax()).min())
    if max_dd > -0.15:
        score += 2; reasoning.append(f"Resilient (max drawdown {max_dd:.1%})")
    elif max_dd > -0.30:
        score += 1; reasoning.append(f"Moderate drawdown ({max_dd:.1%})")
    else:
        reasoning.append(f"Severe drawdown ({max_dd:.1%})")

    return {"score": score, "max_score": 8, "details": "; ".join(reasoning)}


def analyze_antifragility(metrics: list, line_items: list, market_cap) -> dict:
    if not metrics and not line_items:
        return {"score": 0, "max_score": 10, "details": "Insufficient data"}

    score = 0
    reasoning = []
    lm = metrics[0] if metrics else None
    cash = get_metric(line_items, "cash_and_equivalents")
    debt = get_metric(line_items, "total_debt")
    assets = get_metric(line_items, "total_assets")

    if cash is not None and debt is not None:
        net_cash = cash - debt
        if net_cash > 0 and market_cap and cash > 0.20 * market_cap:
            score += 3; reasoning.append(f"War chest: net cash ${net_cash:,.0f}")
        elif net_cash > 0:
            score += 2; reasoning.append(f"Net cash positive (${net_cash:,.0f})")
        elif assets and debt < 0.30 * assets:
            score += 1; reasoning.append("Manageable net debt")
        else:
            reasoning.append("Leveraged — not antifragile")
    else:
        reasoning.append("Cash/debt data unavailable")

    de = getattr(lm, "debt_to_equity", None) if lm else None
    if de is not None:
        if de < 0.3:
            score += 2; reasoning.append(f"Low leverage (D/E {de:.2f})")
        elif de < 0.7:
            score += 1; reasoning.append(f"Moderate leverage (D/E {de:.2f})")
        else:
            reasoning.append(f"High leverage (D/E {de:.2f})")
    else:
        reasoning.append("D/E data unavailable")

    op_margins = [m.operating_margin for m in metrics if m.operating_margin is not None]
    if len(op_margins) >= 3:
        mean_m = sum(op_margins) / len(op_margins)
        std_m = (sum((m - mean_m) ** 2 for m in op_margins) / len(op_margins)) ** 0.5
        cv = std_m / abs(mean_m) if mean_m != 0 else float("inf")
        if cv < 0.15 and mean_m > 0.15:
            score += 3; reasoning.append(f"Stable high margins (avg {mean_m:.1%}, CV {cv:.2f})")
        elif cv < 0.30 and mean_m > 0.10:
            score += 2; reasoning.append(f"Reasonable margin stability (avg {mean_m:.1%})")
        elif cv < 0.30:
            score += 1; reasoning.append(f"Somewhat stable margins (CV {cv:.2f})")
        else:
            reasoning.append(f"Volatile margins (CV {cv:.2f})")
    else:
        reasoning.append("Insufficient margin history")

    fcf_vals = get_metric_series(line_items, "free_cash_flow") if line_items else []
    if fcf_vals:
        pos = sum(1 for v in fcf_vals if v > 0)
        if pos == len(fcf_vals):
            score += 2; reasoning.append(f"Consistent FCF ({pos}/{len(fcf_vals)} periods)")
        elif pos > len(fcf_vals) / 2:
            score += 1; reasoning.append(f"Majority positive FCF ({pos}/{len(fcf_vals)})")
        else:
            reasoning.append(f"Inconsistent FCF ({pos}/{len(fcf_vals)})")
    else:
        reasoning.append("FCF data unavailable")

    return {"score": score, "max_score": 10, "details": "; ".join(reasoning)}


def analyze_convexity(metrics: list, line_items: list, prices_df: pd.DataFrame, market_cap) -> dict:
    if not metrics and not line_items and prices_df.empty:
        return {"score": 0, "max_score": 10, "details": "Insufficient data"}

    score = 0
    reasoning = []
    rd = get_metric(line_items, "research_and_development") if line_items else None
    rev = get_metric(line_items, "revenue") if line_items else None
    if rd is not None and rev and rev > 0:
        ratio = abs(rd) / rev
        if ratio > 0.15:
            score += 3; reasoning.append(f"Significant R&D optionality ({ratio:.1%} of revenue)")
        elif ratio > 0.08:
            score += 2; reasoning.append(f"Meaningful R&D ({ratio:.1%})")
        elif ratio > 0.03:
            score += 1; reasoning.append(f"Modest R&D ({ratio:.1%})")
        else:
            reasoning.append(f"Minimal R&D ({ratio:.1%})")
    else:
        reasoning.append("R&D data unavailable")

    if not prices_df.empty and len(prices_df) >= 20:
        rets = prices_df["close"].pct_change().dropna()
        up = rets[rets > 0]
        dn = rets[rets < 0]
        if len(up) > 10 and len(dn) > 10:
            ratio = up.mean() / abs(dn.mean())
            if ratio > 1.3:
                score += 2; reasoning.append(f"Convex return profile (up/down {ratio:.2f})")
            elif ratio > 1.0:
                score += 1; reasoning.append(f"Slight positive asymmetry ({ratio:.2f})")
            else:
                reasoning.append(f"Concave returns ({ratio:.2f})")
    else:
        reasoning.append("Insufficient price data")

    cash = get_metric(line_items, "cash_and_equivalents") if line_items else None
    if cash and market_cap and market_cap > 0:
        cr = cash / market_cap
        if cr > 0.30:
            score += 3; reasoning.append(f"Cash as call option ({cr:.0%} of market cap)")
        elif cr > 0.15:
            score += 2; reasoning.append(f"Strong cash ({cr:.0%})")
        elif cr > 0.05:
            score += 1; reasoning.append(f"Moderate cash ({cr:.0%})")
        else:
            reasoning.append(f"Low cash ({cr:.0%})")
    else:
        reasoning.append("Cash/market cap data unavailable")

    lm = metrics[0] if metrics else None
    fcf_yield = None
    if line_items and market_cap and market_cap > 0:
        fcf = get_metric(line_items, "free_cash_flow")
        if fcf:
            fcf_yield = fcf / market_cap
    if fcf_yield is None and lm:
        fcf_yield = getattr(lm, "free_cash_flow_yield", None)

    if fcf_yield is not None:
        if fcf_yield > 0.10:
            score += 2; reasoning.append(f"High FCF yield ({fcf_yield:.1%})")
        elif fcf_yield > 0.05:
            score += 1; reasoning.append(f"Decent FCF yield ({fcf_yield:.1%})")
        else:
            reasoning.append(f"Low FCF yield ({fcf_yield:.1%})")
    else:
        reasoning.append("FCF yield unavailable")

    return {"score": score, "max_score": 10, "details": "; ".join(reasoning)}


def analyze_fragility(metrics: list, line_items: list) -> dict:
    if not metrics:
        return {"score": 0, "max_score": 8, "details": "Insufficient data"}

    score = 0
    reasoning = []
    lm = metrics[0]

    de = getattr(lm, "debt_to_equity", None)
    if de is not None:
        if de > 2.0:
            reasoning.append(f"Extremely fragile (D/E {de:.2f})")
        elif de > 1.0:
            score += 1; reasoning.append(f"Elevated leverage (D/E {de:.2f})")
        elif de > 0.5:
            score += 2; reasoning.append(f"Moderate leverage (D/E {de:.2f})")
        else:
            score += 3; reasoning.append(f"Low leverage (D/E {de:.2f})")
    else:
        reasoning.append("D/E data unavailable")

    ic = getattr(lm, "interest_coverage", None)
    if ic is not None:
        if ic > 10:
            score += 2; reasoning.append(f"Interest coverage {ic:.1f}x")
        elif ic > 5:
            score += 1; reasoning.append(f"Comfortable coverage ({ic:.1f}x)")
        else:
            reasoning.append(f"Low coverage ({ic:.1f}x)")
    else:
        reasoning.append("Interest coverage unavailable")

    eg_vals = [m.earnings_growth for m in metrics if m.earnings_growth is not None]
    if len(eg_vals) >= 3:
        mean_eg = sum(eg_vals) / len(eg_vals)
        std_eg = (sum((e - mean_eg) ** 2 for e in eg_vals) / len(eg_vals)) ** 0.5
        if std_eg < 0.20:
            score += 2; reasoning.append(f"Stable earnings (std {std_eg:.2f})")
        elif std_eg < 0.50:
            score += 1; reasoning.append(f"Moderate volatility (std {std_eg:.2f})")
        else:
            reasoning.append(f"Highly volatile earnings (std {std_eg:.2f})")
    else:
        reasoning.append("Insufficient earnings history")

    nm = getattr(lm, "net_margin", None)
    if nm is not None:
        if nm > 0.15:
            score += 1; reasoning.append(f"Fat margins ({nm:.1%})")
        elif nm >= 0.05:
            reasoning.append(f"Moderate margins ({nm:.1%})")
        else:
            reasoning.append(f"Paper-thin margins ({nm:.1%})")
    else:
        reasoning.append("Net margin unavailable")

    return {"score": max(score, 0), "max_score": 8, "details": "; ".join(reasoning)}


def analyze_skin_in_game(insider_trades: list) -> dict:
    if not insider_trades:
        return {"score": 1, "max_score": 4, "details": "No insider trade data — neutral"}

    score = 0
    reasoning = []

    # Sign convention: shares > 0 = purchase, shares < 0 = sale (negated in api.py).
    bought = sum(t.shares or 0 for t in insider_trades if (t.shares or 0) > 0)
    sold = abs(sum(t.shares or 0 for t in insider_trades if (t.shares or 0) < 0))
    net = bought - sold

    if net > 0:
        ratio = net / max(sold, 1)
        if ratio > 2.0:
            score = 4; reasoning.append(f"Strong insider buying (ratio {ratio:.1f}x)")
        elif ratio > 0.5:
            score = 3; reasoning.append(f"Moderate insider conviction")
        else:
            score = 2; reasoning.append(f"Net insider buying {net:,} shares")
    else:
        reasoning.append(f"Insiders selling (net {net:,} shares)")

    return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}


def analyze_volatility_regime(prices_df: pd.DataFrame) -> dict:
    if prices_df.empty or len(prices_df) < 30:
        return {"score": 0, "max_score": 6, "details": "Insufficient price data"}

    score = 0
    reasoning = []
    returns = prices_df["close"].pct_change().dropna()
    hist_vol = returns.rolling(21).std() * math.sqrt(252)

    if len(hist_vol.dropna()) >= 63:
        vol_ma = hist_vol.rolling(63).mean()
        curr = safe_float(hist_vol.iloc[-1])
        avg = safe_float(vol_ma.iloc[-1])
    elif len(hist_vol.dropna()) >= 21:
        curr = safe_float(hist_vol.iloc[-1])
        avg = safe_float(hist_vol.mean())
    else:
        return {"score": 0, "max_score": 6, "details": "Insufficient vol data"}

    regime = curr / avg if avg > 0 else 1.0
    if regime < 0.7:
        reasoning.append(f"Dangerously low vol (regime {regime:.2f}) — turkey problem")
    elif regime < 0.9:
        score += 1; reasoning.append(f"Below-average vol ({regime:.2f})")
    elif regime <= 1.3:
        score += 3; reasoning.append(f"Normal vol regime ({regime:.2f})")
    elif regime <= 2.0:
        score += 4; reasoning.append(f"Elevated vol ({regime:.2f}) — antifragile opportunity")
    else:
        score += 2; reasoning.append(f"Extreme vol ({regime:.2f}) — crisis mode")

    if len(hist_vol.dropna()) >= 42:
        vov = hist_vol.rolling(21).std().dropna()
        if len(vov) > 0:
            curr_vov = safe_float(vov.iloc[-1])
            med_vov = safe_float(vov.median())
            if med_vov > 0:
                if curr_vov > 2 * med_vov:
                    score += 2; reasoning.append(f"Highly unstable vol (vov {curr_vov:.4f})")
                elif curr_vov > med_vov:
                    score += 1; reasoning.append(f"Elevated vol-of-vol ({curr_vov:.4f})")
                else:
                    reasoning.append(f"Stable vol-of-vol ({curr_vov:.4f})")

    return {"score": score, "max_score": 6, "details": "; ".join(reasoning)}


def analyze_black_swan_sentinel(news: list, prices_df: pd.DataFrame) -> dict:
    score = 2
    reasoning = []

    neg_ratio = 0.0
    if news:
        total = len(news)
        neg = sum(1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"])
        neg_ratio = neg / total if total > 0 else 0
    else:
        reasoning.append("No news data")

    volume_spike = 1.0
    recent_return = 0.0
    if not prices_df.empty and len(prices_df) >= 10:
        if "volume" in prices_df.columns:
            recent_vol = prices_df["volume"].iloc[-5:].mean()
            avg_vol = prices_df["volume"].iloc[-63:].mean() if len(prices_df) >= 63 else prices_df["volume"].mean()
            volume_spike = recent_vol / avg_vol if avg_vol > 0 else 1.0
        if len(prices_df) >= 5:
            recent_return = safe_float(prices_df["close"].iloc[-1] / prices_df["close"].iloc[-5] - 1)

    if neg_ratio > 0.7 and volume_spike > 2.0:
        score = 0; reasoning.append(f"Black swan warning — {neg_ratio:.0%} neg news, {volume_spike:.1f}x volume")
    elif neg_ratio > 0.5 or volume_spike > 2.5:
        score = 1; reasoning.append(f"Elevated stress ({neg_ratio:.0%} neg, {volume_spike:.1f}x volume)")
    elif neg_ratio > 0.3 and abs(recent_return) > 0.10:
        score = 1; reasoning.append(f"Moderate stress ({recent_return:.1%} move)")
    elif neg_ratio < 0.3 and volume_spike < 1.5:
        score = 3; reasoning.append("No black swan signals")
    else:
        reasoning.append(f"Normal conditions (neg {neg_ratio:.0%}, vol {volume_spike:.1f}x)")

    if neg_ratio > 0.4 and volume_spike < 1.5 and score < 4:
        score = min(score + 1, 4)
        reasoning.append("Contrarian opportunity — neg sentiment without panic")

    return {"score": score, "max_score": 4, "details": "; ".join(reasoning)}