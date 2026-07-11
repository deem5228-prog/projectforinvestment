# Judge/Consensus agent to aggregate decisions from other agents
"""
Judge Agent
===========
Investment Committee ที่รับ output จาก 4 agents แล้วตัดสิน BUY / HOLD / SELL
พร้อม confidence %, strengths, risks, และ position sizing recommendation

Flow:
  run_all_agents(ticker, end_date)
    → warren_buffett_agent
    → nassim_taleb_agent
    → hedge_fund_agent
    → quant_agent
    → aggregate_signals()
    → _generate_verdict()
    → JudgeReport
"""

import json
import logging
import time
import concurrent.futures
from dataclasses import dataclass, asdict
from typing import Literal

from src.llm import call_llm_json
from src.normalizer.normalizer import normalize

from src.agents.buffett_agent   import warren_buffett_agent
from src.agents.taleb_agent     import nassim_taleb_agent
from src.agents.hedge_fund_agent import hedge_fund_agent
from src.agents.quant_agent     import quant_agent

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    name: str
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float      # 0-100
    reasoning: str
    weight: float          # relative weight in committee vote


@dataclass
class JudgeReport:
    ticker: str
    verdict: Literal["BUY", "HOLD", "SELL"]
    confidence: float      # 0-100 overall confidence
    score_pct: float       # weighted score as % of max
    agents: list[dict]     # each agent's raw result
    strengths: list[str]
    risks: list[str]
    position_size: str     # "full" | "half" | "small" | "none"
    reasoning: str         # LLM narrative summary


# ── Agent weights (sum to 1.0) ────────────────────────────────────────────────
#   Buffett    — long-term quality anchor
#   Taleb      — risk/fragility veto power (high weight for downside)
#   Hedge Fund — catalyst & timing signal
#   Quant      — objective numerical check

AGENT_WEIGHTS = {
    "warren_buffett": 0.30,
    "nassim_taleb":   0.25,
    "hedge_fund":     0.25,
    "quant":          0.20,
}

SIGNAL_SCORES = {"bullish": 1.0, "neutral": 0.5, "bearish": 0.0}


# ── Entry point ───────────────────────────────────────────────────────────────

def judge_agent(ticker: str, end_date: str, parallel: bool = True,
                on_progress=None) -> JudgeReport:
    """
    Run all 4 analysts then synthesize a final verdict.

    Args:
        ticker:   Stock ticker symbol, e.g. "AAPL"
        end_date: ISO date string, e.g. "2024-12-31"
        parallel: Run agents concurrently (faster) or sequentially (easier to debug)
        on_progress: Optional callback(step_id, status, detail) for live progress

    Returns:
        JudgeReport with BUY / HOLD / SELL + full breakdown
    """
    def _emit(step_id, status, detail=""):
        """Fire progress callback if provided."""
        if on_progress:
            try:
                on_progress(step_id, status, detail)
            except Exception:
                pass  # Never let callback errors crash the pipeline

    logger.info("[judge] starting analysis for %s", ticker)

    # ── Step 0: Normalize data once (shared by all agents) ────────────────
    _emit("normalizer", "running", "Fetching financial data from Yahoo Finance…")
    try:
        normalized_data = normalize(ticker, end_date)
        n_metrics = len(normalized_data.get("metrics", []))
        n_items = len(normalized_data.get("line_items", []))
        logger.info("[judge] normalizer done — %d metrics, %d line_items",
                    n_metrics, n_items)
        _emit("normalizer", "done", f"{n_metrics} metrics, {n_items} line items")
    except Exception as e:
        logger.warning("[judge] normalizer failed, agents will self-fetch: %s", e)
        normalized_data = None
        _emit("normalizer", "error", str(e))

    # ── Step 1: Run all agents ────────────────────────────────────────────
    raw = _run_agents(ticker, end_date, parallel=parallel,
                      normalized_data=normalized_data, on_progress=_emit)

    # ── Step 2: Package into AgentResult objects ──────────────────────────
    results = [
        AgentResult(
            name=name,
            signal=raw[name]["signal"],
            confidence=raw[name]["confidence"],
            reasoning=raw[name]["reasoning"],
            weight=AGENT_WEIGHTS[name],
        )
        for name in ["warren_buffett", "nassim_taleb", "hedge_fund", "quant"]
        if name in raw
    ]

    # ── Step 3: Aggregate signals → preliminary verdict ───────────────────
    _emit("aggregation", "running", "Computing weighted vote & Taleb veto…")
    aggregation = _aggregate_signals(results)

    # ── Step 4: Extract strengths & risks from agent reasonings ──────────
    strengths, risks = _extract_strengths_risks(results)

    # ── Step 5: Position sizing ───────────────────────────────────────────
    position_size = _position_size(
        aggregation["weighted_score"],
        aggregation["consensus_strength"],
        raw.get("nassim_taleb", {}).get("signal", "neutral"),
    )
    score = round(aggregation['weighted_score'] * 100)
    _emit("aggregation", "done", f"Score {score}%, position: {position_size}")

    # ── Step 6: LLM narrative verdict ────────────────────────────────────
    _emit("verdict", "running", "Gemini LLM synthesizing final verdict…")
    report_data = {
        "ticker": ticker,
        "aggregation": aggregation,
        "agents": [asdict(r) for r in results],
        "strengths": strengths,
        "risks": risks,
        "position_size": position_size,
    }
    verdict_out = _generate_verdict(ticker, report_data)
    _emit("verdict", "done", f"{verdict_out['verdict']} ({verdict_out['confidence']}% confidence)")

    report = JudgeReport(
        ticker=ticker,
        verdict=verdict_out["verdict"],
        confidence=verdict_out["confidence"],
        score_pct=round(aggregation["weighted_score"] * 100, 1),
        agents=[asdict(r) for r in results],
        strengths=strengths,
        risks=risks,
        position_size=position_size,
        reasoning=verdict_out["reasoning"],
    )

    logger.info(
        "[judge] %s → %s (confidence %d%%, position %s)",
        ticker, report.verdict, report.confidence, report.position_size,
    )
    return report


# ── Runner ─────────────────────────────────────────────────────────────────────

def _run_agents(ticker: str, end_date: str, parallel: bool,
                normalized_data: dict | None = None,
                on_progress=None) -> dict:
    """Run all 4 agents, returning dict of name → result dict."""

    tasks = {
        "warren_buffett": warren_buffett_agent,
        "nassim_taleb":   nassim_taleb_agent,
        "hedge_fund":     hedge_fund_agent,
        "quant":          quant_agent,
    }

    results = {}

    if parallel:
        # Emit all as running
        for name in tasks:
            if on_progress:
                on_progress(f"agent_{name}", "running", "Running in parallel…")
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
            futures = {
                ex.submit(fn, ticker, end_date, normalized_data=normalized_data): name
                for name, fn in tasks.items()
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    results[name] = future.result()
                    signal = results[name]["signal"]
                    conf = results[name]["confidence"]
                    logger.info("[judge] %s agent done → %s", name, signal)
                    if on_progress:
                        on_progress(f"agent_{name}", "done", f"{signal} ({conf}%)")
                except Exception as e:
                    logger.warning("[judge] %s agent failed: %s", name, e)
                    results[name] = {"signal": "neutral", "confidence": 50, "reasoning": f"Error: {e}"}
                    if on_progress:
                        on_progress(f"agent_{name}", "error", str(e))
    else:
        for i, (name, fn) in enumerate(tasks.items()):
            if on_progress:
                on_progress(f"agent_{name}", "running", "Analyzing…")
            try:
                results[name] = fn(ticker, end_date, normalized_data=normalized_data)
                signal = results[name]["signal"]
                conf = results[name]["confidence"]
                logger.info("[judge] %s agent done → %s", name, signal)
                if on_progress:
                    on_progress(f"agent_{name}", "done", f"{signal} ({conf}%)")
            except Exception as e:
                logger.warning("[judge] %s agent failed: %s", name, e)
                results[name] = {"signal": "neutral", "confidence": 50, "reasoning": f"Error: {e}"}
                if on_progress:
                    on_progress(f"agent_{name}", "error", str(e))
            # Space out LLM calls to respect free-tier rate limit.
            if i < len(tasks) - 1:
                logger.info("[judge] waiting 12s before next agent (rate limit)...")
                time.sleep(12)

    return results


# ── Signal aggregation ─────────────────────────────────────────────────────────

def _aggregate_signals(results: list[AgentResult]) -> dict:
    """
    Weighted vote across agents.

    weighted_score: 0.0 (all bearish) → 1.0 (all bullish)
    Threshold: >0.62 = bullish, <0.38 = bearish, else neutral
    """
    if not results:
        return {
            "weighted_score": 0.5,
            "preliminary_verdict": "HOLD",
            "bullish_weight": 0.0,
            "bearish_weight": 0.0,
            "neutral_weight": 0.0,
            "consensus_strength": 0.0,
            "dissent": [],
        }

    total_weight = sum(r.weight for r in results)
    weighted_score = sum(
        SIGNAL_SCORES[r.signal] * r.weight * (r.confidence / 100)
        for r in results
    ) / total_weight if total_weight > 0 else 0.5

    bullish_w = sum(r.weight for r in results if r.signal == "bullish") / total_weight
    bearish_w = sum(r.weight for r in results if r.signal == "bearish") / total_weight
    neutral_w = sum(r.weight for r in results if r.signal == "neutral") / total_weight

    # Consensus strength: how aligned are agents (0 = split, 1 = unanimous)
    max_side = max(bullish_w, bearish_w, neutral_w)
    consensus_strength = max_side

    # Preliminary verdict
    if weighted_score >= 0.62:
        prelim = "BUY"
    elif weighted_score <= 0.38:
        prelim = "SELL"
    else:
        prelim = "HOLD"

    # Taleb veto: if Taleb is bearish with >70 confidence, cap at HOLD
    taleb = next((r for r in results if r.name == "nassim_taleb"), None)
    if taleb and taleb.signal == "bearish" and taleb.confidence >= 70 and prelim == "BUY":
        prelim = "HOLD"
        logger.info("[judge] Taleb veto: BUY → HOLD (fragility concern)")

    # Dissenting agents (going against majority)
    majority_signal = "bullish" if prelim == "BUY" else ("bearish" if prelim == "SELL" else "neutral")
    dissent = [r.name for r in results if r.signal != majority_signal and r.confidence >= 60]

    return {
        "weighted_score": round(weighted_score, 4),
        "preliminary_verdict": prelim,
        "bullish_weight": round(bullish_w, 3),
        "bearish_weight": round(bearish_w, 3),
        "neutral_weight": round(neutral_w, 3),
        "consensus_strength": round(consensus_strength, 3),
        "dissent": dissent,
    }


# ── Strengths & risks extraction ───────────────────────────────────────────────

def _extract_strengths_risks(results: list[AgentResult]) -> tuple[list[str], list[str]]:
    """
    Parse agent reasonings to surface key strengths and risks.
    Bullish agents → strengths; bearish agents → risks.
    Neutral agents split by keyword matching.
    """
    AGENT_LABELS = {
        "warren_buffett": "Buffett",
        "nassim_taleb":   "Taleb",
        "hedge_fund":     "Hedge Fund",
        "quant":          "Quant",
    }

    RISK_KEYWORDS = [
        "overvalued", "expensive", "fragile", "leverage", "debt", "declining",
        "negative", "risk", "weak", "loss", "short", "crash", "warning",
        "poor", "burn", "dilut",
    ]

    strengths, risks = [], []

    for r in results:
        label = AGENT_LABELS.get(r.name, r.name)
        text = r.reasoning.strip()

        if r.signal == "bullish":
            strengths.append(f"[{label}] {text}")
        elif r.signal == "bearish":
            risks.append(f"[{label}] {text}")
        else:
            # Classify neutral reasoning by keyword
            lower = text.lower()
            if any(k in lower for k in RISK_KEYWORDS):
                risks.append(f"[{label}] {text}")
            else:
                strengths.append(f"[{label}] {text}")

    return strengths, risks


# ── Position sizing ────────────────────────────────────────────────────────────

def _position_size(
    weighted_score: float,
    consensus_strength: float,
    taleb_signal: str,
) -> str:
    """
    Recommend position size based on conviction and risk.

    full  = >4% portfolio weight  (high conviction BUY, low risk)
    half  = 2-4%                  (moderate conviction)
    small = 0.5-2%                (low conviction or Taleb concern)
    none  = 0%                    (SELL or too risky)
    """
    if weighted_score <= 0.38:
        return "none"

    # Taleb concern overrides sizing upward
    if taleb_signal == "bearish":
        if weighted_score >= 0.65:
            return "small"
        return "none"

    if weighted_score >= 0.70 and consensus_strength >= 0.70:
        return "full"
    elif weighted_score >= 0.58:
        return "half"
    elif weighted_score >= 0.45:
        return "small"
    else:
        return "none"


# ── LLM verdict ───────────────────────────────────────────────────────────────

def _generate_verdict(ticker: str, report_data: dict) -> dict:
    """
    Call Gemini to write a short investment committee narrative,
    confirm the verdict, and set final confidence.
    """
    agg = report_data["aggregation"]
    agents_summary = [
        {
            "agent": r["name"],
            "signal": r["signal"],
            "confidence": r["confidence"],
            "reasoning": r["reasoning"],
        }
        for r in report_data["agents"]
    ]

    facts = {
        "ticker": ticker,
        "preliminary_verdict": agg["preliminary_verdict"],
        "weighted_score_pct": round(agg["weighted_score"] * 100, 1),
        "bullish_weight": agg["bullish_weight"],
        "bearish_weight": agg["bearish_weight"],
        "consensus_strength": agg["consensus_strength"],
        "dissenting_agents": agg["dissent"],
        "agents": agents_summary,
        "strengths": report_data["strengths"],
        "risks": report_data["risks"],
        "position_size": report_data["position_size"],
    }

    system_prompt = (
        "You are the chair of an investment committee. "
        "Four analysts have voted: Warren Buffett (value/quality), "
        "Nassim Taleb (risk/antifragility), Hedge Fund PM (catalyst/momentum), "
        "Quant (statistical/DCF).\n\n"
        "Your job:\n"
        "1. Confirm or adjust the preliminary verdict (BUY/HOLD/SELL) "
        "based on the full picture.\n"
        "2. Set final confidence (0-100). Reduce confidence if agents strongly dissent "
        "or data is thin.\n"
        "3. Write a 2-3 sentence investment committee reasoning that cites the most "
        "important factors across all analysts. Be specific — mention numbers, ratios, "
        "or signals where available.\n\n"
        "Verdict override rules:\n"
        "— Upgrade HOLD→BUY only if ≥3 agents bullish AND no Taleb veto.\n"
        "— Downgrade BUY→HOLD if Taleb bearish >65 confidence.\n"
        "— SELL if ≥3 agents bearish OR Taleb+Quant both bearish >70.\n\n"
        "Confidence rules:\n"
        "— Unanimous (4/4): 85-95\n"
        "— Strong majority (3/4): 70-84\n"
        "— Split (2/4): 50-69\n"
        "— Minority only (1/4): 30-49\n\n"
        "Do not invent data. Return JSON only."
    )

    user_prompt = (
        f"Facts:\n{json.dumps(facts, separators=(',', ':'), ensure_ascii=False)}\n\n"
        "Return exactly:\n"
        "{\n"
        '  "verdict": "BUY" | "HOLD" | "SELL",\n'
        '  "confidence": int,\n'
        '  "reasoning": "2-3 sentence investment committee narrative"\n'
        "}"
    )

    fallback = {
        "verdict": agg["preliminary_verdict"],
        "confidence": agg["weighted_score"] * 100,
        "reasoning": (
            f"Committee verdict based on weighted agent scores "
            f"(bullish {agg['bullish_weight']:.0%}, "
            f"bearish {agg['bearish_weight']:.0%}). "
            f"Dissenting agents: {agg['dissent'] or 'none'}."
        ),
    }

    data = call_llm_json(system_prompt, user_prompt, fallback=fallback, max_tokens=400)
    if data.get("verdict") not in ("BUY", "HOLD", "SELL"):
        logger.warning("[judge] invalid verdict '%s', using fallback", data.get("verdict"))
        return fallback
    return data


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_report(report: JudgeReport) -> None:
    """Print a formatted investment committee report to stdout."""
    verdict_emoji = {"BUY": "🟢", "HOLD": "🟡", "SELL": "🔴"}
    size_label = {
        "full":  "Full position  (>4% portfolio)",
        "half":  "Half position  (2-4% portfolio)",
        "small": "Small position (0.5-2%)",
        "none":  "No position",
    }

    print("\n" + "═" * 60)
    print(f"  INVESTMENT COMMITTEE REPORT — {report.ticker}")
    print("═" * 60)
    print(f"  Verdict   : {verdict_emoji.get(report.verdict, '')} {report.verdict}")
    print(f"  Confidence: {report.confidence}%")
    print(f"  Score     : {report.score_pct}% of max")
    print(f"  Position  : {size_label.get(report.position_size, report.position_size)}")
    print()
    print("  Reasoning:")
    for line in report.reasoning.split(". "):
        if line.strip():
            print(f"    • {line.strip().rstrip('.')}.")
    print()
    print("  ── Agent Votes ──────────────────────────────────────")
    signal_icon = {"bullish": "↑", "neutral": "→", "bearish": "↓"}
    for a in report.agents:
        icon = signal_icon.get(a["signal"], "?")
        print(f"    {icon} {a['name']:<16} {a['signal']:<8} {a['confidence']:>3}%  {a['reasoning']}")
    print()
    if report.strengths:
        print("  ── Strengths ─────────────────────────────────────────")
        for s in report.strengths:
            print(f"    + {s}")
        print()
    if report.risks:
        print("  ── Risks ─────────────────────────────────────────────")
        for r in report.risks:
            print(f"    - {r}")
    print("═" * 60 + "\n")