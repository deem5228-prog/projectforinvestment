# ── main.py ─────────────────────────────────────────────────────────────────
# FastAPI backend — AI Stock Analysis System
# Orchestrates: Judge Agent (which internally runs 4 AI Agents) → Response
# ────────────────────────────────────────────────────────────────────────────

import sys
import os
import logging
import asyncio
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import json
import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ── Path & Env Setup ────────────────────────────────────────────────────────

# Ensure the project root (parent of src/) is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

# ── Internal Imports ─────────────────────────────────────────────────────────

from src.agents.judge_agent import judge_agent, JudgeReport
from src.data.api import is_valid_ticker


# ── Request / Response Schemas ───────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """POST /analyze request body."""
    ticker: str = Field(..., description="Stock ticker symbol, e.g. AAPL")
    date: str = Field(
        default=None,
        description="Analysis end date (YYYY-MM-DD). Defaults to today.",
    )

    def get_end_date(self) -> str:
        return self.date or datetime.now().strftime("%Y-%m-%d")


class BatchRequest(BaseModel):
    """POST /analyze/batch request body."""
    tickers: list[str] = Field(..., description="List of ticker symbols, e.g. ['AAPL', 'GOOGL']")
    date: str | None = Field(
        default=None,
        description="Analysis end date (YYYY-MM-DD). Defaults to today.",
    )


class AgentResult(BaseModel):
    name: str | None = None
    signal: str | None = None
    confidence: float | None = None
    reasoning: str | None = None


class AnalyzeResponse(BaseModel):
    ticker: str
    verdict: str
    confidence: float
    position_size: str
    weighted_score: float
    consensus_strength: str
    agents: list[AgentResult]
    strengths: list[str]
    risks: list[str]
    verdict_reasoning: str


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str


# ── Thread Pool (for blocking calls) ─────────────────────────────────────────

_executor = ThreadPoolExecutor(max_workers=4)


# ── Core Analysis Pipeline ───────────────────────────────────────────────────

def _run_judge(ticker: str, end_date: str) -> JudgeReport:
    """
    Run the full analysis pipeline via judge_agent.
    judge_agent internally:
      1. Runs 4 AI agents in parallel (Buffett, Taleb, Hedge Fund, Quant)
      2. Each agent fetches its own data from yfinance
      3. Aggregates results with weighted voting + Taleb veto
      4. Calls Gemini LLM for final verdict narrative
    """
    return judge_agent(ticker, end_date, parallel=False)


def _report_to_response(report: JudgeReport) -> dict:
    """Convert JudgeReport dataclass to API response dict."""

    # Map consensus_strength from float (0-1) to label
    score_pct = report.score_pct
    if score_pct >= 70:
        consensus_label = "strong"
    elif score_pct >= 50:
        consensus_label = "moderate"
    else:
        consensus_label = "weak"

    return {
        "ticker": report.ticker.upper(),
        "verdict": report.verdict,
        "confidence": report.confidence,
        "position_size": report.position_size,
        "weighted_score": round(report.score_pct / 100, 4),
        "consensus_strength": consensus_label,
        "agents": [
            AgentResult(
                name=a.get("name"),
                signal=a.get("signal"),
                confidence=a.get("confidence"),
                reasoning=a.get("reasoning"),
            )
            for a in report.agents
        ],
        "strengths": report.strengths,
        "risks": report.risks,
        "verdict_reasoning": report.reasoning,
    }


async def run_analysis(ticker: str, end_date: str) -> dict:
    """
    Full analysis pipeline:
      1. Validate ticker exists
      2. Judge agent runs 4 AI agents in parallel
      3. Each agent fetches data from yfinance independently
      4. Judge aggregates into final verdict
    """
    loop = asyncio.get_event_loop()

    # Validate ticker
    is_valid = await loop.run_in_executor(_executor, is_valid_ticker, ticker)
    if not is_valid:
        raise ValueError(f"Ticker symbol '{ticker}' is not found or has no data on Yahoo Finance.")

    logger.info("=" * 60)
    logger.info("🚀 ANALYSIS START: %s (end_date=%s)", ticker, end_date)
    logger.info("=" * 60)

    # Run judge_agent in thread pool (it blocks for data fetching + LLM calls)
    report = await loop.run_in_executor(_executor, _run_judge, ticker, end_date)

    logger.info(
        "   ↳ Verdict: %s (confidence=%d%%, position=%s)",
        report.verdict, report.confidence, report.position_size,
    )

    response = _report_to_response(report)

    logger.info("=" * 60)
    logger.info("✅ ANALYSIS COMPLETE: %s → %s", ticker, response["verdict"])
    logger.info("=" * 60)

    return response


# ── FastAPI App ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup & shutdown events."""
    logger.info("🟢 AI Stock Analysis System starting up...")
    logger.info("   Server: http://localhost:%s", os.getenv("BACKEND_PORT", "8000"))
    logger.info("   Docs:   http://localhost:%s/docs", os.getenv("BACKEND_PORT", "8000"))
    yield
    _executor.shutdown(wait=False)
    logger.info("🔴 Server shutting down...")


app = FastAPI(
    title="AI Stock Analysis System",
    description=(
        "ระบบวิเคราะห์หุ้นอัตโนมัติด้วย Multi-Agent AI — "
        "จำลองมุมมองจากนักลงทุนระดับโลก 4 สไตล์ "
        "(Buffett · Taleb · Hedge Fund · Quant) "
        "แล้วให้ Judge Agent สรุปคำแนะนำสุดท้าย"
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS (allow React frontend) ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        timestamp=datetime.now().isoformat(),
    )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_stock(request: AnalyzeRequest):
    """
    วิเคราะห์หุ้นด้วย Multi-Agent AI

    **Flow:**
    1. Judge Agent สั่งรัน 4 AI Agents พร้อมกัน (Buffett, Taleb, Hedge Fund, Quant)
    2. แต่ละ Agent ดึงข้อมูลจาก yfinance และวิเคราะห์อิสระจากกัน
    3. Judge Agent รวมผลด้วย weighted voting + Taleb veto
    4. Claude LLM สรุป verdict สุดท้าย

    **Request Body:**
    ```json
    { "ticker": "AAPL", "date": "2024-12-31" }
    ```
    """
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")

    end_date = request.get_end_date()

    try:
        result = await run_analysis(ticker, end_date)
        return AnalyzeResponse(**result)
    except ValueError as ve:
        logger.warning("⚠️ Validation failed for %s: %s", ticker, ve)
        raise HTTPException(
            status_code=400,
            detail=str(ve),
        )
    except Exception as e:
        logger.error("❌ Analysis failed for %s: %s", ticker, e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed for {ticker}: {str(e)}",
        )


@app.post("/analyze/stream", tags=["Analysis"])
async def analyze_stock_stream(request: AnalyzeRequest):
    """
    SSE streaming endpoint — same analysis as /analyze but sends
    real-time progress events so the frontend can show a live pipeline tracker.

    Event types:
      - progress: {step, status, detail}  — pipeline step update
      - result:   full analysis JSON      — final result
      - error:    {detail}                — fatal error
    """
    ticker = request.ticker.strip().upper()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker is required")
    end_date = request.get_end_date()

    # Thread-safe queue for progress events
    progress_q: queue.Queue = queue.Queue()

    def on_progress(step_id: str, status: str, detail: str = ""):
        """Called from judge_agent (in a worker thread) to push events."""
        progress_q.put({"type": "progress", "step": step_id, "status": status, "detail": detail})

    def _run_pipeline():
        """Blocking pipeline — runs in a background thread."""
        try:
            # Validate ticker
            if not is_valid_ticker(ticker):
                progress_q.put({"type": "error", "detail": f"Ticker '{ticker}' not found on Yahoo Finance."})
                return

            on_progress("validate", "done", f"Ticker {ticker} validated")

            report = judge_agent(ticker, end_date, parallel=False, on_progress=on_progress)
            response = _report_to_response(report)
            progress_q.put({"type": "result", "data": response})
        except Exception as e:
            logger.error("❌ Stream analysis failed for %s: %s", ticker, e, exc_info=True)
            progress_q.put({"type": "error", "detail": f"Analysis failed: {str(e)}"})

    async def event_generator():
        """Async generator that yields SSE-formatted events."""
        # Start the blocking pipeline in a background thread
        thread = threading.Thread(target=_run_pipeline, daemon=True)
        thread.start()

        while True:
            # Use async sleep instead of blocking queue.get to keep event loop alive
            # This allows FastAPI to flush SSE chunks to the client immediately
            await asyncio.sleep(0.3)

            # Drain all available events from the queue
            events_batch = []
            while not progress_q.empty():
                try:
                    events_batch.append(progress_q.get_nowait())
                except queue.Empty:
                    break

            for event in events_batch:
                yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"

                # Terminal events — stop the generator
                if event["type"] in ("result", "error"):
                    return

            # If thread is dead and queue is empty, we're done
            if not thread.is_alive() and progress_q.empty():
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(request: BatchRequest):
    """
    วิเคราะห์หุ้นหลายตัวพร้อมกัน

    **Request Body:**
    ```json
    { "tickers": ["AAPL", "GOOGL", "MSFT"], "date": "2024-12-31" }
    ```
    """
    tickers = request.tickers
    date = request.date

    if not tickers or len(tickers) > 10:
        raise HTTPException(
            status_code=400,
            detail="Provide 1-10 tickers",
        )

    end_date = date or datetime.now().strftime("%Y-%m-%d")

    results = []
    for ticker in tickers:
        try:
            result = await run_analysis(ticker.strip().upper(), end_date)
            results.append(result)
        except Exception as e:
            logger.error("❌ Batch analysis failed for %s: %s", ticker, e)
            results.append({
                "ticker": ticker.upper(),
                "verdict": "ERROR",
                "confidence": 0,
                "position_size": "none",
                "weighted_score": 0.0,
                "consensus_strength": "none",
                "agents": [],
                "strengths": [],
                "risks": [f"Analysis failed: {str(e)}"],
                "verdict_reasoning": f"Error: {str(e)}",
            })

    return {"results": results, "count": len(results)}


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("BACKEND_PORT", "8000"))
    logger.info("Starting server on port %d...", port)

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=port,
        reload=True,
        reload_dirs=[os.path.join(PROJECT_ROOT, "src")],
        log_level="info",
    )