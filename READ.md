# 📈 AI Stock Analysis System

ระบบวิเคราะห์หุ้นอัตโนมัติด้วย Multi-Agent AI — จำลองมุมมองจากนักลงทุนระดับโลก 4 สไตล์ แล้วให้ Judge Agent สรุปคำแนะนำสุดท้าย

---

## 🏗️ Architecture ภาพรวม

```
User Input (Ticker)
        │
        ▼
┌───────────────────────────────────────┐
│           Layer 1 — Data Layer        │
│   api.py · models.py · cache.py       │
│         (yfinance backend)            │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│        Layer 2 — Normalizer           │
│             normalizer.py             │
│   59 computed metrics · 11 กลุ่ม     │
└──────────────────┬────────────────────┘
                   │
       ┌───────────┴───────────┐
       │   Layer 3 — AI Agents  │  (parallel)
       │                       │
  ┌────┴────┐  ┌──────┐  ┌────┴─────┐  ┌──────┐
  │ Buffett │  │Taleb │  │Hedge Fund│  │Quant │
  └────┬────┘  └──┬───┘  └────┬─────┘  └──┬───┘
       └──────────┴───────────┴────────────┘
                               │
                               ▼
┌───────────────────────────────────────┐
│        Layer 4 — Judge Agent          │
│            judge_agent.py             │
│  Weighted Vote · Taleb Veto · Verdict │
└──────────────────┬────────────────────┘
                   │
                   ▼
        FastAPI Endpoint (main.py)
                   │
                   ▼
        React Dashboard (frontend)
```

---

## 📁 โครงสร้างโปรเจกต์

```
project/
├── src/
│   ├── data/
│   │   ├── api.py              # ดึงข้อมูลจาก yfinance
│   │   ├── models.py           # Pydantic data models
│   │   └── cache.py            # In-memory cache system
│   │
│   ├── analysis/
│   │   └── normalizer.py       # คำนวณ 59 metrics จากข้อมูลดิบ
│   │
│   ├── agents/
│   │   ├── buffett_agent.py    # Value investing analysis
│   │   ├── taleb_agent.py      # Tail-risk & antifragility
│   │   ├── hedge_fund_agent.py # Momentum & catalyst
│   │   └── quant_agent.py      # Statistical & quantitative
│   │
│   └── orchestration/
│       └── judge_agent.py      # รวมผล + สรุป verdict สุดท้าย
│
├── backend/
│   └── main.py                 # FastAPI endpoints
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── Dashboard.jsx
│   │   └── hooks/useAnalysis.js
│   └── package.json
│
├── .env
├── requirements.txt
└── README.md
```

---

## 🤖 AI Agents

### Warren Buffett Agent (`buffett_agent.py`)
วิเคราะห์แบบ Value Investing เน้นธุรกิจที่มี moat แข็งแกร่ง

| Function | วิเคราะห์อะไร |
|---|---|
| Earnings Power | ROIC vs WACC, FCF yield, margin stability |
| Business Quality | Moat indicators, revenue predictability |
| Valuation | Owner earnings, DCF, margin of safety |
| Management | Insider ownership, capital allocation |

---

### Nassim Taleb Agent (`taleb_agent.py`)
วิเคราะห์ความเปราะบางของหุ้นและ tail risk

| Function | วิเคราะห์อะไร |
|---|---|
| Fragility | Debt load, fixed cost leverage, cash runway |
| Convexity | Asymmetric upside vs downside |
| Tail Risk | Max drawdown, vol clustering, black swan exposure |
| Antifragility | Cash buffer, optionality, crisis performance |

> ⚠️ **Taleb Veto**: ถ้า Taleb bearish ≥ 70% confidence จะ block BUY ลงเป็น HOLD เสมอ

---

### Hedge Fund Agent (`hedge_fund_agent.py`)
วิเคราะห์แบบ Hedge Fund เน้น catalyst และ momentum

| Function | Max Score | วิเคราะห์อะไร |
|---|---|---|
| `analyze_momentum` | 10 | MA alignment, 52W position, RSI |
| `analyze_catalyst` | 8 | News sentiment, revenue acceleration |
| `analyze_relative_value` | 8 | EV/EBITDA, FCF yield, PEG |
| `analyze_short_squeeze` | 6 | Recovery from low, vol spike, beta |
| `analyze_earnings_quality` | 8 | FCF conversion, accruals, margin trend |
| `analyze_flow_signal` | 6 | Insider net buying, trade size |
| `analyze_macro_overlay` | 6 | Trend stability, Sharpe, drawdown |
| **Total** | **52** | |

---

### Quant Agent (`quant_agent.py`)
วิเคราะห์เชิงสถิติล้วนๆ ไม่มีความเห็นเชิงคุณภาพ

| Function | Max Score | วิเคราะห์อะไร |
|---|---|---|
| `analyze_dcf_valuation` | 12 | 3-stage DCF จาก FCF จริง |
| `analyze_multiples` | 10 | EV/EBITDA, PEG, FCF yield, P/B |
| `analyze_multifactor` | 12 | Value + Quality + Momentum + Low-Vol |
| `analyze_mean_reversion` | 8 | Z-score, Bollinger Band position |
| `analyze_quality_screen` | 10 | Piotroski F-Score (9 binary signals) |
| `analyze_statistical_edge` | 8 | Autocorrelation, skewness, Calmar |
| `analyze_capital_efficiency` | 8 | ROIC, capex, Greenblatt Magic Formula |
| **Total** | **68** | |

---

## ⚖️ Judge Agent (`judge_agent.py`)

รวมผลจาก 4 agents แล้วออก verdict สุดท้าย

**Weighted Vote:**

| Agent | น้ำหนัก |
|---|---|
| Buffett | 30% |
| Taleb | 25% |
| Hedge Fund | 25% |
| Quant | 20% |

คะแนนถ่วงด้วย confidence ของแต่ละ agent เพิ่มเติม

**Position Sizing:**

| Condition | Position Size |
|---|---|
| High score + strong consensus | `full` |
| Medium score | `half` |
| Low score หรือ mixed signals | `small` |
| Taleb veto หรือ bearish | `none` |

---

## 🧮 Normalizer (`normalizer.py`)

แปลงข้อมูลดิบเป็น 59 computed metrics ใน 11 กลุ่ม

| กลุ่ม | ตัวอย่าง metrics |
|---|---|
| Income | revenue, ebitda, ebitda_margin |
| Balance Sheet | net_cash, invested_capital, tangible_equity |
| Cash Flow | free_cash_flow, owner_earnings, fcf_yield |
| Returns | roic, roce, roe, roa |
| Leverage | interest_coverage, net_debt_ebitda, debt_to_assets |
| Multiples | ev, ev_ebitda, ev_fcf, earnings_yield |
| Growth | rev_cagr_3y, ni_cagr_3y, fcf_growth |
| Efficiency | capex_intensity, rd_intensity, reinvestment_rate |
| Price | ann_vol, sharpe, calmar, max_drawdown, price_52w_pct |
| Composite | piotroski_f (0-9), magic_formula_score |

---

## 🗄️ Data Layer

### Data Sources
ข้อมูลทั้งหมดดึงจาก **yfinance** ผ่าน `api.py`

| Function | ข้อมูล |
|---|---|
| `get_prices()` | OHLCV price history |
| `get_financial_metrics()` | PE, EV/EBITDA, Beta, ROE ฯลฯ |
| `get_income_statement()` | Revenue, Net Income, EBITDA |
| `get_balance_sheet()` | Assets, Debt, Equity |
| `get_cash_flow_statement()` | Operating CF, Capex, FCF |
| `get_insider_trades()` | การซื้อขายของ insiders |
| `get_company_news()` | ข่าวล่าสุด + sentiment |

### Cache System (`cache.py`)
In-memory cache รองรับ composite key `{ticker}_{start}_{end}` และ merge อัตโนมัติเพื่อกัน duplicate

---

## 🚀 Getting Started

### 1. ติดตั้ง dependencies

```bash
pip install fastapi uvicorn anthropic yfinance pydantic python-dotenv
```

### 2. ตั้งค่า environment variables

```bash
# .env
ANTHROPIC_API_KEY=sk-ant-...
```

### 3. รัน backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. รัน frontend

```bash
cd frontend
npm install
npm run dev
```

---

## 📡 API Endpoints

```
POST /analyze
Body: { "ticker": "AAPL", "date": "2024-12-31" }

Response:
{
  "ticker": "AAPL",
  "verdict": "BUY",
  "position_size": "half",
  "weighted_score": 0.72,
  "consensus_strength": "moderate",
  "agents": {
    "buffett": { "signal": "bullish", "confidence": 0.80, ... },
    "taleb":   { "signal": "neutral", "confidence": 0.60, ... },
    "hedge_fund": { ... },
    "quant":   { ... }
  },
  "strengths": [...],
  "risks": [...],
  "verdict_reasoning": "..."
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Data | yfinance |
| Backend | Python · FastAPI · Pydantic |
| AI | Anthropic Claude API (claude-sonnet-4-6) |
| Frontend | React · Tailwind CSS · Recharts |
| Cache | In-memory (Python dict) |

---

## ⚠️ Disclaimer

ระบบนี้สร้างขึ้นเพื่อการศึกษาและวิจัยเท่านั้น ไม่ใช่คำแนะนำทางการเงิน ผลการวิเคราะห์จาก AI ไม่รับประกันผลตอบแทนและไม่ควรใช้เป็นพื้นฐานเดียวในการตัดสินใจลงทุน