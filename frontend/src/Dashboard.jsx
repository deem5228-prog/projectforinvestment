import React, { useState } from 'react'
import { useAnalysis } from './useAnalysis'

// ── Verdict colour config ────────────────────────────────────────────────────
const VERDICT_CFG = {
  BUY:  { color: '#22c55e', glow: '#22c55e40', icon: '▲', label: 'BUY' },
  HOLD: { color: '#f59e0b', glow: '#f59e0b40', icon: '◆', label: 'HOLD' },
  SELL: { color: '#ef4444', glow: '#ef444440', icon: '▼', label: 'SELL' },
  ERROR:{ color: '#6b7280', glow: '#6b728040', icon: '✕', label: 'ERROR' },
}

const SIGNAL_CFG = {
  bullish: { color: '#22c55e', icon: '↑' },
  neutral: { color: '#f59e0b', icon: '→' },
  bearish: { color: '#ef4444', icon: '↓' },
}

const AGENT_NAMES = {
  warren_buffett: 'Warren Buffett',
  nassim_taleb:   'Nassim Taleb',
  hedge_fund:     'Hedge Fund PM',
  quant:          'Quant',
}

// ── Sub-components ───────────────────────────────────────────────────────────

function AgentCard({ agent }) {
  const cfg = SIGNAL_CFG[agent.signal] || { color: '#6b7280', icon: '?' }
  return (
    <div className="agent-card" style={{ borderColor: cfg.color + '50' }}>
      <div className="agent-header">
        <span className="agent-icon" style={{ color: cfg.color }}>{cfg.icon}</span>
        <span className="agent-name">{AGENT_NAMES[agent.name] || agent.name}</span>
        <span className="agent-signal" style={{ color: cfg.color }}>{agent.signal}</span>
      </div>
      <div className="agent-confidence">
        <div className="confidence-bar-track">
          <div
            className="confidence-bar-fill"
            style={{ width: `${agent.confidence}%`, background: cfg.color }}
          />
        </div>
        <span className="confidence-pct" style={{ color: cfg.color }}>{agent.confidence}%</span>
      </div>
      {agent.reasoning && (
        <p className="agent-reasoning">{agent.reasoning}</p>
      )}
    </div>
  )
}

function ResultPanel({ result }) {
  const cfg = VERDICT_CFG[result.verdict] || VERDICT_CFG.HOLD
  const scorePct = Math.round(result.weighted_score * 100)

  return (
    <div className="result-panel" style={{ '--glow': cfg.glow }}>
      {/* ── Header verdict block ── */}
      <div className="verdict-block" style={{ borderColor: cfg.color + '60' }}>
        <div className="verdict-ticker">{result.ticker}</div>
        <div className="verdict-badge" style={{ color: cfg.color, boxShadow: `0 0 32px ${cfg.glow}` }}>
          <span className="verdict-icon">{cfg.icon}</span>
          <span className="verdict-text">{cfg.label}</span>
        </div>
        <div className="verdict-meta">
          <div className="meta-item">
            <span className="meta-label">Confidence</span>
            <span className="meta-value" style={{ color: cfg.color }}>{result.confidence}%</span>
          </div>
          <div className="meta-divider" />
          <div className="meta-item">
            <span className="meta-label">Score</span>
            <span className="meta-value">{scorePct}%</span>
          </div>
          <div className="meta-divider" />
          <div className="meta-item">
            <span className="meta-label">Position</span>
            <span className="meta-value position-badge">{result.position_size}</span>
          </div>
          <div className="meta-divider" />
          <div className="meta-item">
            <span className="meta-label">Consensus</span>
            <span className="meta-value">{result.consensus_strength}</span>
          </div>
        </div>
      </div>

      {/* ── Reasoning ── */}
      {result.verdict_reasoning && (
        <div className="section reasoning-section">
          <h3 className="section-title">📋 Committee Reasoning</h3>
          <p className="reasoning-text">{result.verdict_reasoning}</p>
        </div>
      )}

      {/* ── Agents grid ── */}
      {result.agents && result.agents.length > 0 && (
        <div className="section">
          <h3 className="section-title">🤖 Agent Votes</h3>
          <div className="agents-grid">
            {result.agents.map((a, i) => <AgentCard key={i} agent={a} />)}
          </div>
        </div>
      )}

      {/* ── DCF Fair Value ── */}
      {result.dcf_valuation && !result.dcf_valuation.error && (
        <DcfCard dcf={result.dcf_valuation} />
      )}

      {/* ── Strengths & Risks ── */}
      <div className="strengths-risks-row">
        {result.strengths && result.strengths.length > 0 && (
          <div className="section sr-section">
            <h3 className="section-title">✅ Strengths</h3>
            <ul className="sr-list">
              {result.strengths.map((s, i) => <li key={i} className="sr-item strength-item">{s}</li>)}
            </ul>
          </div>
        )}
        {result.risks && result.risks.length > 0 && (
          <div className="section sr-section">
            <h3 className="section-title">⚠️ Risks</h3>
            <ul className="sr-list">
              {result.risks.map((r, i) => <li key={i} className="sr-item risk-item">{r}</li>)}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

function DcfCard({ dcf }) {
  const upside = dcf.upside_pct
  const isUp = upside != null && upside >= 0
  const color = dcf.verdict === 'Undervalued' ? '#22c55e'
              : dcf.verdict === 'Overvalued'  ? '#ef4444'
              : '#f59e0b'

  const formatNum = (n) => {
    if (n == null) return '—'
    if (Math.abs(n) >= 1e9) return (n / 1e9).toFixed(1) + 'B'
    if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(1) + 'M'
    if (Math.abs(n) >= 1e3) return (n / 1e3).toFixed(1) + 'K'
    return n.toLocaleString()
  }

  return (
    <div className="section dcf-section" style={{ borderColor: color + '40' }}>
      <h3 className="section-title">💰 DCF Fair Value</h3>

      <div className="dcf-hero">
        <div className="dcf-price-block">
          <div className="dcf-label">Fair Value / Share</div>
          <div className="dcf-fair-price" style={{ color }}>
            {dcf.fair_value_per_share?.toFixed(2) ?? '—'}
          </div>
        </div>

        {dcf.current_price && (
          <div className="dcf-price-block">
            <div className="dcf-label">Current Price</div>
            <div className="dcf-current-price">
              {dcf.current_price.toFixed(2)}
            </div>
          </div>
        )}

        {upside != null && (
          <div className="dcf-price-block">
            <div className="dcf-label">Upside / Downside</div>
            <div className="dcf-upside" style={{ color }}>
              {isUp ? '▲' : '▼'} {upside > 0 ? '+' : ''}{upside.toFixed(1)}%
            </div>
          </div>
        )}

        <div className="dcf-price-block">
          <div className="dcf-label">Verdict</div>
          <div className="dcf-verdict-badge" style={{ color, borderColor: color + '60' }}>
            {dcf.verdict}
          </div>
        </div>
      </div>

      {dcf.details && (
        <details className="dcf-details">
          <summary className="dcf-details-summary">📐 DCF Breakdown</summary>
          <div className="dcf-details-grid">
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Base FCF</span>
              <span className="dcf-detail-value">{formatNum(dcf.details.base_fcf)}</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Growth (Stage 1)</span>
              <span className="dcf-detail-value">{dcf.details.growth_rate_stage1}%</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Growth (Stage 2)</span>
              <span className="dcf-detail-value">{dcf.details.growth_rate_stage2}%</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Terminal Growth</span>
              <span className="dcf-detail-value">{dcf.details.terminal_growth}%</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Discount Rate</span>
              <span className="dcf-detail-value">{dcf.details.discount_rate}%</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Safety Margin</span>
              <span className="dcf-detail-value">{dcf.details.safety_margin}%</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">PV Stage 1</span>
              <span className="dcf-detail-value">{formatNum(dcf.details.pv_stage1)}</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">PV Stage 2</span>
              <span className="dcf-detail-value">{formatNum(dcf.details.pv_stage2)}</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">PV Terminal</span>
              <span className="dcf-detail-value">{formatNum(dcf.details.pv_terminal)}</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Enterprise Value</span>
              <span className="dcf-detail-value">{formatNum(dcf.details.enterprise_value)}</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Net Debt</span>
              <span className="dcf-detail-value">{formatNum((dcf.details.total_debt || 0) - (dcf.details.cash || 0))}</span>
            </div>
            <div className="dcf-detail-item">
              <span className="dcf-detail-label">Equity Value</span>
              <span className="dcf-detail-value" style={{ color }}>{formatNum(dcf.details.equity_value)}</span>
            </div>
          </div>
        </details>
      )}
    </div>
  )
}

// ── Pipeline Tracker ─────────────────────────────────────────────────────────

const STATUS_ICONS = {
  pending: '○',
  running: '◌',
  done:    '●',
  error:   '✕',
}

const STATUS_COLORS = {
  pending: 'var(--text-muted)',
  running: 'var(--accent)',
  done:    'var(--green)',
  error:   'var(--red)',
}

function PipelineTracker({ steps, ticker }) {
  return (
    <div className="pipeline-tracker">
      <div className="pipeline-header">
        <span className="pipeline-icon">⚙️</span>
        <span className="pipeline-title">Analysis Pipeline</span>
        <span className="pipeline-ticker">{ticker}</span>
      </div>

      <div className="pipeline-steps">
        {steps.map((step, i) => (
          <div
            key={step.id}
            className={`pipeline-step pipeline-step--${step.status}`}
          >
            {/* Connector line */}
            {i > 0 && (
              <div
                className="pipeline-connector"
                style={{
                  background: step.status === 'pending'
                    ? 'var(--border)'
                    : STATUS_COLORS[step.status],
                }}
              />
            )}

            {/* Step indicator */}
            <div className="step-row">
              <div
                className={`step-indicator step-indicator--${step.status}`}
                style={{ color: STATUS_COLORS[step.status] }}
              >
                {step.status === 'running' ? (
                  <span className="step-spinner" />
                ) : (
                  STATUS_ICONS[step.status]
                )}
              </div>

              <div className="step-content">
                <div className="step-label-row">
                  <span className="step-icon">{step.icon}</span>
                  <span className="step-label">{step.label}</span>
                </div>
                {step.detail && (
                  <span className="step-detail">{step.detail}</span>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  // Default to 1 year ago — yfinance has complete annual data for past dates
  const oneYearAgo = new Date()
  oneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1)
  const defaultDate = oneYearAgo.toISOString().slice(0, 10)

  const [ticker, setTicker] = useState('')
  const [date, setDate]     = useState(defaultDate)
  const { result, loading, error, steps, analyze } = useAnalysis()

  const todayISO = new Date().toISOString().slice(0, 10)

  const handleSubmit = (e) => {
    e.preventDefault()
    analyze(ticker, date || null)
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter') handleSubmit(e)
  }

  return (
    <div className="page">
      {/* ── Background particles ── */}
      <div className="bg-orb orb-1" />
      <div className="bg-orb orb-2" />
      <div className="bg-orb orb-3" />

      {/* ── Header ── */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <span className="logo-text">InvestAI</span>
        </div>
        <p className="header-tagline">Multi-Agent Stock Analysis · Powered by Gemini</p>
      </header>

      {/* ── Input Card ── */}
      <main className="main">
        <section className="input-card">
          <h1 className="card-title">Analyze a Stock</h1>
          <p className="card-subtitle">
            Four AI perspectives — Buffett, Taleb, Hedge Fund &amp; Quant — synthesised into one verdict.
          </p>

          <form className="input-form" onSubmit={handleSubmit}>
            <div className="input-row">
              <div className="input-group flex-2">
                <label htmlFor="ticker-input" className="input-label">Ticker Symbol</label>
                <input
                  id="ticker-input"
                  type="text"
                  className="input-field"
                  placeholder="AAPL, TSLA, MSFT…"
                  value={ticker}
                  onChange={e => setTicker(e.target.value.toUpperCase())}
                  onKeyDown={handleKeyDown}
                  maxLength={10}
                  autoComplete="off"
                  spellCheck={false}
                  disabled={loading}
                />
              </div>
              <div className="input-group flex-1">
                <label htmlFor="date-input" className="input-label">Analysis Date <span className="optional">(optional)</span></label>
                <input
                  id="date-input"
                  type="date"
                  className="input-field date-field"
                  value={date}
                  max={todayISO}
                  onChange={e => setDate(e.target.value)}
                  disabled={loading}
                />
              </div>
            </div>

            <button
              id="analyze-btn"
              type="submit"
              className={`analyze-btn ${loading ? 'loading' : ''}`}
              disabled={loading || !ticker.trim()}
            >
              {loading ? (
                <>
                  <span className="btn-spinner" />
                  <span>Analyzing…</span>
                </>
              ) : (
                <>
                  <span>⚡</span>
                  <span>Analyze</span>
                </>
              )}
            </button>
          </form>

          {error && (
            <div className="error-banner" role="alert">
              <span className="error-icon">✕</span>
              <span>{error}</span>
            </div>
          )}
        </section>

        {/* ── Pipeline Tracker (replaces old spinner) ── */}
        {loading && <PipelineTracker steps={steps} ticker={ticker} />}

        {/* ── Results ── */}
        {result && !loading && <ResultPanel result={result} />}
      </main>

      <footer className="footer">
        <p>InvestAI · For educational purposes only. Not financial advice.</p>
      </footer>
    </div>
  )
}
