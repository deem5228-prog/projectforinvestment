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

function LoadingSpinner() {
  return (
    <div className="loading-container">
      <div className="spinner-rings">
        <div className="ring ring-1" />
        <div className="ring ring-2" />
        <div className="ring ring-3" />
      </div>
      <p className="loading-label">Running 4 AI agents in parallel…</p>
      <p className="loading-sublabel">Buffett · Taleb · Hedge Fund · Quant</p>
    </div>
  )
}

// ── Main Dashboard ───────────────────────────────────────────────────────────

export default function Dashboard() {
  const [ticker, setTicker] = useState('')
  const [date, setDate]     = useState('')
  const { result, loading, error, analyze } = useAnalysis()

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

        {/* ── Loading state ── */}
        {loading && <LoadingSpinner />}

        {/* ── Results ── */}
        {result && !loading && <ResultPanel result={result} />}
      </main>

      <footer className="footer">
        <p>InvestAI · For educational purposes only. Not financial advice.</p>
      </footer>
    </div>
  )
}
