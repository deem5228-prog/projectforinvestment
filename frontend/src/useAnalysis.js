import { useState, useCallback, useRef } from 'react'

const STREAM_URL = '/analyze/stream'

// Pipeline steps in display order
const PIPELINE_STEPS = [
  { id: 'validate',            label: 'Validate Ticker',      icon: '🔍' },
  { id: 'normalizer',          label: 'Fetch Financial Data',  icon: '📊' },
  { id: 'dcf_valuation',       label: 'DCF Fair Value',        icon: '💰' },
  { id: 'agent_warren_buffett', label: 'Buffett Agent',        icon: '🏛️' },
  { id: 'agent_nassim_taleb',  label: 'Taleb Agent',         icon: '🦢' },
  { id: 'agent_hedge_fund',    label: 'Hedge Fund Agent',    icon: '📈' },
  { id: 'agent_quant',         label: 'Quant Agent',         icon: '🔢' },
  { id: 'aggregation',         label: 'Aggregate & Vote',    icon: '⚖️' },
  { id: 'verdict',             label: 'Final Verdict (LLM)', icon: '🤖' },
]

/**
 * useAnalysis — SSE streaming hook.
 *
 * Returns:
 *   result    — final analysis JSON (null until complete)
 *   loading   — boolean
 *   error     — string or null
 *   steps     — array of { id, label, icon, status, detail } for pipeline tracker
 *   analyze   — function(ticker, date) to start analysis
 */
export function useAnalysis() {
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)
  const [steps, setSteps]     = useState(() =>
    PIPELINE_STEPS.map(s => ({ ...s, status: 'pending', detail: '' }))
  )
  const abortRef = useRef(null)

  const updateStep = useCallback((stepId, status, detail = '') => {
    setSteps(prev => prev.map(s =>
      s.id === stepId ? { ...s, status, detail } : s
    ))
  }, [])

  const resetSteps = useCallback(() => {
    setSteps(PIPELINE_STEPS.map(s => ({ ...s, status: 'pending', detail: '' })))
  }, [])

  const analyze = useCallback(async (ticker, date) => {
    if (!ticker || !ticker.trim()) {
      setError('Please enter a ticker symbol.')
      return
    }

    // Abort any previous request
    if (abortRef.current) {
      abortRef.current.abort()
    }
    const controller = new AbortController()
    abortRef.current = controller

    setLoading(true)
    setError(null)
    setResult(null)
    resetSteps()

    try {
      const payload = {
        ticker: ticker.trim().toUpperCase(),
        date: date || null,
      }

      // Use fetch (not axios) for SSE streaming via POST
      const response = await fetch(STREAM_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
        signal: controller.signal,
      })

      if (!response.ok) {
        const errBody = await response.json().catch(() => ({}))
        throw new Error(errBody.detail || `Server error ${response.status}`)
      }

      // Read SSE stream
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // Parse SSE frames: "event: xxx\ndata: {...}\n\n"
        const frames = buffer.split('\n\n')
        buffer = frames.pop() || '' // Keep incomplete frame in buffer

        for (const frame of frames) {
          if (!frame.trim()) continue

          // Extract data line
          const dataLine = frame.split('\n').find(l => l.startsWith('data: '))
          if (!dataLine) continue

          try {
            const event = JSON.parse(dataLine.slice(6))

            if (event.type === 'progress') {
              updateStep(event.step, event.status, event.detail || '')
            } else if (event.type === 'result') {
              setResult(event.data)
              setLoading(false)
              return
            } else if (event.type === 'error') {
              setError(event.detail || 'Unknown error')
              setLoading(false)
              return
            }
          } catch (parseErr) {
            console.warn('[useAnalysis] Failed to parse SSE event:', parseErr)
          }
        }
      }

      // Stream ended without result/error event — check if we got a result
      if (!result) {
        setError('Connection closed before analysis completed.')
      }
    } catch (err) {
      if (err.name === 'AbortError') return // User cancelled
      setError(err.message || 'Unknown error')
    } finally {
      setLoading(false)
    }
  }, [resetSteps, updateStep])

  return { result, loading, error, steps, analyze }
}
