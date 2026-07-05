import { useState, useCallback } from 'react'
import axios from 'axios'

const API_BASE = '/analyze'

/**
 * useAnalysis — hook that POSTs to the backend and manages state.
 *
 * Usage:
 *   const { result, loading, error, analyze } = useAnalysis()
 *   analyze('AAPL', '2024-12-31')
 */
export function useAnalysis() {
  const [result, setResult]   = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  const analyze = useCallback(async (ticker, date) => {
    if (!ticker || !ticker.trim()) {
      setError('Please enter a ticker symbol.')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const payload = {
        ticker: ticker.trim().toUpperCase(),
        date: date || null,   // null → backend defaults to today
      }

      const { data } = await axios.post(API_BASE, payload, {
        timeout: 120_000,   // 2 min — analysis can take a while
        headers: { 'Content-Type': 'application/json' },
      })

      setResult(data)
    } catch (err) {
      if (axios.isAxiosError(err)) {
        const msg =
          err.response?.data?.detail ||
          err.response?.data?.message ||
          err.message
        setError(msg)
      } else {
        setError(err.message || 'Unknown error')
      }
    } finally {
      setLoading(false)
    }
  }, [])

  return { result, loading, error, analyze }
}
