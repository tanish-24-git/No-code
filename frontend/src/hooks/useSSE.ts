import { useState, useEffect, useRef } from 'react'

export interface LogEvent {
  timestamp: string
  run_id: string
  agent: string
  level: string
  message: string
  [key: string]: any
}

interface UseSSEOptions {
  onMessage?: (event: LogEvent) => void
  onError?: (error: Event) => void
  autoConnect?: boolean
}

export function useSSE(url: string | null, options: UseSSEOptions = {}) {
  const { onMessage, onError, autoConnect = true } = options
  const [logs, setLogs] = useState<LogEvent[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const eventSourceRef = useRef<EventSource | null>(null)

  useEffect(() => {
    if (!url || !autoConnect) return

    // Create EventSource connection
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onopen = () => {
      setIsConnected(true)
      setError(null)
    }

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        // Skip keepalive messages
        if (data.type === 'keepalive') return

        const logEvent: LogEvent = data
        
        setLogs((prev) => [...prev, logEvent])
        onMessage?.(logEvent)
      } catch (err) {
        console.error('Failed to parse SSE message:', err)
      }
    }

    eventSource.onerror = (err) => {
      setIsConnected(false)
      setError('Connection lost')
      onError?.(err)
      
      // Auto-reconnect after 3 seconds
      setTimeout(() => {
        if (eventSourceRef.current?.readyState === EventSource.CLOSED) {
          eventSource.close()
          // Trigger re-render to reconnect
          setError('Reconnecting...')
        }
      }, 3000)
    }

    return () => {
      eventSource.close()
      setIsConnected(false)
    }
  }, [url, autoConnect, onMessage, onError])

  const clearLogs = () => setLogs([])

  const disconnect = () => {
    eventSourceRef.current?.close()
    setIsConnected(false)
  }

  return {
    logs,
    isConnected,
    error,
    clearLogs,
    disconnect,
  }
}
