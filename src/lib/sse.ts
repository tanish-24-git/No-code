'use client';

/**
 * Per-session event stream. All frames are unnamed SSE `message` events with
 * the kind inside the JSON — one onmessage handler, no per-kind listener
 * registry to keep in sync. The server sends `id:` on every frame, so
 * EventSource's native Last-Event-ID reconnect resumes without duplicates
 * (a seen-id set guards the edge cases anyway).
 */
import { useEffect, useRef, useState } from 'react';
import type { AgentEvent } from './events';

export function useSessionEvents(sessionId: string | null) {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const seen = useRef<Set<string>>(new Set());

  useEffect(() => {
    setEvents([]);
    setConnected(false);
    seen.current = new Set();
    if (!sessionId) return;

    const es = new EventSource(`/api/sessions/${sessionId}/events`);
    es.onopen = () => setConnected(true);
    es.onerror = () => setConnected(false);
    es.onmessage = (e: MessageEvent<string>) => {
      try {
        const event = JSON.parse(e.data) as AgentEvent;
        if (event.sessionId !== sessionId) return;
        if (seen.current.has(event.id)) return;
        seen.current.add(event.id);
        setEvents((prev) => [...prev, event]);
      } catch {
        // keepalive comments / malformed frames
      }
    };
    return () => es.close();
  }, [sessionId]);

  return { events, connected };
}
