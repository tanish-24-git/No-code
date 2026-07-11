import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { ulid } from 'ulid';
import type { AgentEvent, EventKind } from './events';

type Subscriber = (e: AgentEvent) => void;

/**
 * Per-session event bus. emit() persists to the session's events.jsonl FIRST
 * (append-before-fan-out: a subscriber crash never loses an event), then fans
 * out to live SSE subscribers. Synchronous append keeps ordering trivial —
 * single process, low volume (train.metric is throttled at the watcher).
 */
export class EventBus {
  private subs = new Map<string, Set<Subscriber>>();

  constructor(private readonly dataDir: string) {}

  private eventsFile(sessionId: string): string {
    return path.join(this.dataDir, 'sessions', sessionId, 'events.jsonl');
  }

  emit<T>(sessionId: string, kind: EventKind, payload: T, agentRunId?: string): AgentEvent<T> {
    const event: AgentEvent<T> = {
      id: ulid(),
      ts: new Date().toISOString(),
      sessionId,
      agentRunId,
      kind,
      payload,
    };
    const file = this.eventsFile(sessionId);
    mkdirSync(path.dirname(file), { recursive: true });
    appendFileSync(file, JSON.stringify(event) + '\n', 'utf8');
    for (const cb of this.subs.get(sessionId) ?? []) {
      try {
        cb(event as AgentEvent);
      } catch {
        // one bad subscriber never blocks the others
      }
    }
    return event;
  }

  subscribe(sessionId: string, cb: Subscriber): () => void {
    let set = this.subs.get(sessionId);
    if (!set) {
      set = new Set();
      this.subs.set(sessionId, set);
    }
    set.add(cb);
    return () => {
      set!.delete(cb);
      if (set!.size === 0) this.subs.delete(sessionId);
    };
  }

  /** Replay persisted events, optionally only those after `sinceId` (ulid compare = lexicographic). */
  replay(sessionId: string, sinceId?: string): AgentEvent[] {
    const file = this.eventsFile(sessionId);
    if (!existsSync(file)) return [];
    const out: AgentEvent[] = [];
    for (const line of readFileSync(file, 'utf8').split('\n')) {
      if (!line.trim()) continue;
      try {
        const e = JSON.parse(line) as AgentEvent;
        if (!sinceId || e.id > sinceId) out.push(e);
      } catch {
        // tolerate a torn tail line from a crash
      }
    }
    return out;
  }
}
