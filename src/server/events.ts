/**
 * Event taxonomy v2 — the single contract between harness and UI.
 * Every event is appended to data/sessions/<id>/events.jsonl BEFORE fan-out,
 * so the stream is replayable and the UI state is fully reconstructable.
 *
 * Mirrored on the client in src/lib/events.ts — keep the two in sync.
 */

export type EventKind =
  // chat
  | 'chat.message'
  | 'chat.delta'
  | 'user.ask'
  | 'user.answer'
  | 'approval.requested'
  | 'approval.decided'
  | 'error'
  // tools
  | 'tool.called'
  | 'tool.result'
  // budget
  | 'budget.usage'
  | 'budget.warning'
  | 'budget.exceeded'
  | 'budget.topup'
  // agent graph
  | 'agent.defined'
  | 'agent.spawned'
  | 'agent.status'
  | 'agent.message'
  | 'agent.artifact'
  | 'task.update'
  // training
  | 'train.metric'
  | 'train.phase'
  | 'train.anomaly'
  // session / models
  | 'session.status'
  | 'session.done'
  | 'model.registered';

export interface AgentEvent<T = Record<string, unknown>> {
  /** ulid — monotonic, doubles as SSE `id:` for Last-Event-ID resume. */
  id: string;
  ts: string;
  sessionId: string;
  /** Which agent run produced this (undefined = harness/system). */
  agentRunId?: string;
  kind: EventKind;
  payload: T;
}

// ---- payload shapes (the frequently-used ones; the rest are ad-hoc) ----

export interface ChatMessagePayload {
  role: 'user' | 'assistant' | 'system';
  text: string;
}

export interface ChatDeltaPayload {
  channel: 'text' | 'thinking';
  delta: string;
}

export interface ErrorPayload {
  message: string;
}

export interface SessionStatusPayload {
  status: string;
}

export interface BudgetUsagePayload {
  lastCallUsd: number;
  spentUsd: number;
  budgetUsd: number;
  inputTokens: number;
  outputTokens: number;
  estimated: boolean;
}
