/**
 * Event taxonomy v2 — client mirror of src/server/events.ts (keep in sync)
 * plus client mirrors of the session shapes. Replaces the old ~60-kind
 * types.ts contract.
 */

export type EventKind =
  | 'chat.message'
  | 'chat.delta'
  | 'user.ask'
  | 'user.answer'
  | 'approval.requested'
  | 'approval.decided'
  | 'error'
  | 'tool.called'
  | 'tool.result'
  | 'budget.usage'
  | 'budget.warning'
  | 'budget.exceeded'
  | 'budget.topup'
  | 'agent.defined'
  | 'agent.spawned'
  | 'agent.status'
  | 'agent.message'
  | 'agent.artifact'
  | 'task.update'
  | 'train.metric'
  | 'train.phase'
  | 'train.anomaly'
  | 'session.status'
  | 'session.done'
  | 'model.registered';

export interface AgentEvent<T = Record<string, unknown>> {
  id: string; // ulid — lexicographic sort == chronological
  ts: string;
  sessionId: string;
  agentRunId?: string;
  kind: EventKind;
  payload: T;
}

// ---- session shapes (mirror of src/server/session.ts) ----

export type SessionStatus =
  | 'idle'
  | 'running'
  | 'awaiting_approval'
  | 'awaiting_user'
  | 'paused_budget'
  | 'training'
  | 'done'
  | 'failed'
  | 'interrupted';

export interface LedgerEntry {
  at: string;
  agentRunId: string;
  usd: number;
  inputTokens: number;
  outputTokens: number;
  estimated: boolean;
}

export interface PendingApproval {
  approvalId: string;
  tool: string;
  summary: string;
  payload: Record<string, unknown>;
}

export interface PendingQuestion {
  questionId: string;
  question: string;
  kind: 'text' | 'single' | 'multi' | 'yes_no';
  options?: string[];
}

export interface SessionRecord {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  status: SessionStatus;
  datasetFiles: string[];
  planApproved: boolean;
  budgetUsd: number;
  ledger: LedgerEntry[];
  pendingApproval: PendingApproval | null;
  pendingQuestion: PendingQuestion | null;
}

export interface SessionListItem {
  id: string;
  title: string;
  status: SessionStatus;
  createdAt: string;
  updatedAt: string;
  datasetFiles: string[];
}

export const STATUS_LABEL: Record<SessionStatus, string> = {
  idle: 'Idle',
  running: 'Working',
  awaiting_approval: 'Awaiting approval',
  awaiting_user: 'Awaiting your answer',
  paused_budget: 'Budget pause',
  training: 'Training',
  done: 'Complete',
  failed: 'Failed',
  interrupted: 'Interrupted',
};

export const STATUS_TONE: Record<SessionStatus, string> = {
  idle: 'bg-white/5 text-white/40',
  running: 'bg-info/10 text-info',
  awaiting_approval: 'bg-warn/10 text-warn',
  awaiting_user: 'bg-warn/10 text-warn',
  paused_budget: 'bg-warn/10 text-warn',
  training: 'bg-info/10 text-info',
  done: 'bg-success/10 text-success',
  failed: 'bg-danger/10 text-danger',
  interrupted: 'bg-white/5 text-white/40',
};

export function spentUsd(ledger: LedgerEntry[] | undefined): number {
  return (ledger ?? []).reduce((sum, e) => sum + e.usd, 0);
}
