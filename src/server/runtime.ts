import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { ulid } from 'ulid';
import { loadConfig, type AppConfig } from './config';
import { EventBus } from './bus';
import { SessionStore, type SessionRecord, type SessionStatus } from './session';
import { runAgentLoop } from './agent/loop';
import { ensureMemory } from './agent/memory';
import { SteeringQueue } from './agent/steering';
import { orchestrator } from './agent/definitions/orchestrator';

interface ActiveRun {
  steering: SteeringQueue;
  abort: AbortController;
  promise: Promise<void>;
}

/**
 * The long-lived harness runtime. Route handlers grab it via getRuntime();
 * it must survive dev-mode HMR recompiles, hence the globalThis stash
 * (plain module scope is wiped on every recompile).
 */
export class SessionManager {
  readonly bus: EventBus;
  readonly store: SessionStore;
  private readonly active = new Map<string, ActiveRun>();

  constructor(readonly config: AppConfig) {
    this.bus = new EventBus(config.dataDir);
    this.store = new SessionStore(config.dataDir);
    mkdirSync(path.join(config.dataDir, 'sessions'), { recursive: true });
    mkdirSync(config.workspacesDir, { recursive: true });
  }

  // ---- session lifecycle ----

  createSession(title?: string): SessionRecord {
    const record = this.store.create(title, this.config.budget.totalUsd);
    ensureMemory(this.config.dataDir, record.id);
    mkdirSync(this.workspaceDir(record.id), { recursive: true });
    this.bus.emit(record.id, 'session.status', { status: record.status });
    return record;
  }

  listSessions(): SessionRecord[] {
    return this.store.list();
  }

  getSession(id: string): SessionRecord | null {
    return this.store.get(id);
  }

  deleteSession(id: string): boolean {
    const act = this.active.get(id);
    if (act) {
      act.abort.abort();
      this.active.delete(id);
    }
    return this.store.delete(id);
  }

  workspaceDir(sessionId: string): string {
    return path.join(this.config.workspacesDir, sessionId);
  }

  isLoopActive(sessionId: string): boolean {
    return this.active.has(sessionId);
  }

  // ---- conversation ----

  /**
   * User message entry point. Loop active → steering queue (drained between
   * tool calls). Otherwise → start the orchestrator from persisted history.
   */
  handleUserMessage(sessionId: string, text: string): { queued: boolean } {
    this.bus.emit(sessionId, 'chat.message', { role: 'user', text });

    const act = this.active.get(sessionId);
    if (act) {
      act.steering.push(text);
      return { queued: true };
    }

    const history = this.store.loadHistory(sessionId, 'orchestrator');
    const session = this.store.get(sessionId);
    if (session?.status === 'interrupted') {
      history.push({
        role: 'user',
        content: `[server restarted — reassess state from your history and memory before continuing] ${text}`,
      });
    } else {
      history.push({ role: 'user', content: text });
    }
    this.startOrchestrator(sessionId, history);
    return { queued: false };
  }

  /** Re-invoke the orchestrator with a synthetic (non-user) notification turn. */
  wake(sessionId: string, notification: string): void {
    const act = this.active.get(sessionId);
    if (act) {
      act.steering.push(notification);
      return;
    }
    const history = this.store.loadHistory(sessionId, 'orchestrator');
    history.push({ role: 'user', content: notification });
    this.startOrchestrator(sessionId, history);
  }

  cancelSession(sessionId: string): boolean {
    const act = this.active.get(sessionId);
    if (!act) return false;
    act.abort.abort();
    return true;
  }

  private startOrchestrator(sessionId: string, history: import('ai').ModelMessage[]): void {
    const steering = new SteeringQueue();
    const abort = new AbortController();
    this.setStatus(sessionId, 'running');

    const promise = runAgentLoop({
      sessionId,
      agentRunId: 'orchestrator',
      definition: orchestrator,
      messages: history,
      cfg: this.config,
      bus: this.bus,
      store: this.store,
      steering,
      abort: abort.signal,
      depth: 0,
    })
      .then((result) => {
        const session = this.store.get(sessionId);
        // Don't clobber states set by tools mid-run (training / pauses).
        const holdStates: SessionStatus[] = ['training', 'paused_budget', 'awaiting_approval', 'awaiting_user'];
        if (session && !holdStates.includes(session.status)) {
          this.setStatus(sessionId, result.subtype === 'paused_budget' ? 'paused_budget' : 'idle');
        }
      })
      .catch((err) => {
        this.bus.emit(sessionId, 'error', { message: `Harness error: ${err instanceof Error ? err.message : err}` });
        this.setStatus(sessionId, 'idle');
      })
      .finally(() => {
        this.active.delete(sessionId);
      });

    this.active.set(sessionId, { steering, abort, promise });
  }

  setStatus(sessionId: string, status: SessionStatus): void {
    const rec = this.store.update(sessionId, (r) => {
      r.status = status;
    });
    if (rec) this.bus.emit(sessionId, 'session.status', { status });
  }

  // ---- boot ----

  /** Called once at server boot from instrumentation.ts. */
  async resumeAll(): Promise<void> {
    for (const session of this.store.list()) {
      if (session.status === 'running') {
        // The loop died with the old process; next user message resumes it.
        this.store.update(session.id, (r) => {
          r.status = 'interrupted';
        });
      }
      // M5: re-attach training watchers / synthesize wake for finished runs.
    }
    console.log('[finetune-studio] runtime ready', {
      dataDir: this.config.dataDir,
      workspacesDir: this.config.workspacesDir,
      approvalMode: this.config.approvalMode,
      sessions: this.store.list().length,
    });
  }
}

const g = globalThis as unknown as { __ftRuntime?: SessionManager };

export function getRuntime(): SessionManager {
  return (g.__ftRuntime ??= new SessionManager(loadConfig()));
}

export { ulid };
