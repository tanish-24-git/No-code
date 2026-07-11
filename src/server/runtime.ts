import { existsSync, readFileSync, statSync } from 'node:fs';
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
import { TrainingWatcher } from './watch/trainingWatcher';
import { isAlive, killTree } from './exec/terminal';

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
  /** In-flight interactive waits (approvals / questions). Lost on restart —
   *  the persisted pending* fields on the session re-arm the UI cards, and
   *  the answering routes fall back to a wake() when no waiter exists. */
  private readonly approvalWaiters = new Map<string, (approved: boolean) => void>();
  private readonly questionWaiters = new Map<string, (answer: string) => void>();
  private readonly watchers = new Map<string, TrainingWatcher>();

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
    // Root node of the agent graph.
    this.bus.emit(
      record.id,
      'agent.spawned',
      {
        agentRunId: 'orchestrator',
        parentRunId: null,
        definitionId: 'orchestrator',
        name: 'Orchestrator',
        ephemeral: false,
        agentKind: 'llm',
      },
      'orchestrator',
    );
    return record;
  }

  /** Push a steering note into an active loop (no chat echo). */
  steerSession(sessionId: string, note: string): boolean {
    const act = this.active.get(sessionId);
    if (!act) return false;
    act.steering.push(note);
    return true;
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

    // Hard-wired stop intent: kill session processes BEFORE the LLM sees it.
    if (/^\s*(stop|abort|cancel|halt|kill)\b/i.test(text)) {
      const killed = this.killSessionProcesses(sessionId);
      if (killed > 0) {
        this.bus.emit(sessionId, 'chat.message', {
          role: 'system',
          text: `Stopped ${killed} running process(es) immediately.`,
        });
      }
    }

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
    this.killSessionProcesses(sessionId);
    const act = this.active.get(sessionId);
    // Reject any interactive waits and clear the persisted pendings.
    const session = this.store.get(sessionId);
    if (session?.pendingApproval) this.decideApproval(sessionId, session.pendingApproval.approvalId, false, true);
    if (session?.pendingQuestion) {
      const w = this.questionWaiters.get(session.pendingQuestion.questionId);
      this.questionWaiters.delete(session.pendingQuestion.questionId);
      this.store.update(sessionId, (r) => {
        r.pendingQuestion = null;
      });
      w?.('[canceled by user]');
    }
    if (!act) return false;
    act.abort.abort();
    return true;
  }

  // ---- interactive waits (used by tools via ToolCtx) ----

  /** Emit approval.requested, persist the pending card, await the decision. */
  requestApproval(args: {
    sessionId: string;
    agentRunId: string;
    tool: string;
    summary: string;
    body?: string;
    payload?: Record<string, unknown>;
  }): Promise<boolean> {
    const approvalId = ulid();
    const { sessionId } = args;
    const prevStatus = this.store.get(sessionId)?.status ?? 'running';
    this.store.update(sessionId, (r) => {
      r.pendingApproval = {
        approvalId,
        tool: args.tool,
        summary: args.summary,
        payload: args.payload ?? {},
      };
      r.status = 'awaiting_approval';
    });
    this.bus.emit(sessionId, 'session.status', { status: 'awaiting_approval' });
    this.bus.emit(
      sessionId,
      'approval.requested',
      { approvalId, tool: args.tool, summary: args.summary, body: args.body },
      args.agentRunId,
    );
    return new Promise<boolean>((resolve) => {
      this.approvalWaiters.set(approvalId, (approved) => {
        this.store.update(sessionId, (r) => {
          r.pendingApproval = null;
          r.status = prevStatus === 'awaiting_approval' ? 'running' : prevStatus;
        });
        this.bus.emit(sessionId, 'session.status', { status: 'running' });
        resolve(approved);
      });
    });
  }

  /** Route entry: resolve an approval card. Returns false when unknown. */
  decideApproval(sessionId: string, approvalId: string, approved: boolean, silent = false): boolean {
    const session = this.store.get(sessionId);
    const pending = session?.pendingApproval;
    if (!session || !pending || pending.approvalId !== approvalId) return false;
    if (!silent) this.bus.emit(sessionId, 'approval.decided', { approvalId, approved });
    const waiter = this.approvalWaiters.get(approvalId);
    this.approvalWaiters.delete(approvalId);
    if (waiter) {
      waiter(approved);
    } else {
      // Server restarted since the card was raised: clear + wake the loop.
      this.store.update(sessionId, (r) => {
        r.pendingApproval = null;
        r.status = 'idle';
      });
      this.wake(
        sessionId,
        `[approval decision] ${pending.tool}: ${approved ? 'APPROVED' : 'DENIED'} — continue accordingly.`,
      );
    }
    return true;
  }

  /** Emit user.ask, persist the pending question, await the answer. */
  askUser(args: {
    sessionId: string;
    agentRunId: string;
    question: string;
    kind: 'text' | 'single' | 'multi' | 'yes_no';
    options?: string[];
  }): Promise<string> {
    const questionId = ulid();
    const { sessionId } = args;
    this.store.update(sessionId, (r) => {
      r.pendingQuestion = { questionId, question: args.question, kind: args.kind, options: args.options };
      r.status = 'awaiting_user';
    });
    this.bus.emit(sessionId, 'session.status', { status: 'awaiting_user' });
    this.bus.emit(
      sessionId,
      'user.ask',
      { questionId, question: args.question, kind: args.kind, options: args.options },
      args.agentRunId,
    );
    return new Promise<string>((resolve) => {
      this.questionWaiters.set(questionId, (answer) => {
        this.store.update(sessionId, (r) => {
          r.pendingQuestion = null;
          r.status = 'running';
        });
        this.bus.emit(sessionId, 'session.status', { status: 'running' });
        resolve(answer);
      });
    });
  }

  /** Route entry: answer a pending question. Returns false when it doesn't match. */
  answerQuestion(sessionId: string, questionId: string, value: string): boolean {
    const session = this.store.get(sessionId);
    const pending = session?.pendingQuestion;
    if (!session || !pending || pending.questionId !== questionId) return false;
    this.bus.emit(sessionId, 'user.answer', { questionId, value });
    const waiter = this.questionWaiters.get(questionId);
    this.questionWaiters.delete(questionId);
    if (waiter) {
      waiter(value);
    } else {
      this.store.update(sessionId, (r) => {
        r.pendingQuestion = null;
        r.status = 'idle';
      });
      this.wake(sessionId, `[answer to your earlier question "${pending.question}"] ${value}`);
    }
    return true;
  }

  // ---- training watch (M5: the detach/wake mechanic) ----

  /** Attach the pure-code watcher; the LLM loop suspends until it fires. */
  attachTrainingWatcher(args: {
    sessionId: string;
    pid: number;
    logFile: string;
    metricsFile: string;
    stallMinutes: number;
    initialLogOffset?: number;
    initialMetricsOffset?: number;
  }): void {
    this.watchers.get(args.sessionId)?.dispose();
    // Fresh attach: start at the metrics file's CURRENT size — a previous
    // run's appended metrics must not poison this run's loss statistics
    // (false divergence). Resume passes explicit offsets instead.
    const metricsStart =
      args.initialMetricsOffset ??
      (existsSync(args.metricsFile) ? statSync(args.metricsFile).size : 0);
    const logStart = args.initialLogOffset ?? 0; // each bg run gets a unique log file
    this.store.update(args.sessionId, (r) => {
      r.watcher = {
        pid: args.pid,
        logFile: args.logFile,
        metricsFile: args.metricsFile,
        logOffset: logStart,
        metricsOffset: metricsStart,
        stallMinutes: args.stallMinutes,
      };
      r.status = 'training';
    });
    this.bus.emit(args.sessionId, 'session.status', { status: 'training' });
    const watcher = new TrainingWatcher({
      ...args,
      initialLogOffset: logStart,
      initialMetricsOffset: metricsStart,
      bus: this.bus,
      store: this.store,
      onWake: (notification) => {
        this.watchers.delete(args.sessionId);
        this.store.update(args.sessionId, (r) => {
          r.watcher = null;
          if (r.status === 'training') r.status = 'idle';
        });
        this.wake(args.sessionId, notification);
      },
    });
    this.watchers.set(args.sessionId, watcher);
    watcher.start();
  }

  /** Kill all live processes of a session + dispose its watcher. */
  killSessionProcesses(sessionId: string): number {
    const session = this.store.get(sessionId);
    let killed = 0;
    for (const proc of session?.processes ?? []) {
      if (isAlive(proc.pid)) {
        killTree(proc.pid);
        killed++;
      }
    }
    const watcher = this.watchers.get(sessionId);
    if (watcher) {
      watcher.dispose();
      this.watchers.delete(sessionId);
      this.store.update(sessionId, (r) => {
        r.watcher = null;
        if (r.status === 'training') r.status = 'idle';
      });
    }
    return killed;
  }

  /** Budget top-up: raise the ceiling and resume a budget-paused loop. */
  topUpBudget(sessionId: string, addUsd: number): { newBudgetUsd: number } | null {
    const rec = this.store.update(sessionId, (r) => {
      r.budgetUsd = Math.round((r.budgetUsd + addUsd) * 100) / 100;
      r.budgetWarned = false; // re-arm the soft warning for the new level
    });
    if (!rec) return null;
    this.bus.emit(sessionId, 'budget.topup', { addUsd, newBudgetUsd: rec.budgetUsd });
    if (rec.status === 'paused_budget') {
      this.setStatus(sessionId, 'idle');
      this.wake(sessionId, `[budget increased to $${rec.budgetUsd.toFixed(2)} — continue where you left off]`);
    }
    return { newBudgetUsd: rec.budgetUsd };
  }

  /** Dataset upload hook: record + notify the loop (steer or wake). */
  notifyDatasetUpload(sessionId: string, fileName: string, sizeBytes: number): void {
    this.store.update(sessionId, (r) => {
      if (!r.datasetFiles.includes(fileName)) r.datasetFiles.push(fileName);
    });
    this.bus.emit(sessionId, 'agent.artifact', {
      artifactKind: 'dataset',
      label: fileName,
      path: `dataset/${fileName}`,
    });
    this.bus.emit(sessionId, 'chat.message', {
      role: 'system',
      text: `Dataset uploaded: ${fileName} (${(sizeBytes / 1024 / 1024).toFixed(2)} MB)`,
    });
    const note = `[dataset uploaded] dataset/${fileName} (${(sizeBytes / 1024 / 1024).toFixed(2)} MB). Inspect it and tell the user what you found; ask for their goal if unknown.`;
    const act = this.active.get(sessionId);
    if (act) act.steering.push(note);
    else this.wake(sessionId, note);
  }

  private startOrchestrator(sessionId: string, history: import('ai').ModelMessage[]): void {
    const steering = new SteeringQueue();
    const abort = new AbortController();
    this.setStatus(sessionId, 'running');
    this.bus.emit(sessionId, 'agent.status', { status: 'working' }, 'orchestrator');

    const promise = runAgentLoop({
      sessionId,
      agentRunId: 'orchestrator',
      definition: orchestrator,
      messages: history,
      cfg: this.config,
      bus: this.bus,
      store: this.store,
      manager: this,
      workspaceDir: this.workspaceDir(sessionId),
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
        this.bus.emit(
          sessionId,
          'agent.status',
          { status: result.subtype === 'error_during_execution' ? 'failed' : 'done' },
          'orchestrator',
        );
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
      } else if (session.status === 'training' && session.watcher) {
        const w = session.watcher;
        if (isAlive(w.pid)) {
          // Training survived the restart — re-attach at the stored offsets.
          console.log(`[finetune-studio] re-attaching training watcher (session ${session.id}, pid ${w.pid})`);
          this.attachTrainingWatcher({
            sessionId: session.id,
            pid: w.pid,
            logFile: w.logFile,
            metricsFile: w.metricsFile,
            stallMinutes: w.stallMinutes,
            initialLogOffset: w.logOffset,
            initialMetricsOffset: w.metricsOffset,
          });
        } else {
          // The run ended while we were down — classify from the log tail.
          const tail = this.readTail(w.logFile, 1_500);
          const failed = /error|failed|traceback/i.test(tail);
          this.store.update(session.id, (r) => {
            r.watcher = null;
            r.status = 'idle';
          });
          this.bus.emit(session.id, 'train.phase', { phase: failed ? 'failed' : 'finished', pid: w.pid });
          this.wake(
            session.id,
            `[training notification] the server restarted; training process ${w.pid} is no longer running — it ${
              failed ? 'appears to have FAILED' : 'appears to have finished'
            }. Log tail:\n${tail}\nVerify output/ and continue accordingly.`,
          );
        }
      }
    }
    console.log('[finetune-studio] runtime ready', {
      dataDir: this.config.dataDir,
      workspacesDir: this.config.workspacesDir,
      approvalMode: this.config.approvalMode,
      sessions: this.store.list().length,
    });
  }

  private readTail(file: string, chars: number): string {
    try {
      if (!existsSync(file)) return '(log file missing)';
      const size = statSync(file).size;
      const text = readFileSync(file, 'utf8');
      return text.slice(Math.max(0, size - chars));
    } catch {
      return '(unreadable log)';
    }
  }
}

const g = globalThis as unknown as { __ftRuntime?: SessionManager };

export function getRuntime(): SessionManager {
  return (g.__ftRuntime ??= new SessionManager(loadConfig()));
}

export { ulid };
