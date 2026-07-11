import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { ulid } from 'ulid';
import type { ModelMessage } from 'ai';

export type SessionStatus =
  | 'idle' // waiting for the user
  | 'running' // agent loop active
  | 'awaiting_approval' // plan/command approval card pending
  | 'awaiting_user' // ask_user question pending
  | 'paused_budget' // budget exceeded, top-up needed
  | 'training' // detached training run in progress, loop suspended
  | 'done'
  | 'failed'
  | 'interrupted'; // server died mid-loop; next message resumes

export interface LedgerEntry {
  at: string;
  agentRunId: string;
  usd: number;
  inputTokens: number;
  outputTokens: number;
  estimated: boolean;
}

export interface ProcessRecord {
  pid: number;
  command: string;
  logFile: string;
  startedAt: string;
  kind: 'training' | 'other';
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

export interface WatcherState {
  pid: number;
  logFile: string;
  metricsFile: string;
  logOffset: number;
  metricsOffset: number;
  stallMinutes: number;
}

export interface SessionRecord {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  status: SessionStatus;
  datasetFiles: string[];
  planApproved: boolean;
  /** Effective budget: env value at creation + top-ups. */
  budgetUsd: number;
  ledger: LedgerEntry[];
  /** Soft-warning emitted (once per budget level). */
  budgetWarned: boolean;
  processes: ProcessRecord[];
  pendingApproval: PendingApproval | null;
  pendingQuestion: PendingQuestion | null;
  watcher: WatcherState | null;
  /** Dynamically-created agent definitions (M4), persisted for resume. */
  ephemeralAgents: Record<string, unknown>[];
  /** "Approve & allow similar" prefixes (every-command mode). */
  commandAllowPrefixes: string[];
}

const SAFE_ID = /^[0-9A-HJKMNP-TV-Z]{26}$/; // ulid alphabet — also the path-traversal guard

export function isValidSessionId(id: string): boolean {
  return SAFE_ID.test(id);
}

/** JSON-on-disk session store with atomic writes (temp + rename). */
export class SessionStore {
  constructor(private readonly dataDir: string) {}

  sessionsRoot(): string {
    return path.join(this.dataDir, 'sessions');
  }

  sessionDir(id: string): string {
    return path.join(this.sessionsRoot(), id);
  }

  private sessionFile(id: string): string {
    return path.join(this.sessionDir(id), 'session.json');
  }

  private historyFile(id: string, agentRunId: string): string {
    return path.join(this.sessionDir(id), 'history', `${agentRunId}.json`);
  }

  create(title: string | undefined, budgetUsd: number): SessionRecord {
    const id = ulid();
    const now = new Date().toISOString();
    const record: SessionRecord = {
      id,
      title: title?.trim() || `Session ${now.slice(0, 16).replace('T', ' ')}`,
      createdAt: now,
      updatedAt: now,
      status: 'idle',
      datasetFiles: [],
      planApproved: false,
      budgetUsd,
      ledger: [],
      budgetWarned: false,
      processes: [],
      pendingApproval: null,
      pendingQuestion: null,
      watcher: null,
      ephemeralAgents: [],
      commandAllowPrefixes: [],
    };
    mkdirSync(path.join(this.sessionDir(id), 'history'), { recursive: true });
    this.save(record);
    return record;
  }

  get(id: string): SessionRecord | null {
    if (!isValidSessionId(id)) return null;
    const file = this.sessionFile(id);
    if (!existsSync(file)) return null;
    try {
      return JSON.parse(readFileSync(file, 'utf8')) as SessionRecord;
    } catch {
      return null;
    }
  }

  list(): SessionRecord[] {
    const root = this.sessionsRoot();
    if (!existsSync(root)) return [];
    const out: SessionRecord[] = [];
    for (const entry of readdirSync(root)) {
      if (!isValidSessionId(entry)) continue;
      const rec = this.get(entry);
      if (rec) out.push(rec);
    }
    // ulid ids sort chronologically; newest first
    return out.sort((a, b) => (a.id < b.id ? 1 : -1));
  }

  save(record: SessionRecord): void {
    record.updatedAt = new Date().toISOString();
    this.atomicWrite(this.sessionFile(record.id), JSON.stringify(record, null, 2));
  }

  /** Load-mutate-save; single-process single-writer, so no lock needed. */
  update(id: string, fn: (rec: SessionRecord) => void): SessionRecord | null {
    const rec = this.get(id);
    if (!rec) return null;
    fn(rec);
    this.save(rec);
    return rec;
  }

  delete(id: string): boolean {
    if (!isValidSessionId(id)) return false;
    const dir = this.sessionDir(id);
    if (!existsSync(dir)) return false;
    rmSync(dir, { recursive: true, force: true });
    return true;
  }

  loadHistory(id: string, agentRunId: string): ModelMessage[] {
    const file = this.historyFile(id, agentRunId);
    if (!existsSync(file)) return [];
    try {
      return JSON.parse(readFileSync(file, 'utf8')) as ModelMessage[];
    } catch {
      return [];
    }
  }

  saveHistory(id: string, agentRunId: string, messages: ModelMessage[]): void {
    const file = this.historyFile(id, agentRunId);
    mkdirSync(path.dirname(file), { recursive: true });
    this.atomicWrite(file, JSON.stringify(messages, null, 2));
  }

  private atomicWrite(file: string, content: string): void {
    mkdirSync(path.dirname(file), { recursive: true });
    const tmp = `${file}.${process.pid}.tmp`;
    writeFileSync(tmp, content, 'utf8');
    // Windows: rename fails with EPERM/EBUSY when another handle briefly has
    // the destination open (concurrent reads, AV scans). Retry with backoff —
    // the graceful-fs approach.
    let delay = 2;
    for (let attempt = 0; ; attempt++) {
      try {
        renameSync(tmp, file);
        return;
      } catch (err) {
        const code = (err as NodeJS.ErrnoException).code;
        if (attempt >= 20 || !['EPERM', 'EBUSY', 'EACCES'].includes(code ?? '')) throw err;
        // Synchronous backoff (Node allows Atomics.wait on the main thread).
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, delay);
        delay = Math.min(delay * 2, 50);
      }
    }
  }
}
