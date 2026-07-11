import { closeSync, existsSync, openSync, readSync, statSync } from 'node:fs';
import path from 'node:path';
import type { EventBus } from '../bus';
import type { SessionStore } from '../session';
import { isAlive } from '../exec/terminal';

/**
 * The pure-code training monitor — zero LLM tokens while training runs.
 *
 * Polls the run's log + metrics files (byte-offset incremental reads,
 * offsets checkpointed for crash-resume), streams train.metric to the UI
 * (throttled), classifies anomalies with a regex alternation that covers
 * EVERY terminal state ("silence is not success": nan/inf, OOM, crash,
 * divergence, stall, process death), and wakes the agent loop exactly once.
 */

export type AnomalyKind = 'nan' | 'oom' | 'crash' | 'divergence' | 'stall';

export interface WatcherArgs {
  sessionId: string;
  pid: number;
  logFile: string;
  metricsFile: string;
  stallMinutes: number;
  bus: EventBus;
  store: SessionStore;
  /** Called ONCE per watcher lifetime with the wake notification. */
  onWake: (notification: string) => void;
  /** Resume offsets after a harness restart. */
  initialLogOffset?: number;
  initialMetricsOffset?: number;
}

const POLL_MS = 500;
const METRIC_EMIT_MS = 500; // ≤2 events/sec to the UI
const LOG_TAIL_CHARS = 2_000;

const ANOMALY_PATTERNS: { kind: AnomalyKind; re: RegExp }[] = [
  { kind: 'oom', re: /CUDA out of memory|torch\.OutOfMemoryError|MemoryError|Killed\b/i },
  { kind: 'crash', re: /Traceback \(most recent call last\)|FATAL|Segmentation fault/i },
];

export class TrainingWatcher {
  private timer: ReturnType<typeof setInterval> | null = null;
  private metricFlushTimer: ReturnType<typeof setInterval> | null = null;
  private logOffset: number;
  private metricsOffset: number;
  private lastMetricAt = Date.now();
  private bestLoss = Number.POSITIVE_INFINITY;
  private risingCount = 0;
  private lastLoss: number | null = null;
  private steps = 0;
  private lastEpoch: number | undefined;
  private startedAt = Date.now();
  private woken = false;
  private pendingMetric: Record<string, unknown> | null = null;
  private logTail = '';
  private missingPidPolls = 0;
  private pollCount = 0;

  constructor(private readonly args: WatcherArgs) {
    this.logOffset = args.initialLogOffset ?? 0;
    this.metricsOffset = args.initialMetricsOffset ?? 0;
  }

  start(): void {
    this.timer = setInterval(() => this.poll(), POLL_MS);
    this.metricFlushTimer = setInterval(() => this.flushMetric(), METRIC_EMIT_MS);
    this.args.bus.emit(this.args.sessionId, 'train.phase', {
      phase: 'running',
      pid: this.args.pid,
      logFile: path.basename(this.args.logFile),
    });
  }

  dispose(): void {
    if (this.timer) clearInterval(this.timer);
    if (this.metricFlushTimer) clearInterval(this.metricFlushTimer);
    this.timer = null;
    this.metricFlushTimer = null;
  }

  private poll(): void {
    if (this.woken) return;
    try {
      this.pollCount++;
      this.readNewMetrics();
      const newLog = this.readNewLog();
      if (newLog) this.classifyLog(newLog);
      this.checkStall();
      this.checkProcess();
      // Offsets are only needed for crash-resume — persist sparsely to avoid
      // hammering session.json (Windows rename contention).
      if (this.pollCount % 10 === 0) this.persistOffsets();
    } catch {
      // never let the watcher crash the harness
    }
  }

  // ---- incremental reads ----

  private readNewBytes(file: string, offset: number): { text: string; nextOffset: number } {
    if (!existsSync(file)) return { text: '', nextOffset: offset };
    const size = statSync(file).size;
    if (size <= offset) return { text: '', nextOffset: offset };
    const fd = openSync(file, 'r');
    try {
      const len = Math.min(size - offset, 1024 * 1024);
      const buf = Buffer.alloc(len);
      readSync(fd, buf, 0, len, offset);
      return { text: buf.toString('utf8'), nextOffset: offset + len };
    } finally {
      closeSync(fd);
    }
  }

  private readNewMetrics(): void {
    const { text, nextOffset } = this.readNewBytes(this.args.metricsFile, this.metricsOffset);
    this.metricsOffset = nextOffset;
    if (!text) return;
    for (const rawLine of text.split('\n')) {
      if (!rawLine.trim()) continue;
      // Python's json.dumps emits bare NaN/Infinity for non-finite floats,
      // which JSON.parse rejects — normalize so the nan anomaly still fires.
      const line = rawLine.replace(/\bNaN\b/g, '"nan"').replace(/-?\bInfinity\b/g, '"inf"');
      let m: Record<string, unknown>;
      try {
        m = JSON.parse(line);
      } catch {
        continue;
      }
      this.lastMetricAt = Date.now();
      this.steps++;
      const loss = Number(m.loss);
      if (Number.isFinite(loss)) {
        if (loss < this.bestLoss) this.bestLoss = loss;
        if (this.lastLoss !== null && loss > this.lastLoss) this.risingCount++;
        else this.risingCount = 0;
        this.lastLoss = loss;
        if (typeof m.epoch === 'number') this.lastEpoch = m.epoch;
        this.pendingMetric = m;
        // Divergence: loss exploding vs best, or monotonically rising for a while.
        if (this.steps > 10 && (loss > this.bestLoss * 3 + 1e-9 || this.risingCount >= 20)) {
          this.anomaly('divergence', `loss=${loss} vs best=${this.bestLoss} (rising ${this.risingCount} steps)`);
        }
      } else if (typeof m.loss === 'number' || String(m.loss).match(/nan|inf/i)) {
        this.anomaly('nan', `non-finite loss at step ${m.step ?? this.steps}: ${String(m.loss)}`);
      }
    }
  }

  private readNewLog(): string {
    const { text, nextOffset } = this.readNewBytes(this.args.logFile, this.logOffset);
    this.logOffset = nextOffset;
    if (text) this.logTail = (this.logTail + text).slice(-LOG_TAIL_CHARS);
    return text;
  }

  private classifyLog(chunk: string): void {
    for (const { kind, re } of ANOMALY_PATTERNS) {
      if (re.test(chunk)) {
        this.anomaly(kind, this.logTail.slice(-600));
        return;
      }
    }
    if (/loss[=:\s]+(nan|inf)/i.test(chunk)) {
      this.anomaly('nan', this.logTail.slice(-600));
    }
  }

  // ---- terminal states ----

  private checkStall(): void {
    const stallMs = this.args.stallMinutes * 60_000;
    const sinceStart = Date.now() - this.startedAt;
    const sinceMetric = Date.now() - this.lastMetricAt;
    // Give startup (downloads, tokenization) double grace before calling stall.
    const grace = this.steps === 0 ? stallMs * 2 : stallMs;
    if (sinceStart > grace && sinceMetric > grace) {
      this.anomaly('stall', `no training metrics for ${Math.round(sinceMetric / 60_000)} minutes (pid alive)`);
    }
  }

  private checkProcess(): void {
    if (isAlive(this.args.pid)) {
      this.missingPidPolls = 0;
      return;
    }
    // Debounce one poll, then drain remaining output before judging.
    this.missingPidPolls++;
    if (this.missingPidPolls < 3) return;
    this.readNewMetrics();
    const finalChunk = this.readNewLog();
    if (finalChunk) this.classifyLog(finalChunk);
    if (this.woken) return; // classifier already fired

    const failed = /error|failed|traceback/i.test(this.logTail);
    const minutes = Math.round((Date.now() - this.startedAt) / 60_000);
    this.args.bus.emit(this.args.sessionId, 'train.phase', {
      phase: failed ? 'failed' : 'finished',
      pid: this.args.pid,
    });
    this.wake(
      failed
        ? `[training notification] process ${this.args.pid} EXITED WITH ERRORS after ${minutes}m · ${this.steps} steps · last loss ${this.lastLoss ?? 'n/a'} · log tail:\n${this.logTail}`
        : `[training notification] training process ${this.args.pid} finished after ${minutes}m · ${this.steps} steps · final loss ${this.lastLoss ?? 'n/a'}${this.lastEpoch !== undefined ? ` · epoch ${this.lastEpoch}` : ''} · log tail:\n${this.logTail.slice(-800)}`,
    );
  }

  private anomaly(kind: AnomalyKind, evidence: string): void {
    if (this.woken) return;
    this.args.bus.emit(this.args.sessionId, 'train.anomaly', { anomalyKind: kind, evidence: evidence.slice(0, 800) });
    this.wake(
      `[training anomaly: ${kind}] pid ${this.args.pid} · step ${this.steps} · last loss ${this.lastLoss ?? 'n/a'} · evidence:\n${evidence.slice(0, 1_500)}\nDiagnose and recover: read the full log if needed, fix the config, kill leftovers, relaunch and re-attach watch_training.`,
    );
  }

  private wake(notification: string): void {
    if (this.woken) return;
    this.woken = true;
    this.dispose();
    this.args.onWake(notification);
  }

  // ---- UI + persistence ----

  private flushMetric(): void {
    if (!this.pendingMetric) return;
    const m = this.pendingMetric;
    this.pendingMetric = null;
    this.args.bus.emit(this.args.sessionId, 'train.metric', {
      step: m.step ?? this.steps,
      epoch: m.epoch,
      loss: m.loss,
      lr: m.lr,
    });
  }

  private persistOffsets(): void {
    this.args.store.update(this.args.sessionId, (r) => {
      if (r.watcher && r.watcher.pid === this.args.pid) {
        r.watcher.logOffset = this.logOffset;
        r.watcher.metricsOffset = this.metricsOffset;
      }
    });
  }
}
