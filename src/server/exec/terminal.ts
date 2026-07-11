import { spawn } from 'node:child_process';
import { openSync } from 'node:fs';
import { execFile } from 'node:child_process';

export interface ForegroundResult {
  exitCode: number | null;
  /** stdout+stderr interleaved, capped (head + tail preserved). */
  output: string;
  durationMs: number;
  killed: boolean;
}

const OUTPUT_CAP = 25_000; // chars — the context-protection cap from research
const HEAD_KEEP = 10_000;
const TAIL_KEEP = 14_000;

export function capOutput(s: string): string {
  if (s.length <= OUTPUT_CAP) return s;
  return (
    s.slice(0, HEAD_KEEP) +
    `\n…[${s.length - HEAD_KEEP - TAIL_KEEP} chars truncated]…\n` +
    s.slice(-TAIL_KEEP)
  );
}

/** Kill a process tree. Windows needs taskkill /T; POSIX kills the group. */
export function killTree(pid: number): void {
  if (process.platform === 'win32') {
    execFile('taskkill', ['/PID', String(pid), '/T', '/F'], () => {});
  } else {
    try {
      process.kill(-pid, 'SIGKILL');
    } catch {
      try {
        process.kill(pid, 'SIGKILL');
      } catch {
        // already gone
      }
    }
  }
}

/**
 * Run a shell command to completion. `shell: true` because commands come from
 * the model as strings — the approval gate (not shell quoting) is the
 * security boundary here.
 */
export function runForeground(opts: {
  command: string;
  cwd: string;
  timeoutMs: number;
  env?: Record<string, string>;
  abort?: AbortSignal;
}): Promise<ForegroundResult> {
  const started = Date.now();
  return new Promise((resolve) => {
    const child = spawn(opts.command, {
      shell: true,
      cwd: opts.cwd,
      windowsHide: true,
      env: { ...process.env, ...opts.env },
      detached: process.platform !== 'win32', // POSIX: own group so killTree(-pid) works
    });

    let out = '';
    let done = false;
    let killed = false;

    const finish = (exitCode: number | null) => {
      if (done) return;
      done = true;
      clearTimeout(timer);
      opts.abort?.removeEventListener('abort', onAbort);
      resolve({ exitCode, output: capOutput(out), durationMs: Date.now() - started, killed });
    };

    const onAbort = () => {
      killed = true;
      if (child.pid) killTree(child.pid);
    };

    const timer = setTimeout(() => {
      killed = true;
      if (child.pid) killTree(child.pid);
    }, opts.timeoutMs);
    opts.abort?.addEventListener('abort', onAbort);

    child.stdout?.on('data', (c: Buffer) => (out += c.toString('utf8')));
    child.stderr?.on('data', (c: Buffer) => (out += c.toString('utf8')));
    child.on('error', (err) => {
      out += `\n[spawn error] ${err.message}`;
      finish(null);
    });
    child.on('close', (code) => finish(code));
  });
}

export interface BackgroundHandle {
  pid: number;
  logFile: string;
}

/**
 * Launch a detached process whose stdio goes to a log file. Survives a
 * harness restart (the watcher re-attaches by pid + byte offset).
 */
export function spawnBackground(opts: {
  command: string;
  cwd: string;
  logFile: string;
  env?: Record<string, string>;
}): BackgroundHandle {
  const fd = openSync(opts.logFile, 'a');
  const child = spawn(opts.command, {
    shell: true,
    cwd: opts.cwd,
    windowsHide: true,
    detached: true,
    env: { ...process.env, ...opts.env },
    stdio: ['ignore', fd, fd],
  });
  child.unref();
  if (!child.pid) throw new Error('failed to spawn background process');
  return { pid: child.pid, logFile: opts.logFile };
}

/** Cheap liveness check by pid. */
export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}
