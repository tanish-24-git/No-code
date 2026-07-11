import { mkdirSync } from 'node:fs';
import path from 'node:path';
import { z } from 'zod';
import { ulid } from 'ulid';
import { resolveInWorkspace, toPosix } from '../../paths';
import { runForeground, spawnBackground } from '../../exec/terminal';
import { ensureVenv } from '../../exec/uv';
import { registerTool, type ToolCtx, type ToolOutcome } from './index';

const inputSchema = z.object({
  command: z.string().min(1).describe('Shell command to execute (runs in the session workspace).'),
  cwd: z.string().optional().describe('Working directory relative to the workspace root (default ".").'),
  timeout_sec: z
    .number()
    .int()
    .positive()
    .max(3600)
    .optional()
    .describe('Foreground timeout in seconds (default 300).'),
  run_in_background: z
    .boolean()
    .optional()
    .describe(
      'For long-running work (training). Returns {pid, logFile} immediately; attach watch_training afterwards and END YOUR TURN.',
    ),
  description: z.string().optional().describe('One line: what this command does (shown to the user).'),
});

type Input = z.infer<typeof inputSchema>;

/** Runners where the SUBCOMMAND defines what the command does. */
const RUNNER_COMMANDS = new Set(['uv', 'python', 'python3', 'py', 'npm', 'npx', 'node', 'git', 'pip', 'pip3', 'huggingface-cli']);

/**
 * The granularity of "Approve & allow similar": the program name — plus the
 * subcommand for runners (`uv pip` vs `uv run`, `git status` vs `git push`).
 */
export function commandPrefix(command: string): string {
  const tokens = command.trim().split(/\s+/);
  const first = (tokens[0] ?? '').toLowerCase();
  return RUNNER_COMMANDS.has(first) && tokens.length > 1 ? `${tokens[0]} ${tokens[1]}` : tokens[0] ?? '';
}

async function gateApproval(input: Input, ctx: ToolCtx): Promise<ToolOutcome | null> {
  if (ctx.cfg.approvalMode !== 'every-command') return null; // 'plan' + 'auto': auto-run (D6)
  const session = ctx.store.get(ctx.sessionId);
  const prefix = commandPrefix(input.command);
  // Match on the STORED prefix being a prefix-token match of this command
  // (so an allowlisted `uv pip` covers `uv pip install pandas`).
  if (session?.commandAllowPrefixes.some((p) => p === prefix || prefix.startsWith(`${p} `))) return null;
  const approved = await ctx.manager.requestApproval({
    sessionId: ctx.sessionId,
    agentRunId: ctx.agentRunId,
    tool: 'run_terminal',
    summary: input.description ?? `Run: ${input.command.slice(0, 80)}`,
    body: `$ ${input.command}\n(cwd: ${input.cwd ?? '.'})`,
    payload: { command: input.command },
  });
  if (!approved) {
    return {
      text: 'User DENIED this command. Do not retry it verbatim — adjust your approach or ask the user.',
      isError: true,
    };
  }
  return null;
}

registerTool<Input>({
  name: 'run_terminal',
  description:
    'Execute a shell command inside the session workspace. Use for running generated Python via `uv run python scripts/x.py`, installing deps via `uv pip install ...`, and inspecting data. Long-running training MUST use run_in_background:true. Do NOT use for reading/writing files (use read_file/write_file).',
  inputSchema,
  parallelSafe: false,
  async execute(input, ctx) {
    const cwd = resolveInWorkspace(ctx.workspaceDir, input.cwd ?? '.');

    const denied = await gateApproval(input, ctx);
    if (denied) return denied;

    // Lazy per-session venv the first time uv is used here.
    if (/^\s*uv\s/.test(input.command)) {
      const venv = await ensureVenv(ctx.workspaceDir);
      if (venv.note) return { text: venv.note, isError: true };
    }

    if (input.run_in_background) {
      const logsDir = path.join(ctx.workspaceDir, 'logs');
      mkdirSync(logsDir, { recursive: true });
      const logFile = path.join(logsDir, `bg-${ulid()}.log`);
      const handle = spawnBackground({ command: input.command, cwd, logFile });
      ctx.store.update(ctx.sessionId, (r) => {
        r.processes.push({
          pid: handle.pid,
          command: input.command,
          logFile,
          startedAt: new Date().toISOString(),
          kind: /train/i.test(input.command) ? 'training' : 'other',
        });
      });
      return {
        text: `Started in background: pid=${handle.pid}, logFile=${toPosix(
          path.relative(ctx.workspaceDir, logFile),
        )}. Attach watch_training to be woken on completion/anomaly, then end your turn.`,
      };
    }

    const res = await runForeground({
      command: input.command,
      cwd,
      timeoutMs: (input.timeout_sec ?? 300) * 1000,
      env: { PYTHONUNBUFFERED: '1', PYTHONIOENCODING: 'utf-8' },
      abort: ctx.abort,
    });
    const status = res.killed
      ? `KILLED (timeout/cancel) after ${(res.durationMs / 1000).toFixed(1)}s`
      : `exit ${res.exitCode} in ${(res.durationMs / 1000).toFixed(1)}s`;
    return {
      text: `${status}\n${res.output || '(no output)'}`,
      isError: res.killed || res.exitCode !== 0,
    };
  },
});
