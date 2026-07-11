import { z } from 'zod';
import { registerTool } from './index';
import { resolveInWorkspace } from '../../paths';
import { isAlive } from '../../exec/terminal';

registerTool<{ pid: number; logFile: string; metricsFile?: string; stall_minutes?: number }>({
  name: 'watch_training',
  description:
    'Attach the zero-cost training watcher to a background run (from run_terminal run_in_background). It streams metrics to the UI and WAKES the agent on completion, nan/inf loss, OOM, crash, divergence, or stall. After attaching: END YOUR TURN with a short status — never poll.',
  inputSchema: z.object({
    pid: z.number().int().positive(),
    logFile: z.string().min(1).describe('Workspace-relative log path from run_terminal.'),
    metricsFile: z.string().optional().describe('Workspace-relative metrics JSONL (default logs/metrics.jsonl).'),
    stall_minutes: z.number().int().min(2).max(240).optional().describe('No-metrics stall threshold (default 15).'),
  }),
  parallelSafe: false,
  async execute(input, ctx) {
    const logFile = resolveInWorkspace(ctx.workspaceDir, input.logFile);
    const metricsFile = resolveInWorkspace(ctx.workspaceDir, input.metricsFile ?? 'logs/metrics.jsonl');
    if (!isAlive(input.pid)) {
      return {
        text: `pid ${input.pid} is not running — the process died immediately. Read the log (${input.logFile}) to see why before retrying.`,
        isError: true,
      };
    }
    ctx.manager.attachTrainingWatcher({
      sessionId: ctx.sessionId,
      pid: input.pid,
      logFile,
      metricsFile,
      stallMinutes: input.stall_minutes ?? 15,
    });
    return {
      text: `Watcher attached to pid ${input.pid}. END YOUR TURN NOW with a one-line status for the user — you will be woken automatically on completion or anomaly. Zero tokens are spent while training runs.`,
    };
  },
});
