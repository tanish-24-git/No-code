import { z } from 'zod';
import { registerTool } from './index';

const ARTIFACT_KINDS = ['dataset', 'script', 'config', 'checkpoint', 'metrics', 'eval-report', 'model', 'model-card'] as const;

registerTool<{
  status: 'working' | 'done' | 'failed' | 'progress';
  summary: string;
  artifact?: { kind: (typeof ARTIFACT_KINDS)[number]; label: string; path?: string };
}>({
  name: 'report_status',
  description:
    'Report progress to the UI (drives the live agent graph). Use `progress` for milestones mid-task, `done`/`failed` right before your final message. Attach an artifact when you produced one (script, processed dataset, model, report).',
  inputSchema: z.object({
    status: z.enum(['working', 'done', 'failed', 'progress']),
    summary: z.string().min(1).max(300),
    artifact: z
      .object({
        kind: z.enum(ARTIFACT_KINDS),
        label: z.string().min(1).max(120),
        path: z.string().optional(),
      })
      .optional(),
  }),
  parallelSafe: true,
  async execute(input, ctx) {
    ctx.bus.emit(
      ctx.sessionId,
      'agent.status',
      { status: input.status === 'progress' ? 'working' : input.status, summary: input.summary },
      ctx.agentRunId,
    );
    if (input.artifact) {
      ctx.bus.emit(
        ctx.sessionId,
        'agent.artifact',
        { artifactKind: input.artifact.kind, label: input.artifact.label, path: input.artifact.path },
        ctx.agentRunId,
      );
    }
    return { text: 'status reported' };
  },
});
