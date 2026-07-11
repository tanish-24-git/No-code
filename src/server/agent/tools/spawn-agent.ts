import { z } from 'zod';
import { ulid } from 'ulid';
import { registerTool } from './index';
import { resolveDefinition } from '../registry';
import { runAgentLoop } from '../loop';
import { SteeringQueue } from '../steering';

const MAX_RESULT_CHARS = 8_000; // ~2K tokens back to the orchestrator

registerTool<{
  agent: string;
  objective: string;
  output_format?: string;
  boundaries?: string;
  context_files?: string[];
}>({
  name: 'spawn_agent',
  description:
    'Delegate a task to a specialist agent (see "Available specialist agents" in your context). The worker runs with a FRESH context and returns only its final message — give it a complete delegation packet: objective, output format, boundaries. Emit MULTIPLE spawn_agent calls in ONE turn to run independent workers in parallel (e.g. dataset-analyst ∥ hardware-profiler).',
  inputSchema: z.object({
    agent: z.string().min(1).describe('Definition id, e.g. "dataset-analyst".'),
    objective: z.string().min(1).describe('Complete, self-contained task description.'),
    output_format: z.string().optional().describe('Exactly what the final answer should look like.'),
    boundaries: z.string().optional().describe('What the worker must NOT do / scope limits.'),
    context_files: z.array(z.string()).max(10).optional().describe('Workspace paths the worker should look at.'),
  }),
  parallelSafe: true,
  async execute(input, ctx) {
    if (ctx.depth >= 1) {
      return { text: 'workers cannot spawn agents — return your findings and let the orchestrator delegate', isError: true };
    }
    const definition = resolveDefinition(ctx.store, ctx.sessionId, input.agent);
    if (!definition || definition.id === 'orchestrator') {
      return { text: `unknown agent "${input.agent}" — pick one from the catalog or create_agent first`, isError: true };
    }

    const agentRunId = `${definition.id}-${ulid().slice(-6).toLowerCase()}`;
    ctx.bus.emit(
      ctx.sessionId,
      'agent.spawned',
      {
        agentRunId,
        parentRunId: ctx.agentRunId,
        definitionId: definition.id,
        name: definition.name,
        ephemeral: definition.ephemeral ?? false,
        objective: input.objective.slice(0, 200),
        agentKind: 'llm',
      },
      agentRunId,
    );
    ctx.bus.emit(
      ctx.sessionId,
      'agent.message',
      { fromRunId: ctx.agentRunId, toRunId: agentRunId, messageKind: 'task', contentPreview: input.objective.slice(0, 160) },
      ctx.agentRunId,
    );
    ctx.bus.emit(ctx.sessionId, 'agent.status', { status: 'working' }, agentRunId);

    const packet = [
      `# Objective\n${input.objective}`,
      input.output_format ? `# Output format\n${input.output_format}` : '',
      input.boundaries ? `# Boundaries\n${input.boundaries}` : '',
      input.context_files?.length ? `# Relevant files\n${input.context_files.join('\n')}` : '',
    ]
      .filter(Boolean)
      .join('\n\n');

    const result = await runAgentLoop({
      sessionId: ctx.sessionId,
      agentRunId,
      definition,
      messages: [{ role: 'user', content: packet }],
      cfg: ctx.cfg,
      bus: ctx.bus,
      store: ctx.store,
      manager: ctx.manager,
      workspaceDir: ctx.workspaceDir,
      steering: new SteeringQueue(),
      abort: ctx.abort,
      depth: ctx.depth + 1,
    });

    const ok = result.subtype === 'success';
    const finalText = (result.finalText || '(no final message)').slice(0, MAX_RESULT_CHARS);
    ctx.bus.emit(ctx.sessionId, 'agent.status', { status: ok ? 'done' : 'failed', summary: finalText.slice(0, 160) }, agentRunId);
    ctx.bus.emit(
      ctx.sessionId,
      'agent.message',
      { fromRunId: agentRunId, toRunId: ctx.agentRunId, messageKind: 'result', contentPreview: finalText.slice(0, 160) },
      agentRunId,
    );

    if (!ok) {
      return {
        text: `${definition.name} FAILED (${result.subtype}) — partial output:\n${finalText}`,
        isError: true,
      };
    }
    return { text: `${definition.name} finished:\n${finalText}` };
  },
});
