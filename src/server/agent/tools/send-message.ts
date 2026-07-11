import { z } from 'zod';
import { registerTool } from './index';

registerTool<{ to: string; content: string }>({
  name: 'send_message',
  description:
    'Send a note to another agent run (usually "orchestrator"). Delivered between its turns. Use report_status for progress — this is for information another agent needs to act on.',
  inputSchema: z.object({
    to: z.string().min(1).describe('Target agentRunId, e.g. "orchestrator".'),
    content: z.string().min(1).max(2000),
  }),
  parallelSafe: true,
  async execute(input, ctx) {
    ctx.bus.emit(
      ctx.sessionId,
      'agent.message',
      { fromRunId: ctx.agentRunId, toRunId: input.to, messageKind: 'chat', contentPreview: input.content.slice(0, 160) },
      ctx.agentRunId,
    );
    if (input.to === 'orchestrator') {
      ctx.manager.steerSession(ctx.sessionId, `[message from ${ctx.agentRunId}] ${input.content}`);
      return { text: 'delivered to orchestrator (read between its turns)' };
    }
    // Point-to-point delivery to concurrent workers arrives in M5+ if needed;
    // the event above still renders the communication edge in the graph.
    return { text: `noted for ${input.to} (visible in the agent graph)` };
  },
});
