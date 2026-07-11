import { z } from 'zod';
import { registerTool } from './index';

/**
 * The one big gate (D6). The orchestrator proposes its master plan once;
 * after approval every generated script auto-runs. In APPROVAL_MODE=auto the
 * plan is auto-approved (still emitted so the UI shows it).
 */
registerTool<{
  title?: string;
  plan_markdown: string;
  phases: string[];
  estimated_cost_usd?: number;
}>({
  name: 'propose_plan',
  description:
    'Present your master plan for user approval BEFORE any preprocessing or training work. Include: what you learned about the dataset, the training approach (model, method, key hyperparameters), the phases you will execute, and the estimated LLM budget cost. Call it again with a revised plan if the user denies.',
  inputSchema: z.object({
    title: z.string().optional(),
    plan_markdown: z.string().min(1).describe('The full plan, markdown.'),
    phases: z.array(z.string()).min(1).describe('Ordered phase names, e.g. ["preprocess","train","evaluate","publish"].'),
    estimated_cost_usd: z.number().min(0).optional(),
  }),
  parallelSafe: false,
  async execute(input, ctx) {
    const body =
      input.plan_markdown +
      `\n\nPhases: ${input.phases.join(' → ')}` +
      (input.estimated_cost_usd !== undefined ? `\nEstimated LLM cost: ~$${input.estimated_cost_usd.toFixed(2)}` : '');

    if (ctx.cfg.approvalMode === 'auto') {
      ctx.store.update(ctx.sessionId, (r) => {
        r.planApproved = true;
      });
      ctx.bus.emit(
        ctx.sessionId,
        'chat.message',
        { role: 'system', text: `Plan auto-approved (APPROVAL_MODE=auto).\n\n${body}` },
        ctx.agentRunId,
      );
      return { text: 'Plan auto-approved (APPROVAL_MODE=auto). Proceed.' };
    }

    const approved = await ctx.manager.requestApproval({
      sessionId: ctx.sessionId,
      agentRunId: ctx.agentRunId,
      tool: 'propose_plan',
      summary: input.title ?? 'Training plan — approve to run',
      body,
    });
    if (!approved) {
      return {
        text: 'User DENIED the plan. Ask what they want changed (or use their comment), then propose a revised plan.',
        isError: true,
      };
    }
    ctx.store.update(ctx.sessionId, (r) => {
      r.planApproved = true;
    });
    return { text: 'Plan APPROVED. Execute it autonomously — no further permission needed except HF upload and budget top-ups.' };
  },
});
