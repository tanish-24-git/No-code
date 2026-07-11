import { z } from 'zod';
import { registerTool } from './index';

registerTool<{ question: string; kind?: 'text' | 'single' | 'multi' | 'yes_no'; options?: string[] }>({
  name: 'ask_user',
  description:
    'Ask the user ONE question and wait for the answer. Use sparingly — for decisions with real quality impact (training goal, output format, risky tradeoffs), not for things you can decide or discover yourself.',
  inputSchema: z.object({
    question: z.string().min(1),
    kind: z.enum(['text', 'single', 'multi', 'yes_no']).optional().describe('Answer widget (default text).'),
    options: z.array(z.string()).max(8).optional().describe('Choices for single/multi.'),
  }),
  parallelSafe: false,
  async execute(input, ctx) {
    const answer = await ctx.manager.askUser({
      sessionId: ctx.sessionId,
      agentRunId: ctx.agentRunId,
      question: input.question,
      kind: input.kind ?? 'text',
      options: input.options,
    });
    return { text: `User answered: ${answer}` };
  },
});
