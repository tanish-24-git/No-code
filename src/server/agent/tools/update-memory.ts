import { z } from 'zod';
import { registerTool } from './index';
import { updateMemorySection } from '../memory';

registerTool<{ section: string; content: string }>({
  name: 'update_memory',
  description:
    'Update one section of the session memory (FINETUNE.md) — it is re-injected into EVERY model call, surviving compaction. Sections: Mission, Dataset facts, Hardware, Plan & decisions, User directives, Current status. Record durable user preferences under "User directives" IMMEDIATELY when stated.',
  inputSchema: z.object({
    section: z.string().min(1),
    content: z.string().min(1).max(4000),
  }),
  parallelSafe: true,
  async execute(input, ctx) {
    const res = updateMemorySection(ctx.cfg.dataDir, ctx.sessionId, input.section, input.content);
    if (!res.ok) return { text: res.error, isError: true };
    return { text: `memory section "${input.section}" updated` };
  },
});
