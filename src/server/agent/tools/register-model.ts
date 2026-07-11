import { z } from 'zod';
import { registerTool } from './index';
import { resolveInWorkspace } from '../../paths';
import { addModel } from '../../models/registry';
import { existsSync } from 'node:fs';

registerTool<{
  name: string;
  base_model: string;
  adapter_path: string;
  metrics?: Record<string, unknown>;
  hf_repo_id?: string;
  notes?: string;
}>({
  name: 'register_model',
  description:
    'Record a finished fine-tune in the local model registry (drives the Models page where the user can chat with it). Call after training + evaluation with the final metrics.',
  inputSchema: z.object({
    name: z.string().min(2).max(80).describe('Display name, e.g. "qwen2.5-0.5b-support-bot".'),
    base_model: z.string().min(2).describe('HF id of the base model.'),
    adapter_path: z.string().min(1).describe('Workspace-relative path to the saved model dir (e.g. output/model).'),
    metrics: z.record(z.unknown()).optional().describe('Final metrics: final_loss, eval_loss, perplexity…'),
    hf_repo_id: z.string().optional(),
    notes: z.string().max(500).optional(),
  }),
  parallelSafe: true,
  async execute(input, ctx) {
    const abs = resolveInWorkspace(ctx.workspaceDir, input.adapter_path);
    if (!existsSync(abs)) {
      return { text: `adapter path ${input.adapter_path} does not exist — train first`, isError: true };
    }
    const record = addModel(ctx.cfg.dataDir, {
      name: input.name,
      baseModel: input.base_model,
      adapterPath: abs,
      sessionId: ctx.sessionId,
      hfRepoId: input.hf_repo_id,
      metrics: input.metrics,
      notes: input.notes,
    });
    ctx.bus.emit(
      ctx.sessionId,
      'model.registered',
      { modelId: record.id, name: record.name, baseModel: record.baseModel, hfRepoId: record.hfRepoId },
      ctx.agentRunId,
    );
    ctx.bus.emit(
      ctx.sessionId,
      'agent.artifact',
      { artifactKind: 'model', label: record.name, path: input.adapter_path },
      ctx.agentRunId,
    );
    return { text: `model registered (id ${record.id}) — visible on the Models page` };
  },
});
