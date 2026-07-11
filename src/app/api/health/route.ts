import { loadConfig, llmConfigured, missingLlmKeys } from '@/server/config';
import { getSysInfo } from '@/server/sysinfo';

export const dynamic = 'force-dynamic';

export async function GET() {
  const cfg = loadConfig();
  const sys = await getSysInfo();
  return Response.json({
    status: 'ok',
    llm: {
      configured: llmConfigured(cfg),
      missing: missingLlmKeys(cfg),
      model: cfg.llm.model ?? null,
      apiStyle: cfg.llm.apiStyle,
      thinking: cfg.llm.thinking.mode,
      freeTier: cfg.pricing.inputPerM === 0 && cfg.pricing.outputPerM === 0,
    },
    budget: {
      totalUsd: cfg.budget.totalUsd,
    },
    approvalMode: cfg.approvalMode,
    uv: sys.uv,
    gpu: sys.gpu,
    cpu: sys.cpu,
    ramTotalGb: sys.ramTotalGb,
    platform: sys.platform,
  });
}
