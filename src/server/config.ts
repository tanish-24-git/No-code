import path from 'node:path';
import { z } from 'zod';

/**
 * Env-driven configuration. The provider is fully custom — base URL, model,
 * wire style, prices, capabilities and budget all come from .env. Pricing
 * lives here (not in code) because a custom base URL makes model ids opaque
 * to any built-in price registry.
 */

/** `LLM_THINKING` normalized: off, a reasoning-effort keyword, or a raw token budget. */
export type ThinkingConfig =
  | { mode: 'off' }
  | { mode: 'effort'; effort: 'minimal' | 'low' | 'medium' | 'high' }
  | { mode: 'budget'; tokens: number };

const EFFORT_KEYWORDS = ['minimal', 'low', 'medium', 'high'] as const;

function parseThinking(raw: string | undefined): ThinkingConfig {
  const v = (raw ?? 'off').trim().toLowerCase();
  if (!v || v === 'off' || v === 'false' || v === '0') return { mode: 'off' };
  if ((EFFORT_KEYWORDS as readonly string[]).includes(v)) {
    return { mode: 'effort', effort: v as 'minimal' | 'low' | 'medium' | 'high' };
  }
  const n = Number.parseInt(v, 10);
  if (Number.isFinite(n) && n > 0) return { mode: 'budget', tokens: n };
  return { mode: 'off' };
}

const boolish = z
  .string()
  .optional()
  .transform((v) => ['true', '1', 'yes', 'on'].includes((v ?? '').trim().toLowerCase()));

const EnvSchema = z.object({
  LLM_BASE_URL: z.string().url().optional(),
  LLM_API_KEY: z.string().optional(),
  LLM_MODEL: z.string().optional(),
  LLM_API_STYLE: z.enum(['openai', 'anthropic']).default('openai'),
  LLM_MODEL_WORKER: z.string().optional(),

  LLM_PRICE_INPUT: z.coerce.number().min(0).optional(),
  LLM_PRICE_OUTPUT: z.coerce.number().min(0).optional(),
  LLM_PRICE_CACHE_READ: z.coerce.number().min(0).optional(),
  LLM_PRICE_CACHE_WRITE: z.coerce.number().min(0).optional(),

  LLM_THINKING: z.string().optional(),
  LLM_WEB_SEARCH: boolish,
  LLM_CONTEXT_WINDOW: z.coerce.number().int().positive().default(128_000),
  LLM_MAX_OUTPUT_TOKENS: z.coerce.number().int().positive().default(8_192),

  LLM_BUDGET_USD: z.coerce.number().min(0).default(2),
  LLM_BUDGET_SOFT: z.coerce.number().min(0).max(1).default(0.85),
  LLM_BUDGET_FINALIZE_RESERVE: z.coerce.number().min(0).max(0.5).default(0.05),

  APPROVAL_MODE: z.enum(['plan', 'every-command', 'auto']).default('plan'),
  MAX_TURNS: z.coerce.number().int().positive().default(80),

  HF_TOKEN: z.string().optional(),
  DATA_DIR: z.string().default('./data'),
  WORKSPACES_DIR: z.string().default('./workspaces'),
});

export interface AppConfig {
  llm: {
    baseUrl?: string;
    apiKey?: string;
    model?: string;
    apiStyle: 'openai' | 'anthropic';
    workerModel?: string;
    thinking: ThinkingConfig;
    webSearch: boolean;
    contextWindow: number;
    maxOutputTokens: number;
  };
  pricing: {
    /** USD per 1M tokens. `undefined` = unpriced (meter will halt-and-ask). 0 = free tier. */
    inputPerM?: number;
    outputPerM?: number;
    cacheReadPerM?: number;
    cacheWritePerM?: number;
  };
  budget: {
    totalUsd: number;
    softFraction: number;
    finalizeReserve: number;
  };
  approvalMode: 'plan' | 'every-command' | 'auto';
  maxTurns: number;
  hfToken?: string;
  dataDir: string;
  workspacesDir: string;
}

/** Treat present-but-empty env vars ("LLM_API_KEY=") as unset. */
function cleanedEnv(): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(process.env)) {
    if (typeof v === 'string' && v.trim() !== '') out[k] = v.trim();
  }
  return out;
}

export function loadConfig(): AppConfig {
  const env = EnvSchema.parse(cleanedEnv());
  return {
    llm: {
      baseUrl: env.LLM_BASE_URL,
      apiKey: env.LLM_API_KEY,
      model: env.LLM_MODEL,
      apiStyle: env.LLM_API_STYLE,
      workerModel: env.LLM_MODEL_WORKER ?? env.LLM_MODEL,
      thinking: parseThinking(env.LLM_THINKING),
      webSearch: env.LLM_WEB_SEARCH,
      contextWindow: env.LLM_CONTEXT_WINDOW,
      maxOutputTokens: env.LLM_MAX_OUTPUT_TOKENS,
    },
    pricing: {
      inputPerM: env.LLM_PRICE_INPUT,
      outputPerM: env.LLM_PRICE_OUTPUT,
      cacheReadPerM:
        env.LLM_PRICE_CACHE_READ ?? (env.LLM_PRICE_INPUT !== undefined ? env.LLM_PRICE_INPUT * 0.1 : undefined),
      cacheWritePerM:
        env.LLM_PRICE_CACHE_WRITE ?? (env.LLM_PRICE_INPUT !== undefined ? env.LLM_PRICE_INPUT * 1.25 : undefined),
    },
    budget: {
      totalUsd: env.LLM_BUDGET_USD,
      softFraction: env.LLM_BUDGET_SOFT,
      finalizeReserve: env.LLM_BUDGET_FINALIZE_RESERVE,
    },
    approvalMode: env.APPROVAL_MODE,
    maxTurns: env.MAX_TURNS,
    hfToken: env.HF_TOKEN,
    dataDir: path.resolve(env.DATA_DIR),
    workspacesDir: path.resolve(env.WORKSPACES_DIR),
  };
}

/** Which required LLM settings are missing (drives ConfigBanner + /api/health). */
export function missingLlmKeys(cfg: AppConfig): string[] {
  const missing: string[] = [];
  if (!cfg.llm.baseUrl) missing.push('LLM_BASE_URL');
  if (!cfg.llm.model) missing.push('LLM_MODEL');
  // API key is optional (local endpoints like Ollama don't need one) but
  // pricing is required for budget enforcement — explicit 0 means free tier.
  if (cfg.pricing.inputPerM === undefined) missing.push('LLM_PRICE_INPUT');
  if (cfg.pricing.outputPerM === undefined) missing.push('LLM_PRICE_OUTPUT');
  return missing;
}

export function llmConfigured(cfg: AppConfig): boolean {
  return missingLlmKeys(cfg).length === 0;
}
