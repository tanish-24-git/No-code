import type { AppConfig } from '../config';
import type { LedgerEntry } from '../session';
import type { RawUsage } from './provider';

/**
 * Cost math. Pricing comes from env ($/1M tokens). Rules:
 *  - Provider-reported dollar cost (OpenRouter usage.cost) beats local math.
 *  - OpenAI-style completion_tokens ALREADY includes reasoning tokens — never double-count.
 *  - Anthropic-style: cache reads/writes priced separately (defaults 0.1x / 1.25x input).
 *  - `estimated: true` marks entries where usage was missing and we fell back to chars/4.
 */

export interface CostResult {
  usd: number;
  estimated: boolean;
}

export function costOfUsage(usage: RawUsage, cfg: AppConfig): CostResult {
  if (usage.providerCostUsd !== undefined) {
    return { usd: usage.providerCostUsd, estimated: false };
  }
  const inPerM = cfg.pricing.inputPerM ?? 0;
  const outPerM = cfg.pricing.outputPerM ?? 0;
  const cacheReadPerM = cfg.pricing.cacheReadPerM ?? inPerM * 0.1;
  const cacheWritePerM = cfg.pricing.cacheWritePerM ?? inPerM * 1.25;
  // For openai-style, cacheReadTokens are a subset of inputTokens (prompt_tokens_details.cached_tokens);
  // bill the cached share at the cache rate and the remainder at the input rate.
  const freshInput = Math.max(0, usage.inputTokens - (usage.cacheReadTokens ?? 0));
  const usd =
    (freshInput * inPerM +
      (usage.cacheReadTokens ?? 0) * cacheReadPerM +
      (usage.cacheWriteTokens ?? 0) * cacheWritePerM +
      usage.outputTokens * outPerM) /
    1_000_000;
  return { usd, estimated: false };
}

/** chars/4 fallback when the provider returned no usage at all. */
export function estimateUsage(promptChars: number, completionChars: number): RawUsage {
  return {
    inputTokens: Math.ceil(promptChars / 4),
    outputTokens: Math.ceil(completionChars / 4),
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
  };
}

export function spentUsd(ledger: LedgerEntry[]): number {
  return ledger.reduce((sum, e) => sum + e.usd, 0);
}

/** True when both prices are explicitly 0 — free tier, budget effectively unlimited. */
export function isFreeTier(cfg: AppConfig): boolean {
  return cfg.pricing.inputPerM === 0 && cfg.pricing.outputPerM === 0;
}

// ── Gate A: pre-flight projection (checked BEFORE dispatching a model call) ──

export interface PreflightResult {
  ok: boolean;
  spentUsd: number;
  budgetUsd: number;
  /** Worst-case cost of the call about to be made. */
  callCeilingUsd: number;
  /** How much more budget is needed to safely proceed. */
  deltaNeededUsd: number;
}

/**
 * `spent + callCeiling > budget × (1 − finalizeReserve)` → pause BEFORE the
 * call is dispatched (a reactive-only check would spend the tokens first).
 * The finalize reserve keeps a slice for evaluation + publishing; agents
 * marked `finalize` gate against the FULL budget — that's the reserve's job.
 */
export function preflightGate(args: {
  cfg: AppConfig;
  spent: number;
  budgetUsd: number;
  estPromptTokens: number;
  finalize: boolean;
}): PreflightResult {
  const { cfg, spent, budgetUsd, estPromptTokens, finalize } = args;
  if (isFreeTier(cfg)) {
    return { ok: true, spentUsd: spent, budgetUsd, callCeilingUsd: 0, deltaNeededUsd: 0 };
  }
  const inPerM = cfg.pricing.inputPerM ?? 0;
  const outPerM = cfg.pricing.outputPerM ?? 0;
  const callCeilingUsd = (estPromptTokens * inPerM + cfg.llm.maxOutputTokens * outPerM) / 1_000_000;
  const effective = budgetUsd * (finalize ? 1 : 1 - cfg.budget.finalizeReserve);
  const projected = spent + callCeilingUsd;
  const ok = projected <= effective;
  return {
    ok,
    spentUsd: spent,
    budgetUsd,
    callCeilingUsd,
    deltaNeededUsd: ok ? 0 : Math.ceil((projected - effective) * 100) / 100,
  };
}
