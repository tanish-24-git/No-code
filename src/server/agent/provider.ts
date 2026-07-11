import { createAnthropic } from '@ai-sdk/anthropic';
import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import type { LanguageModel } from 'ai';
import type { AppConfig } from '../config';

/**
 * Fully generic provider adapter. The wire protocol comes from LLM_API_STYLE;
 * everything else (base URL, model, key) from env. The custom `fetch` below is
 * the metering shim — the single place where we:
 *   (a) force usage reporting on streams (stream_options.include_usage),
 *   (b) inject thinking config in the provider's native dialect,
 *   (c) tee the response and capture the RAW usage object (incl. fields the
 *       SDK normalizes away: OpenRouter usage.cost, Anthropic cache tokens).
 */

export interface RawUsage {
  inputTokens: number;
  outputTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  /** Exact dollars charged, when the provider reports it (e.g. OpenRouter usage.cost). */
  providerCostUsd?: number;
}

export interface MeterHooks {
  onUsage?: (usage: RawUsage) => void;
  onWarning?: (message: string) => void;
}

const ANTHROPIC_EFFORT_BUDGETS: Record<string, number> = {
  minimal: 1024,
  low: 4096,
  medium: 8192,
  high: 16384,
};

/** Endpoints that rejected our thinking injection — don't retry it every call. */
const thinkingUnsupported = new Set<string>();

export function createChatModel(cfg: AppConfig, modelId: string, hooks: MeterHooks = {}): LanguageModel {
  const fetchImpl = makeMeteredFetch(cfg, modelId, hooks) as typeof fetch;
  if (cfg.llm.apiStyle === 'anthropic') {
    const provider = createAnthropic({
      baseURL: cfg.llm.baseUrl,
      apiKey: cfg.llm.apiKey ?? 'not-needed',
      fetch: fetchImpl,
    });
    return provider(modelId);
  }
  const provider = createOpenAICompatible({
    name: 'custom',
    baseURL: cfg.llm.baseUrl!,
    apiKey: cfg.llm.apiKey ?? 'not-needed',
    fetch: fetchImpl,
    includeUsage: true,
  });
  return provider(modelId);
}

type AnyFetch = (input: any, init?: any) => Promise<Response>;

function makeMeteredFetch(cfg: AppConfig, modelId: string, hooks: MeterHooks): AnyFetch {
  const endpointKey = `${cfg.llm.baseUrl}|${modelId}`;

  return async (input, init) => {
    let body: any;
    if (init?.body && typeof init.body === 'string') {
      try {
        body = JSON.parse(init.body);
      } catch {
        body = undefined;
      }
    }

    let injectedThinking = false;
    if (body && typeof body === 'object') {
      if (cfg.llm.apiStyle === 'openai') {
        if (body.stream) {
          body.stream_options = { include_usage: true, ...(body.stream_options ?? {}) };
        }
        if (cfg.llm.thinking.mode !== 'off' && !thinkingUnsupported.has(endpointKey)) {
          if (cfg.llm.thinking.mode === 'effort') {
            body.reasoning_effort = cfg.llm.thinking.effort;
          } else {
            // Raw token budget — OpenRouter dialect; stripped on 400 below.
            body.reasoning = { max_tokens: cfg.llm.thinking.tokens };
          }
          injectedThinking = true;
        }
      } else {
        // anthropic style
        if (cfg.llm.thinking.mode !== 'off' && !thinkingUnsupported.has(endpointKey) && !body.thinking) {
          const tokens =
            cfg.llm.thinking.mode === 'budget'
              ? cfg.llm.thinking.tokens
              : ANTHROPIC_EFFORT_BUDGETS[cfg.llm.thinking.effort];
          body.thinking = { type: 'enabled', budget_tokens: tokens };
          // Anthropic constraints: max_tokens must exceed the thinking budget,
          // and temperature must be unset when extended thinking is on.
          if (typeof body.max_tokens === 'number' && body.max_tokens <= tokens) {
            body.max_tokens = tokens + cfg.llm.maxOutputTokens;
          }
          delete body.temperature;
          injectedThinking = true;
        }
      }
      init = { ...init, body: JSON.stringify(body) };
    }

    let res = await fetch(input, init);

    // Provider rejected our thinking dialect → retry once without, remember.
    if (!res.ok && injectedThinking && (res.status === 400 || res.status === 422)) {
      thinkingUnsupported.add(endpointKey);
      hooks.onWarning?.(
        `Provider rejected thinking config (HTTP ${res.status}) — retrying without. ` +
          `Set LLM_THINKING=off to silence this.`,
      );
      const stripped = { ...body };
      delete stripped.reasoning_effort;
      delete stripped.reasoning;
      delete stripped.thinking;
      res = await fetch(input, { ...init, body: JSON.stringify(stripped) });
    }

    if (!res.ok || !res.body) return res;

    const contentType = res.headers.get('content-type') ?? '';
    if (contentType.includes('text/event-stream')) {
      const [toSdk, toMeter] = res.body.tee();
      void consumeSseUsage(toMeter, cfg.llm.apiStyle, hooks).catch(() => {});
      return new Response(toSdk, { status: res.status, statusText: res.statusText, headers: res.headers });
    }
    if (contentType.includes('application/json')) {
      void res
        .clone()
        .json()
        .then((json) => {
          const usage = extractJsonUsage(json, cfg.llm.apiStyle);
          if (usage) hooks.onUsage?.(usage);
        })
        .catch(() => {});
    }
    return res;
  };
}

/** Read the teed SSE stream and surface the final usage object. */
async function consumeSseUsage(
  stream: ReadableStream<Uint8Array>,
  style: 'openai' | 'anthropic',
  hooks: MeterHooks,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  // anthropic accumulators
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheRead = 0;
  let cacheWrite = 0;
  let sawAnthropicUsage = false;
  // openai: last chunk that carried usage wins
  let openaiUsage: any = null;

  const handleData = (data: string) => {
    if (!data || data === '[DONE]') return;
    let json: any;
    try {
      json = JSON.parse(data);
    } catch {
      return;
    }
    if (style === 'openai') {
      if (json.usage) openaiUsage = json.usage;
    } else {
      if (json.type === 'message_start' && json.message?.usage) {
        const u = json.message.usage;
        inputTokens = u.input_tokens ?? 0;
        cacheRead = u.cache_read_input_tokens ?? 0;
        cacheWrite = u.cache_creation_input_tokens ?? 0;
        outputTokens = u.output_tokens ?? 0;
        sawAnthropicUsage = true;
      } else if (json.type === 'message_delta' && json.usage) {
        outputTokens = json.usage.output_tokens ?? outputTokens;
        sawAnthropicUsage = true;
      }
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx: number;
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line = buffer.slice(0, idx).trimEnd();
        buffer = buffer.slice(idx + 1);
        if (line.startsWith('data:')) handleData(line.slice(5).trim());
      }
    }
  } finally {
    reader.releaseLock();
  }

  if (style === 'openai' && openaiUsage) {
    hooks.onUsage?.({
      inputTokens: openaiUsage.prompt_tokens ?? 0,
      outputTokens: openaiUsage.completion_tokens ?? 0,
      cacheReadTokens: openaiUsage.prompt_tokens_details?.cached_tokens ?? 0,
      cacheWriteTokens: 0,
      providerCostUsd: typeof openaiUsage.cost === 'number' ? openaiUsage.cost : undefined,
    });
  } else if (style === 'anthropic' && sawAnthropicUsage) {
    hooks.onUsage?.({
      inputTokens,
      outputTokens,
      cacheReadTokens: cacheRead,
      cacheWriteTokens: cacheWrite,
    });
  }
}

function extractJsonUsage(json: any, style: 'openai' | 'anthropic'): RawUsage | null {
  const u = json?.usage;
  if (!u) return null;
  if (style === 'openai') {
    return {
      inputTokens: u.prompt_tokens ?? 0,
      outputTokens: u.completion_tokens ?? 0,
      cacheReadTokens: u.prompt_tokens_details?.cached_tokens ?? 0,
      cacheWriteTokens: 0,
      providerCostUsd: typeof u.cost === 'number' ? u.cost : undefined,
    };
  }
  return {
    inputTokens: u.input_tokens ?? 0,
    outputTokens: u.output_tokens ?? 0,
    cacheReadTokens: u.cache_read_input_tokens ?? 0,
    cacheWriteTokens: u.cache_creation_input_tokens ?? 0,
  };
}

/** Map a thrown provider error to a user-readable message (never a generic placeholder). */
export function classifyLlmError(err: unknown): string {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();
  if (lower.includes('401') || lower.includes('unauthorized') || lower.includes('invalid api key')) {
    return `Authentication failed — check LLM_API_KEY. (${truncate(msg)})`;
  }
  if (lower.includes('403') || lower.includes('forbidden')) {
    return `Access denied by provider — key may lack access to this model. (${truncate(msg)})`;
  }
  if (lower.includes('429') || lower.includes('rate limit') || lower.includes('quota')) {
    return `Provider rate/quota limit hit. Wait a moment and send a message to continue. (${truncate(msg)})`;
  }
  if (lower.includes('context length') || lower.includes('context_length') || lower.includes('too many tokens') || lower.includes('maximum context')) {
    return `Context window exceeded — lower LLM_CONTEXT_WINDOW so compaction kicks in earlier. (${truncate(msg)})`;
  }
  if (lower.includes('404') || lower.includes('model_not_found') || lower.includes('does not exist')) {
    return `Model not found — check LLM_MODEL against your provider. (${truncate(msg)})`;
  }
  if (lower.includes('econnrefused') || lower.includes('enotfound') || lower.includes('fetch failed') || lower.includes('network')) {
    return `Cannot reach LLM_BASE_URL — is the endpoint up? (${truncate(msg)})`;
  }
  if (lower.includes('500') || lower.includes('502') || lower.includes('503') || lower.includes('overloaded')) {
    return `Provider-side error — usually transient; send a message to retry. (${truncate(msg)})`;
  }
  return `LLM call failed: ${truncate(msg)}`;
}

function truncate(s: string, n = 300): string {
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
