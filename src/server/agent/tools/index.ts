import { tool, type ToolSet } from 'ai';
import type { z } from 'zod';
import type { AppConfig } from '../../config';
import type { EventBus } from '../../bus';
import type { SessionStore } from '../../session';
import type { SessionManager } from '../../runtime';

/**
 * Tool registry. Tools are declared to the AI SDK WITHOUT execute — the loop
 * runs them itself (that's what makes approval suspension, steering between
 * calls, and parallel fan-out possible). Each agent sees only its allowlist.
 */

export interface ToolCtx {
  sessionId: string;
  agentRunId: string;
  cfg: AppConfig;
  bus: EventBus;
  store: SessionStore;
  manager: SessionManager;
  workspaceDir: string;
  abort: AbortSignal;
  depth: number;
}

export interface ToolOutcome {
  /** What the model sees (capped by the caller at 25K chars). */
  text: string;
  isError?: boolean;
}

export interface ToolSpec<I = unknown> {
  name: string;
  description: string;
  inputSchema: z.ZodType<I>;
  /** Safe to run concurrently with other parallel-safe calls in one turn. */
  parallelSafe: boolean;
  execute: (input: I, ctx: ToolCtx) => Promise<ToolOutcome>;
}

// `any` erases the per-tool input generic inside the registry; executeToolCall
// re-validates with the tool's own zod schema before execute() runs.
type AnyToolSpec = ToolSpec<any>;

const REGISTRY = new Map<string, AnyToolSpec>();

export function registerTool<I>(spec: ToolSpec<I>): void {
  REGISTRY.set(spec.name, spec as AnyToolSpec);
}

export function getTool(name: string): AnyToolSpec | undefined {
  return REGISTRY.get(name);
}

/** SDK-facing tool set (schemas only, no execute) filtered by allowlist. */
export function buildToolSet(allowlist: string[]): ToolSet {
  const set: ToolSet = {};
  for (const name of allowlist) {
    const spec = REGISTRY.get(name);
    if (!spec) continue; // tools from later milestones simply aren't offered yet
    set[name] = tool({ description: spec.description, inputSchema: spec.inputSchema });
  }
  return set;
}

export function isParallelSafe(name: string): boolean {
  return REGISTRY.get(name)?.parallelSafe ?? false;
}

/** Execute one tool call; never throws — errors come back as outcomes. */
export async function executeToolCall(
  name: string,
  input: unknown,
  ctx: ToolCtx,
): Promise<ToolOutcome> {
  const spec = REGISTRY.get(name);
  if (!spec) {
    return { text: `unknown tool "${name}" — use only the tools you were given`, isError: true };
  }
  const parsed = spec.inputSchema.safeParse(input ?? {});
  if (!parsed.success) {
    return {
      text: `invalid arguments for ${name}: ${parsed.error.issues.map((i) => `${i.path.join('.')}: ${i.message}`).join('; ')}`,
      isError: true,
    };
  }
  try {
    return await spec.execute(parsed.data, ctx);
  } catch (err) {
    return { text: `${name} failed: ${err instanceof Error ? err.message : String(err)}`, isError: true };
  }
}

// NOTE: tool modules are imported for their registration side effects in
// ./all.ts (NOT here — ESM hoisting would run them before REGISTRY exists).
