import type { SessionStore } from '../session';
import type { AgentDefinition } from './types';
import { orchestrator } from './definitions/orchestrator';
import { BUILTIN_WORKERS } from './definitions/workers';

/**
 * Agent-definition registry: built-ins + per-session ephemeral definitions
 * minted at runtime by create_agent (persisted in session.json for resume).
 */

const BUILTINS = new Map<string, AgentDefinition>(
  [orchestrator, ...BUILTIN_WORKERS].map((d) => [d.id, d]),
);

export function resolveDefinition(
  store: SessionStore,
  sessionId: string,
  id: string,
): AgentDefinition | null {
  const builtin = BUILTINS.get(id);
  if (builtin) return builtin;
  const session = store.get(sessionId);
  const eph = session?.ephemeralAgents.find((a) => (a as { id?: string }).id === id);
  return (eph as unknown as AgentDefinition | undefined) ?? null;
}

/** Rendered list for the orchestrator's system prompt. */
export function agentCatalog(store: SessionStore, sessionId: string): string {
  const lines: string[] = [];
  for (const d of BUILTIN_WORKERS) {
    lines.push(`- ${d.id} — ${d.description}`);
  }
  const session = store.get(sessionId);
  for (const raw of session?.ephemeralAgents ?? []) {
    const d = raw as unknown as AgentDefinition;
    if (d?.id && d?.description) lines.push(`- ${d.id} — ${d.description} (created this session)`);
  }
  return lines.join('\n');
}

/** Tools a dynamically-created agent may be granted. */
export const SPAWNABLE_TOOLS = [
  'run_terminal',
  'write_file',
  'read_file',
  'list_dir',
  'report_status',
  'web_search',
  'web_fetch',
  'ask_user',
] as const;
