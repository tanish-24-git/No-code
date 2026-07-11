/**
 * Agent-definition-as-data. A "specialized agent" is nothing but a system
 * prompt + tool allowlist + model tier — which is what makes dynamic agent
 * creation (create_agent) possible: the orchestrator mints one of these at
 * runtime and it is immediately spawnable.
 */
export interface AgentDefinition {
  /** Stable id — also the ReactFlow node id in the Agent Graph. */
  id: string;
  name: string;
  /** "When to spawn me" — shown to the orchestrator when it delegates. */
  description: string;
  systemPrompt: string;
  /** Tool allowlist, enforced by the tool registry. */
  tools: string[];
  /** Override model; workers default to LLM_MODEL_WORKER. */
  model?: string;
  /** Created at runtime by create_agent; persisted in session.json for resume. */
  ephemeral?: boolean;
  /** agentRunId of the creator — the UI draws a distinct "spawned definition" edge. */
  createdBy?: string;
  /** Finalization agents (evaluator, publisher) gate against the FULL budget,
   *  skipping the finalize reserve — that's what the reserve is FOR. */
  finalize?: boolean;
}

export type LoopResultSubtype =
  | 'success'
  | 'error_max_turns'
  | 'error_max_budget_usd'
  | 'paused_budget'
  | 'canceled'
  | 'error_during_execution';

export interface LoopResult {
  subtype: LoopResultSubtype;
  finalText: string;
}
