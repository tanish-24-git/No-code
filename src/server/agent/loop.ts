import { streamText, type ModelMessage, type AssistantModelMessage, type ToolModelMessage } from 'ai';
import type { AppConfig } from '../config';
import type { EventBus } from '../bus';
import type { SessionStore } from '../session';
import type { SessionManager } from '../runtime';
import { capOutput } from '../exec/terminal';
import { renderMemory } from './memory';
import { classifyLlmError, createChatModel, type RawUsage } from './provider';
import { costOfUsage, estimateUsage, isFreeTier, preflightGate, spentUsd } from './budget';
import type { SteeringQueue } from './steering';
import type { AgentDefinition, LoopResult } from './types';
import { agentCatalog } from './registry';
import { buildToolSet, executeToolCall, isParallelSafe, type ToolCtx, type ToolOutcome } from './tools/index';
import './tools/all'; // registers all tools (side effect)

/**
 * The agentic loop — the heart of the harness.
 *
 * Stateless-server shape: full history array; the model streams a turn; tool
 * calls are executed by US (tools are declared without execute so the SDK
 * stops after each step); results are appended and the model re-invoked.
 * A turn with no tool calls ends the run. Steering messages are drained at
 * the top of every iteration so the user can interject between tool calls.
 */

export interface AgentRunArgs {
  sessionId: string;
  agentRunId: string;
  definition: AgentDefinition;
  messages: ModelMessage[];
  cfg: AppConfig;
  bus: EventBus;
  store: SessionStore;
  manager: SessionManager;
  workspaceDir: string;
  steering: SteeringQueue;
  abort: AbortSignal;
  /** Orchestrator = 0; spawned workers = parent + 1 (spawn depth cap lives in spawn_agent). */
  depth: number;
}

interface PendingToolCall {
  toolCallId: string;
  toolName: string;
  input: unknown;
}

export async function runAgentLoop(run: AgentRunArgs): Promise<LoopResult> {
  const { sessionId, agentRunId, definition, messages, cfg, bus, store, steering, abort } = run;
  let turns = 0;
  let lastText = '';
  // Workers stay silent in chat — their narration lives in the agent graph;
  // only their final summary returns to the orchestrator.
  const chatty = run.depth === 0;

  while (true) {
    if (abort.aborted) return { subtype: 'canceled', finalText: lastText };
    if (turns >= cfg.maxTurns) {
      bus.emit(sessionId, 'chat.message', {
        role: 'system',
        text: `Reached the ${cfg.maxTurns}-turn cap. Send a message to continue.`,
      });
      return { subtype: 'error_max_turns', finalText: lastText };
    }

    // 1. Interstitial steering: user messages sent mid-run land here.
    for (const text of steering.drain()) {
      messages.push({ role: 'user', content: `[user interjection] ${text}` });
    }

    // 2. System prompt is rebuilt EVERY turn — memory + budget survive compaction.
    const system = composeSystem(run);

    // 3. Gate A — pre-flight projection BEFORE dispatching the call.
    {
      const session = store.get(sessionId);
      const estPromptTokens = Math.ceil((system.length + JSON.stringify(messages).length) / 4);
      const gate = preflightGate({
        cfg,
        spent: spentUsd(session?.ledger ?? []),
        budgetUsd: session?.budgetUsd ?? cfg.budget.totalUsd,
        estPromptTokens,
        finalize: definition.finalize ?? false,
      });
      if (!gate.ok) {
        store.saveHistory(sessionId, agentRunId, messages);
        bus.emit(
          sessionId,
          'budget.exceeded',
          {
            spentUsd: gate.spentUsd,
            budgetUsd: gate.budgetUsd,
            deltaNeeded: gate.deltaNeededUsd,
            projection: { nextCallCeilingUsd: gate.callCeilingUsd },
          },
          agentRunId,
        );
        return { subtype: 'paused_budget', finalText: lastText };
      }
    }

    // 4. Model call. Usage lands via the metering fetch shim.
    let rawUsage: RawUsage | null = null;
    const model = createChatModel(cfg, resolveModel(cfg, definition), {
      onUsage: (u) => {
        rawUsage = u;
      },
      onWarning: (msg) => bus.emit(sessionId, 'chat.message', { role: 'system', text: msg }, agentRunId),
    });

    turns++;
    let text = '';
    const toolCalls: PendingToolCall[] = [];
    try {
      const result = streamText({
        model,
        system,
        messages,
        tools: buildToolSet(definition.tools),
        maxOutputTokens: cfg.llm.maxOutputTokens,
        abortSignal: abort,
      });
      for await (const part of result.fullStream) {
        switch (part.type) {
          case 'text-delta':
            text += part.text;
            if (chatty) bus.emit(sessionId, 'chat.delta', { channel: 'text', delta: part.text }, agentRunId);
            break;
          case 'reasoning-delta':
            if (chatty) bus.emit(sessionId, 'chat.delta', { channel: 'thinking', delta: part.text }, agentRunId);
            break;
          case 'tool-call':
            toolCalls.push({ toolCallId: part.toolCallId, toolName: part.toolName, input: part.input });
            break;
          case 'error':
            throw part.error instanceof Error ? part.error : new Error(String(part.error));
          default:
            break;
        }
      }
    } catch (err) {
      if (abort.aborted) return { subtype: 'canceled', finalText: lastText };
      bus.emit(sessionId, 'error', { message: classifyLlmError(err) }, agentRunId);
      return { subtype: 'error_during_execution', finalText: lastText };
    }

    // 4. Post-flight accounting (M3 adds the pre-flight projection gate).
    recordUsage(run, rawUsage, system, messages, text);

    if (toolCalls.length === 0) {
      if (text.trim()) {
        messages.push({ role: 'assistant', content: text });
        if (chatty) bus.emit(sessionId, 'chat.message', { role: 'assistant', text }, agentRunId);
        lastText = text;
      }
      store.saveHistory(sessionId, agentRunId, messages);
      // 5. No tool calls: done — unless the user interjected while we streamed.
      if (steering.size > 0) continue;
      return { subtype: 'success', finalText: lastText };
    }

    // 5'. Tool turn: record the assistant message (text narration + calls)…
    const assistantMsg: AssistantModelMessage = {
      role: 'assistant',
      content: [
        ...(text.trim() ? ([{ type: 'text' as const, text }] as const) : []),
        ...toolCalls.map((tc) => ({
          type: 'tool-call' as const,
          toolCallId: tc.toolCallId,
          toolName: tc.toolName,
          input: tc.input,
        })),
      ],
    };
    messages.push(assistantMsg);
    if (text.trim()) {
      if (chatty) bus.emit(sessionId, 'chat.message', { role: 'assistant', text }, agentRunId);
      lastText = text;
    }

    // …then execute. All-parallel-safe turns fan out (this is how the
    // orchestrator runs analyst ∥ profiler concurrently).
    const outcomes = await executeToolCalls(run, toolCalls);

    const toolMsg: ToolModelMessage = {
      role: 'tool',
      content: toolCalls.map((tc, i) => ({
        type: 'tool-result' as const,
        toolCallId: tc.toolCallId,
        toolName: tc.toolName,
        output: outcomes[i].isError
          ? ({ type: 'error-text' as const, value: capOutput(outcomes[i].text) })
          : ({ type: 'text' as const, value: capOutput(outcomes[i].text) }),
      })),
    };
    messages.push(toolMsg);
    store.saveHistory(sessionId, agentRunId, messages);

    if (abort.aborted) return { subtype: 'canceled', finalText: lastText };
  }
}

async function executeToolCalls(run: AgentRunArgs, toolCalls: PendingToolCall[]): Promise<ToolOutcome[]> {
  const { sessionId, agentRunId, bus } = run;
  const ctx: ToolCtx = {
    sessionId,
    agentRunId,
    cfg: run.cfg,
    bus: run.bus,
    store: run.store,
    manager: run.manager,
    workspaceDir: run.workspaceDir,
    abort: run.abort,
    depth: run.depth,
  };

  const runOne = async (tc: PendingToolCall): Promise<ToolOutcome> => {
    bus.emit(
      sessionId,
      'tool.called',
      { toolCallId: tc.toolCallId, tool: tc.toolName, argsPreview: JSON.stringify(tc.input ?? {}).slice(0, 200) },
      agentRunId,
    );
    const outcome = run.abort.aborted
      ? { text: 'canceled', isError: true }
      : await executeToolCall(tc.toolName, tc.input, ctx);
    bus.emit(
      sessionId,
      'tool.result',
      { toolCallId: tc.toolCallId, ok: !outcome.isError, resultPreview: outcome.text.slice(0, 200) },
      agentRunId,
    );
    return outcome;
  };

  if (toolCalls.length > 1 && toolCalls.every((tc) => isParallelSafe(tc.toolName))) {
    return Promise.all(toolCalls.map(runOne));
  }
  const outcomes: ToolOutcome[] = [];
  for (const tc of toolCalls) {
    outcomes.push(await runOne(tc));
  }
  return outcomes;
}

function resolveModel(cfg: AppConfig, definition: AgentDefinition): string {
  if (definition.model) return definition.model;
  if (definition.id === 'orchestrator') return cfg.llm.model!;
  return cfg.llm.workerModel ?? cfg.llm.model!;
}

function composeSystem(run: AgentRunArgs): string {
  const { cfg, sessionId, definition, store } = run;
  const parts: string[] = [definition.systemPrompt];

  if (definition.tools.includes('spawn_agent')) {
    parts.push(`# Available specialist agents (spawn_agent)\n${agentCatalog(store, sessionId)}`);
  }

  const memory = renderMemory(cfg.dataDir, sessionId);
  if (memory.trim()) {
    parts.push(`# Session memory (FINETUNE.md — standing facts and user directives; always honor these)\n${memory}`);
  }

  const session = store.get(sessionId);
  if (session) {
    const spent = spentUsd(session.ledger);
    parts.push(
      `# Status\nBudget: $${spent.toFixed(4)} spent of $${session.budgetUsd.toFixed(2)}. ` +
        `Session status: ${session.status}. Plan approved: ${session.planApproved ? 'yes' : 'no'}.`,
    );
  }
  return parts.join('\n\n');
}

function recordUsage(
  run: AgentRunArgs,
  rawUsage: RawUsage | null,
  system: string,
  messages: ModelMessage[],
  completionText: string,
): void {
  const { cfg, sessionId, agentRunId, bus, store } = run;
  let usage = rawUsage;
  let estimated = false;
  if (!usage) {
    const promptChars = system.length + JSON.stringify(messages).length;
    usage = estimateUsage(promptChars, completionText.length);
    estimated = true;
  }
  const cost = costOfUsage(usage, cfg);
  const updated = store.update(sessionId, (rec) => {
    rec.ledger.push({
      at: new Date().toISOString(),
      agentRunId,
      usd: cost.usd,
      inputTokens: usage!.inputTokens,
      outputTokens: usage!.outputTokens,
      estimated,
    });
  });
  if (updated) {
    const spent = spentUsd(updated.ledger);
    bus.emit(
      sessionId,
      'budget.usage',
      {
        lastCallUsd: cost.usd,
        spentUsd: spent,
        budgetUsd: updated.budgetUsd,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        estimated,
      },
      agentRunId,
    );
    // Gate B — post-flight soft warning (once per budget level).
    if (
      !isFreeTier(cfg) &&
      !updated.budgetWarned &&
      spent >= updated.budgetUsd * cfg.budget.softFraction
    ) {
      store.update(sessionId, (rec) => {
        rec.budgetWarned = true;
      });
      bus.emit(sessionId, 'budget.warning', {
        spentUsd: spent,
        budgetUsd: updated.budgetUsd,
        message: `Budget ${Math.round((spent / updated.budgetUsd) * 100)}% consumed.`,
      });
    }
  }
}
