import { streamText, type ModelMessage } from 'ai';
import type { AppConfig } from '../config';
import type { EventBus } from '../bus';
import type { SessionStore } from '../session';
import { renderMemory } from './memory';
import { classifyLlmError, createChatModel, type RawUsage } from './provider';
import { costOfUsage, estimateUsage, spentUsd } from './budget';
import type { SteeringQueue } from './steering';
import type { AgentDefinition, LoopResult } from './types';

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
  steering: SteeringQueue;
  abort: AbortSignal;
  /** Orchestrator = 0; spawned workers = parent + 1 (spawn depth cap lives in spawn_agent). */
  depth: number;
}

export async function runAgentLoop(run: AgentRunArgs): Promise<LoopResult> {
  const { sessionId, agentRunId, definition, messages, cfg, bus, store, steering, abort } = run;
  let turns = 0;
  let lastText = '';

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

    // 3. Model call. Usage lands via the metering fetch shim.
    let rawUsage: RawUsage | null = null;
    const model = createChatModel(cfg, resolveModel(cfg, definition), {
      onUsage: (u) => {
        rawUsage = u;
      },
      onWarning: (msg) => bus.emit(sessionId, 'chat.message', { role: 'system', text: msg }, agentRunId),
    });

    turns++;
    let text = '';
    try {
      const result = streamText({
        model,
        system,
        messages,
        maxOutputTokens: cfg.llm.maxOutputTokens,
        abortSignal: abort,
        // M2+: tools (declared without execute; we run them ourselves)
      });
      for await (const part of result.fullStream) {
        switch (part.type) {
          case 'text-delta':
            text += part.text;
            bus.emit(sessionId, 'chat.delta', { channel: 'text', delta: part.text }, agentRunId);
            break;
          case 'reasoning-delta':
            bus.emit(sessionId, 'chat.delta', { channel: 'thinking', delta: part.text }, agentRunId);
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

    if (text.trim()) {
      messages.push({ role: 'assistant', content: text });
      bus.emit(sessionId, 'chat.message', { role: 'assistant', text }, agentRunId);
      lastText = text;
    }
    store.saveHistory(sessionId, agentRunId, messages);

    // M2: execute tool calls here; loop again when the turn used tools.

    // 5. No tool calls: done — unless the user interjected while we streamed.
    if (steering.size > 0) continue;
    return { subtype: 'success', finalText: lastText };
  }
}

function resolveModel(cfg: AppConfig, definition: AgentDefinition): string {
  if (definition.model) return definition.model;
  if (definition.id === 'orchestrator') return cfg.llm.model!;
  return cfg.llm.workerModel ?? cfg.llm.model!;
}

function composeSystem(run: AgentRunArgs): string {
  const { cfg, sessionId, definition, store } = run;
  const parts: string[] = [definition.systemPrompt];

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
    bus.emit(
      sessionId,
      'budget.usage',
      {
        lastCallUsd: cost.usd,
        spentUsd: spentUsd(updated.ledger),
        budgetUsd: updated.budgetUsd,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        estimated,
      },
      agentRunId,
    );
  }
}
