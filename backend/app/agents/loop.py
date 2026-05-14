"""AgenticLoop: model-driven think -> tool -> observe -> repeat.

This is the heart of the redesigned runtime. There is exactly one loop
agent registered on the bus (in wiring.py). It triggers on:

    SessionStarted     - greet the user with a static message
    DatasetUploaded    - kick off the loop so the model can inspect
    IntakeCompleted    - upload-time intake is done; the loop takes over
    UserMessage        - cancel the currently running task, queue the new
                         message, and restart with the new context visible

Per-session bookkeeping:
    _running[sid]      - the live asyncio.Task running _run()
    _inbox[sid]        - queue of user messages awaiting integration

Cancellation semantics:
    User messages cancel the running loop's LLM stream and any cancel-safe
    tool call. The loop catches CancelledError, persists state, and
    immediately restarts so the new user message is part of the next turn.
    Cancel-unsafe tools (run_training, export, push_hf) are wrapped in
    asyncio.shield: the cancellation is deferred until they return.

This loop does NOT replace the legacy event-DAG agents wholesale; it calls
their wrapped-as-tool versions. They keep all their narration, decision
logging, and approval gates - which the frontend already renders.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional

from app.agents.base import BaseAgent
from app.agents.prompts import (
    AGENTIC_SYSTEM_PROMPT,
    GREETING_MESSAGE,
    INTAKE_ACK_TEMPLATE,
)
from app.agents.providers import stream_chat_async
from app.events.bus import EventBus
from app.events.types import AgentEvent
from app.services import dataset_service, session_service
from app.services import directives as directives_service
from app.tools.registry import ToolContext, get_tool, llm_tool_schemas, run_tool
from app.utils.config import settings


log = logging.getLogger("finetune-studio.agents.loop")


MAX_TURNS = 50
MAX_TOOL_RETRIES = 3


class AgenticLoop(BaseAgent):
    name = "AgenticLoop"
    role = "Autonomous fine-tuning engineer (think -> tool -> observe loop)"
    directive_scope = "global"
    triggers = (
        "SessionStarted",
        "IntakeCompleted",
        "UserMessage",
    )

    # Class-level so re-instantiation during reload preserves running work.
    _running: dict[str, asyncio.Task] = {}
    _inbox: dict[str, asyncio.Queue[str]] = {}
    _tool_failures: dict[str, dict[str, int]] = {}

    def __init__(self, bus: EventBus) -> None:
        super().__init__(bus)
        # Trigger import side-effects so the tool registry is populated by
        # the time the loop's first turn happens.
        import app.tools.agent_tools  # noqa: F401

    async def safe_handle(self, event: AgentEvent) -> None:
        """Override the BaseAgent reentrancy guard for UserMessage events.

        UserMessage must NEVER be dropped - if the user types twice in
        quick succession, both messages need to land. handle() returns
        quickly (it just queues + cancels + schedules), so racing is
        not a correctness issue.
        """
        if event.kind == "UserMessage":
            try:
                await self.handle(event)
            except Exception:
                log.exception("AgenticLoop UserMessage handler crashed")
            return
        await super().safe_handle(event)

    async def handle(self, event: AgentEvent) -> None:
        sid = event.session_id

        if event.kind == "SessionStarted":
            await self.emit_message(sid, GREETING_MESSAGE, parent=event.id)
            return

        if event.kind == "UserMessage":
            text = (event.payload or {}).get("text", "")
            if text:
                self._inbox.setdefault(sid, asyncio.Queue()).put_nowait(text)
                # Always record user messages as global directives so
                # wrapped agents (task_inference, model_selection,
                # strategy) that read directives see the user's voice
                # even if they fire in a later turn.
                try:
                    directives_service.record(sid, text, scope="global", actor="user")
                except Exception:
                    log.exception("failed to record user directive")
            existing = self._running.get(sid)
            if existing and not existing.done():
                # Interrupt the running loop; the next iteration will pick
                # up the inbox message after the CancelledError catch.
                existing.cancel()
            else:
                self._running[sid] = asyncio.create_task(
                    self._run(sid, parent_event_id=event.id),
                )
            return

        # SessionStarted greeting already returned. IntakeCompleted kicks
        # off the loop if it is not already running. We do NOT cancel an
        # in-flight loop for these events; only UserMessage cancels.
        existing = self._running.get(sid)
        if existing and not existing.done():
            return
        self._running[sid] = asyncio.create_task(
            self._run(sid, parent_event_id=event.id),
        )

    # ── The loop ─────────────────────────────────────────────────────────

    async def _run(self, sid: str, *, parent_event_id: Optional[str] = None) -> None:
        """One run = until the model says stop, hits the turn cap, or
        bails on an error. Cancellation by a UserMessage exits cleanly and
        is rescheduled by the caller."""
        try:
            await self._run_loop(sid, parent_event_id=parent_event_id)
        except asyncio.CancelledError:
            log.info("AgenticLoop cancelled for session %s; rescheduling", sid)
            await self._note_interrupt(sid)
            # Reschedule so the new user message is consumed.
            self._running[sid] = asyncio.create_task(
                self._run(sid, parent_event_id=parent_event_id),
            )
        except Exception as e:
            log.exception("AgenticLoop crashed for session %s", sid)
            await self.emit_error(sid, f"AgenticLoop crashed: {e}")

    async def _run_loop(self, sid: str, *, parent_event_id: Optional[str]) -> None:
        # On startup: if intake-ack hasn't been posted for this session yet
        # but a dataset exists, post a brief ack so the user knows we are
        # paying attention.
        await self._maybe_post_intake_ack(sid, parent_event_id)

        # Drain any queued user messages and weave them into a synthetic
        # "since last turn" user message for the first turn.
        inbox = self._inbox.setdefault(sid, asyncio.Queue())
        first_turn_user: list[str] = []
        while not inbox.empty():
            try:
                first_turn_user.append(inbox.get_nowait())
            except asyncio.QueueEmpty:
                break

        messages: list[dict[str, Any]] = []
        tool_history: list[dict[str, Any]] = []  # text-summarized tool log

        if first_turn_user:
            messages.append({"role": "user", "content": "\n\n".join(first_turn_user)})

        provider, model, base_url, api_key = self._resolve_provider(sid)
        if not provider or not model:
            await self.emit_message(
                sid,
                "I need an LLM provider configured before I can plan. "
                "Open Settings and set LLM_PROVIDER / LLM_MODEL / API key, "
                "then send me a message and we will continue.",
                parent=parent_event_id,
            )
            return

        for turn in range(1, MAX_TURNS + 1):
            # Drain any user messages that arrived between turns.
            extra: list[str] = []
            while not inbox.empty():
                try:
                    extra.append(inbox.get_nowait())
                except asyncio.QueueEmpty:
                    break
            if extra:
                messages.append({"role": "user", "content": "\n\n".join(extra)})

            # Build the system prompt freshly each turn so directives +
            # current state are always up-to-date.
            system_prompt = self._compose_system(sid, tool_history)
            tools = llm_tool_schemas()

            assistant_text = ""
            tool_calls: list[dict[str, Any]] = []
            stop_reason = "stop"

            try:
                async for chunk in stream_chat_async(
                    provider=provider,
                    api_key=api_key,
                    model=model,
                    base_url=base_url,
                    system=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_tokens=4096,
                ):
                    kind = chunk.get("kind")
                    if kind == "thinking":
                        await self.think_delta(sid, chunk.get("delta", ""),
                                               parent=parent_event_id)
                    elif kind == "text":
                        delta = chunk.get("delta", "")
                        assistant_text += delta
                        await self.stream_message(sid, delta,
                                                  parent=parent_event_id)
                    elif kind == "tool_call":
                        tool_calls.append({
                            "id": chunk.get("id", ""),
                            "name": chunk.get("name", ""),
                            "args": chunk.get("args") or {},
                        })
                    elif kind == "error":
                        await self.emit_error(sid, f"LLM stream error: {chunk.get('error')}")
                    elif kind == "stop":
                        stop_reason = chunk.get("reason", "stop")
            except asyncio.CancelledError:
                raise

            # Record what the assistant said this turn into the working
            # conversation. Skip empty assistant turns to keep context tight.
            if assistant_text.strip():
                messages.append({"role": "assistant", "content": assistant_text})
                # Mark the final-message boundary so the FE can finalize
                # the assistant bubble.
                await self.stream_message(sid, "", is_final=True,
                                          parent=parent_event_id)

            if not tool_calls:
                # No more work this turn. Decide whether to stop.
                if stop_reason in ("end_turn", "stop", "stop_sequence") and not extra:
                    log.info("loop session=%s terminated naturally after %d turns", sid, turn)
                    return
                if stop_reason == "error":
                    return
                # Otherwise, the model produced text without a tool call but
                # has more to do. Continue to next turn (the user might
                # message in the meantime).
                continue

            # Run the tools. They may be cancel-safe or not.
            tool_outputs: list[dict[str, Any]] = []
            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"] or {}
                tool_def = get_tool(name)
                if tool_def is None:
                    res = {"error": f"unknown tool: {name}"}
                else:
                    if self._tool_failures.setdefault(sid, {}).get(name, 0) >= MAX_TOOL_RETRIES:
                        res = {"error": f"tool {name} exceeded retry budget; pick a different approach"}
                    else:
                        res = await self._dispatch_tool(sid, tool_def, args)
                        if "error" in res:
                            self._tool_failures[sid][name] = self._tool_failures[sid].get(name, 0) + 1
                        else:
                            # On a successful call, reset the per-tool counter.
                            self._tool_failures[sid][name] = 0
                tool_outputs.append({"name": name, "id": tc["id"], "result": res})
                tool_history.append({
                    "turn": turn,
                    "tool": name,
                    "args": args,
                    "result": _summarize_for_history(res),
                })

            # Inject a synthetic user message describing the tool results so
            # the next turn sees them. (We do not pass back the structured
            # tool_call/tool_result message pair because we abstract across
            # providers - the text projection is enough for the model to
            # reason on.)
            messages.append({
                "role": "user",
                "content": _format_tool_results_for_model(tool_outputs),
            })

        log.warning("AgenticLoop hit MAX_TURNS=%d for session %s", MAX_TURNS, sid)
        await self.emit_message(
            sid,
            f"I hit my per-run turn budget ({MAX_TURNS}). Send me a message "
            "to continue from where I left off.",
            parent=parent_event_id,
        )

    # ── Tool dispatch with cancel handling ───────────────────────────────

    async def _dispatch_tool(
        self, sid: str, tool_def, args: dict[str, Any],
    ) -> dict[str, Any]:
        ctx = ToolContext(
            session_id=sid,
            bus=self.bus,
            extras={"agent": self.name},
        )
        coro = run_tool(tool_def.name, args, ctx)
        if tool_def.cancel_safe:
            return await coro
        # Cancel-unsafe (training, export). Shield against task cancellation
        # so a UserMessage during training does not corrupt the job. The
        # message stays in the inbox and is consumed on the next turn.
        return await asyncio.shield(coro)

    # ── System-prompt composition ────────────────────────────────────────

    def _compose_system(self, sid: str, tool_history: list[dict[str, Any]]) -> str:
        parts: list[str] = [AGENTIC_SYSTEM_PROMPT]
        directive_md = directives_service.render_for_prompt(sid, scope="global")
        if directive_md:
            parts.append(directive_md)
        state_md = self._render_state(sid)
        if state_md:
            parts.append(state_md)
        if tool_history:
            parts.append(_render_tool_history(tool_history))
        return "\n\n".join(parts)

    def _render_state(self, sid: str) -> str:
        session = session_service.get(sid)
        if not session:
            return ""
        artifacts = session.artifacts or {}
        bits: list[str] = ["## Current session state"]
        bits.append(f"- state: {session.state.value}")
        bits.append(f"- confidence: {session.confidence}")
        if session.dataset_id:
            bits.append(f"- dataset_id: {session.dataset_id}")
            ds = dataset_service.get_dataset(session.dataset_id)
            if ds:
                kind = (ds.analysis or {}).get("kind", "unknown")
                bits.append(f"- dataset: {ds.name} ({ds.row_count} rows, kind={kind}, type={ds.file_type})")
        if session.pipeline_id:
            bits.append(f"- pipeline_id: {session.pipeline_id}")
        if session.job_id:
            bits.append(f"- job_id: {session.job_id}")
        for key in ("hardware", "profile", "task_inference", "chosen_model",
                    "strategy", "data_health", "evaluation", "export"):
            v = artifacts.get(key)
            if v:
                try:
                    s = json.dumps(v, default=str)
                except Exception:
                    s = str(v)
                bits.append(f"- {key}: {s[:400]}")
        return "\n".join(bits)

    # ── Provider / model resolution ──────────────────────────────────────

    def _resolve_provider(self, sid: str) -> tuple[str, str, str, str]:
        session = self.get_session(sid)
        provider = ""
        model = ""
        base_url = ""
        api_key = ""
        if session:
            provider = getattr(session, "llm_provider", "") or ""
            model = getattr(session, "llm_model", "") or ""
            base_url = getattr(session, "llm_base_url", "") or ""
            if hasattr(session, "decrypted_api_key"):
                try:
                    api_key = session.decrypted_api_key() or ""
                except Exception:
                    api_key = ""
        provider = provider or (settings.llm_provider or "")
        model = model or (settings.llm_model or "")
        base_url = base_url or (settings.llm_base_url or "")
        api_key = api_key or (settings.llm_api_key or "")
        return provider, model, base_url, api_key

    # ── Misc helpers ─────────────────────────────────────────────────────

    async def _note_interrupt(self, sid: str) -> None:
        await self.emit(
            "BlackboardUpdated",
            sid,
            payload={"section": "interrupt", "agent": self.name,
                     "note": "user message received; replanning"},
        )

    async def _maybe_post_intake_ack(self, sid: str, parent: Optional[str]) -> None:
        session = self.get_session(sid)
        if not session or not session.dataset_id:
            return
        if (session.artifacts or {}).get("loop_intake_ack_posted"):
            return
        ds = dataset_service.get_dataset(session.dataset_id)
        if not ds:
            return
        text = INTAKE_ACK_TEMPLATE.format(
            dataset_name=ds.name,
            row_count=ds.row_count,
            kind=(ds.analysis or {}).get("kind", "unknown"),
        )
        await self.emit_message(sid, text, parent=parent)
        session_service.attach_artifact(session, "loop_intake_ack_posted", True)


# ── Module-level helpers ──────────────────────────────────────────────────

def _summarize_for_history(result: dict[str, Any]) -> str:
    """A short string snapshot of a tool result for the running text log."""
    try:
        s = json.dumps(result, default=str)
    except Exception:
        s = str(result)
    return s[:800]


def _render_tool_history(history: list[dict[str, Any]]) -> str:
    bits = ["## Recent tool calls in this run"]
    for entry in history[-20:]:  # last 20 keep context tight
        args = entry.get("args") or {}
        try:
            args_s = json.dumps(args, default=str)
        except Exception:
            args_s = str(args)
        bits.append(
            f"- turn {entry['turn']}: {entry['tool']}({args_s[:120]}) -> "
            f"{entry['result'][:300]}"
        )
    return "\n".join(bits)


def _format_tool_results_for_model(outputs: list[dict[str, Any]]) -> str:
    """Format the tool outputs as a synthetic user message the model sees
    next turn. Compact but complete."""
    lines = ["[tool results]"]
    for out in outputs:
        name = out.get("name", "?")
        try:
            res_s = json.dumps(out.get("result"), default=str)
        except Exception:
            res_s = str(out.get("result"))
        lines.append(f"- {name}: {res_s[:1500]}")
    return "\n".join(lines)
