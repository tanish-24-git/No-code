"""BaseAgent: the contract every specialized agent implements.

An agent's life:
    1. Bus delivers an event of `triggers` kind.
    2. Agent reads session state + the federated blackboard + global directives.
    3. Agent thinks (think), plans (announce), optionally asks for approval.
    4. Agent calls tools or LLM with shared-context.
    5. Agent emits new events (which schedule other agents).

Agents never call other agents directly - coupling is via the bus.

Three narration patterns coexist on this class:

    * Phase narration (Antigravity-style)   announce / materialize_node
                                            / complete. Drives ChatGPT-style
                                            chat-bubble plan cards with inline
                                            approve / comment buttons.

    * Streaming helpers                     think / plan / ask / garnish /
                                            stream_executing. Mirrored to
                                            the federated blackboard.

    * Shared-context LLM                    call_llm builds a context block
                                            from the master plan + global
                                            user directives + prior artifacts
                                            so user intent never gets siloed
                                            in one phase.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Optional, Type, TypeVar

from pydantic import BaseModel

from app.agents.model_tiers import Tier, pick_model
from app.agents.providers import stream_chat
from app.agents.schemas import LLMSchemaError, parse_llm_json
from app.events.bus import EventBus
from app.events.types import AgentEvent, EventKind, StreamTone
from app.services import session_service
from app.services.blackboard import get_blackboard
from app.services.directives import render_for_prompt
from app.services.phase_service import (
    PhasePlan,
    announce_edge,
    announce_node,
    announce_phase,
    complete_phase,
)
from app.tools.registry import ToolContext, run_tool
from app.utils.config import settings


log = logging.getLogger("finetune-studio.agents")

T = TypeVar("T", bound=BaseModel)


class BaseAgent:
    name: str = "BaseAgent"
    role: str = ""
    allowed_tools: tuple[str, ...] = ()
    triggers: tuple[EventKind, ...] = ()

    # Sub-classes can declare which directive scope they read.
    directive_scope: str = "global"

    # When True the agent's narrative chat helpers (emit_message,
    # stream_message, and announce() with requires_approval=False) are
    # suppressed. Artifact writes, decision_log, materialize_node,
    # complete(), emit_error, and approval-gate announces are unaffected.
    # The AgenticLoop sets this on wrapped sub-agents so the loop's own
    # LLM is the only voice in the chat.
    silent: bool = False

    # Default tier for this agent's LLM calls. "strong" keeps the user's
    # configured model unchanged. "fast" / "medium" auto-route to a
    # cheaper same-provider sibling via app.agents.model_tiers — set
    # this on wrapped specialty agents whose reasoning is mechanical
    # narration (intake, profile, hardware) to preserve free-tier RPD
    # for the strategic decisions. Per-call override available via the
    # ``tier`` parameter on call_llm / call_llm_typed.
    tier_preference: Tier = "strong"

    def __init__(self, bus: EventBus) -> None:
        self.bus = bus
        # Tracks session IDs with an active handle() invocation. Keeps
        # PhaseApproved events from re-entering an already-running handler.
        self._active_sessions: set[str] = set()

    async def handle(self, event: AgentEvent) -> None:
        raise NotImplementedError

    async def safe_handle(self, event: AgentEvent) -> None:
        """Wrapper that prevents reentrant handle() for the same session."""
        sid = event.session_id
        if sid in self._active_sessions:
            log.debug("%s: skipping reentrant handle for session %s (event=%s)",
                      self.name, sid, event.kind)
            return
        self._active_sessions.add(sid)
        try:
            await self.handle(event)
        finally:
            self._active_sessions.discard(sid)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _ctx(self, session_id: str) -> ToolContext:
        return ToolContext(
            session_id=session_id,
            bus=self.bus,
            extras={"agent": self.name},
        )

    async def call_tool(self, name: str, args: dict[str, Any], session_id: str) -> dict[str, Any]:
        if self.allowed_tools and name not in self.allowed_tools:
            return {"error": f"agent {self.name} not authorized for tool {name}"}
        return await run_tool(name, args, self._ctx(session_id))

    async def emit(
        self,
        kind: EventKind,
        session_id: str,
        *,
        payload: Optional[dict[str, Any]] = None,
        rationale: Optional[str] = None,
        confidence: Optional[float] = None,
        parent_event_id: Optional[str] = None,
        decision_id: Optional[str] = None,
    ) -> AgentEvent:
        ev = AgentEvent(
            session_id=session_id,
            kind=kind,
            actor=self.name,
            payload=payload or {},
            rationale=rationale,
            confidence=confidence,
            parent_event_id=parent_event_id,
            decision_id=decision_id,
        )
        await self.bus.publish(ev)
        return ev

    async def emit_message(self, session_id: str, text: str, *, parent: Optional[str] = None) -> Optional[AgentEvent]:
        if self.silent:
            return None
        return await self.emit(
            "AssistantMessage",
            session_id,
            payload={"text": _normalize_md(text)},
            parent_event_id=parent,
        )

    async def stream_message(
        self,
        session_id: str,
        delta: str,
        *,
        is_final: bool = False,
        parent: Optional[str] = None,
    ) -> None:
        if self.silent:
            return
        await self.emit(
            "AssistantMessage",
            session_id,
            payload={"delta": delta, "is_final": is_final},
            parent_event_id=parent,
        )

    async def emit_error(self, session_id: str, message: str) -> AgentEvent:
        return await self.emit("Error", session_id, payload={"error": message})

    def get_session(self, session_id: str):
        return session_service.get(session_id)

    # ── Phase narration (Antigravity-style) ────────────────────────────────

    async def announce(
        self,
        session_id: str,
        *,
        phase: str,
        title: str,
        summary: str,
        steps: list[str] | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: list[str] | None = None,
        requires_approval: bool = False,
        parent: Optional[str] = None,
    ) -> None:
        # Suppress narration-only phase cards in silent mode. Approval
        # cards still fire so the user retains an interaction surface.
        if self.silent and not requires_approval:
            return
        plan = PhasePlan(
            phase=phase,
            title=title,
            summary=summary,
            steps=steps or [],
            inputs=inputs or {},
            outputs=outputs or [],
            requires_approval=requires_approval,
        )
        await announce_phase(
            self.bus,
            session_id=session_id,
            actor=self.name,
            phase=phase,
            plan=plan,
            parent_event_id=parent,
        )

    async def materialize_node(
        self,
        session_id: str,
        node: dict[str, Any],
        *,
        parent: Optional[str] = None,
    ) -> None:
        await announce_node(
            self.bus,
            session_id=session_id,
            actor=self.name,
            node=node,
            parent_event_id=parent,
        )

    async def materialize_edge(
        self,
        session_id: str,
        edge: dict[str, Any],
        *,
        parent: Optional[str] = None,
    ) -> None:
        await announce_edge(
            self.bus,
            session_id=session_id,
            actor=self.name,
            edge=edge,
            parent_event_id=parent,
        )

    async def complete(
        self,
        session_id: str,
        *,
        phase: str,
        summary: str,
        artifacts: Optional[dict[str, Any]] = None,
        parent: Optional[str] = None,
    ) -> None:
        await complete_phase(
            self.bus,
            session_id=session_id,
            actor=self.name,
            phase=phase,
            summary=summary,
            artifacts=artifacts,
            parent_event_id=parent,
        )

    async def wait_for_approval(self, session_id: str, phase: str) -> Optional[str]:
        """Pause until the user clicks Approve or submits a Comment.

        Returns the comment text if one was provided, or None if approved.

        The comment is also recorded as a *global* directive so every
        subsequent agent sees it even if they never run another approval
        gate. This is the fix for "agents ignore my earlier instructions".
        """
        from app.api.schemas.session import FSMState
        from app.services import directives as directives_service

        # Pending-approval bookkeeping lives in artifacts, not the FSM.
        session = self.get_session(session_id)
        if session:
            pending = list(session.artifacts.get("pending_approvals") or [])
            if phase not in pending:
                pending.append(phase)
            session_service.attach_artifact(session, "pending_approvals", pending)

        # Move the FSM to AWAITING_APPROVAL so the FE renders interaction
        # buttons. Remember the previous state so we restore it cleanly.
        prev_state = session.state if session else None
        if session and session.state != FSMState.AWAITING_APPROVAL:
            try:
                session_service.advance_state(
                    session, FSMState.AWAITING_APPROVAL,
                    reason=f"awaiting {phase} approval",
                )
            except ValueError:
                # Some terminal states cannot transition; in that case we
                # silently treat it as auto-approve (already-completed run).
                return None

        loop = asyncio.get_running_loop()
        future: asyncio.Future[Optional[str]] = loop.create_future()

        async def _on_approved(ev: AgentEvent) -> None:
            if ev.session_id == session_id and ev.payload.get("phase") == phase:
                if not future.done():
                    future.set_result(None)

        async def _on_commented(ev: AgentEvent) -> None:
            if ev.session_id == session_id and ev.payload.get("phase") == phase:
                if not future.done():
                    future.set_result(ev.payload.get("text"))

        self.bus.on("PhaseApproved", _on_approved)
        self.bus.on("PhaseCommented", _on_commented)

        comment: Optional[str] = None
        try:
            comment = await future
        except asyncio.CancelledError:
            comment = None
            raise
        finally:
            self.bus.off("PhaseApproved", _on_approved)
            self.bus.off("PhaseCommented", _on_commented)

            # Drain pending bookkeeping. Reload to see fresh writes.
            session = self.get_session(session_id)
            remaining: list[str] = []
            if session:
                remaining = [p for p in (session.artifacts.get("pending_approvals") or []) if p != phase]
                session_service.attach_artifact(session, "pending_approvals", remaining)

            # Persist the comment as a global directive *before* restoring
            # state, so any agent triggered by the state change can read it.
            if comment:
                directives_service.record(
                    session_id, comment, source_phase=phase, actor="user",
                )
                await self.emit_message(
                    session_id,
                    f"User feedback ({phase}): {comment}",
                )

            # Restore previous state ONLY if no other agent is also waiting.
            if (
                session
                and session.state == FSMState.AWAITING_APPROVAL
                and not remaining
            ):
                target = (
                    prev_state
                    if prev_state and prev_state != FSMState.AWAITING_APPROVAL
                    else FSMState.PROFILING
                )
                try:
                    session_service.advance_state(
                        session, target,
                        reason=f"{phase} {'comment received' if comment else 'approved'}",
                    )
                except ValueError as e:
                    log.warning("could not restore state after %s: %s", phase, e)

        return comment

    # ── Streaming helpers (blackboard-mirrored) ────────────────────────────

    async def think(
        self,
        session_id: str,
        thought: str,
        *,
        parent: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> AgentEvent:
        get_blackboard().post(session_id, "thoughts", agent=self.name, content=thought, meta=meta)
        return await self.emit(
            "AgentThinking",
            session_id,
            payload={"stream": "thinking", "agent": self.name, "text": thought, "meta": meta or {}},
            parent_event_id=parent,
        )

    async def think_delta(
        self,
        session_id: str,
        delta: str,
        *,
        is_final: bool = False,
        parent: Optional[str] = None,
    ) -> None:
        await self.emit(
            "AgentThinking",
            session_id,
            payload={"stream": "thinking", "agent": self.name, "delta": delta, "is_final": is_final},
            parent_event_id=parent,
        )

    async def plan(
        self,
        session_id: str,
        steps: list[str],
        *,
        title: Optional[str] = None,
        parent: Optional[str] = None,
    ) -> AgentEvent:
        get_blackboard().post(session_id, "plans", agent=self.name, content=title or "plan", meta={"steps": steps})
        return await self.emit(
            "AgentPlanning",
            session_id,
            payload={"stream": "planning", "agent": self.name, "title": title or "plan", "steps": steps},
            parent_event_id=parent,
        )

    async def ask(
        self,
        session_id: str,
        question: str,
        *,
        parent: Optional[str] = None,
        impact: str = "medium",
    ) -> AgentEvent:
        get_blackboard().post(session_id, "questions", agent=self.name, content=question, meta={"impact": impact})
        return await self.emit(
            "AgentAsking",
            session_id,
            payload={"stream": "asking", "agent": self.name, "question": question, "impact": impact},
            parent_event_id=parent,
        )

    async def garnish(
        self,
        session_id: str,
        node: dict[str, Any],
        *,
        parent: Optional[str] = None,
    ) -> AgentEvent:
        get_blackboard().post(session_id, "nodes", agent=self.name, content=node.get("type", "node"), meta=node)
        return await self.emit(
            "AgentGarnishing",
            session_id,
            payload={"stream": "garnishing", "agent": self.name, "node": node},
            parent_event_id=parent,
        )

    async def stream_executing(
        self,
        session_id: str,
        message: str,
        *,
        parent: Optional[str] = None,
    ) -> AgentEvent:
        return await self.emit(
            "AgentExecuting",
            session_id,
            payload={"stream": "executing", "agent": self.name, "text": message},
            parent_event_id=parent,
        )

    def blackboard(self):
        return get_blackboard()

    def stream_kind_for(self, tone: StreamTone) -> EventKind:
        return {
            "thinking": "AgentThinking",
            "planning": "AgentPlanning",
            "asking": "AgentAsking",
            "garnishing": "AgentGarnishing",
            "executing": "AgentExecuting",
        }[tone]

    # ── LLM with shared context ───────────────────────────────────────────

    def _build_context_block(self, session_id: str) -> str:
        """Render the standing-context block injected into every LLM call.

        Includes:
            * recent user directives in the agent's scope
            * the agreed master plan
            * key prior artifacts (chosen_model, strategy, hardware, profile)
        """
        session = self.get_session(session_id)
        parts: list[str] = []

        directive_md = render_for_prompt(session_id, scope=self.directive_scope)
        if directive_md:
            parts.append(directive_md)

        if session:
            artifacts = session.artifacts or {}
            plan = artifacts.get("master_plan")
            if isinstance(plan, dict) and plan.get("steps"):
                parts.append("**Approved master plan:**\n" + "\n".join(
                    f"{i+1}. {s}" for i, s in enumerate(plan["steps"])
                ))
            for key in ("hardware", "profile", "task_inference",
                        "chosen_model", "strategy", "data_health"):
                v = artifacts.get(key)
                if v:
                    try:
                        snippet = json.dumps(v, default=str)[:600]
                    except Exception:
                        snippet = str(v)[:600]
                    parts.append(f"**{key}:** {snippet}")

        return "\n\n".join(parts) if parts else ""

    async def call_llm(
        self,
        session_id: str,
        prompt: str,
        *,
        system: str = "",
        stream_thoughts: bool = True,
        include_tools: bool = False,
        include_context: bool = True,
        parent: Optional[str] = None,
        tier: Optional[Tier] = None,
    ) -> str:
        """Run a single LLM turn with shared context injected.

        Reads the configured provider from the session (encrypted at rest)
        or falls back to global settings. Returns "" when no provider is
        configured - callers should branch on that.

        The ``tier`` parameter (or, when omitted, the agent's
        ``tier_preference`` class attribute) may swap the model for a
        cheaper same-provider sibling. The substitution never upgrades
        the user's pick; see app.agents.model_tiers for the rules.
        """
        session = self.get_session(session_id)
        if not session:
            return ""

        provider = (
            getattr(session, "llm_provider", None)
            or settings.llm_provider
            or ""
        )
        model = (
            getattr(session, "llm_model", None)
            or settings.llm_model
            or ""
        )
        base_url = (
            getattr(session, "llm_base_url", None)
            or settings.llm_base_url
            or ""
        )
        api_key = ""
        if hasattr(session, "decrypted_api_key"):
            try:
                api_key = session.decrypted_api_key()
            except Exception:
                api_key = ""
        if not api_key:
            api_key = settings.llm_api_key or ""

        if not provider or not model:
            await self.emit_error(
                session_id,
                "LLM call aborted: no provider/model configured. "
                "Open Settings and set LLM_PROVIDER + LLM_MODEL + API key, "
                "then retry.",
            )
            return ""

        # Tier routing: optionally swap to a cheaper same-provider model
        # for narration-grade calls. Never upgrades above the user's pick.
        effective_tier: Tier = tier if tier is not None else self.tier_preference
        model = pick_model(provider, model, effective_tier)

        # Augment the system prompt with directives + artifacts.
        ctx_block = self._build_context_block(session_id) if include_context else ""
        full_system = system or ""
        if ctx_block:
            full_system = (full_system + "\n\n" if full_system else "") + ctx_block
        if not include_tools:
            tool_guard = (
                "CRITICAL OUTPUT RULES (override all other instructions):\n"
                "- No tools are registered for this call. Do NOT emit tool calls, "
                "function calls, browser_search, python, code interpreter, or any "
                "structured tool invocation. The runtime has none registered and "
                "will discard them.\n"
                "- Think out loud freely. Thinking blocks and deliberation are "
                "welcome; the runtime streams them to the user.\n"
                "- If asked for JSON, return EXACTLY ONE fenced "
                "```json ... ``` block after your thinking.\n"
                "- If asked for plain text, return one or two sentences after "
                "your thinking."
            )
            full_system = (full_system + "\n\n" if full_system else "") + tool_guard

        messages = [{"role": "user", "content": prompt}]
        full_text = ""
        try:
            for chunk in stream_chat(
                provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
                messages=messages,
                use_tools=include_tools,
                extra_system=full_system,
            ):
                full_text += chunk
                if stream_thoughts:
                    await self.think_delta(session_id, chunk, parent=parent)
        except Exception as e:
            log.warning("call_llm failed: %s", e)
            if stream_thoughts:
                await self.think_delta(session_id, "", is_final=True, parent=parent)
            # Always surface LLM failures to the UI. The full provider
            # message is preserved verbatim — context-length errors,
            # 429s, invalid keys, and network blips all read clearly.
            await self.emit_error(
                session_id,
                _classify_llm_error(e, provider=provider, model=model),
            )
            return full_text

        if stream_thoughts:
            await self.think_delta(session_id, "", is_final=True, parent=parent)
        return full_text

    async def call_llm_typed(
        self,
        session_id: str,
        prompt: str,
        schema: Type[T],
        *,
        system: str = "",
        stream_thoughts: bool = True,
        include_context: bool = True,
        parent: Optional[str] = None,
        max_retries: int = 1,
        tier: Optional[Tier] = None,
    ) -> Optional[T]:
        """Same as call_llm, but parses the response against a pydantic
        schema. Retries once on a parse failure with a corrective hint.

        Returns None if no provider is configured OR all retries fail.
        Callers decide whether to fall back to a deterministic computation.
        """
        # Inject the actual JSON schema so the LLM doesn't have to guess
        # field names or enum casing. This is the single most effective
        # defence against cascading validation failures in StrategyChoice,
        # GraphProposal, RecoveryPlan, etc. — without it the model mirrors
        # whatever casing the prompt uses (e.g. "QLoRA"), which Pydantic
        # Literal validators reject case-sensitively.
        schema_block = _format_schema_for_prompt(schema)
        strict_footer = (
            "\n\n=== RESPONSE SCHEMA ===\n"
            + schema_block
            + "\n=== END SCHEMA ===\n\n"
            "OUTPUT FORMAT — read carefully:\n"
            "1. Your reply MUST start with the marker `<<<JSON>>>` and end "
            "with `<<<END>>>`.\n"
            "2. Between those markers, emit EXACTLY ONE fenced "
            "```json ... ``` block. Nothing else.\n"
            "3. Do NOT write explanations, preambles, 'let me think', "
            "'here is the JSON', commentary, or analysis outside the "
            "fence. Anything outside the markers is discarded.\n"
            "4. Enum / Literal values are CASE-SENSITIVE — use the exact "
            "lowercase strings shown in the schema above.\n"
            "5. If you are streaming thoughts, finish them BEFORE the "
            "`<<<JSON>>>` marker — never interleave thinking with JSON.\n"
            "6. The JSON object must be complete and balanced; do not "
            "truncate it. If it would exceed your output budget, shorten "
            "string fields (especially 'rationale') rather than omit "
            "closing braces."
        )
        attempt_prompt = prompt + strict_footer
        last_parse_error: Optional[str] = None
        for i in range(max_retries + 1):
            raw = await self.call_llm(
                session_id, attempt_prompt,
                system=system,
                stream_thoughts=stream_thoughts,
                include_context=include_context,
                parent=parent,
                tier=tier,
            )
            if not raw:
                # call_llm already emitted a UI-visible error (no provider
                # configured, stream exception). Caller decides next steps.
                return None
            try:
                return parse_llm_json(raw, schema)
            except LLMSchemaError as e:
                last_parse_error = str(e)
                if i == max_retries:
                    log.info("call_llm_typed gave up parsing %s after %d tries: %s",
                             schema.__name__, max_retries + 1, e)
                    # Surface the schema failure to the UI so the user
                    # sees WHY the agent stalled (which field was wrong,
                    # which enum value mismatched, etc.).
                    snippet = (raw or "").strip()
                    if len(snippet) > 400:
                        snippet = snippet[:400] + "..."
                    await self.emit_error(
                        session_id,
                        f"[{self.name}] LLM returned an invalid "
                        f"{schema.__name__} JSON after {max_retries + 1} "
                        f"attempt(s). Parse error: {last_parse_error}. "
                        f"This usually means the model is hitting its "
                        f"context limit, has weak JSON-mode support, or "
                        f"misread the schema. First 400 chars of the "
                        f"final reply: {snippet}",
                    )
                    return None
                attempt_prompt = (
                    prompt
                    + strict_footer
                    + f"\n\nRETRY: your previous reply was not valid "
                    f"{schema.__name__} JSON: {e}. Common causes: emitted "
                    "prose before/after the fence, truncated mid-object "
                    "(ran out of output tokens), or used uppercase enum "
                    "values. Fix and re-emit ONLY the fenced JSON inside "
                    "<<<JSON>>> / <<<END>>> markers."
                )
        return None


def _classify_llm_error(e: Exception, *, provider: str = "", model: str = "") -> str:
    """Translate a provider exception into a user-readable chat message.

    Pattern-matches the most common signals from the OpenAI, Anthropic,
    and Gemini SDKs (status codes embedded in the str, error type
    names, key substrings). Falls back to the raw exception message
    so the user always sees the underlying cause — never a generic
    "AI failed" placeholder.
    """
    msg = str(e)
    low = msg.lower()
    tag = f"[{provider}/{model}] " if provider else ""

    # Rate-limit / quota
    if any(s in low for s in (
        "rate limit", "rate_limit", "ratelimit", "429",
        "tokens per minute", "tokens-per-minute", "tpm",
        "requests per minute", "rpm", "quota", "exceeded",
        "resource_exhausted", "throttle",
    )):
        return (
            f"{tag}Provider rate limit hit: {msg}. Wait ~60s and try again, "
            f"add more keys in Settings → 'multiple-key rotation', or "
            f"upgrade your tier."
        )

    # Context-length / token-budget
    if any(s in low for s in (
        "context length", "context_length", "context window",
        "maximum context", "max_tokens", "too long",
        "prompt is too long", "context limit",
    )):
        return (
            f"{tag}Context window exceeded: {msg}. The conversation has "
            f"grown past this model's max input. Start a fresh session, "
            f"or switch to a larger-context model in Settings."
        )

    # Auth / key
    if any(s in low for s in (
        "401", "403", "unauthorized", "forbidden",
        "invalid api key", "invalid_api_key", "authentication",
        "permission denied", "api key not valid",
    )):
        return (
            f"{tag}Authentication failed: {msg}. Check the API key in "
            f"Settings — it may be invalid, revoked, or missing the "
            f"permission scope the model requires."
        )

    # Model not found / unsupported
    if any(s in low for s in (
        "model not found", "model_not_found", "unknown model",
        "does not exist", "not supported",
    )):
        return (
            f"{tag}Model unavailable: {msg}. Verify the model id in "
            f"Settings — typos and deprecated ids are the usual cause."
        )

    # Network / connectivity
    if any(s in low for s in (
        "connection", "timeout", "timed out", "dns",
        "name resolution", "unreachable", "refused",
    )):
        return (
            f"{tag}Network error reaching the provider: {msg}. Check "
            f"your internet connection or the provider's status page."
        )

    # Server / 5xx
    if any(s in low for s in ("500", "502", "503", "504", "internal server", "service unavailable")):
        return (
            f"{tag}Provider server error: {msg}. The provider is having "
            f"trouble; retry in a minute."
        )

    # Default — show the raw exception so the user still sees the cause.
    return f"{tag}LLM call failed: {msg}"


def _format_schema_for_prompt(schema: Type[BaseModel]) -> str:
    """Render a compact prompt-ready view of a pydantic schema.

    We do not dump the full JSON Schema (it can be 100+ lines for nested
    models and burns prompt budget for little gain). Instead we extract
    just the surface every Literal/enum LLM mistake hits: field name,
    type hint, default, and the exact list of valid enum values.
    """
    try:
        full = schema.model_json_schema()
    except Exception:
        return f"(could not introspect {schema.__name__})"

    lines: list[str] = []
    defs = full.get("$defs") or full.get("definitions") or {}

    def _resolve(ref: str) -> dict[str, Any]:
        key = ref.rsplit("/", 1)[-1]
        return defs.get(key) or {}

    def _describe_type(field_schema: dict[str, Any]) -> str:
        if "enum" in field_schema:
            vals = field_schema["enum"]
            return "enum: " + " | ".join(repr(v) for v in vals)
        if "const" in field_schema:
            return f"const {field_schema['const']!r}"
        if "anyOf" in field_schema:
            return " or ".join(_describe_type(v) for v in field_schema["anyOf"])
        if "$ref" in field_schema:
            ref_schema = _resolve(field_schema["$ref"])
            if "enum" in ref_schema:
                return "enum: " + " | ".join(repr(v) for v in ref_schema["enum"])
            return ref_schema.get("title", "object")
        t = field_schema.get("type")
        if t == "array":
            items = field_schema.get("items") or {}
            return f"array<{_describe_type(items)}>"
        if t == "object":
            return "object"
        if isinstance(t, list):
            return " or ".join(t)
        return t or "any"

    props = full.get("properties") or {}
    required = set(full.get("required") or [])
    for name, spec in props.items():
        req = " (required)" if name in required else ""
        default = spec.get("default", "<no default>")
        type_desc = _describe_type(spec)
        desc = spec.get("description") or ""
        line = f"- {name}: {type_desc}{req}, default={default!r}"
        if desc:
            line += f"  // {desc[:80]}"
        lines.append(line)

    # Cap at a sane budget so deeply-nested models don't blow context.
    body = "\n".join(lines[:60])
    if len(lines) > 60:
        body += f"\n... ({len(lines) - 60} more fields elided)"
    return f"Schema: {schema.__name__}\n{body}"


def _normalize_md(text: str) -> str:
    """Light normalization that does NOT eat code or regex characters.

    Earlier versions stripped every ``*`` which mangled regex literals,
    code samples, and any LLM output containing ``*``. We now only
    collapse multi-line fenced bold and leave everything else alone.
    """
    if not text:
        return text
    # Trim trailing whitespace per line; the FE renders whitespace-pre-wrap.
    return "\n".join(line.rstrip() for line in str(text).splitlines()).strip()
