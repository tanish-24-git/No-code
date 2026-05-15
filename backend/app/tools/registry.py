"""Tool registry. A tool is a pure(ish) async function with metadata.

Side-effect class is used by policies to decide whether a tool can run
autonomously, requires approval, or must be sandboxed:

    read              read-only inspection
    write_session     mutates session state only
    write_resource    mutates a persistent resource (pipeline, job, model)
    external          calls outside the process (HF Hub push, etc.)
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Literal, Optional

from app.events.bus import EventBus
from app.events.types import AgentEvent


log = logging.getLogger("finetune-studio.tools")

SideEffect = Literal["read", "write_session", "write_resource", "external"]
CostClass = Literal["cheap", "medium", "expensive"]


@dataclass
class ToolContext:
    """Bag of context passed to every tool. Tools must not import the bus
    or the session service directly — they receive it here."""
    session_id: str
    bus: Optional[EventBus] = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDef:
    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: dict[str, Any]
    fn: Callable[[dict[str, Any], ToolContext], Awaitable[dict[str, Any]]]
    side_effect: SideEffect = "read"
    requires_approval: bool = False
    cost_class: CostClass = "cheap"
    # AgenticLoop metadata:
    # cancel_safe=True  - if the user interrupts mid-call, the runtime may
    #                     cancel this tool with asyncio.CancelledError and
    #                     return {"interrupted": True} to the model.
    # cancel_safe=False - the call is wrapped in asyncio.shield so user
    #                     interrupts are queued and processed when the tool
    #                     returns. Use for training, export, HF push.
    # interactive=True  - the tool will emit user-facing events (questions,
    #                     plans) and may await user response. The loop must
    #                     not treat a slow response as a stall.
    cancel_safe: bool = True
    interactive: bool = False
    # supports_full_output=False (default): the central observation-budget
    #     middleware in run_tool() will truncate string fields longer than
    #     FT_OBSERVATION_BUDGET before the result reaches the LLM. Use this
    #     for tools whose outputs the LLM never needs in full.
    # supports_full_output=True: the tool already manages its own size
    #     (web_fetch caps at 8000 chars; raw_extractor caps structured
    #     output at 200K chars). The middleware leaves the result alone.
    supports_full_output: bool = False


REGISTRY: dict[str, ToolDef] = {}


def tool(
    *,
    name: str,
    description: str,
    input_schema: dict[str, Any],
    output_schema: Optional[dict[str, Any]] = None,
    side_effect: SideEffect = "read",
    requires_approval: bool = False,
    cost_class: CostClass = "cheap",
    cancel_safe: bool = True,
    interactive: bool = False,
    supports_full_output: bool = False,
) -> Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]]:
    """Decorator. Registers the function as a tool under `name`.

    cancel_safe / interactive are honored by the AgenticLoop - see ToolDef.
    supports_full_output bypasses the central observation-budget truncator.
    """
    def deco(fn: Callable[..., Awaitable[dict[str, Any]]]):
        if name in REGISTRY:
            # Re-registration is a no-op during dev reloads. We log and skip
            # rather than raising so module re-imports don't crash startup.
            log.debug("tool %s already registered; skipping duplicate", name)
            return fn
        REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            input_schema=input_schema,
            output_schema=output_schema or {"type": "object"},
            fn=fn,
            side_effect=side_effect,
            requires_approval=requires_approval,
            cost_class=cost_class,
            cancel_safe=cancel_safe,
            interactive=interactive,
            supports_full_output=supports_full_output,
        )
        return fn
    return deco


def get_tool(name: str) -> Optional[ToolDef]:
    return REGISTRY.get(name)


async def run_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Dispatch + audit. Emits AgentToolCalled with timing.

    Wraps the tool body in two pieces of middleware:
        Pre: strip None and empty-string values from args. Weaker models
        (Groq's llama-3.3-70b, Gemini Flash) routinely emit ``"family":
        null`` or ``"size_b": ""`` for optional fields; we drop them so
        every tool body sees a clean dict.
        Post: if the tool did not declare supports_full_output=True, walk
        the result dict and truncate any string field longer than the
        observation budget (default 2000 chars, tunable via
        FT_OBSERVATION_BUDGET). Protects the LLM's context window from
        runaway tool outputs across many turns.

    Returns a structured result dict; on failure returns {"error": ...} rather
    than raising, because tool failure is part of the agent control flow.
    """
    t = REGISTRY.get(name)
    if not t:
        return {"error": f"unknown tool: {name}"}

    cleaned_args = _strip_nulls_and_empties(args or {})

    started = time.perf_counter()
    try:
        result = await t.fn(cleaned_args, ctx)
        if not t.supports_full_output:
            result = _apply_observation_budget(result, _observation_budget())
        ms = round((time.perf_counter() - started) * 1000, 1)
        if ctx.bus is not None:
            await ctx.bus.publish(AgentEvent(
                session_id=ctx.session_id,
                kind="AgentToolCalled",
                actor=ctx.extras.get("agent", "unknown"),
                payload={"tool": name, "ms": ms, "ok": "error" not in result},
            ))
        return result
    except Exception as e:
        ms = round((time.perf_counter() - started) * 1000, 1)
        log.exception("tool %s crashed", name)
        if ctx.bus is not None:
            await ctx.bus.publish(AgentEvent(
                session_id=ctx.session_id,
                kind="AgentToolCalled",
                actor=ctx.extras.get("agent", "unknown"),
                payload={"tool": name, "ms": ms, "ok": False, "error": str(e)},
            ))
        return {"error": str(e)}


def list_descriptors() -> list[dict[str, Any]]:
    """For the /api/tools introspection endpoint."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
            "output_schema": t.output_schema,
            "side_effect": t.side_effect,
            "requires_approval": t.requires_approval,
            "cost_class": t.cost_class,
            "cancel_safe": t.cancel_safe,
            "interactive": t.interactive,
            "supports_full_output": t.supports_full_output,
        }
        for t in REGISTRY.values()
    ]


# Tools exposed to the AgenticLoop by name. The registry contains many
# legacy tools (dataset.*, model.*, alchemy.*, ...) that are called
# internally by wrapped agents; the loop should only see high-level
# composite tools, otherwise the tool-schema overhead exhausts the
# provider's tokens-per-minute budget on rate-limited free tiers.
_AGENT_LOOP_TOOLS: tuple[str, ...] = (
    "probe_hardware",
    "profile_dataset",
    "grade_data_health",
    "infer_task_type",
    "select_base_model",
    "propose_training_strategy",
    "build_pipeline",
    "run_training",
    "evaluate_model",
    "export_artifact",
    "propose_plan",
    "ask_user",
    "record_decision",
    "walk_session_uploads",
    "extract_raw_text",
    "synthesize_unified_dataset",
    "search_hf_models",
    "web_search",
    "web_fetch",
    "propose_recovery",
)


def llm_tool_schemas() -> list[dict[str, Any]]:
    """Tools exposed to the AgenticLoop's tool-use channel.

    Filtered to high-level composite tools only - the wrapped specialty
    agents' internal helpers (dataset.profile_tokens, model.search_hf,
    etc.) stay hidden so the per-turn schema budget stays tight.
    """
    out: list[dict[str, Any]] = []
    for name in _AGENT_LOOP_TOOLS:
        t = REGISTRY.get(name)
        if t is None:
            continue
        out.append({
            "name": t.name,
            "description": t.description,
            "input_schema": t.input_schema,
        })
    return out


def _safe_json(value: Any) -> str:
    return json.dumps(value, default=str)


# ── Central middleware helpers ──────────────────────────────────────────

def _observation_budget() -> int:
    try:
        return max(200, int(os.getenv("FT_OBSERVATION_BUDGET", "2000")))
    except ValueError:
        return 2000


def _strip_nulls_and_empties(args: dict[str, Any]) -> dict[str, Any]:
    """Drop keys whose value is None or an empty string.

    Weaker LLMs (Groq llama-3.3-70b, Gemini Flash) routinely emit null or
    "" for unused optional parameters, then tool bodies that rely on
    ``args.get("k")`` get unexpected falsy values that bypass their
    defaults. Stripping here at the dispatcher means every tool body
    sees a clean dict without having to patch each tool individually.
    """
    if not isinstance(args, dict):
        return args
    return {
        k: v for k, v in args.items()
        if v is not None and not (isinstance(v, str) and v == "")
    }


def _apply_observation_budget(result: Any, max_chars: int) -> Any:
    """Recursively truncate string fields in ``result`` longer than
    ``max_chars``. Non-string values pass through untouched. Designed to
    be cheap and idempotent.

    The marker ``...[truncated]`` makes the cap visible to the model so
    it can ask for more (e.g. by re-calling with a different argument)
    rather than assuming the field ended there.
    """
    if isinstance(result, str):
        if len(result) > max_chars:
            return result[:max_chars] + "...[truncated]"
        return result
    if isinstance(result, dict):
        return {k: _apply_observation_budget(v, max_chars) for k, v in result.items()}
    if isinstance(result, list):
        return [_apply_observation_budget(v, max_chars) for v in result]
    return result
