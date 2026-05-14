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
) -> Callable[[Callable[..., Awaitable[dict[str, Any]]]], Callable[..., Awaitable[dict[str, Any]]]]:
    """Decorator. Registers the function as a tool under `name`.

    cancel_safe / interactive are honored by the AgenticLoop - see ToolDef.
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
        )
        return fn
    return deco


def get_tool(name: str) -> Optional[ToolDef]:
    return REGISTRY.get(name)


async def run_tool(name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    """Dispatch + audit. Emits AgentToolCalled with timing.

    Returns a structured result dict; on failure returns {"error": ...} rather
    than raising, because tool failure is part of the agent control flow.
    """
    t = REGISTRY.get(name)
    if not t:
        return {"error": f"unknown tool: {name}"}

    started = time.perf_counter()
    try:
        result = await t.fn(args or {}, ctx)
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
