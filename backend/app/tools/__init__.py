"""Tool registry. Every external action an agent can take goes through here.

Tools declare a JSON input schema and a side-effect class so policies can
gate them (e.g. require_approval for expensive or irreversible ops).

Tools are registered at import time; importing app.tools wires them all up.
"""
from app.tools.registry import REGISTRY, ToolDef, ToolContext, get_tool, run_tool, tool

# Side-effecting imports register their tools.
from app.tools import (  # noqa: F401
    alchemy,
    audit,
    clarify,
    data,
    dataset,
    eval as eval_tools,
    export,
    hardware,
    hf_search,
    job,
    llm,
    metrics,
    model,
    pipeline,
    recovery,
    sandbox,
    strategy,
    task,
    # AgenticLoop additions:
    web,
    agent_tools,
)

__all__ = ["REGISTRY", "ToolDef", "ToolContext", "tool", "get_tool", "run_tool"]
