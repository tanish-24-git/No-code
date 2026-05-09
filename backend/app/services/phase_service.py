"""Phase narration helpers.

Each agent wraps its work in a Phase. A phase is:

    PhaseStarted        - "I'm about to do X. Here is the plan."
    PhasePlanProposed   - the plan markdown the user can read
    [optional approval] - user clicks approve, comments, or stays silent
    NodeMaterialized    - tells the canvas to pop a node into existence
    PhaseCompleted      - "X is done; here is what I produced"

The plan + approval pattern is the same one Antigravity IDE uses: the
agent never silently jumps ahead. For non-critical phases, the pause is
short and auto-continues; for critical phases (model search, training)
the agent waits for an explicit approve / comment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from app.events.bus import EventBus
from app.events.types import AgentEvent


# Canonical phase names. Keep in sync with the FE node map.
PHASES = (
    "intake",         # Dataset metadata read
    "profile",        # Token / duplicate / missing / imbalance scan
    "hardware",       # Device + VRAM probe
    "task",           # Infer task type
    "model_search",   # HF Hub search + ranking
    "strategy",       # Hyperparameters + PEFT variant
    "plan",           # Pipeline graph build
    "audit",          # Critic review (only if auditor enabled)
    "execute",        # Training start
    "monitor",        # Live metrics
    "evaluate",       # Post-training eval
    "sandbox",        # Clean-room benchmarks
    "export",         # Save local / push to HF
)


# Phases that require explicit user approval before they advance.
CRITICAL_PHASES = frozenset({"model_search", "plan", "execute", "export"})


# Optional: which canvas node corresponds to which phase. Multiple phases
# can reference the same node (e.g. monitor + evaluate live on the train
# node). The frontend uses this map to know which node to materialize.
PHASE_TO_NODE: dict[str, str] = {
    "intake":       "dataset",
    "profile":      "preprocess",
    "hardware":     "preprocess",
    "task":         "preprocess",
    "model_search": "model",
    "strategy":     "strategy",
    "plan":         "strategy",
    "audit":        "preprocess",
    "execute":      "train",
    "monitor":      "train",
    "evaluate":     "evaluate",
    "sandbox":      "evaluate",
    "export":       "export",
}


@dataclass
class PhasePlan:
    """A markdown plan card surfaced to the user before a phase runs."""
    phase: str
    title: str
    summary: str           # one-line summary of what the agent will do
    steps: list[str]       # bullet list of concrete steps
    inputs: dict[str, Any] # what the agent is reading
    outputs: list[str]     # what the agent will produce
    requires_approval: bool

    def as_markdown(self) -> str:
        lines = [f"## Phase: {self.title}", "", self.summary, ""]
        if self.steps:
            lines.append("**Plan:**")
            for s in self.steps:
                lines.append(f"- {s}")
            lines.append("")
        if self.outputs:
            lines.append("**Will produce:** " + ", ".join(self.outputs))
        if self.requires_approval:
            lines.append("\n_This phase needs your approval before it continues._")
        return "\n".join(lines)


async def announce_phase(
    bus: EventBus,
    *,
    session_id: str,
    actor: str,
    phase: str,
    plan: PhasePlan,
    parent_event_id: Optional[str] = None,
) -> None:
    """Emit PhaseStarted + PhasePlanProposed in one shot.

    The frontend renders the plan as a chat bubble with [approve] / [comment]
    buttons when ``plan.requires_approval`` is true, or as a passive note
    otherwise.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase!r}")
    payload = {
        "phase": phase,
        "node": PHASE_TO_NODE.get(phase),
        "title": plan.title,
        "requires_approval": plan.requires_approval,
        "summary": plan.summary,
    }
    await bus.publish(AgentEvent(
        session_id=session_id, kind="PhaseStarted", actor=actor,
        payload=payload, parent_event_id=parent_event_id,
    ))
    await bus.publish(AgentEvent(
        session_id=session_id, kind="PhasePlanProposed", actor=actor,
        payload={
            **payload,
            "plan_markdown": plan.as_markdown(),
            "steps": plan.steps,
            "inputs": plan.inputs,
            "outputs": plan.outputs,
        },
        parent_event_id=parent_event_id,
    ))


async def announce_node(
    bus: EventBus,
    *,
    session_id: str,
    actor: str,
    node: dict[str, Any],
    parent_event_id: Optional[str] = None,
) -> None:
    """Tell the canvas to pop a single node into existence."""
    await bus.publish(AgentEvent(
        session_id=session_id, kind="NodeMaterialized", actor=actor,
        payload={"node": node}, parent_event_id=parent_event_id,
    ))


async def announce_edge(
    bus: EventBus,
    *,
    session_id: str,
    actor: str,
    edge: dict[str, Any],
    parent_event_id: Optional[str] = None,
) -> None:
    """Tell the canvas to draw a connection between two nodes."""
    await bus.publish(AgentEvent(
        session_id=session_id, kind="EdgeMaterialized", actor=actor,
        payload={"edge": edge}, parent_event_id=parent_event_id,
    ))


async def complete_phase(
    bus: EventBus,
    *,
    session_id: str,
    actor: str,
    phase: str,
    summary: str,
    artifacts: Optional[dict[str, Any]] = None,
    parent_event_id: Optional[str] = None,
) -> None:
    await bus.publish(AgentEvent(
        session_id=session_id, kind="PhaseCompleted", actor=actor,
        payload={
            "phase": phase,
            "summary": summary,
            "artifacts": artifacts or {},
            "node": PHASE_TO_NODE.get(phase),
        },
        parent_event_id=parent_event_id,
    ))
