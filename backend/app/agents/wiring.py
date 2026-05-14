"""Single source of truth for the event -> agent mapping.

v5.0 - AgenticLoop. The static fan-out DAG of 20 agents is gone. The
loop is the orchestrator now: it thinks, picks tools, observes, repeats,
and reacts to user messages mid-flow. Most of the old specialty agents
live on as tool implementations called by the loop (see
``backend/app/tools/agent_tools.py``), not as bus subscribers.

We keep three subscriptions only:

    DatasetIntakeAgent     reactive to upload; populates dataset_facts
                           and emits IntakeCompleted (which wakes the loop)
    AgenticLoop            the model-driven core. Triggers on
                           SessionStarted, IntakeCompleted, UserMessage.
    TrainingMonitorAgent   reactive to training metric events from the
                           training worker thread. Stays reactive because
                           the metrics arrive asynchronously and the
                           training tool is one long-running step inside
                           the loop.
"""
from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.agents.intake import DatasetIntakeAgent
from app.agents.loop import AgenticLoop
from app.agents.monitoring import TrainingMonitorAgent
from app.events.bus import EventBus


log = logging.getLogger("finetune-studio.agents.wiring")


def register_agents(bus: EventBus) -> list[BaseAgent]:
    """Instantiate the three bus-subscribed agents and register them.

    Loop-only agents (Profiling, Hardware, ModelSelection, Strategy,
    PipelineBuilder, Execution, Evaluation, Export, Restructurer, etc.)
    are NOT instantiated here. They are imported and instantiated
    on-demand by the tool wrappers in app/tools/agent_tools.py.
    """
    # Importing agent_tools forces the @tool decorators in that module
    # (and in app.tools.web) to register their tools so the AgenticLoop
    # sees them on its first introspection of the registry.
    import app.tools.agent_tools  # noqa: F401

    agents: list[BaseAgent] = [
        DatasetIntakeAgent(bus),
        AgenticLoop(bus),
        TrainingMonitorAgent(bus),
    ]
    for agent in agents:
        for kind in agent.triggers:
            bus.on(kind, agent.safe_handle)
        log.info("registered %s on %s", agent.name, ",".join(agent.triggers))
    return agents
