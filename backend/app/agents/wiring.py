"""Single source of truth for the event → agent mapping. Called once at
FastAPI startup. Adding a new agent is a one-entry change here."""
from __future__ import annotations

import logging

from app.agents.base import BaseAgent
from app.agents.clarification import ClarificationAgent
from app.agents.evaluation import EvaluationAgent
from app.agents.execution import ExecutionAgent
from app.agents.export import ExportAgent
from app.agents.gates import ApprovalGate, ConfidenceGate
from app.agents.hardware_analysis import HardwareAnalysisAgent
from app.agents.intake import DatasetIntakeAgent
from app.agents.model_selection import ModelSelectionAgent
from app.agents.monitoring import TrainingMonitorAgent
from app.agents.orchestrator import OrchestratorAgent
from app.agents.pipeline_builder import PipelineBuilderAgent
from app.agents.profiling import DatasetProfilingAgent
from app.agents.recovery import RecoveryAgent
from app.agents.strategy import TrainingStrategyAgent
from app.agents.task_inference import TaskInferenceAgent
from app.events.bus import EventBus


log = logging.getLogger("finetune-studio.agents.wiring")


def register_agents(bus: EventBus) -> list[BaseAgent]:
    """Instantiate every agent and subscribe it to its trigger events.

    Returns the list of agent instances so they can be inspected at runtime.
    """
    agents: list[BaseAgent] = [
        OrchestratorAgent(bus),
        DatasetIntakeAgent(bus),
        DatasetProfilingAgent(bus),
        HardwareAnalysisAgent(bus),
        TaskInferenceAgent(bus),
        ConfidenceGate(bus),
        ClarificationAgent(bus),
        ModelSelectionAgent(bus),
        TrainingStrategyAgent(bus),
        PipelineBuilderAgent(bus),
        ApprovalGate(bus),
        ExecutionAgent(bus),
        TrainingMonitorAgent(bus),
        RecoveryAgent(bus),
        EvaluationAgent(bus),
        ExportAgent(bus),
    ]
    for agent in agents:
        for kind in agent.triggers:
            bus.on(kind, agent.handle)
        log.info("registered %s on %s", agent.name, ",".join(agent.triggers))
    return agents
