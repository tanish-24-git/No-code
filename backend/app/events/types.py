"""Canonical event taxonomy. Every state transition in the agent runtime is an
event of one of these kinds. Add a new kind here when you add a new agent
behaviour, never inline strings elsewhere."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


EventKind = Literal[
    # Session lifecycle
    "SessionStarted", "SessionPaused", "SessionResumed", "SessionClosed",
    # Dataset intake & profiling
    "DatasetUploaded",
    "IntakeStarted", "IntakeCompleted",
    "DatasetProfileStarted", "DatasetProfileCompleted",
    "DataHealthReport",          # Data Alchemist surfaces ‹crap-vs-fortune› verdict
    # Task inference & clarification
    "TaskInferenceStarted", "TaskInferred", "IntentConfidenceLow",
    "UserClarificationRequested", "UserClarificationReceived",
    # Hardware
    "HardwareProfileStarted", "HardwareProfileCompleted",
    # Model selection / strategy / planning
    "CandidateModelsRanked", "StrategyChosen",
    "PipelineDraftRequested", "PipelineDraftCreated",
    "PipelineApprovalRequested", "PipelineApproved", "PipelineRejected",
    # Execute / monitor / recover
    "PipelineExecutionStarted",
    "TrainingMetricUpdated",
    "TrainingAnomalyDetected", "RecoveryPlanGenerated",
    "RetryRequested", "RetryApproved", "RetryDenied",
    # Wrap-up
    "TrainingCompleted", "EvaluationStarted", "EvaluationCompleted",
    "ExportChoiceRequested",
    "SaveLocalRequested", "PushToHFRequested",
    "ExportCompleted",
    # ── Socratic / interactive streaming kinds ────────────────────────────
    # See blueprint §3.1. payload.stream is one of:
    #   "thinking"   — internal reasoning trace ("I am currently…")
    #   "planning"   — step-by-step task breakdown
    #   "asking"     — interactive dialogue ⇒ pauses execution until user answers
    #   "garnishing" — UI finalization (nodes, edges, animations)
    #   "executing"  — live execution log tail
    "AgentThinking",            # purple in UI
    "AgentPlanning",            # blue in UI
    "AgentAsking",              # yellow in UI; pairs with UserClarificationRequested
    "AgentGarnishing",          # cyan in UI; one per node animation step
    "AgentExecuting",           # green in UI; live tail
    # ── Hierarchy / collaboration ─────────────────────────────────────────
    "BlackboardUpdated",        # any agent posts to the federated blackboard
    "AuditCritique",            # Audit Agent (Critic) reviews a plan
    "AuditOverride",            # Critic vetoed; user or master must resolve
    "SandboxBenchmarkStarted",
    "SandboxBenchmarkCompleted",
    "CircuitBreakerTripped",    # too many loops detected on the same kind
    "CheckpointSaved",          # session checkpoint persisted
    # ── Conversational + audit ────────────────────────────────────────────
    "AssistantMessage",         # what the chat panel renders as a chat bubble
    "UserMessage",              # free-text from the user
    "AgentToolCalled",          # one entry per tool call, for the trace
    "AgentDecisionRecorded",    # references a decision_id in the audit log
    "Error",
]


# Visual stream tone — mapped to UI colors per blueprint §11.
StreamTone = Literal["thinking", "planning", "asking", "garnishing", "executing"]


def _new_id() -> str:
    return uuid.uuid4().hex


def _now() -> datetime:
    return datetime.now(timezone.utc)


class AgentEvent(BaseModel):
    """Append-only message on the bus. Persisted to data/events/<session>.jsonl."""

    id: str = Field(default_factory=_new_id)
    session_id: str
    kind: EventKind
    actor: str                              # agent name, "user", or "system"
    parent_event_id: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    rationale: Optional[str] = None
    confidence: Optional[float] = None      # 0..1 when applicable
    decision_id: Optional[str] = None       # link into decision_log
    created_at: datetime = Field(default_factory=_now)

    def as_sse(self) -> str:
        """Format as a Server-Sent Event frame."""
        import json
        body = self.model_dump(mode="json")
        return f"event: {self.kind}\ndata: {json.dumps(body)}\n\n"
