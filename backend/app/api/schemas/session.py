"""AgentSession is the long-running record that ties together a dataset, the
agents working on it, and all events fired against it. Its FSM is the source
of truth for "what stage is this in?".

API keys carried per-session are encrypted at rest. The plaintext value is
held in memory only as long as needed to make a provider call; on disk we
only ever see the Fernet ciphertext.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_serializer, field_validator

from app.utils import crypto


class FSMState(str, Enum):
    INIT = "init"
    PROFILING = "profiling"
    CLARIFYING = "clarifying"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    MONITORING = "monitoring"
    RECOVERING = "recovering"
    EVALUATING = "evaluating"
    AWAITING_EXPORT_CHOICE = "awaiting_export_choice"
    FINALIZING = "finalizing"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Allowed transitions. Pending-approval lives in `pending_approvals` artifact;
# the FSM stays a strict ladder so illegal jumps surface as errors.
ALLOWED_TRANSITIONS: dict[FSMState, set[FSMState]] = {
    FSMState.INIT:                  {FSMState.PROFILING, FSMState.AWAITING_APPROVAL,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.PROFILING:             {FSMState.CLARIFYING, FSMState.PLANNING,
                                     FSMState.AWAITING_APPROVAL,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.CLARIFYING:            {FSMState.CLARIFYING, FSMState.PLANNING,
                                     FSMState.AWAITING_APPROVAL,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.PLANNING:              {FSMState.AWAITING_APPROVAL, FSMState.EXECUTING,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.AWAITING_APPROVAL:     {FSMState.PROFILING, FSMState.CLARIFYING,
                                     FSMState.PLANNING, FSMState.EXECUTING,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.EXECUTING:             {FSMState.MONITORING,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.MONITORING:            {FSMState.RECOVERING, FSMState.EVALUATING,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.RECOVERING:            {FSMState.EXECUTING,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.EVALUATING:            {FSMState.AWAITING_EXPORT_CHOICE,
                                     FSMState.FAILED, FSMState.CANCELLED},
    FSMState.AWAITING_EXPORT_CHOICE: {FSMState.FINALIZING, FSMState.CANCELLED},
    FSMState.FINALIZING:            {FSMState.DONE, FSMState.FAILED},
    FSMState.DONE:                  set(),
    FSMState.FAILED:                set(),
    FSMState.CANCELLED:             set(),
}


class ClarificationQuestion(BaseModel):
    """A focused question put to the user. Pulled from the catalog so the agent
    can never invent free-form questions; this keeps UX predictable."""

    question_id: str
    question: str
    kind: str                       # single_choice | multi_choice | text | yes_no
    options: list[str] = Field(default_factory=list)
    context: Optional[str] = None
    required: bool = True


class ClarificationAnswer(BaseModel):
    question_id: str
    value: Any                       # str for single/text/yes_no, list[str] for multi
    answered_at: datetime


class RecoveryRecord(BaseModel):
    diff_id: str
    operations: list[dict[str, Any]]
    rationale: str
    confidence: float
    estimated_extra_minutes: float
    approved: Optional[bool] = None
    created_at: datetime


class AgentSession(BaseModel):
    """Long-running container. One per dataset upload."""

    id: str
    name: str = "New Session"
    dataset_id: str
    pipeline_id: Optional[str] = None
    job_id: Optional[str] = None
    state: FSMState = FSMState.INIT
    state_entered_at: datetime
    confidence: float = 0.0
    question_budget_max: int = 3
    question_budget_used: int = 0
    pending_question: Optional[ClarificationQuestion] = None
    clarifications: list[ClarificationAnswer] = Field(default_factory=list)
    recovery_history: list[RecoveryRecord] = Field(default_factory=list)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    # Per-session LLM configuration overrides. The api_key is stored as
    # ciphertext on disk; access via ``decrypted_api_key()``.
    llm_provider: Optional[str] = None
    llm_model: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_api_key_enc: Optional[str] = None

    last_event_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    def decrypted_api_key(self) -> str:
        if not self.llm_api_key_enc:
            return ""
        try:
            return crypto.decrypt(self.llm_api_key_enc)
        except Exception:
            return ""

    @field_validator("llm_api_key_enc", mode="before")
    @classmethod
    def _accept_plaintext_or_ciphertext(cls, v: Any) -> Any:
        """Accept either pre-encrypted ciphertext or raw plaintext on input.

        Plaintext arriving over the API gets encrypted before persistence.
        Ciphertext (already encrypted, e.g. on session reload) passes through.
        """
        if v is None or v == "":
            return None
        if isinstance(v, str) and v.startswith("gAAAA"):  # Fernet token prefix
            return v
        if isinstance(v, str):
            return crypto.encrypt(v)
        return v


class SessionListItem(BaseModel):
    id: str
    name: str = "New Session"
    dataset_id: str
    state: FSMState
    pipeline_id: Optional[str] = None
    job_id: Optional[str] = None
    confidence: float
    created_at: datetime
    updated_at: datetime


class SessionMessageRequest(BaseModel):
    text: str = Field(..., min_length=1)


class ClarificationReply(BaseModel):
    value: Any


class ApprovalDecision(BaseModel):
    approve: bool
    reason: Optional[str] = None


class ExportChoice(BaseModel):
    choice: str                     # "local" | "hf" | "both"
    hf_repo_id: Optional[str] = None


class RetryDecision(BaseModel):
    diff_id: str
    approve: bool
    reason: Optional[str] = None
