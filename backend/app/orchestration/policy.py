"""Cost / approval / safety policy. The orchestration layer asks this module
before letting an agent autonomously execute a sensitive step."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Decision = Literal["approve", "block", "require_user_approval"]


@dataclass(frozen=True)
class CostPolicy:
    auto_approve_under_minutes: float = 5.0
    max_recoveries: int = 2
    push_to_hf_requires_user: bool = True
    save_local_requires_user: bool = True


POLICY = CostPolicy()


def decide_plan_approval(estimated_minutes: float, confidence: float) -> Decision:
    if estimated_minutes <= POLICY.auto_approve_under_minutes and confidence >= 0.85:
        return "approve"
    return "require_user_approval"


def decide_recovery_approval(confidence: float, prior_recoveries: int) -> Decision:
    if prior_recoveries >= POLICY.max_recoveries:
        return "block"
    if confidence >= 0.9:
        return "approve"
    return "require_user_approval"


def decide_export(choice: str) -> Decision:
    # User-driven; we never auto-export.
    if choice in ("local", "hf", "both"):
        return "approve"
    return "block"
