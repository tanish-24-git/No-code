"""Confidence thresholds and gating helpers."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Thresholds:
    # Below this, ClarificationAgent activates.
    proceed_to_plan: float = 0.7
    # Auto-approve a plan only above this *and* under cost cap.
    auto_approve_plan: float = 0.85
    # Auto-approve a recovery diff only above this.
    auto_approve_recovery: float = 0.9


THRESHOLDS = Thresholds()


def is_high_confidence(value: float) -> bool:
    return value >= THRESHOLDS.proceed_to_plan
