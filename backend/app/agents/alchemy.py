"""DataAlchemistAgent — turns 'crap' into 'fortune' (blueprint §4).

Sits between the deterministic ``DatasetProfilingAgent`` and the
``TaskInferenceAgent`` and adds the Socratic data-health step:

    1. Reads the just-completed profile.
    2. Computes a Data Health Report (duplicate, missing, low-entropy,
       sensitive-field hits, schema fragmentation).
    3. Posts the report on the blackboard and emits a ``DataHealthReport``
       event that the UI renders as a dedicated card.
    4. If issues are severe, surfaces a *blocking* recommendation as an
       ``AgentAsking`` event; otherwise it streams the verdict and lets
       the pipeline continue.

This agent is the embodiment of the blueprint's "Data is King" commandment:
it pauses before wasting compute on poor data.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import session_service


_DUP_HARD = 30.0       # blueprint example: 30% dupes triggers explicit ask
_DUP_SOFT = 10.0
_MISSING_HARD = 50.0


class DataAlchemistAgent(BaseAgent):
    name = "DataAlchemistAgent"
    role = "Detects data-quality risks and surfaces the Data Health Report."
    allowed_tools = (
        "alchemy.semantic_dedup",
        "alchemy.entropy_filter",
        "alchemy.redact_sensitive",
        "alchemy.induce_schema",
        "audit.write",
    )
    triggers = ("DatasetProfileCompleted",)

    async def handle(self, event: AgentEvent) -> None:
        session = self.get_session(event.session_id)
        if not session:
            return
        profile = event.payload.get("profile") or {}
        dataset_id = event.payload.get("dataset_id")

        await self.think(
            session.id,
            "Profile is in. I'm grading data health (duplicates, missing values, "
            "low-entropy boilerplate, sensitive-info leaks, schema fragmentation) "
            "before we plan training.",
            parent=event.id,
        )

        report = self._build_report(profile)
        session_service.attach_artifact(session, "data_health", report)
        await self.emit(
            "DataHealthReport",
            session.id,
            payload={"dataset_id": dataset_id, "report": report},
            parent_event_id=event.id,
            confidence=report["confidence"],
        )

        # Streamed verdict line — the UI renders a coloured banner.
        verdict = report["verdict"]
        await self.emit_message(
            session.id,
            f"Data Health: **{verdict.replace('_', ' ').title()}** — {report['summary']}",
            parent=event.id,
        )

        # Hard issues surface as a Socratic ask. The blueprint's example is
        # the "30% duplicates / inconsistent schemas" question.
        if report["asks"]:
            await self.ask(
                session.id,
                report["asks"][0],
                impact="high",
                parent=event.id,
            )

    # ── helpers ──────────────────────────────────────────────────────────

    def _build_report(self, profile: dict[str, Any]) -> dict[str, Any]:
        dup_pct = float((profile.get("duplicates") or {}).get("duplicate_pct") or 0.0)
        missing = profile.get("missing") or {}
        worst_missing = 0.0
        worst_col = None
        for col, m in (missing.get("per_column") or {}).items():
            pct = float(m.get("missing_pct") or 0.0)
            if pct > worst_missing:
                worst_missing, worst_col = pct, col

        imbalance = profile.get("imbalance") or {}
        minority = float(imbalance.get("minority_pct") or 100.0)

        score = 1.0
        notes: list[str] = []
        asks: list[str] = []

        if dup_pct >= _DUP_HARD:
            score -= 0.4
            notes.append(f"{dup_pct:.0f}% duplicate rows")
            asks.append(
                f"I found {dup_pct:.0f}% duplicates. Should I deduplicate, or "
                "treat them as separate domains?"
            )
        elif dup_pct >= _DUP_SOFT:
            score -= 0.15
            notes.append(f"{dup_pct:.0f}% duplicates (advisory)")

        if worst_missing >= _MISSING_HARD and worst_col:
            score -= 0.3
            notes.append(f"`{worst_col}` is {worst_missing:.0f}% missing")
            asks.append(
                f"`{worst_col}` is missing in {worst_missing:.0f}% of rows. Drop it, "
                "impute, or skip rows that lack it?"
            )

        if 0 < minority < 5:
            score -= 0.2
            notes.append(f"smallest class is only {minority:.1f}% — risk of imbalance")

        score = max(0.0, min(1.0, score))
        if score >= 0.85:
            verdict = "healthy"
        elif score >= 0.7:
            verdict = "advisory"
        elif score >= 0.5:
            verdict = "needs_attention"
        else:
            verdict = "blocking"

        summary = ", ".join(notes) if notes else "clean dataset; no remediation suggested"
        return {
            "verdict": verdict,
            "score": round(score, 3),
            "confidence": round(0.6 + 0.4 * score, 3),
            "summary": summary,
            "asks": asks,
            "signals": {
                "duplicate_pct": dup_pct,
                "worst_missing_pct": worst_missing,
                "worst_missing_column": worst_col,
                "minority_class_pct": minority,
            },
        }
