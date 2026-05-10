"""DataAlchemistAgent — grades data health for fine-tuning readiness.

Sits between profiling and the rest of the pipeline. Reads the profile,
asks the LLM (when configured) for a strict-JSON DataHealthReport, and
falls back to a deterministic verdict computed from the metrics.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import DataHealthReport
from app.events.types import AgentEvent
from app.services import session_service


class DataAlchemistAgent(BaseAgent):
    name = "DataAlchemistAgent"
    role = "Detects data-quality risks and surfaces the Data Health Report."
    directive_scope = "data"
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

        # 1. Try LLM-typed.
        report_obj = await self.call_llm_typed(
            event.session_id,
            (
                f"Dataset profile:\n"
                f"- Row count: {profile.get('row_count')}\n"
                f"- Duplicates: {profile.get('duplicates', {}).get('duplicate_pct') or 0}%\n"
                f"- Missing values: {profile.get('missing', {}).get('per_column')}\n"
                f"- Class balance: {profile.get('imbalance', {}).get('minority_pct') or 100}%\n\n"
                "Grade fine-tuning readiness as healthy / advisory / "
                "needs_attention / blocking. Return JSON with verdict, "
                "score (0..1), summary, asks."
            ),
            DataHealthReport,
            system="You are a meticulous data scientist.",
            stream_thoughts=False,
            parent=event.id,
        )

        if report_obj:
            report = report_obj.model_dump()
            report.setdefault("confidence", round(0.6 + 0.4 * report["score"], 3))
        else:
            report = self._deterministic_report(profile)

        session_service.attach_artifact(session, "data_health", report)
        await self.emit(
            "DataHealthReport",
            session.id,
            payload={"dataset_id": dataset_id, "report": report},
            parent_event_id=event.id,
            confidence=report.get("confidence"),
        )

        verdict = report["verdict"]
        await self.emit_message(
            session.id,
            f"Data health: **{verdict.replace('_', ' ').title()}** - {report['summary']}",
            parent=event.id,
        )
        if report.get("asks"):
            await self.ask(session.id, report["asks"][0], impact="high", parent=event.id)

    def _deterministic_report(self, profile: dict) -> dict:
        dup_pct = float((profile.get("duplicates") or {}).get("duplicate_pct") or 0.0)
        worst_missing = 0.0
        for col, m in (profile.get("missing", {}).get("per_column") or {}).items():
            pct = float(m.get("missing_pct") or 0.0)
            if pct > worst_missing:
                worst_missing = pct
        minority = float((profile.get("imbalance") or {}).get("minority_pct") or 100.0)

        score = 1.0
        notes: list[str] = []
        asks: list[str] = []
        if dup_pct >= 30:
            score -= 0.4; notes.append(f"{dup_pct:.0f}% duplicate rows")
            asks.append(f"I found {dup_pct:.0f}% duplicates. Should I dedupe?")
        elif dup_pct >= 10:
            score -= 0.15; notes.append(f"{dup_pct:.0f}% duplicates (advisory)")
        if worst_missing >= 50:
            score -= 0.3; notes.append(f"a column is {worst_missing:.0f}% missing")
        if 0 < minority < 5:
            score -= 0.2; notes.append(f"smallest class is only {minority:.1f}%")

        score = max(0.0, min(1.0, score))
        verdict = (
            "healthy" if score >= 0.85
            else "advisory" if score >= 0.7
            else "needs_attention" if score >= 0.5
            else "blocking"
        )
        return {
            "verdict": verdict,
            "score": round(score, 3),
            "confidence": round(0.6 + 0.4 * score, 3),
            "summary": ", ".join(notes) if notes else "clean dataset; no remediation suggested",
            "asks": asks,
        }
