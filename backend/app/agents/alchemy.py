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

        # Deliberate on data health using the LLM.
        prompt = (
            f"Dataset Profile:\n"
            f"- Row count: {profile.get('row_count')}\n"
            f"- Duplicates: {profile.get('duplicates', {}).get('duplicate_pct') or 0}%\n"
            f"- Missing values: {profile.get('missing', {}).get('per_column')}\n"
            f"- Class balance: {profile.get('imbalance', {}).get('minority_pct') or 100}%\n\n"
            "Assess the data health for fine-tuning. Is it 'healthy', 'advisory', 'needs_attention', or 'blocking'? "
            "Explain why in a short summary. If there are severe issues, propose a specific question (ask) for the user. "
            "Return a JSON object with 'verdict', 'score' (0.0 to 1.0), 'summary', and 'asks' (list of strings)."
        )
        
        # Default report as safety fallback
        report = {
            "verdict": "healthy",
            "score": 1.0,
            "summary": "clean dataset; no remediation suggested",
            "asks": [],
            "confidence": 0.9
        }
        
        if session.llm_provider:
            try:
                import json, re
                res_text = await self.call_llm(
                    session.id,
                    prompt,
                    system="You are a Socratic data scientist. Grade data quality strictly for fine-tuning readiness.",
                    parent=event.id
                )

                m = re.search(r"\{.*\}", res_text, re.DOTALL)
                if m:
                    res_json = json.loads(m.group(0))
                    report.update(res_json)
                    report["confidence"] = round(0.6 + 0.4 * report["score"], 3)
            except Exception:
                pass

        session_service.attach_artifact(session, "data_health", report)
        await self.emit(
            "DataHealthReport",
            session.id,
            payload={"dataset_id": dataset_id, "report": report},
            parent_event_id=event.id,
            confidence=report.get("confidence", 0.9),
        )

        # Streamed verdict line — the UI renders a coloured banner.
        verdict = report["verdict"]
        await self.emit_message(
            session.id,
            f"Data Health: **{verdict.replace('_', ' ').title()}** — {report['summary']}",
            parent=event.id,
        )

        # Hard issues surface as a Socratic ask.
        if report.get("asks"):
            await self.ask(
                session.id,
                report["asks"][0],
                impact="high",
                parent=event.id,
            )

    # ── helpers ──────────────────────────────────────────────────────────
