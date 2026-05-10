"""ClarificationAgent: picks the next focused question from the catalog,
emits UserClarificationRequested, and on UserClarificationReceived parses
+ validates the reply, then re-fires TaskInferenceAgent via an event."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.orchestration.catalog import build_question
from app.services import session_service


class ClarificationAgent(BaseAgent):
    name = "ClarificationAgent"
    role = "Ask the smallest useful clarifying question; never invent free-form questions."
    allowed_tools = ("clarify.ask", "clarify.parse_reply", "audit.write")
    triggers = ("IntentConfidenceLow",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return
        if session.question_budget_used >= session.question_budget_max:
            return

        missing = list(event.payload.get("missing") or [])
        info = (session.artifacts.get("dataset_facts") or {}).get("info", {})
        cols = info.get("column_names", [])

        # Deliberate on which question to ask next using the LLM.
        from app.orchestration.catalog import CATALOG
        catalog_desc = "\n".join([f"- {k}: {v.question} ({v.why})" for k, v in CATALOG.items()])
        
        prompt = (
            f"Missing information: {', '.join(missing)}\n"
            f"Dataset columns: {', '.join(cols)}\n"
            f"Question Catalog:\n{catalog_desc}\n\n"
            "Which question should I ask next to proceed with the pipeline? "
            "Return ONLY the question_id."
        )
        
        qid = "q_user_goal"
        if session.llm_provider:
            try:
                qid_raw = await self.call_llm(
                    session_id,
                    prompt,
                    system="You are a data analyst. Pick the best question_id from the catalog to clarify user intent. Return ONLY the ID.",
                    parent=event.id
                )
                qid = qid_raw.strip().strip("'").strip('"').split("\n")[0]
                if qid not in CATALOG:
                    qid = "q_user_goal"
            except Exception:
                pass

        # Context for the question
        ctx = f"I need to clarify the {qid.replace('q_', '')} to configure the pipeline correctly."
        opts = cols if qid in ("q_target_field", "q_input_fields") else None

        q_payload = await self.call_tool(
            "clarify.ask",
            {"question_id": qid, "options": opts, "context": ctx},
            session_id,
        )

        if "error" in q_payload:
            await self.emit_error(session_id, q_payload["error"])
            return

        # Persist as pending question + bump budget.
        from app.api.schemas.session import ClarificationQuestion
        q = ClarificationQuestion(**q_payload)
        session_service.set_pending_question(session, q)

        await self.emit(
            "UserClarificationRequested",
            session_id,
            payload=q_payload,
            parent_event_id=event.id,
        )
        await self.emit_message(session_id, q.question, parent=event.id)
