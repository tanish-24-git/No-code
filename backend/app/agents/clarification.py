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

        # Pick one question at a time, in priority order.
        if "user_goal" in missing:
            qid, opts, ctx = "q_user_goal", None, "I couldn't tell from the data alone."
        elif "target_field" in missing:
            qid, opts, ctx = "q_target_field", cols, "I see these fields and need to know which is the target."
        elif "input_field" in missing:
            qid, opts, ctx = "q_input_fields", cols, "Which fields should the model see as input?"
        else:
            qid, opts, ctx = "q_task_type", None, "I'm not sure between a couple of task types."

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
