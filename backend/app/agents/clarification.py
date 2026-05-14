"""ClarificationAgent: picks the next focused question from the catalog,
emits UserClarificationRequested, and on UserClarificationReceived parses
+ validates the reply, then re-fires TaskInferenceAgent via an event."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
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

        # The agent now generates free-form questions based on the context.
        # No more hardcoded catalog.
        from app.api.schemas.session import ClarificationQuestion
        
        prompt = (
            f"Missing information: {', '.join(missing)}\n"
            f"Dataset columns: {', '.join(cols)}\n\n"
            "What is the single most important question to ask the user to proceed with the pipeline? "
            "Output the question in ClarificationQuestion JSON format."
        )
        
        q = await self.call_llm_typed(
            session_id,
            prompt,
            ClarificationQuestion,
            system="You are a data analyst. Generate a focused question to clarify user intent. Be concise.",
            parent=event.id,
        )

        if not q:
            await self.emit_error(session_id, "AI failed to generate a clarifying question.")
            return

        # Persist as pending question + bump budget.
        session_service.set_pending_question(session, q)

        await self.emit(
            "UserClarificationRequested",
            session_id,
            payload=q.model_dump(mode="json"),
            parent_event_id=event.id,
        )
        await self.emit_message(session_id, q.question, parent=event.id)
