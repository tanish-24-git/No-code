"""OrchestratorAgent: opens the session, posts the first user-visible
message, and (importantly) is the only agent that runs synchronously inside
the upload request handler — everything else is event-driven from the bus."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent


class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Greeter and meta-router. Posts the welcome message on a new session."
    allowed_tools = ()
    triggers = ("SessionStarted",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        await self.emit_message(
            session_id,
            "I've started analyzing your dataset. I'll check schema, quality, "
            "likely task type, and let you know if I need any clarification.",
            parent=event.id,
        )
