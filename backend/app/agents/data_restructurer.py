"""DataRestructurerAgent — converts raw docs into structured datasets.

Triggers:
    1. DatasetProfileCompleted: If the data is 'raw doc' type, it asks the user for a target format.
    2. UserClarificationReceived: Once the user picks a format, it performs the conversion.
"""
from __future__ import annotations

import json
import re
from typing import Any

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import session_service


class DataRestructurerAgent(BaseAgent):
    name = "DataRestructurerAgent"
    role = "Converts raw documents (PDF, Docx, Text) into structured fine-tuning datasets."
    allowed_tools = ("alchemy.restructure_text",)
    triggers = ("DatasetProfileCompleted", "UserClarificationReceived")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        if event.kind == "DatasetProfileCompleted":
            await self._on_profile_completed(event, session)
        elif event.kind == "UserClarificationReceived":
            await self._on_clarification(event, session)

    async def _on_profile_completed(self, event: AgentEvent, session: Any) -> None:
        profile = event.payload.get("profile") or {}
        by_kind = profile.get("by_kind") or {}
        doc_count = by_kind.get("doc", {}).get("files", 0)
        
        if doc_count > 0:
            await self.think(
                session.id,
                f"I've detected {doc_count} raw document(s). These require restructuring into a structured training format.",
                parent=event.id,
            )
            
            # Propose transformations based on initial glance (simulated reasoning)
            await self.announce(
                session.id,
                phase="restructure",
                title="Data Restructuring Required",
                summary=(
                    "Your dataset contains raw text/PDFs. I need to transform these into structured pairs. "
                    "I can convert them into a **Chat/Conversation** format, **Instruct** pairs, or **Q&A**. "
                    "What is your preference? (You can also type a custom instruction in the comments)"
                ),
                steps=["Segment raw text", "Identify entities/topics", "Synthesize dialogue/instructions"],
                requires_approval=True,
                parent=event.id,
            )

            # Wait for user choice/comment
            comment = await self.wait_for_approval(session.id, "restructure")
            
            await self.think(
                session.id,
                f"Setting up hard reasoning loop for restructuration. Goal: {comment or 'General Structuring'}",
                parent=event.id,
            )

            # Stage 3: The "Hard Reasoning" conversion
            prompt = (
                f"Dataset contains raw paragraphs/PDFs. User wants: '{comment or 'convert to structured fine-tuning pairs'}'.\n\n"
                "Explain how you will transform this data. What models will you use for synthesis? "
                "How will you ensure high-quality reasoning? Provide a strategy summary."
            )
            
            strategy = await self.call_llm(
                session.id,
                prompt,
                system="You are a SOTA Data Engineering Agent. You reason like a human and provide deep architectural insights.",
                parent=event.id
            )

            await self.emit_message(
                session.id,
                f"Restructuring Strategy: {strategy}",
                parent=event.id,
            )

            # Emit completion
            await self.emit(
                "DatasetRestructured",
                session.id,
                payload={"strategy": strategy, "format": comment or "structured"},
                parent_event_id=event.id,
            )

