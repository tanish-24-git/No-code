"""DatasetProfilingAgent: token distribution, duplicates, missing values,
class imbalance. Produces a numeric profile that downstream agents read."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import session_service


class DatasetProfilingAgent(BaseAgent):
    name = "DatasetProfilingAgent"
    role = "Numeric profile of dataset (tokens, duplicates, missing, imbalance)"
    allowed_tools = (
        "dataset.profile_tokens",
        "dataset.detect_duplicates",
        "dataset.detect_missing",
        "dataset.detect_imbalance",
        "audit.write",
    )
    triggers = ("IntakeCompleted",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        dataset_id = event.payload.get("dataset_id")
        if not dataset_id:
            return

        await self.emit("DatasetProfileStarted", session_id, payload={"dataset_id": dataset_id})
        await self.think(
            session_id,
            "Computing token-length distribution, exact-row hash dedup, "
            "per-column missing rates, and class balance.",
            parent=event.id,
        )
        await self.emit_message(session_id, "Profiling token lengths, duplicates, missing values, and class balance…", parent=event.id)

        tokens = await self.call_tool("dataset.profile_tokens", {"dataset_id": dataset_id}, session_id)
        dupes = await self.call_tool("dataset.detect_duplicates", {"dataset_id": dataset_id}, session_id)
        missing = await self.call_tool("dataset.detect_missing", {"dataset_id": dataset_id}, session_id)
        imbalance = await self.call_tool("dataset.detect_imbalance", {"dataset_id": dataset_id}, session_id)

        profile = {
            "tokens": tokens,
            "duplicates": dupes,
            "missing": missing,
            "imbalance": imbalance,
            "row_count": dupes.get("row_count") or missing.get("row_count") or 0,
            "p95": tokens.get("p95"),
        }
        session = self.get_session(session_id)
        if session:
            session_service.attach_artifact(session, "profile", profile)

        # Concise chat update.
        bits = []
        if "p95" in tokens:
            bits.append(f"tokens p95 ≈ {tokens['p95']}")
        if "duplicate_pct" in dupes:
            bits.append(f"{dupes['duplicate_pct']}% duplicates")
        if "minority_pct" in imbalance:
            bits.append(f"smallest class {imbalance['minority_pct']}%")
        if bits:
            await self.emit_message(session_id, "Profile: " + ", ".join(bits) + ".", parent=event.id)

        await self.emit(
            "DatasetProfileCompleted",
            session_id,
            payload={"dataset_id": dataset_id, "profile": profile},
            parent_event_id=event.id,
        )
