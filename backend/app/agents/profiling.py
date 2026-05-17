"""DatasetProfilingAgent: token distribution, duplicates, missing values,
class imbalance. Produces a numeric profile that downstream agents read."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import session_service


class DatasetProfilingAgent(BaseAgent):
    name = "DatasetProfilingAgent"
    role = "Numeric profile of dataset (tokens, duplicates, missing, imbalance)"
    directive_scope = "data"
    allowed_tools = (
        "dataset.profile_tokens",
        "dataset.detect_duplicates",
        "dataset.detect_missing",
        "dataset.detect_imbalance",
        "audit.write",
    )
    triggers = ("IntakeCompleted",)
    tier_preference = "fast"  # narration of numeric profile output

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        dataset_id = event.payload.get("dataset_id") or session.dataset_id
        if not dataset_id:
            return

        await self.announce(
            session_id,
            phase="profile",
            title="Profiling the dataset",
            summary="Computing token-length distribution, duplicates, missing values, and class balance.",
            steps=[
                "Whitespace token-length p50/p95/max",
                "Exact-row hash dedup",
                "Per-column missing-rate",
                "Low-cardinality column class balance",
            ],
            outputs=["profile artifact (tokens, duplicates, missing, imbalance)"],
            requires_approval=False,
            parent=event.id,
        )

        await self.materialize_node(
            session_id,
            {"id": "preprocess", "type": "preprocess", "position": {"x": 240, "y": 80},
             "data": {"label": "profile + clean"}},
            parent=event.id,
        )
        await self.materialize_edge(
            session_id,
            {"id": "e-dataset-preprocess", "source": "dataset", "target": "preprocess", "animated": True},
            parent=event.id,
        )

        await self.emit("DatasetProfileStarted", session_id, payload={"dataset_id": dataset_id})

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
        if session:
            session_service.attach_artifact(session, "profile", profile)

        # Headline summary - deterministic, computed from inputs.
        bits: list[str] = []
        if "p95" in tokens:
            bits.append(f"tokens p95 ~ {tokens['p95']}")
        if "duplicate_pct" in dupes:
            bits.append(f"{dupes['duplicate_pct']}% duplicates")
        if "minority_pct" in imbalance:
            bits.append(f"smallest class {imbalance['minority_pct']}%")
        synthesis = "Profile: " + ", ".join(bits) if bits else "Profile complete."

        # If LLM is configured, ask it to interpret the metrics in context.
        llm_msg = await self.call_llm(
            session_id,
            (
                f"Profile metrics:\n"
                f"- tokens: {tokens}\n"
                f"- duplicates: {dupes}\n"
                f"- missing: {missing}\n"
                f"- imbalance: {imbalance}\n\n"
                "In one tight sentence (no preamble), what does this say "
                "about training readiness? Mention any quality risk."
            ),
            system="You are a meticulous MLOps profiler. Be concise.",
            stream_thoughts=False,
            parent=event.id,
        )
        if llm_msg.strip():
            synthesis = llm_msg.strip()

        await self.emit_message(session_id, synthesis, parent=event.id)

        await self.complete(
            session_id,
            phase="profile",
            summary=synthesis[:160],
            artifacts={"row_count": profile["row_count"], "p95": profile.get("p95")},
            parent=event.id,
        )

        await self.emit(
            "DatasetProfileCompleted",
            session_id,
            payload={"dataset_id": dataset_id, "profile": profile},
            parent_event_id=event.id,
        )
