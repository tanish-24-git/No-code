"""DatasetProfilingAgent: token distribution, duplicates, missing values,
class imbalance. Produces a numeric profile that downstream agents read."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
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
    triggers = ("IntakeCompleted", "PhaseApproved")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        dataset_id = event.payload.get("dataset_id") or session.dataset_id
        if not dataset_id:
            return

        # Guard: ignore PhaseApproved events that aren't for us.
        if event.kind == "PhaseApproved" and event.payload.get("phase") != "profile":
            return

        resuming = event.kind == "PhaseApproved" and event.payload.get("phase") == "profile"

        if not resuming:
            # Socratic deliberation before starting the scan.
            await self.think(session_id, "Deliberating on the profiling strategy for your data...", parent=event.id)
            
            facts = (session.artifacts.get("dataset_facts") or {}).get("info", {})
            deliberation_prompt = (
                f"Dataset: {facts.get('name')}\n"
                f"Metadata: {facts.get('row_count')} rows, columns: {facts.get('column_names')}\n\n"
                "What specific data quality risks should I look for in this dataset? "
                "Think about token length, balance, and potential noise. Provide a 1-sentence engineering focus."
            )
            focus = "Checking for token distribution and row-level duplicates."
            if session.llm_provider:
                try:
                    focus = await self.call_llm(session_id, deliberation_prompt, system="You are a meticulous Data Scientist.", parent=event.id)
                except Exception:
                    pass

            await self.announce(
                session_id,
                phase="profile",
                title="Profiling the dataset",
                summary=f"Focus: {focus}",
                steps=[
                    "Whitespace token-length p50/p95/max",
                    "Exact-row hash dedup",
                    "Per-column missing-rate",
                    "Low-cardinality column class balance",
                ],
                outputs=["profile artifact (tokens, duplicates, missing, imbalance)"],
                requires_approval=True,
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

            comment = await self.wait_for_approval(session_id, "profile")
            if comment:
                await self.think(session_id, f"Adjusting focus based on your feedback: '{comment}'")
                # Dynamic adjustment of tools based on comment (simulated)
                if "pii" in comment.lower() or "sensitive" in comment.lower():
                    await self.think(session_id, "I've added a sensitive-info scan to the profiling queue.")

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
        session = self.get_session(session_id)
        if session:
            session_service.attach_artifact(session, "profile", profile)

        # Agentic synthesis of results.
        synthesis = f"Profile: tokens p95 ≈ {tokens.get('p95', 0)}, {dupes.get('duplicate_pct', 0)}% duplicates."
        if session.llm_provider:
            await self.think(session_id, "Analyzing profile metrics for training implications...")
            synth_prompt = (
                f"Metrics:\n- Tokens: {tokens}\n- Duplicates: {dupes}\n- Missing: {missing}\n- Imbalance: {imbalance}\n\n"
                "Synthesize these into a human-readable summary (1-2 sentences) about the dataset's readiness for training."
            )
            try:
                synthesis = await self.call_llm(session_id, synth_prompt, system="You are an expert MLOps engineer.", parent=event.id)
            except Exception:
                pass

        await self.emit_message(session_id, synthesis, parent=event.id)

        await self.complete(
            session_id,
            phase="profile",
            summary=synthesis[:100],
            artifacts={"row_count": profile["row_count"], "p95": profile.get("p95")},
            parent=event.id,
        )

        await self.emit(
            "DatasetProfileCompleted",
            session_id,
            payload={"dataset_id": dataset_id, "profile": profile},
            parent_event_id=event.id,
        )
