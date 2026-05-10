"""ExportAgent: enforces the explicit save-local / push-hf / both
finalization flow. Never auto-publishes; the user must answer
ExportChoiceRequested first."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.api.schemas.session import FSMState
from app.events.types import AgentEvent
from app.services import session_service


class ExportAgent(BaseAgent):
    name = "ExportAgent"
    role = "Save local, push to HF, or both — based on explicit user choice."
    allowed_tools = ("export.save_local", "export.push_hf", "audit.write")
    triggers = ("EvaluationCompleted", "SaveLocalRequested", "PushToHFRequested")

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        if event.kind == "EvaluationCompleted":
            if session.state == FSMState.EVALUATING:
                session_service.advance_state(session, FSMState.AWAITING_EXPORT_CHOICE, reason="awaiting export decision")
            # Deliberate on how to present the export choice.
            eval_data = session.artifacts.get("evaluation") or {}
            prompt = (
                f"Training job {session.job_id} completed.\n"
                f"Final score: {eval_data.get('score')}\n"
                f"Baseline delta: {eval_data.get('vs_baseline', {}).get('delta')}\n\n"
                "Explain that the model is ready and ask if they want to save it locally, "
                "push it to HuggingFace, or both. Be brief but professional."
            )
            
            msg = "Where should I put the trained model? Options: **local**, **HuggingFace**, or **both**."
            if session.llm_provider:
                try:
                    msg = await self.call_llm(
                        session_id,
                        prompt,
                        system="You are a deployment engineer. Present the export choice to the user.",
                        parent=event.id
                    )
                except Exception:
                    pass

            await self.emit_message(session_id, msg, parent=event.id)
            await self.emit(
                "ExportChoiceRequested",
                session_id,
                payload={
                    "options": ["local", "hf", "both"],
                    "job_id": session.job_id,
                },
                parent_event_id=event.id,
            )

            return

        # Both SaveLocalRequested and PushToHFRequested arrive after the user's choice.
        if session.state == FSMState.AWAITING_EXPORT_CHOICE:
            session_service.advance_state(session, FSMState.FINALIZING, reason="finalizing export")

        if event.kind == "SaveLocalRequested":
            await self._save_local(session, event)
        elif event.kind == "PushToHFRequested":
            await self._push_hf(session, event)

        # When everything we were asked to do is done, close the session.
        await self._maybe_complete(session, event)

    async def _save_local(self, session, event: AgentEvent) -> None:
        result = await self.call_tool(
            "export.save_local",
            {"job_id": session.job_id, "name": event.payload.get("name")},
            session.id,
        )
        if "error" in result:
            await self.emit_error(session.id, result["error"])
            return
        # Track in session artifacts.
        exported = session.artifacts.get("export") or {}
        exported["local"] = result
        session_service.attach_artifact(session, "export", exported)
        await self.emit_message(session.id, f"Saved locally at `{result.get('local_path')}`.", parent=event.id)

    async def _push_hf(self, session, event: AgentEvent) -> None:
        repo_id = event.payload.get("repo_id")
        if not repo_id:
            await self.emit_error(session.id, "missing repo_id for HF push")
            return
        # The export.save_local must run first to register the model id.
        local = (session.artifacts.get("export") or {}).get("local")
        if not local:
            saved = await self.call_tool("export.save_local", {"job_id": session.job_id}, session.id)
            if "error" in saved:
                await self.emit_error(session.id, saved["error"])
                return
            exported = session.artifacts.get("export") or {}
            exported["local"] = saved
            session_service.attach_artifact(session, "export", exported)
            local = saved

        result = await self.call_tool(
            "export.push_hf",
            {"model_id": local["model_id"], "repo_id": repo_id},
            session.id,
        )
        if "error" in result:
            await self.emit_error(session.id, result["error"])
            return
        exported = session.artifacts.get("export") or {}
        exported["hf"] = result
        session_service.attach_artifact(session, "export", exported)
        await self.emit_message(session.id, f"Pushing to Hugging Face: `{repo_id}`.", parent=event.id)

    async def _maybe_complete(self, session, event: AgentEvent) -> None:
        # Reload to get freshest state.
        session = self.get_session(session.id) or session
        pending = session.artifacts.get("export_pending") or []
        # Caller marks "both" by setting export_pending=["local","hf"]; once
        # both keys are populated we can finish.
        export = session.artifacts.get("export") or {}
        wanted = set(pending) if pending else _wanted_from_event(event)
        done = set(k for k in ("local", "hf") if k in export)
        if wanted and not wanted.issubset(done):
            return
        await self.emit("ExportCompleted", session.id, payload={"export": export}, parent_event_id=event.id)
        if session.state == FSMState.FINALIZING:
            session_service.advance_state(session, FSMState.DONE, reason="export complete")
        await self.emit("SessionClosed", session.id, payload={"reason": "done"}, parent_event_id=event.id)


def _wanted_from_event(event: AgentEvent) -> set[str]:
    if event.kind == "SaveLocalRequested":
        return {"local"}
    if event.kind == "PushToHFRequested":
        return {"hf"}
    return set()
