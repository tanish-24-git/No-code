"""InferenceSandboxAgent — clean-room benchmark suite (blueprint §2.1).

Once training finishes, this agent loads the freshly trained adapter in an
isolated process boundary (subprocess if available; in-process fallback)
and runs a benchmark suite against the user's stated goals plus a tiny
fixed regression set (MMLU-style multiple-choice samples, GSM8K-style word
problems, a HumanEval-style code completion). Results are streamed back
as ``SandboxBenchmarkCompleted`` and folded into the EvaluationCompleted
payload by the EvaluationAgent.

The sandbox never modifies the trained artefact; the training output is
opened read-only.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent


class InferenceSandboxAgent(BaseAgent):
    name = "InferenceSandboxAgent"
    role = "Isolated post-training evaluation in a clean-room environment."
    allowed_tools = ("sandbox.benchmark", "audit.write")
    triggers = ("EvaluationCompleted",)

    async def handle(self, event: AgentEvent) -> None:
        session = self.get_session(event.session_id)
        if not session or not session.job_id:
            return

        await self.think(
            session.id,
            "Spinning up an isolated inference sandbox so benchmarks see the "
            "exact adapter the user just trained, with no leaked context.",
            parent=event.id,
        )
        await self.emit("SandboxBenchmarkStarted", session.id, payload={"job_id": session.job_id})

        result = await self.call_tool(
            "sandbox.benchmark",
            {"job_id": session.job_id, "suite": "auto"},
            session.id,
        )
        if "error" in result:
            await self.emit_error(session.id, result["error"])
            return

        bench = result.get("benchmarks") or []
        await self.emit_message(
            session.id,
            "Sandbox benchmarks: " + ", ".join(
                f"{b['name']}={b.get('score', 0):.2f}" for b in bench
            ) + ".",
            parent=event.id,
        )
        await self.emit(
            "SandboxBenchmarkCompleted",
            session.id,
            payload={"benchmarks": bench, "summary": result.get("summary")},
            parent_event_id=event.id,
        )
