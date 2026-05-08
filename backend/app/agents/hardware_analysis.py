"""HardwareAnalysisAgent: detects local hardware. Runs in parallel with
profiling so the user sees immediate feedback either way."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import session_service


class HardwareAnalysisAgent(BaseAgent):
    name = "HardwareAnalysisAgent"
    role = "Detect device/VRAM/CPU and recommend training method bounds"
    allowed_tools = ("hardware.detect", "hardware.estimate_throughput", "audit.write")
    triggers = ("DatasetUploaded",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        await self.emit("HardwareProfileStarted", session_id)
        await self.think(
            session_id,
            "Probing local hardware — torch availability, CUDA / MPS, VRAM, and "
            "approximate tokens-per-second so I can scope the model search to "
            "what your machine can actually run.",
            parent=event.id,
        )
        info = await self.call_tool("hardware.detect", {}, session_id)

        session = self.get_session(session_id)
        if session:
            session_service.attach_artifact(session, "hardware", info)

        device = info.get("device", "cpu")
        vram = info.get("vram_gb")
        msg = f"Detected **{device.upper()}**"
        if vram:
            msg += f" with {vram} GB VRAM"
        if info.get("gpu_name"):
            msg += f" ({info['gpu_name']})"
        await self.emit_message(session_id, msg + ".", parent=event.id)

        await self.emit(
            "HardwareProfileCompleted",
            session_id,
            payload={"hardware": info},
            parent_event_id=event.id,
        )
