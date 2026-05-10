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
    triggers = ("DatasetProfileCompleted",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return

        await self.announce(

            session_id,
            phase="hardware",
            title="Probing hardware",
            summary="Detecting device, VRAM, CUDA / MPS support, and recommending bounds.",
            steps=["torch.cuda + torch.backends.mps probe", "VRAM total / free", "throughput estimate"],
            outputs=["hardware artifact"],
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, "hardware")
        
        await self.emit("HardwareProfileStarted", session_id)
        await self.think(session_id, "Probing local devices and estimating training capacity...", parent=event.id)
        
        info = await self.call_tool("hardware.detect", {}, session_id)
        
        # Agentic deliberation on hardware constraints.
        device = info.get("device", "cpu")
        vram = info.get("vram_gb")
        
        await self.think(session_id, f"Found {device.upper()} with {vram or 'no'} GB VRAM. Analyzing training bounds...")
        
        synthesis = f"Hardware: {device.upper()} ({vram or 0}GB VRAM)."
        if session.llm_provider:
            deliberation_prompt = (
                f"Hardware: {device.upper()}, VRAM: {vram}GB.\n"
                f"User instructions: '{comment or 'none'}'\n\n"
                "What are the realistic training bounds for this machine? "
                "Can we run 7B or 13B models? Should we enforce QLoRA? Provide a 1-sentence engineering verdict."
            )
            try:
                synthesis = await self.call_llm(session_id, deliberation_prompt, system="You are a hardware-aware MLOps engineer.", parent=event.id)
            except Exception:
                pass

        session = self.get_session(session_id)
        if session:
            session_service.attach_artifact(session, "hardware", info)

        msg = f"**Hardware Analysis Complete**\n\n{synthesis}"
        await self.emit_message(session_id, msg, parent=event.id)

        await self.complete(
            session_id,
            phase="hardware",
            summary=synthesis[:100],
            artifacts={"device": device, "vram_gb": vram},
            parent=event.id,
        )

        await self.emit(
            "HardwareProfileCompleted",
            session_id,
            payload={"hardware": info},
            parent_event_id=event.id,
        )
