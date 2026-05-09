"""TrainingStrategyAgent: picks method/precision/batch/seq_len/lr/epochs
given a chosen model and the rest of the context."""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.events.types import AgentEvent
from app.services import decision_log, session_service


class TrainingStrategyAgent(BaseAgent):
    name = "TrainingStrategyAgent"
    role = "Choose training strategy + estimate runtime."
    allowed_tools = ("strategy.choose", "strategy.estimate_runtime", "audit.write")
    triggers = ("CandidateModelsRanked",)

    async def handle(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        if not session:
            return
        hw = session.artifacts.get("hardware") or {}
        profile = session.artifacts.get("profile") or {}
        task = session.artifacts.get("task_inference") or {}
        chosen_model = session.artifacts.get("chosen_model") or {}
        priority = self._user_priority(session) or "quality"

        await self.announce(
            session_id,
            phase="strategy",
            title="Picking a training strategy",
            summary=(
                f"Choosing PEFT method, precision, batch size, and seq length "
                f"for {chosen_model.get('repo_id', 'the chosen model')} on "
                f"{hw.get('device', 'cpu').upper()}."
            ),
            steps=[
                "Pick LoRA / QLoRA / DoRA based on VRAM headroom",
                "Set bf16 / fp16 / fp32 by device support",
                "Compute batch * grad_accum to fit VRAM",
                "Estimate runtime",
            ],
            outputs=["strategy artifact", "runtime_estimate"],
            parent=event.id,
        )

        strategy = await self.call_tool(
            "strategy.choose",
            {"model": chosen_model, "hardware": hw, "profile": profile, "task": task, "priority": priority},
            session_id,
        )
        if "error" in strategy:
            await self.emit_error(session_id, strategy["error"])
            return

        runtime = await self.call_tool(
            "strategy.estimate_runtime",
            {"strategy": strategy, "hardware": hw, "profile": profile, "model": chosen_model},
            session_id,
        )
        est_min = float(runtime.get("estimated_minutes", 0.0))

        session_service.attach_artifact(session, "strategy", strategy)
        session_service.attach_artifact(session, "runtime_estimate", {"estimated_minutes": est_min})

        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="strategy",
            inputs={"hardware": hw, "profile": profile, "model": chosen_model, "priority": priority},
            chosen=strategy,
            confidence=0.7,
            rationale=f"priority={priority}",
        )

        # Plan card with the SOTA stack so the user sees DoRA/GaLore/Unsloth.
        stack_bits = [strategy["method"]]
        if strategy.get("adapter_variant") and strategy["adapter_variant"] != "none":
            stack_bits.append(f"+{strategy['adapter_variant'].upper()}")
        if strategy.get("kernel_pack") and strategy["kernel_pack"] != "standard":
            stack_bits.append(f"-{strategy['kernel_pack']}")
        if strategy.get("quantization") and strategy["quantization"] != "none":
            stack_bits.append(f"-{strategy['quantization']}")
        stack = " ".join(stack_bits)

        await self.emit_message(
            session_id,
            f"Strategy: **{stack}** / {strategy['precision']} / "
            f"batch {strategy['batch_size']}x{strategy['gradient_accumulation']} grad accum / "
            f"seq {strategy['max_seq_len']} / {strategy['epochs']}ep. "
            f"~{est_min:.1f} min estimated.",
            parent=event.id,
        )

        await self.complete(
            session_id,
            phase="strategy",
            summary=f"{strategy['method']} / {strategy['precision']} / "
                    f"~{est_min:.1f} min estimated",
            artifacts={"method": strategy["method"], "precision": strategy["precision"]},
            parent=event.id,
        )
        await self.emit(
            "StrategyChosen",
            session_id,
            payload={"strategy": strategy, "estimated_minutes": est_min},
            parent_event_id=event.id,
            decision_id=d.id,
        )

    def _user_priority(self, session) -> str | None:
        for a in reversed(session.clarifications):
            if a.question_id == "q_priority":
                return str(a.value)
        return None
