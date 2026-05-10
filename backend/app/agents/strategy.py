"""TrainingStrategyAgent: picks method/precision/batch/seq_len/lr/epochs.

Always honors any user directive captured upstream (e.g. "use 5 epochs",
"prefer DoRA"). The LLM, when configured, proposes a strict-JSON
StrategyChoice; we validate it. The deterministic fallback is computed
from inputs (hardware, profile, model size, priority) - no hardcoded
"3-5 epochs for small datasets" rule that ignores user intent.
"""
from __future__ import annotations

from app.agents.base import BaseAgent
from app.agents.schemas import StrategyChoice
from app.events.types import AgentEvent
from app.services import decision_log, session_service
from app.services import directives as directives_service


class TrainingStrategyAgent(BaseAgent):
    name = "TrainingStrategyAgent"
    role = "Choose training strategy + estimate runtime."
    directive_scope = "strategy"
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

        await self.materialize_node(
            session_id,
            {"id": "strategy", "type": "strategy", "position": {"x": 640, "y": 80},
             "data": {"label": "strategy"}},
            parent=event.id,
        )
        await self.materialize_edge(
            session_id,
            {"id": "e-model-strategy", "source": "model", "target": "strategy", "animated": True},
            parent=event.id,
        )

        await self.think(
            session_id,
            f"Picking the training strategy for {chosen_model.get('repo_id')} "
            f"on {hw.get('device')}/{hw.get('vram_gb')}GB. Reading directives "
            "for any user preferences first.",
            parent=event.id,
        )

        # 1. Try the LLM with strict schema. Shared context already
        # injects user directives so "use 5 epochs" is honored.
        llm_proposal = await self.call_llm_typed(
            session_id,
            (
                f"Hardware: {hw}\nProfile: {profile}\nTask: {task}\n"
                f"Model: {chosen_model.get('repo_id')} ({chosen_model.get('params_b')}B)\n"
                f"Priority: {priority}\n\n"
                "Choose a fine-tuning strategy. Honor any user directives "
                "above. Use bf16 on CUDA when possible, fp16 on MPS, "
                "fp32 only on CPU. For datasets < 5k rows prefer 3-5 "
                "epochs, > 50k rows 1 epoch."
            ),
            StrategyChoice,
            system="You are a SOTA MLOps architect. Output only valid JSON.",
            stream_thoughts=False,
            parent=event.id,
        )

        if llm_proposal is not None:
            strategy = llm_proposal.model_dump()
        else:
            # Deterministic fallback - computed, not hardcoded.
            tool_out = await self.call_tool(
                "strategy.choose",
                {"model": chosen_model, "hardware": hw, "profile": profile,
                 "task": task, "priority": priority},
                session_id,
            )
            if "error" in tool_out:
                await self.emit_error(session_id, tool_out["error"])
                return
            strategy = tool_out

        # Apply directive overrides (LLM might have ignored them - we
        # belt-and-suspenders here).
        self._apply_directive_overrides(session_id, strategy)

        runtime = await self.call_tool(
            "strategy.estimate_runtime",
            {"strategy": strategy, "hardware": hw, "profile": profile, "model": chosen_model},
            session_id,
        )
        est_min = float(runtime.get("estimated_minutes", 0.0))

        stack_bits = [strategy.get("method", "lora")]
        if strategy.get("adapter_variant") and strategy["adapter_variant"] != "none":
            stack_bits.append(f"+{strategy['adapter_variant'].upper()}")
        if strategy.get("kernel_pack") and strategy["kernel_pack"] != "standard":
            stack_bits.append(f"+{strategy['kernel_pack']}")
        if strategy.get("quantization") and strategy["quantization"] != "none":
            stack_bits.append(f"+{strategy['quantization']}")
        stack = " ".join(stack_bits)

        await self.announce(
            session_id,
            phase="strategy",
            title="Training strategy",
            summary=(
                f"Proposed stack: **{stack}**\n\n"
                f"- Precision: {strategy.get('precision')}\n"
                f"- Batch: {strategy.get('batch_size')} x grad_accum {strategy.get('gradient_accumulation')}\n"
                f"- Seq len: {strategy.get('max_seq_len')}\n"
                f"- LR: {strategy.get('learning_rate')}\n"
                f"- Epochs: {strategy.get('epochs')}\n"
                f"- LoRA rank: {strategy.get('lora_rank')}\n\n"
                f"Estimated runtime: ~{est_min:.1f} min.\n\n"
                f"Comment to override anything (e.g. 'use 5 epochs', 'switch to DoRA')."
            ),
            steps=[
                "Honor user directives",
                "Match precision to device (bf16/fp16/fp32)",
                "Compute batch x grad_accum to fit VRAM",
                "Choose LoRA / QLoRA / DoRA",
            ],
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, "strategy")
        if comment:
            refined = await self.call_llm_typed(
                session_id,
                (
                    f"Current strategy JSON: {strategy}\n"
                    f"User feedback: {comment}\n\n"
                    "Adjust the strategy per the feedback. Keep all other "
                    "fields. Return ONLY the new StrategyChoice JSON."
                ),
                StrategyChoice,
                system="You are a flexible MLOps assistant.",
                stream_thoughts=False,
                parent=event.id,
            )
            if refined is not None:
                strategy = refined.model_dump()
                self._apply_directive_overrides(session_id, strategy)
                await self.think(session_id, "Strategy adjusted per your feedback.", parent=event.id)

        session_service.attach_artifact(session, "strategy", strategy)
        session_service.attach_artifact(session, "runtime_estimate", {"estimated_minutes": est_min})

        d = decision_log.record(
            session_id=session_id,
            agent=self.name,
            kind="strategy",
            inputs={"hardware": hw, "profile": profile, "model": chosen_model,
                    "priority": priority, "comment": comment},
            chosen=strategy,
            confidence=0.9,
            rationale=strategy.get("rationale") or f"priority={priority}",
        )

        await self.complete(
            session_id,
            phase="strategy",
            summary=f"{strategy.get('method')} / {strategy.get('precision')} / "
                    f"{strategy.get('epochs')}ep / ~{est_min:.0f}min",
            artifacts={"method": strategy.get("method")},
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

    def _apply_directive_overrides(self, session_id: str, strategy: dict) -> None:
        """Pull simple numeric / categorical overrides out of user directives."""
        import re
        text = " ".join(
            d.text.lower()
            for d in directives_service.read_for_scope(session_id, "strategy")
        )
        if not text:
            return
        m = re.search(r"(\d+)\s*epoch", text)
        if m:
            try:
                strategy["epochs"] = max(1, int(m.group(1)))
            except ValueError:
                pass
        if "dora" in text:
            strategy["adapter_variant"] = "dora"
        if "qlora" in text or "4bit" in text or "int4" in text:
            strategy["method"] = "qlora"
            strategy["quantization"] = "int4"
        if "unsloth" in text:
            strategy["kernel_pack"] = "unsloth"
        m = re.search(r"lr\s*=?\s*([\d\.eE\-\+]+)", text)
        if m:
            try:
                strategy["learning_rate"] = float(m.group(1))
            except ValueError:
                pass
        m = re.search(r"batch\s*=?\s*(\d+)", text)
        if m:
            try:
                strategy["batch_size"] = max(1, int(m.group(1)))
            except ValueError:
                pass
