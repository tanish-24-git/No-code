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

        await self.think_delta(session_id, "Formulating optimal training strategy (evaluating precision, LoRA/DoRA, batch sizes)...", is_final=False, parent=event.id)
        
        prompt = f"Hardware: {hw}, Profile: {profile}, Task: {task}, Model: {chosen_model}, Priority: {priority}. Decide the fine-tuning strategy. Return JSON with 'method' (lora/dora/full), 'precision' (float16/float32/bfloat16), 'batch_size' (int), 'gradient_accumulation' (int), 'max_seq_len' (int), 'epochs' (int), 'adapter_variant' (none/dora), 'kernel_pack' (standard/unsloth), 'quantization' (none/4bit/8bit), 'rationale' (list of strings)."
        system = "You are an AI ML optimizer. Return valid JSON only."
        
        result_str = await self.call_llm(session_id, prompt, system=system, stream_thoughts=True, parent=event.id)
        
        import json
        try:
            if "```json" in result_str:
                result_str = result_str.split("```json")[1].split("```")[0]
            elif "```" in result_str:
                result_str = result_str.split("```")[1].split("```")[0]
            strategy = json.loads(result_str.strip())
        except Exception:
            strategy = {
                "method": "lora",
                "precision": "float16",
                "batch_size": 1,
                "gradient_accumulation": 8,
                "max_seq_len": 256,
                "epochs": 1,
                "adapter_variant": "none",
                "kernel_pack": "standard",
                "quantization": "none",
                "rationale": ["Default strategy due to JSON parsing error"]
            }
        
        est_min = strategy.get("epochs", 1) * strategy.get("max_seq_len", 256) / 100.0  # mock estimate

        await self.think_delta(session_id, f"\nStrategy chosen: {strategy.get('method')} with batch size {strategy.get('batch_size')}x{strategy.get('gradient_accumulation')}.", is_final=True, parent=event.id)
        
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

        # Streaming "thinking" rationale — blueprint commandment §7.
        for line in strategy.get("rationale") or []:
            await self.think(session_id, line, parent=event.id)

        # Plan card with the SOTA stack so the user sees DoRA/GaLore/Unsloth.
        stack_bits = [strategy["method"]]
        if strategy.get("adapter_variant") and strategy["adapter_variant"] != "none":
            stack_bits.append(f"+{strategy['adapter_variant'].upper()}")
        if strategy.get("kernel_pack") and strategy["kernel_pack"] != "standard":
            stack_bits.append(f"·{strategy['kernel_pack']}")
        if strategy.get("quantization") and strategy["quantization"] != "none":
            stack_bits.append(f"·{strategy['quantization']}")
        stack = " ".join(stack_bits)

        await self.emit_message(
            session_id,
            f"Strategy: **{stack}** / {strategy['precision']} / "
            f"batch {strategy['batch_size']}×{strategy['gradient_accumulation']} grad accum / "
            f"seq {strategy['max_seq_len']} / {strategy['epochs']}ep. "
            f"~{est_min:.1f} min estimated.",
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
