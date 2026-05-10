"""TrainingStrategyAgent: picks method/precision/batch/seq_len/lr/epochs
given a chosen model and the rest of the context."""
from __future__ import annotations

import json
import re

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

        await self.materialize_node(
            session_id,
            {"id": "strategy", "type": "config", "position": {"x": 440, "y": 80}, "data": {"label": "strategy"}},
            parent=event.id,
        )
        await self.materialize_edge(
            session_id,
            {"id": "e-model-strategy", "source": "model", "target": "strategy", "animated": True},
            parent=event.id,
        )

        # Stage 1: Socratic Deliberation.
        await self.think(session_id, "Deliberating on the optimal training architecture for your dataset and hardware...", parent=event.id)
        
        # We start by asking the LLM to propose a strategy from scratch, 
        # instead of relying on a deterministic baseline.
        deliberation_prompt = (
            f"Dataset Profile: {profile.get('row_count')} rows.\n"
            f"Hardware: {hw.get('device')} / {hw.get('vram_gb')}GB VRAM.\n"
            f"Model: {chosen_model.get('repo_id')} ({chosen_model.get('params_b')}B params)\n"
            f"Task: {task.get('chosen')}\n\n"
            "Propose a SOTA training strategy. CRITICAL: NEVER use float32 precision; use float16 or bf16 (if hardware supports it). "
            "For small datasets, use 3-5 epochs. For LoRA, suggest a rank like 16 or 32. "
            "Return a JSON object with: method, adapter_variant, precision, batch_size, gradient_accumulation, learning_rate, epochs, and a short 'rationale'."
        )
        
        strategy = {}
        if session.llm_provider:
            try:
                res = await self.call_llm(session_id, deliberation_prompt, system="You are a SOTA MLOps Architect.", parent=event.id)
                m = re.search(r"\{.*\}", res, re.DOTALL)
                if m:
                    strategy = json.loads(m.group(0))
            except Exception:
                pass
        
        # Fallback to tool if LLM fails or returned incomplete JSON
        if not strategy.get("method"):
            strategy = await self.call_tool(
                "strategy.choose",
                {"model": chosen_model, "hardware": hw, "profile": profile, "task": task, "priority": priority},
                session_id,
            )

        # Present the strategy for approval.
        stack_bits = [strategy.get("method", "PEFT")]
        if strategy.get("adapter_variant") and strategy["adapter_variant"] != "none":
            stack_bits.append(f"+{strategy['adapter_variant'].upper()}")
        stack = " ".join(stack_bits)

        await self.announce(
            session_id,
            phase="strategy",
            title="Training Strategy",
            summary=(
                f"I've deliberated on your setup and propose a **{stack}** stack.\n\n"
                f"**Rationale:** {strategy.get('rationale', 'Optimized for your VRAM and data size.')}\n\n"
                f"- Precision: {strategy.get('precision', 'auto')}\n"
                f"- Batch size: {strategy.get('batch_size')} (Accumulation: {strategy.get('gradient_accumulation')})\n"
                f"- Learning rate: {strategy.get('learning_rate', 'auto')}\n\n"
                "Does this architectural plan work for you? You can approve or comment to request changes (e.g., 'increase LR', 'use GaLore instead')."
            ),
            steps=["Analyze VRAM-to-parameter ratio", "Select optimal PEFT variant", "Compute gradient accumulation bounds"],
            requires_approval=True,
            parent=event.id,
        )

        comment = await self.wait_for_approval(session_id, "strategy")
        if comment and session.llm_provider:
            await self.think(session_id, f"Re-deliberating strategy based on your feedback: '{comment}'")
            prompt = (
                f"Current strategy: {json.dumps(strategy)}\n"
                f"User feedback: '{comment}'\n\n"
                "Adjust the strategy JSON according to the feedback. Return ONLY the new JSON object."
            )
            try:
                adjusted_text = await self.call_llm(session_id, prompt, system="You are a flexible MLOps assistant.", parent=event.id)
                m = re.search(r"\{.*\}", adjusted_text, re.DOTALL)
                if m:
                    strategy = json.loads(m.group(0))
                    await self.think(session_id, f"Strategy adjusted. I've updated the {', '.join(strategy.keys())} as requested.")
            except Exception:
                pass

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
            inputs={"hardware": hw, "profile": profile, "model": chosen_model, "comment": comment},
            chosen=strategy,
            confidence=0.9,
            rationale=f"AI-refined strategy based on hardware/priority",
        )

        await self.complete(
            session_id,
            phase="strategy",
            summary=f"Finalized {strategy.get('method')} strategy",
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

