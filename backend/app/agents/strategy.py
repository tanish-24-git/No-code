"""TrainingStrategyAgent: picks method/precision/batch/seq_len/lr/epochs.

v4.0 — Efficiency-First kernel. Prioritizes **Unsloth** for 2x-5x faster
training and **GaLore** for memory-efficient 7B+ fine-tuning on consumer
hardware (8GB-16GB VRAM). Supports Liger-Kernel Triton-optimized losses.

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
from app.utils.hardware import clamp_strategy_to_vram


class TrainingStrategyAgent(BaseAgent):
    name = "TrainingStrategyAgent"
    role = "Choose training strategy + estimate runtime."
    directive_scope = "strategy"
    allowed_tools = (
        "strategy.choose",
        "strategy.estimate_runtime",
        "hardware.vram_budget",  # programmatic OOM guard (#2)
        "web_search",            # look up recent fine-tuning techniques (#8)
        "model.search_hf",       # verify chosen base model still exists (#8)
        "audit.write",
    )
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

        # The agent now relies entirely on its reasoning for strategy selection.
        # Hardcoded fallbacks and heuristic tools have been decommissioned.
        llm_proposal = await self.call_llm_typed(
            session_id,
            (
                f"Hardware: {hw}\nProfile: {profile}\nTask: {task}\n"
                f"Model: {chosen_model.get('repo_id')} ({chosen_model.get('params_b')}B)\n"
                f"Priority: {priority}\n\n"
                "Design the optimal fine-tuning strategy. You MUST follow these rules. "
                "All enum field values in your JSON MUST be the exact lowercase strings "
                "shown in the schema (e.g. 'qlora', 'galore', 'bf16' — not 'QLoRA', "
                "'GaLore', 'BF16'). Pydantic Literal validation is case-sensitive and "
                "will reject capitalised variants.\n\n"
                "1. **Unsloth first**: if the model architecture is supported by Unsloth "
                "(llama, mistral, gemma, phi, qwen families), set use_unsloth=true and "
                "kernel_pack='unsloth'. 2x-5x speedup at zero cost.\n\n"
                "2. **GaLore for VRAM constraints**: if the model has 7B+ parameters and "
                "available VRAM is <=16GB, set optimizer='galore' or 'galore_q'. GaLore "
                "projects gradients to a low-rank subspace, cutting optimizer memory by 65%%. "
                "Enables 7B fine-tuning on 12GB VRAM without full 4-bit quantization.\n\n"
                "3. **Liger kernel**: if CUDA is available, set use_liger=true for "
                "Triton-optimized cross-entropy and RMS-norm — ~20%% throughput gain.\n\n"
                "4. **Combine techniques**: for extreme VRAM constraints (<=12GB + 7B model), "
                "use method='qlora' + optimizer='galore' + use_unsloth=true together.\n\n"
                "5. Base your decision on the model architecture, VRAM constraints, and "
                "dataset profile. Output ONLY the StrategyChoice JSON."
            ),
            StrategyChoice,
            system=(
                "You are an autonomous MLOps architect specializing in efficient fine-tuning. "
                "You prioritize Unsloth and GaLore for maximum training throughput on "
                "consumer hardware. Design the best job for this model specimen."
            ),
            stream_thoughts=False,
            parent=event.id,
        )

        if llm_proposal:
            strategy = llm_proposal.model_dump()
        else:
            # Safety net: the LLM call failed (no provider, repeated bad
            # JSON, transient error). Rather than bail the pipeline into a
            # deadlock — the previous behaviour, which left the frontend
            # spinning forever — we synthesize a known-safe default from
            # the user's hardware + chosen model. The user sees a clear
            # message and can comment to override anything they don't
            # like, which is the normal flow anyway.
            strategy = self._safe_default_strategy(chosen_model, hw, priority)
            await self.emit_message(
                session_id,
                "The LLM strategy planner did not return a valid response, "
                "so I picked a conservative default based on your hardware. "
                "Review the values below and comment to change anything "
                "(e.g. 'use 5 epochs', 'switch to qlora').",
                parent=event.id,
            )

        # Apply directive overrides (LLM might have ignored them - we
        # belt-and-suspenders here).
        self._apply_directive_overrides(session_id, strategy)

        # Programmatic VRAM clamp. The LLM-proposed batch/seq_len are
        # estimated against the user's actual GPU; if they don't fit
        # within 85%% of available VRAM we halve them deterministically
        # until they do. Prevents CUDA OOM mid-run on consumer GPUs.
        try:
            model_repo = chosen_model.get("repo_id") or ""
            clamp_strategy_to_vram(strategy, model_repo)
            clamp_info = strategy.pop("_vram_clamp", None)
            if clamp_info:
                frm = clamp_info["from"]
                to = clamp_info["to"]
                budget = clamp_info.get("budget", {})
                await self.emit_message(
                    session_id,
                    f"VRAM clamp: requested batch={frm['batch_size']} "
                    f"seq_len={frm['max_seq_len']} needed "
                    f"~{budget.get('required_gb', '?')}GB; GPU has "
                    f"{budget.get('available_gb', '?')}GB. Clamped to "
                    f"batch={to['batch_size']} seq_len={to['max_seq_len']}.",
                    parent=event.id,
                )
                strategy["rationale"] = (strategy.get("rationale") or "") + (
                    f" [VRAM clamp applied: bs {frm['batch_size']}->"
                    f"{to['batch_size']}, seq {frm['max_seq_len']}->{to['max_seq_len']}]"
                )
        except Exception:
            # Clamp is best-effort: a model-card lookup failure should not
            # block the run. trainer.py still has a CPU Safe-Mode guard
            # for the worst case.
            pass

        runtime = await self.call_tool(
            "strategy.estimate_runtime",
            {"strategy": strategy, "hardware": hw, "profile": profile, "model": chosen_model},
            session_id,
        )
        est_min = float(runtime.get("estimated_minutes", 0.0))

        stack_bits = [strategy.get("method", "lora")]
        if strategy.get("use_unsloth"):
            stack_bits.append("+Unsloth")
        if strategy.get("adapter_variant") and strategy["adapter_variant"] != "none":
            stack_bits.append(f"+{strategy['adapter_variant'].upper()}")
        if strategy.get("optimizer") and strategy["optimizer"] not in ("adamw",):
            stack_bits.append(f"+{strategy['optimizer'].upper()}")
        if strategy.get("use_liger"):
            stack_bits.append("+Liger")
        if strategy.get("kernel_pack") and strategy["kernel_pack"] != "standard":
            if "Unsloth" not in str(stack_bits):  # avoid duplicate
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

    def _safe_default_strategy(
        self,
        chosen_model: dict,
        hw: dict,
        priority: str,
    ) -> dict:
        """Synthesize a conservative-but-functional StrategyChoice when the
        LLM planner fails. Used to prevent the pipeline from deadlocking
        on a typed-LLM validation failure.

        The defaults are intentionally conservative — small batch, short
        seq, LoRA (not QLoRA) — so they fit on almost any GPU. The user
        can comment to upgrade once the strategy card renders.
        """
        vram_gb = float(hw.get("vram_gb") or 0.0)
        device = (hw.get("device") or "cpu").lower()
        params_b = float(chosen_model.get("params_b") or 7.0)

        # Method: QLoRA only when VRAM is genuinely tight; otherwise LoRA.
        method = "lora"
        quantization = "none"
        if device == "cuda" and vram_gb and (vram_gb < 12 or params_b >= 13):
            method = "qlora"
            quantization = "int4"

        # Precision: bf16 on CUDA, fp32 elsewhere.
        precision = "bf16" if device == "cuda" else "fp32"

        # Batch / seq scale down with VRAM. On CPU we go absolute floor.
        if device != "cuda":
            batch_size, max_seq_len, grad_accum = 1, 512, 8
        elif vram_gb >= 24:
            batch_size, max_seq_len, grad_accum = 4, 2048, 2
        elif vram_gb >= 12:
            batch_size, max_seq_len, grad_accum = 2, 1024, 4
        else:
            batch_size, max_seq_len, grad_accum = 1, 768, 8

        return {
            "method": method,
            "adapter_variant": "none",
            "precision": precision,
            "quantization": quantization,
            "kernel_pack": "standard",
            "use_unsloth": False,
            "optimizer": "adamw",
            "use_liger": False,
            "batch_size": batch_size,
            "gradient_accumulation": grad_accum,
            "max_seq_len": max_seq_len,
            "learning_rate": 2e-4,
            "epochs": 3 if priority == "quality" else 1,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "early_stopping": True,
            "rationale": (
                "Safe-default fallback (LLM planner unavailable). "
                f"Picked for device={device}, vram={vram_gb}GB, "
                f"model={params_b}B, priority={priority}."
            ),
        }

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
            strategy["use_unsloth"] = True
        if "galore" in text:
            strategy["optimizer"] = "galore"
            strategy["adapter_variant"] = "galore"
        if "galore_q" in text or "galore-q" in text:
            strategy["optimizer"] = "galore_q"
        if "liger" in text:
            strategy["use_liger"] = True
            strategy["kernel_pack"] = "liger"
        if "adam8bit" in text or "8bit adam" in text:
            strategy["optimizer"] = "adam8bit"
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
