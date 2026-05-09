"""AuditAgent — the Critic in the swarm hierarchy (blueprint §2.1).

Responsibility: every plan or strategy that another agent puts on the
blackboard is reviewed here. If the Critic identifies a high-impact risk
(>20% chance of materially affecting model quality), it raises an
``AuditCritique`` event. The downstream behaviour:

    - rationale-only critiques are informational; they appear as a yellow
      "note" card in the UI.
    - blocking critiques set ``payload.blocking=True`` and emit
      ``AuditOverride``, which the Master Orchestrator funnels back to the
      user as a clarification question before the plan is allowed to
      proceed.

The agent does not call an LLM by default — its checks are deterministic
heuristics derived from the blueprint's "10 Commandments". This keeps it
testable and free; an LLM upgrade can be slotted in via tools later.
"""
from __future__ import annotations

from typing import Any

from app.agents.base import BaseAgent
from app.events.types import AgentEvent


# How aggressive the critic is. Higher = more vetoes.
_RISKY_DUPLICATE_PCT = 25.0
_RISKY_MISSING_PCT = 30.0
_RISKY_RUNTIME_MIN = 180.0
_RISKY_LR_LOWER = 1e-6
_RISKY_LR_UPPER = 1e-2


class AuditAgent(BaseAgent):
    name = "AuditAgent"
    role = "Independent critic. Reviews plans and strategies for high-impact risks."
    allowed_tools = ("audit.write",)
    triggers = (
        "DatasetProfileCompleted",
        "StrategyChosen",
        "PipelineDraftCreated",
        "RecoveryPlanGenerated",
    )

    async def handle(self, event: AgentEvent) -> None:
        session = self.get_session(event.session_id)
        if not session:
            return

        if event.kind == "DatasetProfileCompleted":
            await self._audit_data_quality(session, event)
        elif event.kind == "StrategyChosen":
            await self._audit_strategy(session, event)
        elif event.kind == "PipelineDraftCreated":
            await self._audit_pipeline(session, event)
        elif event.kind == "RecoveryPlanGenerated":
            await self._audit_recovery(session, event)

    # ── checks ───────────────────────────────────────────────────────────

    async def _audit_data_quality(self, session, event: AgentEvent) -> None:
        profile = (event.payload.get("profile") or {})
        dupes = (profile.get("duplicates") or {}).get("duplicate_pct") or 0.0
        missing = profile.get("missing") or {}
        # Compute worst per-column missing rate.
        worst = 0.0
        worst_col = None
        for col, m in (missing.get("per_column") or {}).items():
            pct = float(m.get("missing_pct") or 0.0)
            if pct > worst:
                worst, worst_col = pct, col

        notes: list[str] = []
        if dupes >= _RISKY_DUPLICATE_PCT:
            notes.append(f"{dupes:.1f}% duplicate rows could bias the loss surface")
        if worst >= _RISKY_MISSING_PCT and worst_col:
            notes.append(f"column `{worst_col}` is {worst:.1f}% missing")
        if notes:
            await self.think(
                session.id,
                "I'm reviewing the dataset profile and I see signal-degrading risks.",
                meta={"checks": notes},
            )
            await self._raise_critique(
                session.id,
                event,
                kind="data_quality",
                blocking=False,
                summary="; ".join(notes),
                advice="Consider deduplication, missing-value imputation, or scoping the input columns.",
            )

    async def _audit_strategy(self, session, event: AgentEvent) -> None:
        strategy = event.payload.get("strategy") or {}
        est = float(event.payload.get("estimated_minutes") or 0.0)
        lr = float(strategy.get("learning_rate") or 0.0)

        notes: list[str] = []
        if est >= _RISKY_RUNTIME_MIN:
            notes.append(f"estimated runtime {est:.0f} min exceeds the comfortable budget")
        if lr and (lr < _RISKY_LR_LOWER or lr > _RISKY_LR_UPPER):
            notes.append(f"learning rate {lr:g} is outside the safe envelope")
        method = strategy.get("method")
        precision = strategy.get("precision")
        if method == "full" and precision in (None, "fp32", "float32"):
            notes.append("full-parameter training in fp32 will likely OOM")

        # Flag DoRA on tiny models — wasted overhead.
        if strategy.get("adapter_variant") == "dora" and (session.artifacts.get("chosen_model") or {}).get("params_b", 0) < 1.0:
            notes.append("DoRA adds compute that is rarely worth it on <1B models")

        if not notes:
            await self.think(session.id, "Strategy looks sound — no audit overrides.", meta={"strategy": strategy})
            return

        await self._raise_critique(
            session.id,
            event,
            kind="strategy",
            blocking=any("OOM" in n for n in notes),
            summary="; ".join(notes),
            advice="The Architectural Designer can downgrade to QLoRA, reduce rank, or split sequence length.",
        )

    async def _audit_pipeline(self, session, event: AgentEvent) -> None:
        cfg = event.payload.get("config") or {}
        notes: list[str] = []
        # Sanity: ratio integrity.
        if cfg.get("max_seq_len") and cfg.get("batch_size"):
            tokens_per_step = int(cfg["max_seq_len"]) * int(cfg["batch_size"]) * int(cfg.get("gradient_accumulation") or 1)
            if tokens_per_step > 65_536:
                notes.append(f"effective tokens/step is {tokens_per_step}; risk of memory pressure")
        if not cfg.get("early_stopping"):
            notes.append("early_stopping is disabled; runs may overfit on small datasets")
        if notes:
            await self._raise_critique(
                session.id,
                event,
                kind="pipeline",
                blocking=False,
                summary="; ".join(notes),
                advice="Consider enabling gradient checkpointing or shrinking the batch.",
            )

    async def _audit_recovery(self, session, event: AgentEvent) -> None:
        rec = event.payload or {}
        if float(rec.get("confidence") or 0.0) < 0.5:
            await self._raise_critique(
                session.id,
                event,
                kind="recovery",
                blocking=True,
                summary=f"low-confidence recovery diff ({rec.get('confidence')})",
                advice="Surface to the user for explicit approval rather than auto-applying.",
            )

    # ── helper ───────────────────────────────────────────────────────────

    async def _raise_critique(
        self,
        session_id: str,
        event: AgentEvent,
        *,
        kind: str,
        blocking: bool,
        summary: str,
        advice: str,
    ) -> None:
        payload: dict[str, Any] = {
            "kind": kind,
            "blocking": blocking,
            "summary": summary,
            "advice": advice,
            "reviewed_event_id": event.id,
            "reviewed_kind": event.kind,
        }
        self.blackboard().post(
            session_id,
            "critiques",
            agent=self.name,
            content=summary,
            tags=[kind, "blocking" if blocking else "advisory"],
            meta=payload,
        )
        await self.emit(
            "AuditCritique",
            session_id,
            payload=payload,
            parent_event_id=event.id,
            confidence=0.85 if blocking else 0.6,
        )
        if blocking:
            await self.emit(
                "AuditOverride",
                session_id,
                payload=payload,
                parent_event_id=event.id,
            )
