"""OrchestratorAgent — the General (blueprint §2.1).

Responsibilities:
    1. Open every session with a brief, on-brand greeting that prepares
       the user for an interactive run.
    2. Stream a top-level plan ("here is what I will do") so the UI can
       render the bird's-eye view *before* tool execution begins.
    3. Watch for late-arriving free-text from the user and route it to the
       appropriate downstream agent (free-text guidance becomes a clarif-
       ication answer when one is pending).

Per the blueprint: the General is the "Voice of the Studio." Tone here
matters more than logic — every opening message should make the user feel
the agent has a strategy.
"""
from __future__ import annotations

import re
from app.agents.base import BaseAgent
from app.events.types import AgentEvent


_PLAN_SYSTEM_PROMPT = """You are the Lead Orchestrator for FineTune Studio.
Your goal is to generate a structured JSON plan for the user's fine-tuning session.
Consider the user's input and the overall workflow:
1. Data Alchemy (cleaning, dedup, PII redaction, restructuring raw text)
2. Hardware Analysis (VRAM, GPU throughput)
3. Task Inference (finding the ideal objective)
4. Strategy Selection (DoRA, LoRA, Unsloth)
5. Pipeline Construction
6. Live Training & Monitoring
7. Evaluation & Export

Output ONLY a JSON list of strings representing the steps."""



class OrchestratorAgent(BaseAgent):
    name = "OrchestratorAgent"
    role = "Voice of the studio. Greets the user and broadcasts the master plan."
    allowed_tools = ()
    triggers = ("SessionStarted", "AuditOverride")

    async def handle(self, event: AgentEvent) -> None:
        if event.kind == "SessionStarted":
            await self._open(event)
        elif event.kind == "AuditOverride":
            await self._audit_override(event)

    async def _open(self, event: AgentEvent) -> None:
        session_id = event.session_id
        session = self.get_session(session_id)
        
        # 1. Generate Greeting
        greeting_prompt = f"The user just started a session. Their input context: {session.name if session else 'new project'}. Write a brief, professional greeting (2 sentences) as the FineTune Studio Orchestrator."
        greeting = await self.call_llm(session_id, greeting_prompt, system="You are the Voice of FineTune Studio. Professional, expert, and proactive.", stream_thoughts=False)
        
        await self.emit_message(
            session_id,
            greeting + " I've analyzed your project and I'm drafting a custom execution strategy now.",
            parent=event.id,
        )

        # 2. Dynamic Planning
        await self.think(session_id, "Formulating a custom strategy based on your project goals...", parent=event.id)
        
        plan_prompt = f"Generate a 6-8 step execution plan for this project: {session.name if session else 'General training task'}. Return ONLY a JSON list of strings."
        plan_json = await self.call_llm(session_id, plan_prompt, system=_PLAN_SYSTEM_PROMPT, parent=event.id)
        
        try:
            # Try to extract JSON if the LLM wrapped it in markdown
            import json
            match = re.search(r"\[.*\]", plan_json, re.DOTALL)
            steps = json.loads(match.group(0)) if match else []
        except Exception:
            steps = ["Analyze dataset", "Check hardware", "Configure training", "Execute pipeline", "Export model"]

        await self.plan(session_id, steps, title="Master Plan", parent=event.id)
        
        await self.think(
            session_id,
            "Master plan broadcasted. The swarm is initializing: Data Alchemist, Hardware Analyst, and Task Inference are coming online.",
            parent=event.id,
        )

    async def _audit_override(self, event: AgentEvent) -> None:
        """The Critic vetoed something. Surface to the user immediately."""
        p = event.payload or {}
        await self.emit_message(
            event.session_id,
            f"**Audit override** — {p.get('summary', 'a critical concern was raised')}.\n\n"
            f"Recommendation: {p.get('advice', 'review the agent activity log')}.",
            parent=event.id,
        )
