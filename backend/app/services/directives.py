"""Global User Directives — the shared intent store.

Every user comment, clarification, and master-plan instruction is recorded
here. Every agent reads this BEFORE calling the LLM so that "use Llama"
spoken in the master-plan phase is honored five phases later by the
ModelSelectionAgent.

Without this, agent "context" was siloed per phase and earlier user input
was silently lost - the well-known root cause of the "the agent ignored
me" complaint.

Storage: piggy-backs on the federated blackboard's "directives" section
(all sections are append-only and persisted to disk).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from app.services.blackboard import get_blackboard


# Canonical scopes a directive can apply to.
SCOPES = (
    "global",          # any agent
    "model",           # ModelSelectionAgent
    "strategy",        # TrainingStrategyAgent
    "data",            # ProfilingAgent / DataAlchemistAgent / DataRestructurer
    "pipeline",        # PipelineBuilderAgent
    "training",        # ExecutionAgent / TrainingMonitor
    "evaluation",      # EvaluationAgent
    "export",          # ExportAgent
)


@dataclass
class Directive:
    text: str
    scope: str
    source_phase: Optional[str]
    ts: str

    def as_md(self) -> str:
        scope = f"[{self.scope}] " if self.scope and self.scope != "global" else ""
        phase = f" (from phase: {self.source_phase})" if self.source_phase else ""
        return f"- {scope}{self.text}{phase}"


# Phase → likely scope mapping. The user's comment on a phase card maps to
# a sensible default scope, but we always also record it under "global" so
# every agent sees it regardless of where it was uttered.
_PHASE_SCOPE: dict[str, str] = {
    "master_plan": "global",
    "intake": "data",
    "profile": "data",
    "restructure": "data",
    "hardware": "training",
    "task": "data",
    "model_search": "model",
    "strategy": "strategy",
    "plan": "pipeline",
    "audit": "global",
    "execute": "training",
    "monitor": "training",
    "evaluate": "evaluation",
    "sandbox": "evaluation",
    "export": "export",
}


def record(
    session_id: str,
    text: str,
    *,
    source_phase: Optional[str] = None,
    scope: Optional[str] = None,
    actor: str = "user",
) -> Directive:
    """Append a user directive to the session's blackboard.

    The directive is stored in two slots so retrieval is O(1) by scope:
        directives           every comment, regardless of scope
        directives_<scope>   subset filtered by scope
    """
    if not text or not text.strip():
        return Directive(text="", scope=scope or "global", source_phase=source_phase, ts="")

    inferred_scope = scope or _PHASE_SCOPE.get(source_phase or "", "global")
    if inferred_scope not in SCOPES:
        inferred_scope = "global"

    bb = get_blackboard()
    entry = bb.post(
        session_id,
        "directives",
        agent=actor,
        content=text.strip(),
        tags=[inferred_scope, source_phase or "unscoped"],
        meta={"scope": inferred_scope, "source_phase": source_phase},
    )
    return Directive(
        text=text.strip(),
        scope=inferred_scope,
        source_phase=source_phase,
        ts=entry["ts"],
    )


def read_for_scope(session_id: str, scope: str = "global") -> list[Directive]:
    """All directives whose scope matches OR equals 'global'.

    The "global" scope is always returned; specific scopes additively pick
    up their own directives. So a comment on the master plan ("use llama")
    flows through to the model_search phase via scope=global, and a
    comment on the model_search phase only flows to model-related agents.
    """
    bb = get_blackboard()
    raw = bb.read(session_id, "directives")
    out: list[Directive] = []
    for r in raw:
        meta = r.get("meta") or {}
        d_scope = meta.get("scope") or "global"
        if d_scope == scope or d_scope == "global" or scope == "all":
            out.append(Directive(
                text=r.get("content", ""),
                scope=d_scope,
                source_phase=meta.get("source_phase"),
                ts=r.get("ts", ""),
            ))
    return out


def render_for_prompt(session_id: str, scope: str = "global") -> str:
    """Markdown-formatted directive list, ready to inject into an LLM prompt."""
    directives = read_for_scope(session_id, scope)
    if not directives:
        return ""
    lines = ["**Standing user directives (read carefully, follow them):**"]
    for d in directives[-20:]:  # most recent 20 - cap context
        lines.append(d.as_md())
    return "\n".join(lines)


# ── Lightweight model-name extraction helper ──────────────────────────────
#
# When a directive mentions a base-model family or specific repo, we
# proactively extract the hint so ModelSelectionAgent can resolve it
# without an extra LLM call.

_MODEL_FAMILIES = (
    "llama", "qwen", "phi", "mistral", "gemma", "smollm",
    "deepseek", "yi", "falcon", "tinyllama", "stablelm", "olmo",
)


def model_hints(session_id: str) -> dict[str, Any]:
    """Return any extracted model-name signals from the directives.

    Returns {family, size_b, repo_id} where any field may be None.
    """
    text = "\n".join(d.text.lower() for d in read_for_scope(session_id, "model"))
    if not text:
        return {"family": None, "size_b": None, "repo_id": None}

    family = next((f for f in _MODEL_FAMILIES if f in text), None)

    size_b: Optional[float] = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB]\b", text)
    if m:
        try:
            size_b = float(m.group(1))
        except ValueError:
            pass

    repo_id: Optional[str] = None
    rm = re.search(r"([\w\-]+/[\w\-\.]+)", text)
    if rm:
        repo_id = rm.group(1)

    return {"family": family, "size_b": size_b, "repo_id": repo_id}
