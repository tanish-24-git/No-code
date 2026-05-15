"""Agent-as-tool wrappers for the AgenticLoop.

The AgenticLoop is a model-driven think -> tool -> observe loop. To preserve
the substantial logic in the existing specialty agents without duplicating
it, each tool here:

    1. Instantiates the corresponding agent
    2. Synthesizes the event that agent originally triggered on
    3. Calls agent.handle() directly (NOT via safe_handle to avoid the
       reentrancy guard, since the loop already serializes tool calls)
    4. Reads the artifact the agent wrote to session storage
    5. Returns a compact result dict the model can reason about

Approval gates inside the wrapped agents (wait_for_approval) propagate
naturally: the tool call blocks until the user clicks Approve or Comment,
exactly mirroring the old behavior. The loop's asyncio.shield handling
treats those tools as cancel-unsafe when interactive=True so user
interrupts during a pending approval get queued, not lost.

Also defines the loop's own native tools that do NOT wrap an agent:
    propose_plan       emit a PhasePlanProposed + wait for approval
    ask_user           emit a UserClarificationRequested + wait for answer
    record_decision    append to the audit log
    synthesize_unified_dataset  the universal "crap-to-trainable" tool
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from app.agents.base import BaseAgent
from app.events.bus import EventBus
from app.events.types import AgentEvent
from app.tools.registry import ToolContext, tool


log = logging.getLogger("finetune-studio.tools.agent_tools")


# ── Helper: a small BaseAgent the tools can borrow for emit / wait ────────

class _LoopHelper(BaseAgent):
    """Used inside tools to emit events and wait for approvals from the
    AgenticLoop's name. Not registered to any trigger."""
    name = "AgenticLoop"
    role = "Autonomous fine-tuning engineer"
    directive_scope = "global"
    triggers = ()


def _helper(ctx: ToolContext) -> _LoopHelper:
    if ctx.bus is None:
        raise RuntimeError("AgenticLoop tools require a bus in ToolContext")
    return _LoopHelper(ctx.bus)


def _session_artifact(session_id: str, key: str) -> Any:
    from app.services import session_service
    s = session_service.get(session_id)
    if not s:
        return None
    return s.artifacts.get(key)


# ══════════════════════════════════════════════════════════════════════════
# Inspection tools (cheap, always safe)
# ══════════════════════════════════════════════════════════════════════════

@tool(
    name="probe_hardware",
    description="Detect local device/VRAM/RAM. Writes `hardware` artifact.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
)
async def probe_hardware(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.hardware_analysis import HardwareAnalysisAgent
    agent = HardwareAnalysisAgent(ctx.bus)
    agent.silent = True
    ev = AgentEvent(
        session_id=ctx.session_id, kind="DatasetProfileCompleted",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    return {"hardware": _session_artifact(ctx.session_id, "hardware")}


@tool(
    name="profile_dataset",
    description="Token lengths, duplicates, missing, imbalance. Writes `profile`.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
)
async def profile_dataset(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.profiling import DatasetProfilingAgent
    from app.services import session_service
    s = session_service.get(ctx.session_id)
    if not s or not s.dataset_id:
        return {"error": "no dataset bound to session"}
    agent = DatasetProfilingAgent(ctx.bus)
    agent.silent = True
    ev = AgentEvent(
        session_id=ctx.session_id, kind="IntakeCompleted",
        actor="AgenticLoop", payload={"dataset_id": s.dataset_id},
    )
    await agent.handle(ev)
    return {"profile": _session_artifact(ctx.session_id, "profile")}


@tool(
    name="grade_data_health",
    description="Data-health verdict (healthy/advisory/needs_attention/blocking). Writes `data_health`.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
)
async def grade_data_health(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.alchemy import DataAlchemistAgent
    from app.services import session_service
    s = session_service.get(ctx.session_id)
    if not s:
        return {"error": "session not found"}
    profile = (s.artifacts or {}).get("profile") or {}
    agent = DataAlchemistAgent(ctx.bus)
    agent.silent = True
    ev = AgentEvent(
        session_id=ctx.session_id, kind="DatasetProfileCompleted",
        actor="AgenticLoop", payload={"dataset_id": s.dataset_id, "profile": profile},
    )
    await agent.handle(ev)
    return {"data_health": _session_artifact(ctx.session_id, "data_health")}


@tool(
    name="infer_task_type",
    description="Classify task type (instruction/chat/qa/classification/lm). Needs profile+hardware first. Writes `task_inference`.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
)
async def infer_task_type(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.task_inference import TaskInferenceAgent
    agent = TaskInferenceAgent(ctx.bus)
    agent.silent = True
    ev = AgentEvent(
        session_id=ctx.session_id, kind="HardwareProfileCompleted",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    return {"task_inference": _session_artifact(ctx.session_id, "task_inference")}


# ══════════════════════════════════════════════════════════════════════════
# Selection / planning (with built-in user-approval gates)
# ══════════════════════════════════════════════════════════════════════════

@tool(
    name="select_base_model",
    description="HF search + rank + user-approval card. Writes `chosen_model`. Honors family/size hints from directives.",
    input_schema={"type": "object", "properties": {}},
    side_effect="external",
    interactive=True,
    cancel_safe=False,
)
async def select_base_model(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.model_selection import ModelSelectionAgent
    agent = ModelSelectionAgent(ctx.bus)
    ev = AgentEvent(
        session_id=ctx.session_id, kind="HardwareProfileCompleted",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    return {"chosen_model": _session_artifact(ctx.session_id, "chosen_model"),
            "candidates": _session_artifact(ctx.session_id, "candidate_models")}


@tool(
    name="propose_training_strategy",
    description="Pick LoRA/QLoRA/DoRA + Unsloth/GaLore/Liger + epochs/lr/batch with approval card. Needs chosen_model. Writes `strategy`, `runtime_estimate`.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
    interactive=True,
    cancel_safe=False,
)
async def propose_training_strategy(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.strategy import TrainingStrategyAgent
    agent = TrainingStrategyAgent(ctx.bus)
    ev = AgentEvent(
        session_id=ctx.session_id, kind="CandidateModelsRanked",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    return {"strategy": _session_artifact(ctx.session_id, "strategy"),
            "runtime_estimate": _session_artifact(ctx.session_id, "runtime_estimate")}


@tool(
    name="build_pipeline",
    description="Assemble pipeline (config+graph) from chosen_model+strategy+profile, with approval card. Sets session.pipeline_id.",
    input_schema={"type": "object", "properties": {}},
    side_effect="write_resource",
    interactive=True,
    cancel_safe=False,
)
async def build_pipeline(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.pipeline_builder import PipelineBuilderAgent
    agent = PipelineBuilderAgent(ctx.bus)
    ev = AgentEvent(
        session_id=ctx.session_id, kind="StrategyChosen",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    from app.services import session_service
    s = session_service.get(ctx.session_id)
    return {"pipeline_id": s.pipeline_id if s else None}


# ══════════════════════════════════════════════════════════════════════════
# Execution (cancel-unsafe; long-running)
# ══════════════════════════════════════════════════════════════════════════

@tool(
    name="run_training",
    description="Launch the fine-tuning job. Long-running, cancel-unsafe (interrupts queue). Sets session.job_id.",
    input_schema={"type": "object", "properties": {}},
    side_effect="write_resource",
    cost_class="expensive",
    cancel_safe=False,
)
async def run_training(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.execution import ExecutionAgent
    agent = ExecutionAgent(ctx.bus)
    agent.silent = True
    ev = AgentEvent(
        session_id=ctx.session_id, kind="PipelineApproved",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    from app.services import session_service
    s = session_service.get(ctx.session_id)
    return {"job_id": s.job_id if s else None}


@tool(
    name="evaluate_model",
    description="Run eval suite + baseline compare on trained adapter. Writes `evaluation`. Needs completed job.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
)
async def evaluate_model(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.evaluation import EvaluationAgent
    agent = EvaluationAgent(ctx.bus)
    agent.silent = True
    ev = AgentEvent(
        session_id=ctx.session_id, kind="TrainingCompleted",
        actor="AgenticLoop", payload={},
    )
    await agent.handle(ev)
    return {"evaluation": _session_artifact(ctx.session_id, "evaluation")}


@tool(
    name="export_artifact",
    description="Save trained model. target=local|hf|both. repo_id required for hf/both. Cancel-unsafe.",
    input_schema={
        "type": "object",
        "properties": {
            "target": {"type": "string", "enum": ["local", "hf", "both"]},
            "repo_id": {"type": ["string", "null"]},
            "name": {"type": ["string", "null"]},
        },
        "required": ["target"],
    },
    side_effect="external",
    cancel_safe=False,
)
async def export_artifact(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.export import ExportAgent
    target = args.get("target", "local")
    repo_id = args.get("repo_id")
    name = args.get("name")
    agent = ExportAgent(ctx.bus)
    agent.silent = True

    if target in ("local", "both"):
        ev = AgentEvent(
            session_id=ctx.session_id, kind="SaveLocalRequested",
            actor="AgenticLoop", payload={"name": name} if name else {},
        )
        await agent.handle(ev)
    if target in ("hf", "both"):
        if not repo_id:
            return {"error": "repo_id is required for hf/both export"}
        ev = AgentEvent(
            session_id=ctx.session_id, kind="PushToHFRequested",
            actor="AgenticLoop", payload={"repo_id": repo_id},
        )
        await agent.handle(ev)

    return {"export": _session_artifact(ctx.session_id, "export")}


# ══════════════════════════════════════════════════════════════════════════
# Interactive tools owned by the loop (not wrapping any existing agent)
# ══════════════════════════════════════════════════════════════════════════

# Synonym map so the LLM can use natural-language phase names ("training",
# "fine-tuning", "data prep", ...) and we still hit canonical PHASES
# entries for proper canvas node materialization. Unknown phases pass
# through (phase_service no longer raises - it just logs a warning).
_PHASE_SYNONYMS: dict[str, str] = {
    "training": "execute",
    "train": "execute",
    "fine tuning": "execute",
    "finetuning": "execute",
    "fine tune": "execute",
    "execution": "execute",
    "eval": "evaluate",
    "evaluation": "evaluate",
    "model selection": "model_search",
    "model search": "model_search",
    "model": "model_search",
    "hyperparameters": "strategy",
    "config": "strategy",
    "configuration": "strategy",
    "data prep": "profile",
    "preprocessing": "profile",
    "preprocess": "profile",
    "data": "profile",
    "upload": "intake",
    "ingest": "intake",
    "deploy": "export",
    "save": "export",
    "publish": "export",
    "pipeline": "plan",
    "device": "hardware",
}


def _coerce_phase(raw: str) -> str:
    """Map a free-form phase string to a canonical PHASES entry when there
    is a sensible synonym; otherwise leave it untouched so the upstream
    log warning fires. Never crashes."""
    if not raw:
        return "master_plan"
    # Normalize: lowercase, collapse hyphens and underscores to spaces.
    key = raw.strip().lower().replace("-", " ").replace("_", " ")
    if key in _PHASE_SYNONYMS:
        return _PHASE_SYNONYMS[key]
    # If the LLM already gave a canonical name (with whatever casing),
    # return the lowercase form for consistency with PHASES.
    return raw.strip().lower().replace("-", "_").replace(" ", "_")


@tool(
    name="propose_plan",
    description="Show user a plan card; wait for Approve or Comment. Comments become global directives.",
    input_schema={
        "type": "object",
        "properties": {
            "phase": {"type": "string"},
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "steps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["phase", "title", "summary", "steps"],
    },
    side_effect="write_session",
    interactive=True,
    cancel_safe=False,
)
async def propose_plan(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    helper = _helper(ctx)
    phase = _coerce_phase(args.get("phase") or "")
    title = args.get("title") or "Plan"
    summary = args.get("summary") or ""
    steps = list(args.get("steps") or [])

    await helper.announce(
        ctx.session_id,
        phase=phase,
        title=title,
        summary=summary,
        steps=steps,
        requires_approval=True,
    )
    comment = await helper.wait_for_approval(ctx.session_id, phase)
    if comment is None:
        return {"approved": True}
    return {"approved": False, "comment": comment}


@tool(
    name="ask_user",
    description="Ask one clarifying question. Blocks until answered. kind is one of 'text' (default), 'single_choice', 'multi_choice', 'yes_no'.",
    input_schema={
        # Permissive schema. Enums + null-unions are dropped here because
        # Groq's strict tool-arg validator rejects the combination ("kind
        # must be one of [text, ..., null]" trips the parser). The tool
        # body coerces unknown kind/impact values back to sensible
        # defaults, so the model can send any string here.
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "kind": {"type": "string"},
            "choices": {"type": "array", "items": {"type": "string"}},
            "impact": {"type": "string"},
        },
        "required": ["question"],
    },
    side_effect="write_session",
    interactive=True,
    cancel_safe=False,
)
async def ask_user(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    import uuid
    from app.api.schemas.session import ClarificationQuestion
    from app.services import session_service

    question = args.get("question") or ""
    if not question.strip():
        return {"error": "question is required"}
    # Coerce free-form values back to a safe enum. The schema is permissive
    # so Groq accepts the tool call; we validate here at runtime.
    raw_kind = (args.get("kind") or "text").strip().lower()
    kind = raw_kind if raw_kind in ("text", "single_choice", "multi_choice", "yes_no") else "text"
    options = list(args.get("choices") or args.get("options") or [])
    raw_impact = (args.get("impact") or "medium").strip().lower()
    impact = raw_impact if raw_impact in ("low", "medium", "high") else "medium"

    qid = "q_" + uuid.uuid4().hex[:10]
    q = ClarificationQuestion(
        question_id=qid,
        kind=kind,
        question=question,
        options=options,
        required=True,
    )
    session = session_service.get(ctx.session_id)
    if not session:
        return {"error": "session not found"}
    session_service.set_pending_question(session, q)

    # Emission shape must be flat - the frontend ClarificationRow
    # (AgentActivity.tsx:458-498) destructures payload.question_id /
    # .question / .kind / .options directly, matching the legacy
    # ClarificationAgent shape. Wrapping in {"question": {...}} crashes
    # React with error #31 when the question card renders.
    flat_payload = q.model_dump(mode="json")
    flat_payload["impact"] = impact
    helper = _helper(ctx)
    await helper.emit(
        "UserClarificationRequested",
        ctx.session_id,
        payload=flat_payload,
    )

    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    async def _on_received(ev: AgentEvent) -> None:
        if ev.session_id != ctx.session_id:
            return
        # The reply endpoint at /api/sessions/{id}/clarifications/{qid}
        # publishes payload={"answer": ClarificationAnswer.model_dump()};
        # unwrap before matching.
        answer = (ev.payload or {}).get("answer") or {}
        if answer.get("question_id") != qid:
            return
        if not future.done():
            future.set_result(answer.get("value"))

    ctx.bus.on("UserClarificationReceived", _on_received)
    try:
        value = await future
    finally:
        ctx.bus.off("UserClarificationReceived", _on_received)

    return {"answer": value, "question_id": qid}


@tool(
    name="record_decision",
    description="Log a decision + rationale to the audit trail.",
    input_schema={
        "type": "object",
        "properties": {
            "kind": {"type": "string"},
            "chosen": {},
            "rationale": {"type": "string"},
            "confidence": {"type": ["number", "null"]},
        },
        "required": ["kind", "chosen", "rationale"],
    },
    side_effect="write_session",
)
async def record_decision(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.services import decision_log
    d = decision_log.record(
        session_id=ctx.session_id,
        agent="AgenticLoop",
        kind=args.get("kind") or "decision",
        inputs={},
        chosen=args.get("chosen"),
        confidence=float(args.get("confidence") or 0.8),
        rationale=args.get("rationale") or "",
    )
    return {"decision_id": d.id}


# ══════════════════════════════════════════════════════════════════════════
# Universal dataset synthesis ("upload any crap" -> trainable)
# ══════════════════════════════════════════════════════════════════════════

@tool(
    name="walk_session_uploads",
    description="List uploaded files for the session with content-sniffed kind.",
    input_schema={"type": "object", "properties": {}},
    side_effect="read",
)
async def walk_session_uploads(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from pathlib import Path
    from app.services import dataset_service, raw_extractor, session_service
    s = session_service.get(ctx.session_id)
    if not s or not s.dataset_id:
        return {"files": []}
    ds = dataset_service.get_dataset(s.dataset_id)
    if not ds:
        return {"files": []}
    root = Path(ds.file_path)
    # If dataset_id points to a single file, return just that one. For
    # directory uploads, the file_path is the aggregated jsonl; we walk the
    # upload_dir for the original sources.
    files = []
    if root.is_dir():
        paths = await raw_extractor.walk_folder(root)
    elif root.parent.exists():
        paths = [root]
    else:
        paths = []
    for p in paths:
        try:
            kind = await raw_extractor.sniff_kind(p)
            files.append({
                "path": str(p),
                "name": p.name,
                "kind": kind,
                "size_bytes": p.stat().st_size if p.exists() else 0,
            })
        except Exception as e:
            log.warning("sniff failed for %s: %s", p, e)
    return {"files": files, "dataset_id": s.dataset_id, "dataset_kind": (ds.analysis or {}).get("kind")}


@tool(
    name="extract_raw_text",
    description="Extract clean text from one file (PDF/DOCX/HTML/text).",
    input_schema={
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    },
    side_effect="read",
    # raw_extractor self-caps structured-file output at 200K chars; the
    # registry middleware would otherwise truncate clean extracted text
    # to 2000 chars, which defeats the point of this tool. The
    # AgenticLoop also passes results through its own 1500-char-per-result
    # cap when rendering observations, so the LLM still sees a tight
    # summary per turn.
    supports_full_output=True,
)
async def extract_raw_text(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.services import raw_extractor
    path = args.get("path") or ""
    if not path:
        return {"error": "path is required"}
    rec = await raw_extractor.extract_text(path)
    if rec is None:
        return {"error": "file is binary or unreadable"}
    return raw_extractor.record_to_dict(rec)


@tool(
    name="synthesize_unified_dataset",
    description="Convert raw_doc upload (PDF/folder/mixed) into a structured training dataset; re-binds session. Use only on raw_doc. target_format defaults to 'instruction'; other values: chat, qa, classification, language_modeling.",
    input_schema={
        # Drop enums to keep Groq's validator happy; the tool body coerces.
        "type": "object",
        "properties": {
            "target_format": {"type": "string"},
            "user_intent": {"type": "string"},
        },
    },
    side_effect="write_resource",
    cost_class="medium",
)
async def synthesize_unified_dataset(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    from app.agents.data_restructurer import DataRestructurerAgent
    from app.services import dataset_service, session_service, directives as directives_service
    s = session_service.get(ctx.session_id)
    if not s or not s.dataset_id:
        return {"error": "no dataset bound to session"}
    ds = dataset_service.get_dataset(s.dataset_id)
    if not ds:
        return {"error": "dataset not found"}
    if not dataset_service.is_raw_doc(ds):
        return {"status": "already_structured",
                "dataset_id": s.dataset_id,
                "row_count": ds.row_count}

    # Seed the restructurer's format directive so its approval gate uses
    # what the model chose. The schema accepts any string for
    # target_format; we coerce here to a known set so downstream code
    # never crashes on a misspelled value.
    raw_target = (args.get("target_format") or "instruction").strip().lower()
    _allowed_targets = {"instruction", "chat", "qa", "classification", "language_modeling"}
    target = raw_target if raw_target in _allowed_targets else "instruction"
    user_intent = args.get("user_intent") or ""
    if user_intent:
        directives_service.record(
            ctx.session_id,
            f"Restructure target format: {target}. Domain context: {user_intent}",
            source_phase="restructure", scope="data", actor="AgenticLoop",
        )
    else:
        directives_service.record(
            ctx.session_id,
            f"Restructure target format: {target}",
            source_phase="restructure", scope="data", actor="AgenticLoop",
        )

    agent = DataRestructurerAgent(ctx.bus)
    ev = AgentEvent(
        session_id=ctx.session_id, kind="DatasetUploaded",
        actor="AgenticLoop",
        payload={"dataset_id": s.dataset_id, "name": ds.name},
    )
    await agent.handle(ev)

    # The restructurer re-binds the session to the new dataset.
    s2 = session_service.get(ctx.session_id)
    new_id = s2.dataset_id if s2 else None
    new_ds = dataset_service.get_dataset(new_id) if new_id else None
    return {
        "from_dataset_id": s.dataset_id,
        "to_dataset_id": new_id,
        "row_count": new_ds.row_count if new_ds else 0,
        "target_format": target,
    }


# ══════════════════════════════════════════════════════════════════════════
# Search HuggingFace Hub (delegates to existing hf_search if present)
# ══════════════════════════════════════════════════════════════════════════

@tool(
    name="search_hf_models",
    description="Search HuggingFace Hub. Filters: family, size_b, task.",
    input_schema={
        "type": "object",
        "properties": {
            # All optional fields accept null because some providers
            # (Groq llama-3.3-70b) emit null instead of omitting the key,
            # which strict schemas would reject server-side.
            "query": {"type": ["string", "null"]},
            "family": {"type": ["string", "null"]},
            "size_b": {"type": ["number", "null"]},
            "task": {"type": ["string", "null"]},
            "top_n": {"type": ["integer", "null"]},
        },
    },
    side_effect="external",
)
async def search_hf_models(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # Delegate to the legacy registered tool which already exists.
    from app.tools.registry import run_tool as _run
    top_n_raw = args.get("top_n")
    try:
        top_n = int(top_n_raw) if top_n_raw is not None else 10
    except (TypeError, ValueError):
        top_n = 10
    payload = {
        "query": args.get("query") or None,
        "family_hint": args.get("family") or None,
        "size_hint_b": args.get("size_b"),
        "task": args.get("task") or "instruction",
        "top_n": top_n,
        "instruct_only": True,
    }
    return await _run("model.search_hf", payload, ctx)


# ══════════════════════════════════════════════════════════════════════════
# Recovery: AgenticLoop-facing wrapper around the existing
# recovery.propose_plan tool. The loop can call this directly when a
# training stub or downstream step reports an anomaly, instead of waiting
# for the dedicated RecoveryAgent to react to a TrainingAnomalyDetected
# event. Backs the existing /api/sessions/{id}/retry endpoint.
# ══════════════════════════════════════════════════════════════════════════

@tool(
    name="propose_recovery",
    description="Given a training/eval anomaly + current pipeline config, propose a config-diff to retry with. Returns operations + rationale + confidence.",
    input_schema={
        "type": "object",
        "properties": {
            "anomaly": {"type": "string"},
            "config": {"type": "object"},
            "extra_minutes": {"type": "number"},
        },
        "required": ["anomaly"],
    },
    side_effect="read",
)
async def propose_recovery(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    # Resolve the current pipeline config from the session if the caller
    # did not pass one explicitly — keeps the LLM's call simple.
    from app.services import pipeline_service, session_service
    from app.tools.registry import run_tool as _run

    cfg = args.get("config")
    if not cfg:
        s = session_service.get(ctx.session_id)
        pipeline = (
            pipeline_service.get(s.pipeline_id)
            if s and getattr(s, "pipeline_id", None)
            else None
        )
        cfg = pipeline.config.model_dump() if pipeline else {}
    payload = {
        "anomaly": args.get("anomaly") or "",
        "config": cfg,
        "extra_minutes": float(args.get("extra_minutes") or 5.0),
    }
    return await _run("recovery.propose_plan", payload, ctx)


# Import side-effect: ensure web + dataset/hf tools are also loaded so the
# AgenticLoop sees them on registry introspection.
from app.tools import web as _web  # noqa: F401
from app.tools import recovery as _recovery  # noqa: F401
