"""Canonical pool of clarification questions. The ClarificationAgent is only
allowed to ask questions defined here. This keeps the UX predictable, the
agent testable, and the question budget meaningful."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.api.schemas.session import ClarificationQuestion


@dataclass(frozen=True)
class CatalogEntry:
    question_id: str
    question: str
    kind: str                     # single_choice | multi_choice | text | yes_no
    default_options: tuple[str, ...] = ()
    why: str = ""                 # internal note; not shown to the user


CATALOG: dict[str, CatalogEntry] = {
    "q_user_goal": CatalogEntry(
        question_id="q_user_goal",
        question="What do you want the model to learn from this dataset?",
        kind="text",
        why="If we have no usable signal anywhere, ask the user directly.",
    ),
    "q_task_type": CatalogEntry(
        question_id="q_task_type",
        question="Which kind of task does this dataset best support?",
        kind="single_choice",
        default_options=("chat", "instruction", "classification", "extraction", "qa"),
        why="When task signals are mixed.",
    ),
    "q_target_field": CatalogEntry(
        question_id="q_target_field",
        question="Which field should the model learn to produce?",
        kind="single_choice",
        why="Unknown target/output field.",
    ),
    "q_input_fields": CatalogEntry(
        question_id="q_input_fields",
        question="Which fields should the model see as input?",
        kind="multi_choice",
        why="Multiple plausible inputs.",
    ),
    "q_style": CatalogEntry(
        question_id="q_style",
        question="What output style should the model use?",
        kind="single_choice",
        default_options=("conversational", "structured_json", "short_label"),
        why="Output style ambiguous.",
    ),
    "q_priority": CatalogEntry(
        question_id="q_priority",
        question="What's most important for this run?",
        kind="single_choice",
        default_options=("quality", "speed", "low_resource"),
        why="Priority drives strategy.choose.",
    ),
    "q_export_choice": CatalogEntry(
        question_id="q_export_choice",
        question="When training finishes, where should the model go?",
        kind="single_choice",
        default_options=("local", "hf", "both"),
        why="Mandatory finalization step.",
    ),
}


def get_question(qid: str) -> Optional[CatalogEntry]:
    return CATALOG.get(qid)


def build_question(
    qid: str,
    *,
    options: Optional[list[str]] = None,
    context: Optional[str] = None,
) -> ClarificationQuestion:
    e = CATALOG[qid]
    opts = list(options) if options is not None else list(e.default_options)
    return ClarificationQuestion(
        question_id=e.question_id,
        question=e.question,
        kind=e.kind,
        options=opts,
        context=context,
        required=True,
    )
