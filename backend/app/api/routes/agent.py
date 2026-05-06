"""Agent chat endpoints. SSE-streamed text from Anthropic with tool use."""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.agents.agent import run_chat
from app.api.schemas.agent import ApplyConfigRequest, ChatRequest
from app.services import pipeline_service


router = APIRouter(prefix="/api/agent", tags=["agent"])


@router.post("/chat")
async def chat(payload: ChatRequest) -> StreamingResponse:
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    async def gen() -> AsyncIterator[str]:
        loop = asyncio.get_event_loop()
        # The Anthropic SDK is sync; run the generator on a thread and bridge
        # chunks back into the asyncio world.
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=256)

        def producer() -> None:
            try:
                for chunk in run_chat(
                    payload.messages,
                    pipeline_id=payload.pipeline_id,
                    dataset_id=payload.dataset_id,
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(f"\n\n[error: {e}]\n"), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, producer)
        while True:
            chunk = await queue.get()
            if chunk is None:
                yield "data: [DONE]\n\n"
                return
            # SSE forbids bare newlines in `data:` payloads; encode them.
            for line in chunk.split("\n"):
                yield f"data: {line}\n"
            yield "\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.post("/apply-config")
def apply_config(payload: ApplyConfigRequest) -> dict:
    """Convenience endpoint: lets the frontend persist a config the agent
    suggested in chat (so the user doesn't need a separate confirmation
    round-trip if they choose to apply manually)."""
    rec = pipeline_service.apply_agent_config(payload.pipeline_id, payload.config, payload.reasoning)
    if not rec:
        raise HTTPException(status_code=404, detail="Pipeline not found")
    return rec.model_dump(mode="json")
