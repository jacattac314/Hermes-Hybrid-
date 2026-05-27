from __future__ import annotations

import json

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hermes import dispatcher, memory, router
from hermes.config import settings
from hermes.prompts import build_worker_prompt
from hermes.schemas import ChatResponse, HealthResponse, UserMessage

app = FastAPI(title="Hermes Hybrid Router", version="0.1.0")


class MemoryClearResponse(BaseModel):
    deleted: int


@app.post("/chat")
async def chat(msg: UserMessage, stream: bool = False) -> ChatResponse | StreamingResponse:
    use_stream = stream or settings.stream_default
    memory_chunks = memory.retrieve(msg.session_id, msg.content)
    chunk_texts = [c.content for c in memory_chunks]

    decision = await router.route(msg.content)

    base_system = decision.worker_system_prompt or "You are a helpful assistant."
    system = build_worker_prompt(base_system, chunk_texts)

    result, used_target = await dispatcher.dispatch(
        decision, msg.content, system, stream=use_stream
    )

    if use_stream:
        async def event_stream():
            full = []
            async for token in result:
                full.append(token)
                yield f"data: {json.dumps({'token': token, 'target': used_target.value})}\n\n"
            assembled = "".join(full)
            memory.save(msg.session_id, f"User: {msg.content}")
            memory.save(msg.session_id, f"Assistant: {assembled}")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    memory.save(msg.session_id, f"User: {msg.content}")
    memory.save(msg.session_id, f"Assistant: {result}")

    return ChatResponse(
        session_id=msg.session_id,
        content=result,
        target_used=used_target,
        memory_chunks_used=len(memory_chunks),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_ok = False
    lmstudio_ok = False
    chroma_ok = False

    async with httpx.AsyncClient(timeout=3.0) as client:
        try:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
        except Exception:
            pass
        try:
            r = await client.get(f"{settings.lmstudio_base_url}/models",
                                 headers={"Authorization": "Bearer lm-studio"})
            lmstudio_ok = r.status_code == 200
        except Exception:
            pass

    try:
        import chromadb
        chromadb.PersistentClient(path=settings.chroma_persist_dir)
        chroma_ok = True
    except Exception:
        pass

    router_ok = lmstudio_ok if settings.router_backend == "lmstudio" else ollama_ok
    return HealthResponse(
        status="ok" if (router_ok and chroma_ok) else "degraded",
        ollama_reachable=ollama_ok,
        lmstudio_reachable=lmstudio_ok,
        chroma_reachable=chroma_ok,
    )


@app.delete("/memory/{session_id}", response_model=MemoryClearResponse)
async def clear_memory(session_id: str) -> MemoryClearResponse:
    deleted = memory.clear(session_id)
    return MemoryClearResponse(deleted=deleted)


@app.get("/memory/{session_id}")
async def get_memory(session_id: str, query: str = "") -> list[dict]:
    if not query:
        raise HTTPException(status_code=400, detail="query param required")
    chunks = memory.retrieve(session_id, query)
    return [c.model_dump() for c in chunks]
