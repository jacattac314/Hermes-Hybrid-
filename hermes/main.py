from __future__ import annotations

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from hermes import memory, router, dispatcher
from hermes.config import settings
from hermes.prompts import build_worker_prompt
from hermes.schemas import ChatResponse, HealthResponse, UserMessage

app = FastAPI(title="Hermes Hybrid Router", version="0.1.0")


class MemoryClearResponse(BaseModel):
    deleted: int


@app.post("/chat", response_model=ChatResponse)
async def chat(msg: UserMessage) -> ChatResponse:
    memory_chunks = memory.retrieve(msg.session_id, msg.content)
    chunk_texts = [c.content for c in memory_chunks]

    decision = await router.route(msg.content)

    system = build_worker_prompt("You are a helpful assistant.", chunk_texts)
    if decision.worker_system_prompt:
        system = build_worker_prompt(decision.worker_system_prompt, chunk_texts)

    content, used_target = await dispatcher.dispatch(decision, msg.content, system)

    memory.save(msg.session_id, f"User: {msg.content}")
    memory.save(msg.session_id, f"Assistant: {content}")

    return ChatResponse(
        session_id=msg.session_id,
        content=content,
        target_used=used_target,
        memory_chunks_used=len(memory_chunks),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ollama_ok = False
    chroma_ok = False

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{settings.ollama_base_url}/api/tags")
            ollama_ok = r.status_code == 200
    except Exception:
        pass

    try:
        import chromadb
        chromadb.PersistentClient(path=settings.chroma_persist_dir)
        chroma_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="ok" if (ollama_ok and chroma_ok) else "degraded",
        ollama_reachable=ollama_ok,
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
