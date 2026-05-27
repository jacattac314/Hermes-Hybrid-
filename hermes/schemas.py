from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class WorkerTarget(str, Enum):
    gemini = "gemini"
    groq = "groq"
    lmstudio = "lmstudio"   # LM Studio / LM Link (OpenAI-compat, localhost:1234)
    local = "local"          # bare Ollama fallback


class UserMessage(BaseModel):
    session_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RouterDecision(BaseModel):
    target: WorkerTarget
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    worker_system_prompt: str | None = None


class MemoryChunk(BaseModel):
    id: str
    content: str
    distance: float
    session_id: str


class ChatResponse(BaseModel):
    session_id: str
    content: str
    target_used: WorkerTarget
    memory_chunks_used: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    chroma_reachable: bool
    lmstudio_reachable: bool = False
