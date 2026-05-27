from __future__ import annotations

from collections.abc import AsyncIterator

import litellm

from hermes.config import settings
from hermes.schemas import RouterDecision, WorkerTarget

litellm.drop_params = True

# Preferred order: lmstudio first (MacBook Studio / LM Link), then cloud, then bare Ollama
_FALLBACK_CHAIN: list[WorkerTarget] = [
    WorkerTarget.lmstudio,
    WorkerTarget.gemini,
    WorkerTarget.groq,
    WorkerTarget.local,
]

_DEFAULT_SYSTEM = "You are a helpful assistant."


def _model_for(target: WorkerTarget) -> str:
    return {
        WorkerTarget.lmstudio: f"openai/{settings.lmstudio_worker_model}",
        WorkerTarget.gemini: settings.gemini_model,
        WorkerTarget.groq: settings.groq_model,
        WorkerTarget.local: f"ollama/{settings.local_worker_model}",
    }[target]


def _extra_kwargs(target: WorkerTarget) -> dict:
    if target == WorkerTarget.lmstudio:
        return {"api_base": settings.lmstudio_base_url, "api_key": "lm-studio"}
    if target == WorkerTarget.gemini:
        return {"api_key": settings.gemini_api_key}
    if target == WorkerTarget.groq:
        return {"api_key": settings.groq_api_key}
    if target == WorkerTarget.local:
        return {"api_base": settings.ollama_base_url}
    return {}


async def dispatch(
    decision: RouterDecision,
    user_content: str,
    system_prompt: str = _DEFAULT_SYSTEM,
    stream: bool = False,
) -> tuple[str | AsyncIterator[str], WorkerTarget]:
    system = decision.worker_system_prompt or system_prompt
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    start_idx = _FALLBACK_CHAIN.index(decision.target) if decision.target in _FALLBACK_CHAIN else 0
    for target in _FALLBACK_CHAIN[start_idx:]:
        try:
            resp = await litellm.acompletion(
                model=_model_for(target),
                messages=messages,
                timeout=settings.dispatcher_timeout_s,
                stream=stream,
                **_extra_kwargs(target),
            )
            if stream:
                return _stream_chunks(resp), target
            return resp.choices[0].message.content, target
        except Exception:
            if target == WorkerTarget.local:
                raise
            continue

    raise RuntimeError("All dispatch targets failed")


async def _stream_chunks(resp) -> AsyncIterator[str]:
    async for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
