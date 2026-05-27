from __future__ import annotations

import json

import httpx

from hermes.config import settings
from hermes.prompts import ROUTER_SYSTEM_PROMPT
from hermes.schemas import RouterDecision, WorkerTarget


async def route(user_content: str) -> RouterDecision:
    if settings.router_backend == "lmstudio":
        return await _route_lmstudio(user_content)
    return await _route_ollama(user_content)


async def _route_lmstudio(user_content: str) -> RouterDecision:
    payload = {
        "model": settings.lmstudio_router_model,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=settings.router_timeout_s) as client:
        resp = await client.post(
            f"{settings.lmstudio_base_url}/chat/completions",
            json=payload,
            headers={"Authorization": "Bearer lm-studio"},
        )
        resp.raise_for_status()

    raw = resp.json()["choices"][0]["message"]["content"]
    return _parse(raw)


async def _route_ollama(user_content: str) -> RouterDecision:
    payload = {
        "model": settings.router_model,
        "messages": [
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "format": "json",
    }
    async with httpx.AsyncClient(timeout=settings.router_timeout_s) as client:
        resp = await client.post(
            f"{settings.ollama_base_url}/api/chat",
            json=payload,
        )
        resp.raise_for_status()

    raw = resp.json()["message"]["content"]
    return _parse(raw)


def _parse(raw: str) -> RouterDecision:
    data = json.loads(raw)
    return RouterDecision(
        target=WorkerTarget(data["target"]),
        confidence=float(data.get("confidence", 1.0)),
        reasoning=data.get("reasoning", ""),
        worker_system_prompt=data.get("worker_system_prompt"),
    )
