from __future__ import annotations

import json

import httpx

from hermes.config import settings
from hermes.prompts import ROUTER_SYSTEM_PROMPT
from hermes.schemas import RouterDecision, WorkerTarget


async def route(user_content: str) -> RouterDecision:
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
    data = json.loads(raw)
    return RouterDecision(
        target=WorkerTarget(data["target"]),
        confidence=float(data.get("confidence", 1.0)),
        reasoning=data.get("reasoning", ""),
        worker_system_prompt=data.get("worker_system_prompt"),
    )
