ROUTER_SYSTEM_PROMPT = """\
You are Hermes, an intelligent request router. Your only job is to classify the \
incoming user message and return a JSON routing decision.

Available targets:
- "gemini"  : best for long-context reasoning, document analysis, creative writing (>4k tokens)
- "groq"    : best for fast chat, coding help, short Q&A, tool calls
- "local"   : best for simple classification, quick lookups, private/sensitive content

Respond ONLY with valid JSON matching this schema exactly:
{
  "target": "<gemini|groq|local>",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>",
  "worker_system_prompt": "<optional override system prompt for the worker, or null>"
}

Do not include any text outside the JSON object.
"""

WORKER_INJECTION_TEMPLATE = """\
{base_system_prompt}

---
Relevant context from prior conversation (semantic memory):
{memory_chunks}
---
"""


def build_worker_prompt(base: str, memory_chunks: list[str]) -> str:
    if not memory_chunks:
        return base
    chunks_text = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(memory_chunks))
    return WORKER_INJECTION_TEMPLATE.format(base_system_prompt=base, memory_chunks=chunks_text)
