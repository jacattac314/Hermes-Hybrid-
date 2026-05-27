# Hermes Hybrid Router

A local-first AI router that classifies every request with **Hermes 3 70B** (via Ollama) before dispatching to the best worker: Gemini, Groq, or a local model. Semantic memory is persisted via ChromaDB.

![Architecture](architecture.svg)

## Structure

```
hermes/
├── schemas.py      # Pydantic contracts
├── config.py       # Env-var settings (pydantic-settings)
├── prompts.py      # Router system prompt + worker injection template
├── memory.py       # ChromaDB RAG layer
├── router.py       # Local Hermes 3 → RouterDecision
├── dispatcher.py   # LiteLLM dispatch + fallback chain
└── main.py         # FastAPI app
tests/test_client.py
```

## Quick start

```bash
# 1. Pull models (once)
ollama pull hermes3:70b && ollama pull llama3.2:3b

# 2. Configure
cp .env.example .env  # paste Gemini + Groq keys

# 3. Install deps
pip install -r requirements.txt

# 4. Run
uvicorn hermes.main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/chat` | Route + respond |
| GET | `/health` | Ollama + Chroma reachability |
| GET | `/memory/{session_id}?query=…` | Semantic recall |
| DELETE | `/memory/{session_id}` | Clear session memory |

## Fallback chain

`Gemini → Groq → Local Ollama`

Failures are silent; the next target is tried automatically.
