# Contributing to Hermes Hybrid Router

Thanks for your interest in contributing! This guide covers how to get a development
environment running and the conventions used in this project.

## Development setup

```bash
# Clone and enter the repo
git clone https://github.com/jacattac314/Hermes-Hybrid-.git
cd Hermes-Hybrid-

# Create and activate a virtual environment (Python 3.10+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env             # fill in keys / pick ROUTER_BACKEND
```

You also need a router model available. Either run **LM Studio / LM Link** with a Hermes 3
model on `http://localhost:1234/v1` (default), or run **Ollama** and set
`ROUTER_BACKEND=ollama` in `.env`.

## Running the app

```bash
uvicorn hermes.main:app --reload --port 8000
```

Interactive API docs are then available at `http://localhost:8000/docs`.

## Running the tests

The suite in `tests/test_client.py` contains **integration tests** that talk to a live server,
so start the app first, then run pytest in another shell:

```bash
uvicorn hermes.main:app --port 8000      # shell 1
pytest                                   # shell 2
```

`test_chat_roundtrip` requires at least one reachable worker (a local model or a valid
`GEMINI_API_KEY` / `GROQ_API_KEY`).

## Project layout

| Module                 | Responsibility                                             |
| ---------------------- | ---------------------------------------------------------- |
| `hermes/schemas.py`    | Pydantic request/response models and the `WorkerTarget` enum |
| `hermes/config.py`     | Settings loaded from `.env` via pydantic-settings          |
| `hermes/prompts.py`    | Router system prompt + memory-injection template           |
| `hermes/memory.py`     | ChromaDB save / retrieve / clear                           |
| `hermes/router.py`     | Classifies a request into a `RouterDecision`               |
| `hermes/dispatcher.py` | Sends the request to a worker with fallback                |
| `hermes/main.py`       | FastAPI app and HTTP endpoints                             |

## Conventions

- **Python style:** follow PEP 8; keep functions small and typed. The codebase uses
  `from __future__ import annotations` and modern `X | Y` type hints.
- **Configuration:** never hard-code secrets or hosts — add a setting to `hermes/config.py`
  and document it in `.env.example`.
- **Adding a worker target:** extend the `WorkerTarget` enum, add a model mapping in
  `dispatcher._model_for`, credentials in `dispatcher._extra_kwargs`, and place it in
  `_FALLBACK_CHAIN`.
- **Secrets:** `.env` and the `.chroma/` directory are git-ignored. Never commit real API keys.

## Pull requests

1. Create a feature branch from the default branch.
2. Keep commits focused and write clear, present-tense commit messages.
3. Make sure the app starts and the tests pass before opening the PR.
4. Describe what changed and why in the PR description.
