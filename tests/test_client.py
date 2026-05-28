"""Integration tests for the Hermes Hybrid Router.

These tests hit a LIVE server, so start the app before running them:

    uvicorn hermes.main:app --port 8000   # shell 1
    pytest                                # shell 2

test_chat_roundtrip also requires at least one reachable worker (a local model,
or a valid GEMINI_API_KEY / GROQ_API_KEY).
"""
import pytest
import httpx

BASE = "http://localhost:8000"

# Valid worker targets the router/dispatcher may report (see hermes.schemas.WorkerTarget)
VALID_TARGETS = ("lmstudio", "gemini", "groq", "local")


@pytest.fixture
def client():
    with httpx.Client(base_url=BASE, timeout=30.0) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "ollama_reachable" in data
    assert "lmstudio_reachable" in data
    assert "chroma_reachable" in data


def test_chat_roundtrip(client):
    payload = {
        "session_id": "test-session-001",
        "content": "What is 2 + 2?",
    }
    r = client.post("/chat", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["session_id"] == "test-session-001"
    assert len(data["content"]) > 0
    assert data["target_used"] in VALID_TARGETS


def test_memory_requires_query(client):
    r = client.get("/memory/test-session-001")
    assert r.status_code == 400


def test_memory_clear(client):
    r = client.delete("/memory/test-session-001")
    assert r.status_code == 200
    assert "deleted" in r.json()
