import pytest
import httpx

BASE = "http://localhost:8000"


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
    assert data["target_used"] in ("gemini", "groq", "local")


def test_memory_requires_query(client):
    r = client.get("/memory/test-session-001")
    assert r.status_code == 400


def test_memory_clear(client):
    r = client.delete("/memory/test-session-001")
    assert r.status_code == 200
    assert "deleted" in r.json()
