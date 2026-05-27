from __future__ import annotations

import uuid

import chromadb
from chromadb.config import Settings as ChromaSettings

from hermes.config import settings
from hermes.schemas import MemoryChunk


def _get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(settings.memory_collection)


def save(session_id: str, content: str) -> str:
    col = _get_collection()
    chunk_id = str(uuid.uuid4())
    col.add(
        ids=[chunk_id],
        documents=[content],
        metadatas=[{"session_id": session_id}],
    )
    return chunk_id


def retrieve(session_id: str, query: str) -> list[MemoryChunk]:
    col = _get_collection()
    if col.count() == 0:
        return []
    results = col.query(
        query_texts=[query],
        n_results=min(settings.memory_top_k, col.count()),
        where={"session_id": session_id},
    )
    chunks: list[MemoryChunk] = []
    for doc, meta, dist, id_ in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
        results["ids"][0],
    ):
        chunks.append(MemoryChunk(id=id_, content=doc, distance=dist, session_id=meta["session_id"]))
    return chunks


def clear(session_id: str) -> int:
    col = _get_collection()
    existing = col.get(where={"session_id": session_id})
    if not existing["ids"]:
        return 0
    col.delete(ids=existing["ids"])
    return len(existing["ids"])
