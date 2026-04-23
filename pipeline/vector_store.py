"""
vector_store.py
Abstraction layer over vector DB backends.

Provides a common VectorStoreClient interface so corpora can target either
the built-in Redis instance or a user-configured custom vector DB endpoint.

Usage:
    from pipeline.vector_store import get_vector_store_client
    client = get_vector_store_client(vs_config)
    client.upsert_chunks(chunks, vectors)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class VectorStoreClient(ABC):
    """Minimal interface that all vector DB backends must implement."""

    @abstractmethod
    def ensure_index(self) -> None:
        """Create or verify the index/collection exists."""

    @abstractmethod
    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Write (or overwrite) chunks with their embedding vectors."""

    @abstractmethod
    def delete_chunks(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        tag_filter: list[str] | None = None,
        usecase_id: str | None = None,
        agent_filter: str | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Vector similarity search; returns list of result dicts."""


# ---------------------------------------------------------------------------
# Redis backend (wraps existing redis_store.py)
# ---------------------------------------------------------------------------

class RedisVectorStore(VectorStoreClient):
    """Delegates to the existing pipeline redis_store module."""

    def ensure_index(self) -> None:
        from pipeline import redis_store
        redis_store.create_index()

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        from pipeline import redis_store
        redis_store.upsert_chunks(chunks, vectors)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        from pipeline import redis_store
        redis_store.delete_chunks(chunk_ids)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        tag_filter: list[str] | None = None,
        usecase_id: str | None = None,
        agent_filter: str | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        from pipeline import redis_store
        return redis_store.search(
            query_vector,
            top_k=top_k,
            tag_filter=tag_filter,
            usecase_id=usecase_id,
            agent_filter=agent_filter,
            source_type=source_type,
        )


# ---------------------------------------------------------------------------
# Custom HTTP backend (OpenAI-compatible /upsert + /search endpoints)
# ---------------------------------------------------------------------------

class CustomVectorStore(VectorStoreClient):
    """
    Generic HTTP-based vector store client.

    Expects the remote endpoint to expose:
      POST {endpoint}/upsert   — body: {collection, chunks: [{id, vector, metadata}]}
      POST {endpoint}/delete   — body: {collection, ids: [str]}
      POST {endpoint}/search   — body: {collection, vector, top_k, filters}

    api_key is sent as Bearer token in Authorization header when set.
    """

    def __init__(self, endpoint: str, api_key: str, collection: str, extra: dict) -> None:
        self._endpoint   = endpoint.rstrip("/")
        self._api_key    = api_key
        self._collection = collection
        self._extra      = extra

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def _post(self, path: str, body: dict) -> dict:
        import json
        import urllib.request

        data = json.dumps(body).encode()
        req  = urllib.request.Request(
            f"{self._endpoint}/{path}",
            data=data,
            headers=self._headers(),
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def ensure_index(self) -> None:
        logger.debug("CustomVectorStore: index management delegated to remote endpoint.")

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        payload = {
            "collection": self._collection,
            "chunks": [
                {
                    "id":       c.chunk_id,
                    "vector":   v,
                    "metadata": {
                        "content": c.content,
                        "source":  c.source,
                        "title":   c.title,
                        "section": c.section,
                        "tags":    c.tags,
                    },
                }
                for c, v in zip(chunks, vectors)
            ],
        }
        self._post("upsert", payload)
        logger.debug("CustomVectorStore: upserted %d chunks to %s", len(chunks), self._endpoint)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        self._post("delete", {"collection": self._collection, "ids": chunk_ids})

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        tag_filter: list[str] | None = None,
        usecase_id: str | None = None,
        agent_filter: str | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        body: dict[str, Any] = {
            "collection": self._collection,
            "vector":     query_vector,
            "top_k":      top_k,
        }
        filters: dict[str, Any] = {}
        if tag_filter:
            filters["tags"] = tag_filter
        if usecase_id:
            filters["usecase_id"] = usecase_id
        if agent_filter:
            filters["agent_filter"] = agent_filter
        if source_type:
            filters["source_type"] = source_type
        if filters:
            body["filters"] = filters
        result = self._post("search", body)
        return result.get("results", [])


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store_client(vs_config: dict[str, Any]) -> VectorStoreClient:
    """
    Return the appropriate VectorStoreClient for a VectorStoreConfig dict
    (as returned by VectorStoreConfigStore.get()).

    vs_config must have at least a ``type`` field ("redis" or "custom").
    """
    vs_type = (vs_config or {}).get("type", "redis")
    if vs_type == "redis":
        return RedisVectorStore()
    if vs_type == "custom":
        return CustomVectorStore(
            endpoint=vs_config.get("endpoint", ""),
            api_key=vs_config.get("api_key", ""),
            collection=vs_config.get("collection", ""),
            extra=vs_config.get("extra") or {},
        )
    raise ValueError(f"Unknown vector store type: {vs_type!r}")
