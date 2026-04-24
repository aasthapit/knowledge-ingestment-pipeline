"""
vector_store.py
Abstraction layer over vector DB backends.

Provides a common VectorStoreClient interface so corpora can target either
the built-in Redis instance, a second parameterized Redis instance, a
user-configured custom HTTP endpoint, or Tachyon (internal GenAI service).

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

    handles_own_embedding: bool = False
    """When True, the backend embeds documents itself; skip the embedder step."""

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
# Redis backend — supports both the default env-var instance and custom instances
# ---------------------------------------------------------------------------

class RedisVectorStore(VectorStoreClient):
    """
    Redis RediSearch backend.

    When instantiated with no parameters, delegates to the module-level
    default client (driven by REDIS_URL / REDIS_INDEX_NAME env vars).

    When instantiated with connection params, creates a dedicated RedisClient
    for that specific Redis instance — enabling multiple Redis targets.
    """

    def __init__(
        self,
        url: str | None = None,
        index_name: str | None = None,
        key_prefix: str | None = None,
        embedding_dims: int | None = None,
    ) -> None:
        from pipeline import redis_store
        if any(p is not None for p in [url, index_name, key_prefix, embedding_dims]):
            from pipeline.config import settings
            self._rc = redis_store.RedisClient(
                url=url or settings.redis_url,
                index_name=index_name or settings.redis_index_name,
                key_prefix=key_prefix or settings.redis_key_prefix,
                embedding_dims=embedding_dims or settings.embedding_dimensions,
            )
        else:
            self._rc = None  # use module-level default

    def _client(self):
        from pipeline import redis_store
        return self._rc if self._rc is not None else redis_store._get_default()

    def ensure_index(self) -> None:
        self._client().create_index()

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._client().upsert_chunks(chunks, vectors)

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        self._client().delete_chunks(chunk_ids)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        tag_filter: list[str] | None = None,
        usecase_id: str | None = None,
        agent_filter: str | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        # tag_filter here may be a pre-built RediSearch filter string or a list
        tf = tag_filter if isinstance(tag_filter, str) else (
            "@tags:{" + "|".join(tag_filter) + "}" if tag_filter else None
        )
        return self._client().search(query_vector, top_k=top_k, tag_filter=tf)


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
# Tachyon backend — internal GenAI service that handles its own vectorization
# ---------------------------------------------------------------------------

class TachyonVectorStore(VectorStoreClient):
    """
    Skeleton client for the internal Tachyon GenAI service.

    Tachyon handles file upload AND vectorization itself, so the embedding
    step is skipped entirely when this backend is in use.

    All methods are stubbed — fill in real HTTP calls once the Tachyon API
    contract is finalised.
    """

    handles_own_embedding: bool = True

    def __init__(self, endpoint: str, api_key: str, collection: str, extra: dict) -> None:
        self._endpoint   = endpoint.rstrip("/")
        self._api_key    = api_key
        self._collection = collection
        self._extra      = extra

    def ensure_index(self) -> None:
        # Tachyon manages its own collections; no explicit index creation needed.
        logger.debug("TachyonVectorStore: index management delegated to Tachyon service.")

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        # vectors are ignored — Tachyon embeds from raw content
        logger.info(
            "TachyonVectorStore: would upload %d chunks to %s (stub — implement when Tachyon API is finalised)",
            len(chunks), self._endpoint or "(no endpoint set)",
        )
        # TODO: POST {endpoint}/upload with raw chunk content once Tachyon API is ready
        # Example payload:
        # {
        #   "collection": self._collection,
        #   "documents": [{"id": c.chunk_id, "content": c.content, "metadata": c.to_dict()} for c in chunks]
        # }

    def delete_chunks(self, chunk_ids: list[str]) -> None:
        logger.info("TachyonVectorStore.delete_chunks: stub — not implemented yet.")

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        tag_filter: list[str] | None = None,
        usecase_id: str | None = None,
        agent_filter: str | None = None,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        logger.info("TachyonVectorStore.search: stub — not implemented yet.")
        return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_vector_store_client(vs_config: dict[str, Any]) -> VectorStoreClient:
    """
    Return the appropriate VectorStoreClient for a VectorStoreConfig dict
    (as returned by VectorStoreConfigStore.get()).

    vs_config must have at least a ``type`` field ("redis", "custom", or "tachyon").

    For parameterized Redis instances, connection details are stored under the
    ``extra`` key: redis_url, index_name, key_prefix, embedding_dims.
    """
    vs_type = (vs_config or {}).get("type", "redis")

    if vs_type == "redis":
        extra = vs_config.get("extra") or {}
        return RedisVectorStore(
            url=extra.get("redis_url") or None,
            index_name=extra.get("index_name") or vs_config.get("collection") or None,
            key_prefix=extra.get("key_prefix") or None,
            embedding_dims=int(extra["embedding_dims"]) if extra.get("embedding_dims") else None,
        )

    if vs_type == "custom":
        return CustomVectorStore(
            endpoint=vs_config.get("endpoint", ""),
            api_key=vs_config.get("api_key", ""),
            collection=vs_config.get("collection", ""),
            extra=vs_config.get("extra") or {},
        )

    if vs_type == "tachyon":
        return TachyonVectorStore(
            endpoint=vs_config.get("endpoint", ""),
            api_key=vs_config.get("api_key", ""),
            collection=vs_config.get("collection", ""),
            extra=vs_config.get("extra") or {},
        )

    raise ValueError(f"Unknown vector store type: {vs_type!r}")
