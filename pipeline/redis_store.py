"""
redis_store.py
Manages the Redis vector index (RediSearch / Redis Stack compatible) and
provides upsert + similarity search operations.

Redis Enterprise requires the Search module to be enabled on the database.
"""
from __future__ import annotations

import json
import logging
import struct
from typing import TYPE_CHECKING, Any

import redis
from redis.commands.search.field import (
    NumericField,
    TagField,
    TextField,
    VectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from pipeline.config import settings

if TYPE_CHECKING:
    from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _index_schema() -> list:
    dims = settings.embedding_dimensions
    return [
        TextField("$.source",  as_name="source",  no_stem=True),
        TextField("$.title",   as_name="title"),
        TextField("$.section", as_name="section"),
        TextField("$.content", as_name="content"),
        TagField( "$.tags.*",  as_name="tags"),
        NumericField("$.metadata.ingested_at", as_name="ingested_at", sortable=True),
        VectorField(
            "$.embedding",
            "FLAT",
            {
                "TYPE": "FLOAT32",
                "DIM": dims,
                "DISTANCE_METRIC": "COSINE",
            },
            as_name="embedding",
        ),
    ]


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def get_client() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=False)


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

def create_index(client: redis.Redis | None = None) -> None:
    """Create the RediSearch index if it does not already exist."""
    client = client or get_client()
    index_name = settings.redis_index_name
    try:
        client.ft(index_name).info()
        logger.info("Index '%s' already exists — skipping creation.", index_name)
    except redis.ResponseError:
        logger.info("Creating index '%s' …", index_name)
        client.ft(index_name).create_index(
            fields=_index_schema(),
            definition=IndexDefinition(
                prefix=[settings.redis_key_prefix],
                index_type=IndexType.JSON,
            ),
        )
        logger.info("Index created.")


def drop_index(client: redis.Redis | None = None, delete_docs: bool = False) -> None:
    """Drop the RediSearch index (optionally also delete all indexed documents)."""
    client = client or get_client()
    try:
        client.ft(settings.redis_index_name).dropindex(delete_documents=delete_docs)
        logger.info("Index '%s' dropped.", settings.redis_index_name)
    except redis.ResponseError as exc:
        logger.warning("Could not drop index: %s", exc)


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def _pack_embedding(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def upsert_chunks(
    chunks: list["Chunk"],
    embeddings: list[list[float]],
    client: redis.Redis | None = None,
) -> None:
    """
    Store each chunk + its embedding as a JSON document in Redis.
    Uses the chunk_id as the document key.
    """
    import time

    client = client or get_client()
    pipe = client.pipeline(transaction=False)
    ts = int(time.time())

    for chunk, vector in zip(chunks, embeddings):
        key = f"{settings.redis_key_prefix}{chunk.chunk_id}"
        doc = chunk.to_dict()
        doc["embedding"] = vector          # stored as JSON array for RedisJSON
        doc["metadata"]["ingested_at"] = ts
        pipe.json().set(key, "$", doc)

    pipe.execute()
    logger.info("Upserted %d chunks into Redis.", len(chunks))


# ---------------------------------------------------------------------------
# Query / retrieval
# ---------------------------------------------------------------------------

def search(
    query_vector: list[float],
    top_k: int = 5,
    tag_filter: str | None = None,
    client: redis.Redis | None = None,
) -> list[dict[str, Any]]:
    """
    KNN vector search with optional tag pre-filter.

    Parameters
    ----------
    query_vector: Embedding of the query.
    top_k:        Number of results to return.
    tag_filter:   RediSearch tag filter string, e.g. ``"@tags:{python|redis}"``.
    """
    client = client or get_client()
    dims = len(query_vector)
    blob = struct.pack(f"{dims}f", *query_vector)

    base_filter = tag_filter if tag_filter else "*"
    q_str = f"({base_filter})=>[KNN {top_k} @embedding $vec AS score]"

    q = (
        Query(q_str)
        .sort_by("score")
        .return_fields("source", "title", "section", "content", "tags", "score")
        .dialect(2)
        .paging(0, top_k)
    )

    results = client.ft(settings.redis_index_name).search(q, query_params={"vec": blob})

    output = []
    for doc in results.docs:
        output.append(
            {
                "chunk_id": doc.id.removeprefix(settings.redis_key_prefix),
                "source": getattr(doc, "source", ""),
                "title": getattr(doc, "title", ""),
                "section": getattr(doc, "section", ""),
                "content": getattr(doc, "content", ""),
                "tags": getattr(doc, "tags", ""),
                "score": float(getattr(doc, "score", 0)),
            }
        )
    return output


def get_chunk(chunk_id: str, client: redis.Redis | None = None) -> dict[str, Any] | None:
    """Retrieve a single chunk by its ID."""
    client = client or get_client()
    key = f"{settings.redis_key_prefix}{chunk_id}"
    raw = client.json().get(key)
    return raw


def delete_chunks(
    chunk_ids: list[str],
    client: redis.Redis | None = None,
) -> int:
    """
    Delete chunks from Redis by their chunk_ids.
    Returns the number of keys actually deleted.
    """
    if not chunk_ids:
        return 0
    client = client or get_client()
    keys = [f"{settings.redis_key_prefix}{cid}" for cid in chunk_ids]
    deleted = client.delete(*keys)
    logger.info("Deleted %d/%d chunks from Redis.", deleted, len(chunk_ids))
    return deleted


def update_tags(
    chunk_id: str,
    tags: list[str],
    client: redis.Redis | None = None,
) -> None:
    """Overwrite the tags on an existing chunk."""
    client = client or get_client()
    key = f"{settings.redis_key_prefix}{chunk_id}"
    client.json().set(key, "$.tags", tags)
    logger.debug("Updated tags for %s → %s", chunk_id, tags)


# ---------------------------------------------------------------------------
# StagingStore — review queue for documents before they reach the vector DB
# ---------------------------------------------------------------------------
#
# Redis key layout:
#   review:queue               List   — doc IDs waiting for review (FIFO)
#   review:doc:{id}            Hash   — document metadata + quality info
#   review:chunks:{id}         List   — JSON-encoded Chunk dicts (pre-embedding)
#   review:pending             Set    — doc IDs with status "pending_review"
#   review:approved            Set    — doc IDs approved for push
#   review:rejected            Set    — doc IDs rejected

_Q_QUEUE    = "review:queue"
_Q_PENDING  = "review:pending"
_Q_APPROVED = "review:approved"
_Q_REJECTED = "review:rejected"


class StagingStore:
    """
    Manages the document review staging area in Redis.

    Documents are staged here before being embedded and pushed to the
    production vector store (Redis RediSearch or Qdrant).
    """

    def __init__(self, client: redis.Redis | None = None) -> None:
        self._r = client or get_client()

    # ── Enqueue / stage ──────────────────────────────────────────────────

    def enqueue(
        self,
        doc_id: str,
        meta: dict,
        chunks: list[dict],
    ) -> None:
        """
        Stage a document for review.

        Parameters
        ----------
        doc_id:  Stable identifier for this document (e.g. UUID5 of source path).
        meta:    Flat dict with at minimum: title, source_path, source_type,
                 quality_score, quality_flags (JSON str), quality_passed (0/1),
                 chunk_count.
        chunks:  List of Chunk.to_dict() dicts — stored as JSON strings.
        """
        import json as _json

        pipe = self._r.pipeline(transaction=True)

        # Document metadata hash
        meta_key = f"review:doc:{doc_id}"
        pipe.hset(meta_key, mapping={k: str(v) for k, v in meta.items()})

        # Chunks list (each chunk serialised as JSON)
        chunks_key = f"review:chunks:{doc_id}"
        pipe.delete(chunks_key)
        for chunk in chunks:
            pipe.rpush(chunks_key, _json.dumps(chunk, ensure_ascii=False))

        # Enqueue to the FIFO review queue and pending set
        pipe.rpush(_Q_QUEUE, doc_id)
        pipe.sadd(_Q_PENDING, doc_id)

        pipe.execute()
        logger.info("Staged %d chunks for doc '%s' (id=%s).", len(chunks), meta.get("title", "?"), doc_id)

    # ── Status transitions ───────────────────────────────────────────────

    def approve(self, doc_id: str) -> None:
        """Mark a document as approved (ready to push to vector store)."""
        pipe = self._r.pipeline(transaction=True)
        pipe.srem(_Q_PENDING, doc_id)
        pipe.srem(_Q_REJECTED, doc_id)
        pipe.sadd(_Q_APPROVED, doc_id)
        pipe.hset(f"review:doc:{doc_id}", "status", "approved")
        pipe.execute()
        logger.info("Approved doc %s.", doc_id)

    def reject(self, doc_id: str, reason: str = "") -> None:
        """Mark a document as rejected and record the reason."""
        pipe = self._r.pipeline(transaction=True)
        pipe.srem(_Q_PENDING, doc_id)
        pipe.srem(_Q_APPROVED, doc_id)
        pipe.sadd(_Q_REJECTED, doc_id)
        pipe.hset(f"review:doc:{doc_id}", mapping={"status": "rejected", "reject_reason": reason})
        pipe.execute()
        logger.info("Rejected doc %s (%s).", doc_id, reason or "no reason given")

    # ── Retrieval ────────────────────────────────────────────────────────

    def get_pending(self) -> list[str]:
        """Return all doc IDs currently in the pending-review set."""
        return [v.decode() if isinstance(v, bytes) else v
                for v in self._r.smembers(_Q_PENDING)]

    def get_approved(self) -> list[str]:
        """Return all doc IDs approved for pushing."""
        return [v.decode() if isinstance(v, bytes) else v
                for v in self._r.smembers(_Q_APPROVED)]

    def get_doc_meta(self, doc_id: str) -> dict | None:
        """Return the metadata hash for a staged document, or None if not found."""
        key = f"review:doc:{doc_id}"
        raw = self._r.hgetall(key)
        if not raw:
            return None
        return {
            (k.decode() if isinstance(k, bytes) else k):
            (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }

    def get_chunks(self, doc_id: str) -> list[dict]:
        """Return the list of staged Chunk dicts for a document."""
        import json as _json

        key = f"review:chunks:{doc_id}"
        raw_list = self._r.lrange(key, 0, -1)
        return [_json.loads(item) for item in raw_list]

    def list_all(self) -> list[dict]:
        """
        Return a summary list of ALL staged documents (pending + approved + rejected).
        Useful for the ``review list`` CLI command.
        """
        import json as _json

        all_ids: set[str] = set()
        for key in (self._r.smembers(_Q_PENDING) or set()):
            all_ids.add(key.decode() if isinstance(key, bytes) else key)
        for key in (self._r.smembers(_Q_APPROVED) or set()):
            all_ids.add(key.decode() if isinstance(key, bytes) else key)
        for key in (self._r.smembers(_Q_REJECTED) or set()):
            all_ids.add(key.decode() if isinstance(key, bytes) else key)

        results = []
        for doc_id in sorted(all_ids):
            meta = self.get_doc_meta(doc_id)
            if meta:
                meta["doc_id"] = doc_id
                # Deserialise quality_flags for display
                try:
                    meta["quality_flags"] = _json.loads(meta.get("quality_flags", "[]"))
                except Exception:
                    meta["quality_flags"] = []
                results.append(meta)
        return results

    # ── Cleanup ──────────────────────────────────────────────────────────

    def remove_doc(self, doc_id: str) -> None:
        """
        Remove all staging data for a document (call after successful push
        or explicit deletion).
        """
        pipe = self._r.pipeline(transaction=True)
        pipe.delete(f"review:doc:{doc_id}")
        pipe.delete(f"review:chunks:{doc_id}")
        pipe.srem(_Q_PENDING, doc_id)
        pipe.srem(_Q_APPROVED, doc_id)
        pipe.srem(_Q_REJECTED, doc_id)
        pipe.execute()
        logger.debug("Removed staging data for doc %s.", doc_id)


def get_staging(client: redis.Redis | None = None) -> StagingStore:
    """Convenience factory — returns a :class:`StagingStore` instance."""
    return StagingStore(client=client)
