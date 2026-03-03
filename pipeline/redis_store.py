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
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
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
