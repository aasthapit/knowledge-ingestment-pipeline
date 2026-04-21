"""
qdrant_store.py
Production vector store backed by Qdrant.

Qdrant is the recommended vector DB for production RAG pipelines:
  - HNSW indexing with cosine / dot-product / Euclidean distance
  - Rich payload filtering (tags, source type, date ranges, …)
  - Docker-deployable: docker run -p 6333:6333 qdrant/qdrant
  - Fully managed cloud option available (cloud.qdrant.io)

Each Qdrant point payload mirrors the Chunk fields plus the full Citation
so users can always trace a result back to its source document and page.
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import TYPE_CHECKING, Any

from pipeline.config import settings

if TYPE_CHECKING:
    from pipeline.chunker import Chunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Payload schema (for reference — Qdrant stores arbitrary dicts)
# ---------------------------------------------------------------------------
#
# {
#   "content":              str,
#   "source":               str,
#   "title":                str,
#   "section":              str,
#   "tags":                 list[str],
#   "quality_score":        float,
#   "ingested_at":          int,   # Unix timestamp
#   "citation": {
#     "source_path":        str,
#     "source_type":        str,   # pdf | docx | html | url | markdown
#     "title":              str,
#     "page_number":        int | None,
#     "page_count":         int | None,
#     "author":             str | None,
#     "created_date":       str | None,
#     "url":                str | None,
#   }
# }


def _get_client():
    """Lazy import and instantiate QdrantClient."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is required for Qdrant storage. "
            "Install it with: uv add qdrant-client"
        ) from exc

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key or None,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def ensure_collection(vector_size: int | None = None) -> None:
    """
    Create the Qdrant collection if it does not already exist.

    Parameters
    ----------
    vector_size:
        Embedding dimension. Defaults to ``settings.embedding_dimensions``.
    """
    try:
        from qdrant_client.models import Distance, VectorParams
    except ImportError as exc:
        raise ImportError("qdrant-client is required") from exc

    size = vector_size or settings.embedding_dimensions
    client = _get_client()
    collection = settings.qdrant_collection

    existing = {c.name for c in client.get_collections().collections}
    if collection in existing:
        logger.debug("Qdrant collection '%s' already exists.", collection)
        return

    logger.info("Creating Qdrant collection '%s' (dim=%d, COSINE).", collection, size)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=size, distance=Distance.COSINE),
    )


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_chunks(
    chunks: list["Chunk"],
    embeddings: list[list[float]],
    quality_scores: dict[str, float] | None = None,
) -> None:
    """
    Upsert chunks + their embeddings into Qdrant.

    Parameters
    ----------
    chunks:
        List of :class:`~pipeline.chunker.Chunk` objects.
    embeddings:
        Parallel list of embedding vectors.
    quality_scores:
        Optional mapping of chunk_id → quality score (from the parent doc).
    """
    try:
        from qdrant_client.models import PointStruct
    except ImportError as exc:
        raise ImportError("qdrant-client is required") from exc

    ensure_collection(len(embeddings[0]) if embeddings else None)
    client = _get_client()
    ts = int(time.time())
    qs = quality_scores or {}

    points = []
    for chunk, vector in zip(chunks, embeddings):
        payload: dict[str, Any] = {
            "content":       chunk.content,
            "source":        chunk.source,
            "title":         chunk.title,
            "section":       chunk.section,
            "tags":          chunk.tags,
            "quality_score": qs.get(chunk.chunk_id, 1.0),
            "ingested_at":   ts,
            "citation":      chunk.metadata.get("citation", {}),
        }
        points.append(
            PointStruct(
                id=chunk.chunk_id,   # Qdrant accepts UUID strings
                vector=vector,
                payload=payload,
            )
        )

    # Batch upsert in groups of 100
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=settings.qdrant_collection, points=batch)

    logger.info("Upserted %d chunks into Qdrant collection '%s'.", len(points), settings.qdrant_collection)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query_vector: list[float],
    top_k: int = 5,
    tag_filter: list[str] | None = None,
    source_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    KNN semantic search with optional payload filters.

    Parameters
    ----------
    query_vector:
        Embedding of the user's query.
    top_k:
        Number of results to return.
    tag_filter:
        Require results to have at least one of these tags.
    source_type_filter:
        Restrict results to a specific source type (e.g. ``"pdf"``).
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
    except ImportError as exc:
        raise ImportError("qdrant-client is required") from exc

    client = _get_client()
    conditions = []

    if tag_filter:
        conditions.append(
            FieldCondition(key="tags", match=MatchAny(any=tag_filter))
        )
    if source_type_filter:
        conditions.append(
            FieldCondition(key="citation.source_type", match=MatchValue(value=source_type_filter))
        )

    qfilter = Filter(must=conditions) if conditions else None

    results = client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_vector,
        limit=top_k,
        query_filter=qfilter,
        with_payload=True,
    )

    output = []
    for r in results:
        p = r.payload or {}
        output.append(
            {
                "chunk_id":  str(r.id),
                "score":     round(r.score, 4),
                "content":   p.get("content", ""),
                "source":    p.get("source", ""),
                "title":     p.get("title", ""),
                "section":   p.get("section", ""),
                "tags":      p.get("tags", []),
                "citation":  p.get("citation", {}),
            }
        )
    return output


# ---------------------------------------------------------------------------
# Single-point retrieval
# ---------------------------------------------------------------------------

def get_chunk(chunk_id: str) -> dict[str, Any] | None:
    """Retrieve a single chunk by its UUID."""
    client = _get_client()
    results = client.retrieve(
        collection_name=settings.qdrant_collection,
        ids=[chunk_id],
        with_payload=True,
        with_vectors=False,
    )
    if not results:
        return None
    p = results[0].payload or {}
    return {"chunk_id": chunk_id, **p}


# ---------------------------------------------------------------------------
# Count
# ---------------------------------------------------------------------------

def count() -> int:
    """Return the total number of vectors in the collection."""
    try:
        client = _get_client()
        info = client.get_collection(settings.qdrant_collection)
        return info.points_count or 0
    except Exception as exc:
        logger.warning("Could not get Qdrant point count: %s", exc)
        return -1
